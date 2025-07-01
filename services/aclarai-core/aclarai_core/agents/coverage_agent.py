"""
Agent for evaluating the coverage of a claim relative to its source.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from aclarai_shared.config import aclaraiConfig
from aclarai_shared.tools.factory import ToolFactory
from aclarai_shared.utils.prompt_loader import load_prompt_template

# Using CodeActAgent for multi-step reasoning as suggested by the technical details
from llama_index.core.agent.workflow import CodeActAgent
from llama_index.core.llms.llm import LLM
from llama_index.core.tools import BaseTool
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential


class DummyCodeExecutor:
    def __init__(self):
        pass

    def execute_code(self, code_string: str) -> Tuple[str, Any]:
        logger.warning(f"CodeActAgent attempted to execute code: {code_string}")
        return "Code execution not supported by this agent.", {}


logger = logging.getLogger(__name__)


class CoverageAgent:
    """
    Agent to assess how completely a claim captures verifiable information from its source.
    Uses a CodeActAgent to allow for multi-step reasoning and tool use.
    """

    def __init__(self, llm: LLM, tool_factory: ToolFactory, config: aclaraiConfig):
        """
        Initializes the CoverageAgent.
        """
        self.llm = llm
        self.config = config
        # Cast the return type to match the expected type for CodeActAgent
        self.tools: list[BaseTool | Callable[..., Any]] = list(
            tool_factory.get_tools_for_agent("coverage_agent")
        )

        # The synchronous `agent.chat()` method is used here, which works well with
        # both synchronous and asynchronous tools.
        self.agent = CodeActAgent(
            llm=self.llm,
            tools=self.tools,
            code_execute_fn=DummyCodeExecutor().execute_code,
            # system_prompt will be part of the loaded prompt template
        )
        self.max_retries = self.config.processing.retries.get("max_attempts", 3)
        self.tool_names = ", ".join(
            [
                tool.metadata.name
                for tool in self.tools
                if isinstance(tool, BaseTool) and tool.metadata.name is not None
            ]
        )

    def evaluate_coverage(
        self,
        claim_id: str,
        claim_text: str,
        source_id: str,
        source_text: str,
    ) -> Tuple[Optional[float], Optional[List[Dict[str, str]]], str]:
        """
        Evaluates the coverage of a given claim by its source.

        Args:
            claim_id: The unique identifier for the claim.
            claim_text: The text of the claim.
            source_id: The unique identifier for the source of the claim.
            source_text: The text of the source from which the claim was derived.

        Returns:
            A tuple containing:
            - coverage_score: float between 0.0 and 1.0, or None if an error occurs
            - omitted_elements: List of dicts with 'text' and 'significance' keys, or None if error
            - status_message: string detailing any error messages or 'success'
        """
        log_details = {
            "service": "aclarai-core",
            "filename_function_name": "coverage_agent.CoverageAgent.evaluate_coverage",
            "aclarai_id_claim": claim_id,
            "aclarai_id_source": source_id,
        }

        try:
            # Load and format the prompt from the external YAML file.
            # The prompt template itself contains the system prompt and user instructions.
            # The CodeActAgent will use the system prompt from the template if provided.
            # The user part of the prompt is what we pass to agent.chat()
            prompt_template_data = load_prompt_template(
                "coverage_evaluation",
                return_dict=True,  # Get the whole prompt structure
                source_text=source_text,
                claim_text=claim_text,
            )

            if (
                not isinstance(prompt_template_data, dict)
                or "template" not in prompt_template_data
            ):
                raise ValueError(
                    "Prompt template for coverage_evaluation is not a dictionary or is missing 'template' key."
                )

            # The 'template' key holds the user-facing part of the prompt after formatting
            user_prompt = prompt_template_data["template"]
            system_prompt = prompt_template_data.get("system_prompt")

            # Update agent's system prompt if provided by the loaded template
            if system_prompt:
                self.agent.update_prompts({"agent_worker:system_prompt": system_prompt})
            else:
                logger.warning(
                    "No system_prompt found in coverage_evaluation.yaml, CodeActAgent might use a default one.",
                    extra=log_details,
                )

        except (FileNotFoundError, ValueError) as e:
            error_message = f"Failed to load coverage prompt template: {e}"
            logger.error(error_message, extra=log_details)
            return None, None, error_message

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=4, max=10),
        )
        def _attempt_evaluation_with_prompt(current_user_prompt: str) -> str:
            """Handle the actual LLM call with retries, ensuring a string response."""
            # CodeActAgent's `chat` method returns an AgentChatResponse
            # The actual response string is in response.response
            logger.debug(
                f"Calling CodeActAgent with prompt: {current_user_prompt}",
                extra=log_details,
            )
            result = self.agent.chat(current_user_prompt)

            if hasattr(result, "response") and isinstance(result.response, str):
                response_str = result.response.strip()
                logger.info(
                    f"Raw response from CodeActAgent: '{response_str}'",
                    extra=log_details,
                )
                return response_str
            else:
                # Fallback or error if the response structure is not as expected
                logger.error(
                    f"LLM did not return expected response structure. Got: {result}",
                    extra=log_details,
                )
                raise ValueError(
                    f"LLM did not return a string response. Got: {type(result)}"
                )

        try:
            logger.info(
                f"Evaluating coverage for claim_id: {claim_id} with source_id: {source_id}, max_retries: {self.max_retries}",
                extra=log_details,
            )
            response_text = _attempt_evaluation_with_prompt(user_prompt)

            # The prompt asks for JSON output with coverage_score and omitted_elements
            try:
                response_data = json.loads(response_text)

                if not isinstance(response_data, dict):
                    raise ValueError("Response is not a JSON object")

                # Extract coverage score
                coverage_score = response_data.get("coverage_score")
                if coverage_score is None or not isinstance(
                    coverage_score, (int, float)
                ):
                    raise ValueError("Missing or invalid coverage_score in response")

                coverage_score = float(coverage_score)
                if not (0.0 <= coverage_score <= 1.0):
                    # Score out of range, treat as an invalid response
                    error_message = (
                        f"LLM returned an out-of-range coverage score for claim_id {claim_id}. "
                        f"Score: {coverage_score}, Response: '{response_text}'"
                    )
                    logger.warning(error_message, extra=log_details)
                    return None, None, error_message

                # Extract omitted elements
                omitted_elements = response_data.get("omitted_elements", [])
                if not isinstance(omitted_elements, list):
                    raise ValueError("omitted_elements must be a list")

                # Validate each omitted element has required fields
                validated_elements = []
                for element in omitted_elements:
                    if isinstance(element, dict) and "text" in element:
                        validated_elements.append(
                            {
                                "text": str(element["text"]),
                                "significance": str(element.get("significance", "")),
                            }
                        )
                    else:
                        logger.warning(
                            f"Invalid omitted element format: {element}. Skipping.",
                            extra=log_details,
                        )

                logger.info(
                    f"Successfully evaluated coverage for claim_id: {claim_id}. Score: {coverage_score}, Omitted elements: {len(validated_elements)}",
                    extra={
                        **log_details,
                        "coverage_score": coverage_score,
                        "omitted_elements_count": len(validated_elements),
                    },
                )
                return coverage_score, validated_elements, "success"

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # This handles cases where response_text is not valid JSON or missing required fields
                error_message = (
                    f"LLM returned an invalid JSON response for claim_id {claim_id}. "
                    f"Error: {e}, Response: '{response_text}'"
                )
                logger.warning(error_message, extra=log_details)
                return None, None, error_message

        except Exception as e:
            final_exception: BaseException = e
            if isinstance(e, RetryError) and e.__cause__:
                final_exception = e.__cause__

            error_message = (
                f"Error during coverage evaluation for claim_id {claim_id} "
                f"after {self.max_retries} retries. Root cause: {final_exception!r}"
            )
            logger.error(error_message, exc_info=True, extra=log_details)
            return None, None, error_message
