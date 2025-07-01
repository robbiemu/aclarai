"""
Agent for evaluating the entailment of a claim by its source.
"""

import logging
from typing import Any, Callable, Optional, Tuple

from aclarai_shared.config import aclaraiConfig
from aclarai_shared.tools.factory import ToolFactory
from aclarai_shared.utils.prompt_loader import load_prompt_template

# Using CodeActAgent as suggested by the issue's technical details for multi-step reasoning
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


class EntailmentAgent:
    """
    Agent to assess if a source text logically entails a claim.
    Uses a CodeActAgent to allow for multi-step reasoning and tool use.
    """

    def __init__(self, llm: LLM, tool_factory: ToolFactory, config: aclaraiConfig):
        """
        Initializes the EntailmentAgent.
        """
        self.llm = llm
        self.config = config
        # Cast the return type to match the expected type for CodeActAgent
        self.tools: list[BaseTool | Callable[..., Any]] = list(
            tool_factory.get_tools_for_agent("entailment_agent")
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

    def evaluate_entailment(
        self,
        claim_id: str,
        claim_text: str,
        source_id: str,
        source_text: str,
        # source_filepath: str, # Added to plan, but not used by agent itself, consumer uses it.
    ) -> Tuple[Optional[float], str]:
        """
        Evaluates the entailment of a given claim by its source.

        Args:
            claim_id: The unique identifier for the claim.
            claim_text: The text of the claim (hypothesis).
            source_id: The unique identifier for the source of the claim.
            source_text: The text of the source from which the claim was derived (premise).
            # source_filepath: Path to the markdown file of the source block.

        Returns:
            A tuple containing the entailment score (float between 0.0 and 1.0,
            or None if an error occurs) and a string detailing any error messages or 'success'.
        """
        log_details = {
            "service": "aclarai-core",
            "filename_function_name": "entailment_agent.EntailmentAgent.evaluate_entailment",
            "aclarai_id_claim": claim_id,
            "aclarai_id_source": source_id,
        }

        try:
            # Load and format the prompt from the external YAML file.
            # The prompt template itself contains the system prompt and user instructions.
            # The CodeActAgent will use the system prompt from the template if provided.
            # The user part of the prompt is what we pass to agent.chat()
            prompt_template_data = load_prompt_template(
                "entailment_evaluation",
                return_dict=True,  # Get the whole prompt structure
                source_text=source_text,
                claim_text=claim_text,
                # Potentially inject tool descriptions if prompt needs them explicitly
                # tool_descriptions = self.agent.get_tool_descriptions() # If needed
            )

            if (
                not isinstance(prompt_template_data, dict)
                or "template" not in prompt_template_data
            ):
                raise ValueError(
                    "Prompt template for entailment_evaluation is not a dictionary or is missing 'template' key."
                )

            # The 'template' key holds the user-facing part of the prompt after formatting
            user_prompt = prompt_template_data["template"]
            system_prompt = prompt_template_data.get("system_prompt")

            # Update agent's system prompt if provided by the loaded template
            if system_prompt:
                self.agent.update_prompts({"agent_worker:system_prompt": system_prompt})
            else:
                logger.warning(
                    "No system_prompt found in entailment_evaluation.yaml, CodeActAgent might use a default one.",
                    extra=log_details,
                )

        except (FileNotFoundError, ValueError) as e:
            error_message = f"Failed to load entailment prompt template: {e}"
            logger.error(error_message, extra=log_details)
            return None, error_message

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
                # As per on-llm_interaction_strategy.md, agent's final output is via print(json.dumps(...))
                # The CodeActAgent might directly return the JSON string if prompted correctly,
                # or it might return the content of the last print statement.
                # The prompt asks for "Output only the float score."
                # If the LLM follows this, result.response will be the score string.
                # If it's a CodeActAgent that *executes* code to print JSON,
                # we might need to parse `result.sources[-1].raw_output` if it's a tool call that prints.
                # However, the prompt instructs "Output only the float score", implying direct output.
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
                f"Evaluating entailment for claim_id: {claim_id} with source_id: {source_id}, max_retries: {self.max_retries}",
                extra=log_details,
            )
            response_text = _attempt_evaluation_with_prompt(user_prompt)

            # The prompt asks for "Output only the float score."
            # So, response_text should be the score itself.
            try:
                score = float(response_text)
                if not (0.0 <= score <= 1.0):
                    # Score out of range, treat as an invalid response
                    error_message = (
                        f"LLM returned an out-of-range score for claim_id {claim_id}. "
                        f"Score: {score}, Response: '{response_text}'"
                    )
                    logger.warning(error_message, extra=log_details)
                    return None, error_message  # Treat as failure to get valid score

                logger.info(
                    f"Successfully evaluated entailment for claim_id: {claim_id}. Score: {score}",
                    extra={**log_details, "entailed_score": score},
                )
                return score, "success"
            except (ValueError, TypeError):
                # This handles cases where response_text is not a float.
                # This could be due to the LLM not following instructions, or returning complex JSON.
                # For now, if it's not a direct float, we treat it as an error for this agent's design.
                error_message = (
                    f"LLM returned an invalid score format for claim_id {claim_id}. "
                    f"Expected a float, got: '{response_text}'"
                )
                logger.warning(error_message, extra=log_details)
                # This is a critical failure in parsing the expected output.
                return None, error_message

        except Exception as e:
            final_exception: BaseException = e
            if isinstance(e, RetryError) and e.__cause__:
                final_exception = e.__cause__

            error_message = (
                f"Error during entailment evaluation for claim_id {claim_id} "
                f"after {self.max_retries} retries. Root cause: {final_exception!r}"
            )
            logger.error(error_message, exc_info=True, extra=log_details)
            return None, error_message
