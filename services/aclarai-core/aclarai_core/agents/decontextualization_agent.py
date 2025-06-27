"""
Agent for evaluating the decontextualization of a claim.
"""

import logging
from typing import Optional, Tuple

from aclarai_shared.config import aclaraiConfig
from aclarai_shared.tools.factory import ToolFactory
from aclarai_shared.utils.prompt_loader import load_prompt_template
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import LLM
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class DecontextualizationAgent:
    """
    Agent to assess if a claim can be understood independently of its source.
    """

    def __init__(self, llm: LLM, tool_factory: ToolFactory, config: aclaraiConfig):
        """
        Initializes the DecontextualizationAgent.
        """
        self.llm = llm
        self.config = config

        # The agent now simply asks for its tools by its role name.
        # The factory handles building the correct tools based on the YAML config.
        self.tools = tool_factory.get_tools_for_agent("decontextualization_agent")

        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm,
            verbose=True,
        )
        self.max_retries = self.config.processing.retries.get("max_attempts", 3)

    async def evaluate_claim_decontextualization(
        self, claim_id: str, claim_text: str, source_id: str, source_text: str
    ) -> Tuple[Optional[float], str]:
        """
        Evaluates the decontextualization of a given claim.

        Args:
            claim_id: The unique identifier for the claim.
            claim_text: The text of the claim.
            source_id: The unique identifier for the source of the claim.
            source_text: The text of the source from which the claim was derived.

        Returns:
            A tuple containing the decontextualization score (float between 0.0 and 1.0, or None if an error occurs)
            and a string detailing any error messages or 'success'.
        """
        log_details = {
            "service": "aclarai-core",
            "filename_function_name": "decontextualization_agent.DecontextualizationAgent.evaluate_claim_decontextualization",
            "aclarai_id_claim": claim_id,
            "aclarai_id_source": source_id,
        }

        try:
            # Load and format the prompt from the external YAML file.
            prompt = load_prompt_template(
                "decontextualization_evaluation",
                claim_text=claim_text,
                source_text=source_text,
            )
        except (FileNotFoundError, ValueError) as e:
            error_message = f"Failed to load decontextualization prompt template: {e}"
            logger.error(error_message, extra=log_details)
            return None, error_message

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=4, max=10),
        )
        async def _attempt_evaluation_with_prompt(current_prompt: str) -> str:
            """Handle the actual LLM call with retries, ensuring a string response."""
            chat_response = await self.agent.achat(current_prompt)

            if isinstance(chat_response.response, str):
                return chat_response.response.strip()

            raise ValueError(
                f"LLM did not return a string response. Got: {chat_response.response}"
            )

        try:
            logger.info(
                f"Evaluating decontextualization for claim_id: {claim_id} with max_retries: {self.max_retries}",
                extra=log_details,
            )
            response_text = await _attempt_evaluation_with_prompt(prompt)

            try:
                score = float(response_text)
                if not (0.0 <= score <= 1.0):
                    raise ValueError("Score out of range.")
                logger.info(
                    f"Successfully evaluated decontextualization for claim_id: {claim_id}. Score: {score}",
                    extra={**log_details, "decontextualization_score": score},
                )
                return score, "success"
            except (ValueError, TypeError):
                error_message = (
                    f"LLM returned an invalid score format or out-of-range value for claim_id {claim_id}. "
                    f"Response: '{response_text}'"
                )
                logger.warning(error_message, extra=log_details)
                return None, error_message

        except Exception as e:
            final_exception: BaseException = e
            # When tenacity raises RetryError, the original exception is stored in `__cause__`.
            if isinstance(e, RetryError) and e.__cause__:
                final_exception = e.__cause__

            error_message = (
                f"Error during decontextualization evaluation for claim_id {claim_id} "
                f"after {self.max_retries} retries. Root cause: {final_exception!r}"
            )

            logger.error(error_message, exc_info=True, extra=log_details)
            return None, error_message
