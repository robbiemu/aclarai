"""
Agent for evaluating the decontextualization of a claim.
"""

import logging
from typing import Any, Optional, Tuple

from aclarai_core.utils.config_manager import ConfigManager
from aclarai_shared.tools.factory import ToolFactory
from llama_index.core.agent import ReActAgent
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.llms import LLM
from llama_index.core.tools import BaseTool, ToolMetadata
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class DecontextualizationAgent:
    """
    Agent to assess if a claim can be understood independently of its source.
    """

    def __init__(
        self, llm: LLM, tool_factory: ToolFactory, config_manager: ConfigManager
    ):
        """
        Initializes the DecontextualizationAgent.

        Args:
            llm: The language model to use.
            tool_factory: Factory to create necessary tools.
            config_manager: Manages access to system configurations.
        """
        self.llm = llm
        self.config_manager = config_manager
        self.tools = self._setup_tools(tool_factory)
        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm,
            verbose=True,  # Consider making this configurable
        )
        self.max_retries = int(
            self.config_manager.get("processing.retries.max_attempts", "3")
        )

    def _setup_tools(self, tool_factory: ToolFactory) -> list[BaseTool]:
        """
        Sets up the tools required by the agent.
        Currently, this uses the VectorSearchTool for 'utterances'.
        """
        vector_search_tool_metadata = ToolMetadata(
            name="vector_search_utterances",
            description=(
                "Searches for similar utterances in the knowledge base. "
                "Use this tool to find existing utterances that are semantically "
                "similar to the input claim. This helps determine if the claim "
                "is ambiguous or relies on unstated context by checking if "
                "very similar phrases appear in diverse contexts."
            ),
        )
        # Ensure the 'utterances' collection is specified for the tool.
        # The ToolFactory should handle this based on config or specific tool requests.
        utterances_tool = tool_factory.get_tool(
            "VectorSearchTool",
            collection_name="utterances",
            metadata=vector_search_tool_metadata,
        )  # type: ignore
        if not utterances_tool:
            logger.warning(
                "VectorSearchTool for 'utterances' could not be created. Decontextualization may be impaired."
            )
            return []
        return [utterances_tool]

    def _get_prompt_template(self) -> str:
        """
        Returns the prompt template for the decontextualization task.
        """
        # This prompt is based on docs/arch/on-evaluation_agents.md (Section "Prompt")
        # and the task description's emphasis on using VectorSearchTool.
        return (
            "You are an expert evaluator. Your task is to determine if a given 'Claim' "
            "can be understood on its own, without needing to refer to its original 'Source' context. "
            "A claim is well-decontextualized if a reader can understand its meaning and verify it "
            "without additional information beyond the claim itself.\n\n"
            "Consider the following:\n"
            "- Pronouns: Are all pronouns (he, she, it, they, etc.) clear or resolved?\n"
            "- Ambiguous References: Are there any terms or phrases that could refer to multiple things?\n"
            "- Missing Context: Is any crucial information (like time, location, specific entities, or scope) missing "
            "  that would be necessary for a fact-checker to verify the claim accurately?\n\n"
            "To help you, you have a 'vector_search_utterances' tool. You can use this tool with the 'Claim' text "
            "as input to see if similar phrases or statements appear in other, potentially very different, contexts "
            "in the knowledge base. If the search returns diverse results for a seemingly specific claim, it might "
            "indicate ambiguity or missing context in the original claim.\n\n"
            "Input:\n"
            'Claim: "{claim_text}"\n'
            'Source: "{source_text}"\n\n'
            "Task:\n"
            "1. Analyze the 'Claim' for any ambiguities or missing context. "
            "2. Optionally, use the 'vector_search_utterances' tool with the 'Claim' text to check for contextual diversity "
            "   of similar phrases. This can help identify if the claim is too generic or relies on implicit context "
            "   not present in the claim itself. "
            "3. Based on your analysis, provide a decontextualization score as a float between 0.0 and 1.0, "
            "   where 0.0 means the claim is completely dependent on its source and cannot be understood alone, "
            "   and 1.0 means the claim is perfectly self-contained and understandable in isolation.\n\n"
            "Output only the float score. For example: 0.75"
        )

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
        prompt = self._get_prompt_template().format(
            claim_text=claim_text, source_text=source_text
        )

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=4, max=10),
        )
        async def _attempt_evaluation_with_prompt(current_prompt: str):
            response: ChatResponse = await self.agent.achat(current_prompt)
            return response.response.strip()

        log_details = {
            "service": "aclarai-core",
            "filename_function_name": "decontextualization_agent.DecontextualizationAgent.evaluate_claim_decontextualization",
            "aclarai_id_claim": claim_id,
            "aclarai_id_source": source_id,
        }

        try:
            logger.info(
                f"Evaluating decontextualization for claim_id: {claim_id} with max_retries: {self.max_retries}",
                extra=log_details,
            )
            # response: ChatResponse = await self.agent.achat(prompt) # Original direct call
            # response_text = response.response.strip()
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
            except ValueError:
                error_message = (
                    f"LLM returned an invalid score format or out-of-range value for claim_id {claim_id}. "
                    f"Response: '{response_text}'"
                )
                logger.warning(error_message, extra=log_details)
                return None, error_message

        except Exception as e:
            error_message = (
                f"Error during decontextualization evaluation for claim_id {claim_id} "
                f"after {self.max_retries} retries: {str(e)}"
            )
            logger.error(error_message, exc_info=True, extra=log_details)
            return None, error_message


# Example usage (conceptual, actual integration will be elsewhere)
async def main_example():
    # This is a simplified example. In a real scenario, these would be properly initialized.
    # from llama_index.llms.openai import OpenAI
    # llm = OpenAI(model="gpt-3.5-turbo")

    # Mock LLM and ToolFactory for testing without real services
    class MockLLM:
        async def achat(
            self, messages, **kwargs
        ):  # Adjusted to match ReActAgent's call
            # Simulate LLM response based on prompt content
            if "Output only the float score" in messages:
                return ChatResponse(message=None, raw={"response": "0.85"})  # type: ignore
            return ChatResponse(message=None, raw={"response": "Some text"})  # type: ignore

        def chat(self, messages, **kwargs):  # Synchronous version for ReActAgent
            if "Output only the float score" in messages:
                return ChatResponse(message=None, raw={"response": "0.85"})  # type: ignore
            return ChatResponse(message=None, raw={"response": "Some text"})  # type: ignore

    class MockTool(BaseTool):
        def __init__(self, name: str, description: str):
            self._name = name
            self._description = description
            self._metadata = ToolMetadata(name=name, description=description)

        @property
        def metadata(self):
            return self._metadata

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return "Tool called with args: " + str(args) + " kwargs: " + str(kwargs)

    class MockToolFactory:
        def get_tool(
            self,
            tool_name: str,
            collection_name: Optional[str] = None,
            metadata: Optional[ToolMetadata] = None,
        ) -> Optional[BaseTool]:
            if tool_name == "VectorSearchTool" and collection_name == "utterances":
                return MockTool(
                    name=metadata.name if metadata else "vector_search_utterances",
                    description=metadata.description
                    if metadata
                    else "Searches utterances",
                )
            return None

    class MockConfigManager:
        def get(self, key: str, default: Any = None) -> Any:
            if key == "processing.retries.max_attempts":
                return 3
            if key.startswith(
                "model.claimify.decontextualization"
            ):  # Or other model keys
                return "mock_llm_model_name"
            return default

    logging.basicConfig(level=logging.INFO)

    # Use the mock LLM and ToolFactory
    mock_llm = MockLLM()
    mock_tool_factory = MockToolFactory()
    mock_config_manager = MockConfigManager()

    # The ReActAgent expects a synchronous chat method for from_tools
    # We need to ensure our MockLLM provides that or use a LLM that LlamaIndex ReActAgent supports
    # For simplicity, I've added a sync `chat` to MockLLM.
    # In a real setup, ensure the LLM is compatible.

    agent_instance = DecontextualizationAgent(
        llm=mock_llm, tool_factory=mock_tool_factory, config_manager=mock_config_manager
    )  # type: ignore

    claim_id_to_test = "claim_123"
    claim_text_to_test = "The program experienced a significant performance increase."
    source_id_to_test = "source_abc"
    source_text_to_test = (
        "After implementing the new caching mechanism, the program experienced a significant performance increase. "
        "This was observed during the stress tests conducted last Tuesday."
    )

    score, message = await agent_instance.evaluate_claim_decontextualization(
        claim_id=claim_id_to_test,
        claim_text=claim_text_to_test,
        source_id=source_id_to_test,
        source_text=source_text_to_test,
    )

    if score is not None:
        print(f"Decontextualization Score: {score}")
    else:
        print(f"Error: {message}")


if __name__ == "__main__":

    # asyncio.run(main_example()) # Commented out to prevent execution in non-test environment
    print("DecontextualizationAgent file created. Example usage commented out.")

"""
This agent will be responsible for:
1. Receiving a claim and its source.
2. Using a configured LLM and potentially a VectorSearchTool (via ToolFactory).
3. Prompting the LLM to evaluate if the claim is understandable in isolation.
   The prompt will guide the LLM to consider pronoun clarity, ambiguity, and missing context.
   It will also instruct the LLM on how to use the VectorSearchTool to check for contextual diversity
   of similar phrases, which can indicate reliance on implicit context.
4. The LLM's output should be a float score between 0.0 and 1.0.
5. The agent will handle retries and default to a null score on persistent failure.
"""
