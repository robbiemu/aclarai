import logging
from typing import Any, Callable, List, Optional, Union

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import BaseTool
from llama_index.llms.openai import OpenAI  # or the LLM class you're using

logger = logging.getLogger(__name__)


def create_retrieval_agent(
    tools: List[Union[BaseTool, Callable[..., Any]]],
    llm: OpenAI,
    system_prompt: str,
    verbose: bool = False,
) -> Optional[FunctionAgent]:
    """
    Create a FunctionAgent for retrieval tasks.

    Args:
      tools: List of tool objects for the agent
      llm: LLM instance (e.g. OpenAI) implementing BaseLLM
      system_prompt: Instructions guiding the agent’s behavior
      verbose: Whether to enable debug logging

    Returns:
      A configured FunctionAgent, or None on failure
    """
    try:
        return FunctionAgent(
            tools=tools,
            llm=llm,
            system_prompt=system_prompt,
            verbose=verbose,
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to create retrieval agent: {e}", exc_info=True)
        return None
