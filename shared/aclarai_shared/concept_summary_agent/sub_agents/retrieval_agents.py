"""
This module defines the specialized sub-agents responsible for retrieving context
for the ConceptSummaryAgent.
"""

import logging
from typing import List, Optional

from llama_index.core.agent import AgentRunner
from llama_index.core.tools import BaseTool
from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)


def create_retrieval_agent(
    tools: List[BaseTool],
    llm: OpenAI,
    system_prompt: str,
    verbose: bool = False,
) -> Optional[AgentRunner]:
    """
    Creates a LlamaIndex agent for a specific retrieval task.

    Args:
        tools: A list of tools for the agent to use.
        llm: The OpenAI LLM instance.
        system_prompt: The system prompt to guide the agent's behavior.
        verbose: Whether to enable verbose logging for the agent.

    Returns:
        An AgentRunner instance or None if creation fails.
    """
    try:
        return AgentRunner.from_llm(
            tools=tools,
            llm=llm,
            system_prompt=system_prompt,
            verbose=verbose,
        )
    except Exception as e:
        logger.error(
            f"Failed to create retrieval agent: {e}",
            extra={
                "service": "aclarai",
                "filename.function_name": "sub_agents.retrieval_agents.create_retrieval_agent",
                "error": str(e),
            },
        )
        return None
