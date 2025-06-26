"""Tool factory for initializing and managing aclarai tools."""

import functools
import logging
from typing import Any, Dict, List, Optional, Union

from llama_index.core.tools import BaseTool

from .implementations.neo4j_tool import Neo4jQueryTool
from .implementations.vector_search_tool import VectorSearchTool
from .implementations.web_search.base import WebSearchTool
from .implementations.web_search.provider import create_provider

logger = logging.getLogger(__name__)


def _freeze(obj: Any) -> Union[frozenset, tuple, Any]:
    """Recursively freeze a container to make it hashable."""
    if isinstance(obj, dict):
        return frozenset((k, _freeze(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    return obj


def _unfreeze(obj: Any) -> Any:
    """Recursively unfreeze a container back to its mutable form."""
    if isinstance(obj, frozenset):
        return {k: _unfreeze(v) for k, v in obj}
    if isinstance(obj, tuple):
        return [_unfreeze(v) for v in obj]
    return obj


@functools.lru_cache(maxsize=None)
def _get_cached_tools(config_hash: frozenset) -> List[BaseTool]:
    """Get a cached list of tools based on a hashable configuration.

    Args:
        config_hash: Hashable frozenset representation of the configuration.

    Returns:
        List of initialized tool instances.
    """
    config = _unfreeze(config_hash)
    tools: List[BaseTool] = []

    # Initialize Neo4j tool if enabled
    if config.get("neo4j", {}).get("enabled", True):
        neo4j_tool = _initialize_neo4j_tool(config.get("neo4j", {}))
        if neo4j_tool:
            tools.append(neo4j_tool)

    # Initialize vector search tool if enabled
    if config.get("vector_search", {}).get("enabled", True):
        vector_tool = _initialize_vector_search_tool(config.get("vector_search", {}))
        if vector_tool:
            tools.append(vector_tool)

    # Initialize web search tool if enabled and configured
    web_config = config.get("web_search", {})
    if web_config.get("enabled", False):
        web_tool = _initialize_web_search_tool(web_config)
        if web_tool:
            tools.append(web_tool)

    return tools


def _initialize_neo4j_tool(config: Dict[str, Any]) -> Optional[Neo4jQueryTool]:
    """Initialize the Neo4j query tool."""
    try:
        tool = Neo4jQueryTool.from_config(config)
        if tool:
            logger.info("Successfully initialized Neo4j query tool")
        return tool
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j tool: {str(e)}")
        return None


def _initialize_vector_search_tool(
    config: Dict[str, Any],
) -> Optional[VectorSearchTool]:
    """Initialize the vector search tool."""
    try:
        tool = VectorSearchTool.from_config(config)
        if tool:
            logger.info("Successfully initialized vector search tool")
        return tool
    except Exception as e:
        logger.error(f"Failed to initialize vector search tool: {str(e)}")
        return None


def _initialize_web_search_tool(config: Dict[str, Any]) -> Optional[WebSearchTool]:
    """Initialize the web search tool."""
    try:
        provider_name = config.get("provider")
        if not provider_name:
            logger.error("No web search provider specified in config")
            return None

        provider = create_provider(provider_name, config)
        if not provider:
            return None

        tool = WebSearchTool(
            provider=provider, max_results=config.get("max_results", 5)
        )
        logger.info(
            "Successfully initialized web search tool",
            extra={"provider": provider_name},
        )
        return tool
    except Exception as e:
        logger.error(f"Failed to initialize web search tool: {str(e)}")
        return None


class ToolFactory:
    """Factory for creating and managing aclarai tools."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the tool factory.

        Args:
            config: Configuration dictionary from aclarai.config.yaml
        """
        self._config = config.get("tools", {})

    def get_tools_for_agent(self, _agent_name: str) -> List[BaseTool]:
        """Get the appropriate set of tools for an agent.

        Args:
            _agent_name: Name of the agent to get tools for (unused)

        Returns:
            List of initialized tool instances
        """
        hashable_config = _freeze(self._config)
        return _get_cached_tools(hashable_config)
