"""Tool factory for initializing and managing aclarai tools."""

import functools
import logging
from typing import Any, Dict, List, Optional, Union

from llama_index.core.tools import BaseTool

from .implementations.neo4j_tool import Neo4jQueryTool
from .implementations.vector_search_tool import VectorSearchTool
from .implementations.web_search.base import WebSearchTool
from .implementations.web_search.provider import create_provider
from .vector_store_manager import VectorStore, VectorStoreManager

logger = logging.getLogger(__name__)


def _freeze(obj: Any) -> Union[frozenset, tuple, Any]:
    """Recursively freeze a container to make it hashable for caching."""
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
    config: Dict[str, Any], vector_store_manager: VectorStoreManager
) -> Optional[VectorSearchTool]:
    """Initialize the vector search tool using the VectorStoreManager."""
    try:
        collection_names = config.get("collections", [])
        if not collection_names:
            logger.warning(
                "Vector search tool enabled but no collections specified in config."
            )
            return None

        vector_stores: Dict[str, VectorStore] = {}
        for name in collection_names:
            store = vector_store_manager.get_store(name)
            if store:
                vector_stores[name] = store
            else:
                logger.error(f"Vector store '{name}' not found in VectorStoreManager.")

        if not vector_stores:
            logger.error("Could not initialize any vector stores for VectorSearchTool.")
            return None

        tool = VectorSearchTool(
            vector_stores=vector_stores,
            similarity_threshold=config.get("similarity_threshold", 0.7),
            max_results=config.get("max_results", 5),
        )
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


@functools.lru_cache(maxsize=None)
def _get_cached_tools(
    config_hash: frozenset, vector_store_manager: VectorStoreManager
) -> List[BaseTool]:
    """
    Get a cached list of tools based on a hashable configuration and manager.

    This function is defined outside the class to prevent memory leaks associated
    with caching instance methods.
    """
    config = _unfreeze(config_hash)
    tools: List[BaseTool] = []

    # Initialize Neo4j tool if enabled
    if config.get("neo4j", {}).get("enabled", True) and (
        neo4j_tool := _initialize_neo4j_tool(config.get("neo4j", {}))
    ):
        tools.append(neo4j_tool)

    # Initialize vector search tool if enabled
    if config.get("vector_search", {}).get("enabled", True) and (
        vector_tool := _initialize_vector_search_tool(
            config.get("vector_search", {}), vector_store_manager
        )
    ):
        tools.append(vector_tool)

    # Initialize web search tool if enabled and configured
    if (
        (web_config := config.get("web_search", {}))
        and web_config.get("enabled", False)
        and (web_tool := _initialize_web_search_tool(web_config))
    ):
        tools.append(web_tool)

    return tools


class ToolFactory:
    """Factory for creating and managing aclarai tools."""

    def __init__(
        self, config: Dict[str, Any], vector_store_manager: VectorStoreManager
    ) -> None:
        """Initialize the tool factory."""
        self._config = config.get("tools", {})
        self.vector_store_manager = vector_store_manager

    def get_tools_for_agent(self, _agent_name: str) -> List[BaseTool]:
        """
        Get the appropriate set of tools for an agent.

        This method acts as a public interface to the cached tool creation
        logic, ensuring that tool initialization only happens once for a given
        configuration.
        """
        hashable_config = _freeze(self._config)
        return _get_cached_tools(hashable_config, self.vector_store_manager)
