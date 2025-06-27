# In shared/aclarai_shared/tools/factory.py

import logging
from typing import Any, Callable, Dict, List, Optional

from llama_index.core.tools import BaseTool, ToolMetadata

from .implementations.neo4j_tool import Neo4jQueryTool
from .implementations.vector_search_tool import VectorSearchTool
from .implementations.web_search.base import WebSearchTool
from .implementations.web_search.provider import create_provider
from .vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)


class ToolFactory:
    """
    Factory for creating and managing agent tools based on system configuration.
    This class reads the `tools` section of the configuration and constructs
    the appropriate toolset for a given agent role.
    """

    def __init__(
        self, config: Dict[str, Any], vector_store_manager: VectorStoreManager
    ) -> None:
        """Initialize the tool factory."""
        self._config = config.get("tools", {})
        self.vector_store_manager = vector_store_manager
        # Instance-level cache to store initialized tools for each agent
        self._agent_tool_cache: Dict[str, List[BaseTool | Callable[..., Any]]] = {}

    def get_tools_for_agent(
        self, agent_name: str
    ) -> List[BaseTool | Callable[..., Any]]:
        """
        Get a cached list of tools for a specific agent role.
        This is the primary entry point for agents to get their tools.
        The result is cached per agent name to prevent re-initializing tools
        on every call for the same agent.
        """
        # Check the instance cache first
        if agent_name in self._agent_tool_cache:
            return self._agent_tool_cache[agent_name]

        # If not cached, build the tools for this agent
        agent_tool_configs = self._config.get("agent_tool_mappings", {}).get(
            agent_name, []
        )
        if not agent_tool_configs:
            logger.warning(f"No tool configuration found for agent: {agent_name}")
            return []

        tools: List[BaseTool | Callable[..., Any]] = []
        for tool_config in agent_tool_configs:
            tool_type = tool_config.get("name")
            params = tool_config.get("params", {})

            tool: Optional[BaseTool] = None
            if tool_type == "VectorSearchTool":
                tool = self._initialize_vector_search_tool(params)
            elif tool_type == "Neo4jQueryTool":
                tool = self._initialize_neo4j_tool(self._config.get("neo4j", {}))
            elif tool_type == "WebSearchTool":
                tool = self._initialize_web_search_tool(
                    self._config.get("web_search", {})
                )
            else:
                logger.warning(
                    f"Unknown tool type '{tool_type}' configured for agent '{agent_name}'"
                )
                continue

            if tool:
                # Allow agent-specific metadata to override the tool's default metadata
                if "metadata" in params:
                    tool.metadata = ToolMetadata(**params["metadata"])
                tools.append(tool)

        # Store the newly created tool list in the cache before returning
        self._agent_tool_cache[agent_name] = tools
        return tools

    def _initialize_neo4j_tool(
        self, config: Dict[str, Any]
    ) -> Optional[Neo4jQueryTool]:
        """Initialize the Neo4j query tool from its configuration section."""
        if not config.get("enabled", True):
            return None
        try:
            tool = Neo4jQueryTool.from_config(config)
            if tool:
                logger.info("Successfully initialized Neo4j query tool.")
            return tool
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j tool: {str(e)}")
            return None

    def _initialize_vector_search_tool(
        self, params: Dict[str, Any]
    ) -> Optional[VectorSearchTool]:
        """Initialize the vector search tool based on specific parameters."""
        collection_name = params.get("collection")
        if not collection_name:
            logger.error("VectorSearchTool config params missing 'collection' key.")
            return None

        store = self.vector_store_manager.get_store(collection_name)
        if not store:
            logger.error(
                f"Vector store '{collection_name}' not found in VectorStoreManager."
            )
            return None

        main_vector_config = self._config.get("vector_search", {})

        return VectorSearchTool(
            vector_stores={collection_name: store},
            similarity_threshold=params.get(
                "similarity_threshold",
                main_vector_config.get("similarity_threshold", 0.7),
            ),
            max_results=params.get(
                "max_results", main_vector_config.get("max_results", 5)
            ),
        )

    def _initialize_web_search_tool(
        self, config: Dict[str, Any]
    ) -> Optional[WebSearchTool]:
        """Initialize the web search tool from its configuration section."""
        if not config.get("enabled", False):
            return None
        try:
            provider_name = config.get("provider")
            if not provider_name:
                logger.error("No web search provider specified in config.")
                return None

            provider = create_provider(provider_name, config)
            if not provider:
                return None

            tool = WebSearchTool(
                provider=provider, max_results=config.get("max_results", 5)
            )
            logger.info(
                f"Successfully initialized web search tool with provider: {provider_name}."
            )
            return tool
        except Exception as e:
            logger.error(f"Failed to initialize web search tool: {str(e)}")
            return None
