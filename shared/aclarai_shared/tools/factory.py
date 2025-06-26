"""Tool factory for initializing and managing aclarai tools."""

import functools
import logging
from typing import Any, Dict, List, Optional

from llama_index.core.tools import BaseTool

from .implementations.neo4j_tool import Neo4jQueryTool
from .implementations.vector_search_tool import VectorSearchTool
from .implementations.web_search.base import WebSearchTool
from .implementations.web_search.provider import create_provider

logger = logging.getLogger(__name__)


class ToolFactory:
    """Factory for creating and managing aclarai tools."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the tool factory.
        
        Args:
            config: Configuration dictionary from aclarai.config.yaml
        """
        self._config = config.get('tools', {})
        
    @functools.lru_cache(maxsize=None)
    def get_tools_for_agent(self, agent_name: str) -> List[BaseTool]:
        """Get the appropriate set of tools for an agent.
        
        Args:
            agent_name: Name of the agent to get tools for
            
        Returns:
            List of initialized tool instances
        """
        tools = []
        
        # Initialize Neo4j tool if enabled
        if self._config.get('neo4j', {}).get('enabled', True):
            neo4j_tool = self._initialize_neo4j_tool()
            if neo4j_tool:
                tools.append(neo4j_tool)
                
        # Initialize vector search tool if enabled
        if self._config.get('vector_search', {}).get('enabled', True):
            vector_tool = self._initialize_vector_search_tool()
            if vector_tool:
                tools.append(vector_tool)
                
        # Initialize web search tool if enabled and configured
        web_config = self._config.get('web_search', {})
        if web_config.get('enabled', False):
            web_tool = self._initialize_web_search_tool()
            if web_tool:
                tools.append(web_tool)
        
        return tools
    
    def _initialize_neo4j_tool(self) -> Optional[Neo4jQueryTool]:
        """Initialize the Neo4j query tool.
        
        Returns:
            Configured Neo4jQueryTool instance or None if initialization fails
        """
        try:
            config = self._config.get('neo4j', {})
            tool = Neo4jQueryTool.from_config(config)
            if tool:
                logger.info("Successfully initialized Neo4j query tool")
            return tool
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j tool: {str(e)}")
            return None
    
    def _initialize_vector_search_tool(self) -> Optional[VectorSearchTool]:
        """Initialize the vector search tool.
        
        Returns:
            Configured VectorSearchTool instance or None if initialization fails
        """
        try:
            config = self._config.get('vector_search', {})
            tool = VectorSearchTool.from_config(config)
            if tool:
                logger.info("Successfully initialized vector search tool")
            return tool
            
        except Exception as e:
            logger.error(f"Failed to initialize vector search tool: {str(e)}")
            return None
    
    def _initialize_web_search_tool(self) -> Optional[WebSearchTool]:
        """Initialize the web search tool.
        
        Returns:
            Configured WebSearchTool instance or None if initialization fails
        """
        try:
            config = self._config.get('web_search', {})
            
            # Create the appropriate provider
            provider_name = config.get('provider')
            if not provider_name:
                logger.error("No web search provider specified in config")
                return None
                
            provider = create_provider(provider_name, config)
            if not provider:
                return None
            
            # Create and return the tool with the configured provider
            tool = WebSearchTool(
                provider=provider,
                max_results=config.get('max_results', 5)
            )
            logger.info(
                "Successfully initialized web search tool",
                extra={'provider': provider_name}
            )
            return tool
            
        except Exception as e:
            logger.error(f"Failed to initialize web search tool: {str(e)}")
            return None
