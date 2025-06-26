"""Tavily web search provider implementation."""

import logging
import os
from typing import Any, Dict, List, Optional

from tavily import TavilyClient

from .base import WebSearchProvider, WebSearchResult

logger = logging.getLogger(__name__)


class TavilySearchProvider(WebSearchProvider):
    """Web search provider using Tavily's API."""
    
    def __init__(self, api_key: str) -> None:
        """Initialize the Tavily provider.
        
        Args:
            api_key: Tavily API key
        """
        self._client = TavilyClient(api_key=api_key)
    
    def search(
        self,
        query: str,
        max_results: int = 5
    ) -> List[WebSearchResult]:
        """Execute a web search using Tavily.
        
        Args:
            query: Search query text
            max_results: Maximum number of results to return
            
        Returns:
            List of standardized search results
        """
        try:
            response = self._client.search(
                query=query,
                max_results=max_results,
                search_depth="basic"
            )
            
            results = []
            for item in response.get('results', []):
                results.append(WebSearchResult(
                    title=item.get('title', 'Untitled'),
                    url=item.get('url', ''),
                    snippet=item.get('content', ''),
                    published_date=item.get('published_date')
                ))
            
            return results
            
        except Exception as e:
            logger.error(
                f"Tavily search failed: {str(e)}",
                extra={'query': query}
            )
            return []
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional['TavilySearchProvider']:
        """Create a Tavily provider instance from configuration.
        
        Args:
            config: Configuration containing:
                - api_key_env_var: Name of environment variable with API key
                
        Returns:
            Configured provider instance or None if config is invalid
        """
        api_key_var = config.get('api_key_env_var')
        if not api_key_var:
            logger.error("Missing API key environment variable name in config")
            return None
            
        api_key = os.getenv(api_key_var)
        if not api_key:
            logger.error(
                f"Environment variable {api_key_var} not set",
                extra={'env_var': api_key_var}
            )
            return None
            
        return cls(api_key=api_key)
