"""Base classes for web search tool implementations."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from llama_index.core.tools import ToolMetadata

from ...base import AclaraiBaseTool

logger = logging.getLogger(__name__)


class WebSearchResult:
    """Standardized container for web search results."""

    def __init__(
        self, title: str, url: str, snippet: str, published_date: Optional[str] = None
    ) -> None:
        """Initialize a web search result.

        Args:
            title: Title of the web page
            url: URL of the web page
            snippet: Text snippet or summary
            published_date: Optional publication date
        """
        self.title = title
        self.url = url
        self.snippet = snippet
        self.published_date = published_date

    def __str__(self) -> str:
        """Format the result as a readable string."""
        parts = [f"Title: {self.title}", f"URL: {self.url}", f"Summary: {self.snippet}"]
        if self.published_date:
            parts.append(f"Published: {self.published_date}")
        return "\n".join(parts)


class WebSearchProvider(ABC):
    """Abstract base class for web search providers."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        """Execute a web search using this provider.

        Args:
            query: Search query text
            max_results: Maximum number of results to return

        Returns:
            List of standardized search results
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional["WebSearchProvider"]:
        """Create a provider instance from configuration.

        Args:
            config: Provider-specific configuration

        Returns:
            Configured provider instance or None if config is invalid
        """
        pass


class WebSearchTool(AclaraiBaseTool):
    """Tool for performing web searches using configurable providers."""

    def __init__(self, provider: WebSearchProvider, max_results: int = 5) -> None:
        """Initialize the web search tool.

        Args:
            provider: Configured search provider instance
            max_results: Maximum number of results to return
        """
        self._provider = provider
        self._max_results = max_results
        super().__init__()

    def __call__(self, input: Any) -> Any:
        """Execute the tool.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tool execution result as a string
        """
        if isinstance(input, dict):
            query = input.get("query", "")
            max_results = input.get("max_results")
        else:
            query = str(input)
            max_results = None
        return self._call(query=query, max_results=max_results)

    @property
    def metadata(self) -> ToolMetadata:
        """Get tool metadata for agent use."""
        return ToolMetadata(
            name="web_search",
            description="Search the web for information about a topic.",
        )

    def _call(self, query: str, max_results: Optional[int] = None) -> str:
        """Execute a web search.

        Args:
            query: Search query text
            max_results: Optional override for max results

        Returns:
            Formatted string containing search results or error message
        """
        try:
            results = self._provider.search(
                query=query, max_results=max_results or self._max_results
            )

            if not results:
                return "No results found."

            formatted = ["Search results:"]
            for i, result in enumerate(results, 1):
                formatted.extend([f"\n[{i}] {str(result)}", "-" * 40])

            return "\n".join(formatted)

        except Exception as e:
            error_msg = f"Failed to execute web search: {str(e)}"
            logger.error(error_msg)
            return error_msg
