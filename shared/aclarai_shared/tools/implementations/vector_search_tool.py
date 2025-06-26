"""Vector search tool implementation."""

import logging
from typing import Any, Dict, List, Optional

from llama_index.core.tools import ToolMetadata
from llama_index.core.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

from ..base import AclaraiBaseTool

logger = logging.getLogger(__name__)


class VectorSearchTool(AclaraiBaseTool):
    """Tool for performing semantic similarity searches across vector stores."""

    def __init__(
        self,
        vector_stores: Dict[str, VectorStore],
        similarity_threshold: float = 0.7,
        max_results: int = 5,
    ) -> None:
        """Initialize the vector search tool.

        Args:
            vector_stores: Dictionary mapping collection names to vector stores
            similarity_threshold: Minimum similarity score for matches (0-1)
            max_results: Maximum number of results to return per collection
        """
        self._vector_stores = vector_stores
        self._similarity_threshold = similarity_threshold
        self._max_results = max_results
        super().__init__()

    @property
    def metadata(self) -> ToolMetadata:
        """Get tool metadata for agent use."""
        return ToolMetadata(
            name="vector_search",
            description=(
                "Search for semantically similar content across collections. "
                f"Available collections: {', '.join(self._vector_stores.keys())}"
            ),
        )

    def _search_collection(
        self, collection: str, query: str, max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search a single vector store collection.

        Args:
            collection: Name of the collection to search
            query: Search query text
            max_results: Optional override for max results

        Returns:
            List of search results with scores
        """
        try:
            store = self._vector_stores[collection]
            query_obj = VectorStoreQuery(
                query_str=query,
                similarity_top_k=max_results or self._max_results,
            )
            results: VectorStoreQueryResult = store.query(query_obj)

            formatted_results = []
            if results.nodes and results.similarities:
                for node, similarity in zip(
                    results.nodes, results.similarities, strict=False
                ):
                    if similarity >= self._similarity_threshold:
                        formatted_results.append(
                            {
                                "collection": collection,
                                "text": node.get_content(),
                                "score": similarity,
                                "metadata": node.metadata,
                            }
                        )

            return formatted_results

        except Exception as e:
            logger.error(
                f"Error searching collection {collection}: {str(e)}",
                extra={"query": query, "collection": collection},
            )
            return []

    def __call__(self, input: Any) -> Any:
        """Executes the tool with the given arguments and keyword arguments."""
        if not isinstance(input, dict):
            raise ValueError("Input must be a dictionary")
        return self._call(**input)

    def _call(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        max_results: Optional[int] = None,
    ) -> str:
        """Execute a semantic search across collections.

        Args:
            query: The text to search for
            collections: Optional list of specific collections to search
            max_results: Optional maximum number of results per collection

        Returns:
            Formatted string containing search results or error message
        """
        try:
            # Determine which collections to search
            search_collections = collections or list(self._vector_stores.keys())
            invalid_collections = set(search_collections) - set(
                self._vector_stores.keys()
            )
            if invalid_collections:
                return (
                    f"Invalid collections specified: {', '.join(invalid_collections)}"
                )

            # Search each collection
            all_results = []
            for collection in search_collections:
                results = self._search_collection(
                    collection=collection, query=query, max_results=max_results
                )
                all_results.extend(results)

            # Sort by score and format results
            all_results.sort(key=lambda x: x["score"], reverse=True)

            if not all_results:
                return "No results found matching the query."

            # Format results as readable text
            formatted = ["Search results:"]
            for result in all_results:
                formatted.append(
                    f"\nCollection: {result['collection']}"
                    f"\nScore: {result['score']:.3f}"
                    f"\nText: {result['text']}"
                    f"\nMetadata: {result['metadata']}\n"
                )

            return "\n".join(formatted)

        except Exception as e:
            error_msg = f"Failed to execute vector search: {str(e)}"
            logger.error(error_msg)
            return error_msg
