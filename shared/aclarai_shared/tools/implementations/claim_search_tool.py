"""
This module defines the ClaimSearchTool for retrieving claims from Neo4j.
"""

import logging
from typing import Any

from aclarai_shared.graph.neo4j_manager import Neo4jGraphManager
from llama_index.core.tools import BaseTool, ToolMetadata, ToolOutput

logger = logging.getLogger(__name__)


class ClaimSearchTool(BaseTool):
    """
    A tool for searching claims related to a concept in a Neo4j graph.
    """

    def __init__(
        self,
        neo4j_manager: Neo4jGraphManager,
        metadata: ToolMetadata,
    ) -> None:
        self._neo4j_manager = neo4j_manager
        self._metadata = metadata

    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """
        Retrieves claims related to a given concept name.

        Args:
            concept_name: The name of the concept to search for.
            limit: The maximum number of claims to return.

        Returns:
            A list of strings, where each string is a claim with its anchor.
        """
        concept_name = kwargs.get("concept_name", args[0] if args else None)
        limit = kwargs.get("limit", args[1] if len(args) > 1 else 5)

        if not concept_name:
            raise ValueError("concept_name must be provided.")

        try:
            query = """
            MATCH (c:Concept {text: $concept_name})<-[r]-(claim:Claim)
            RETURN claim.text AS text, claim.aclarai_id AS aclarai_id, type(r) as relationship
            ORDER BY claim.timestamp DESC
            LIMIT $limit
            """
            params = {"concept_name": concept_name, "limit": limit}
            result = self._neo4j_manager.execute_query(query, params)
            claims = []
            for record in result:
                text = record.get("text")
                aclarai_id = record.get("aclarai_id")
                if text and aclarai_id:
                    claims.append(f"{text} ^{aclarai_id}")
            return ToolOutput(
                content="\n".join(claims),
                tool_name=self.metadata.name or "claim_search_tool",
                raw_output=claims,
                raw_input=concept_name,
            )
        except Exception as e:
            logger.error(
                f"Failed to retrieve claims for concept '{concept_name}': {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "tools.implementations.claim_search_tool.ClaimSearchTool.__call__",
                    "concept_name": concept_name,
                    "error": str(e),
                },
            )
            return ToolOutput(
                content="",
                tool_name=self.metadata.name or "claim_search_tool",
                raw_output=[],
                raw_input=concept_name,
            )

    @property
    def metadata(self) -> ToolMetadata:
        """Return the metadata for the tool."""
        return self._metadata
