"""Neo4j query tool implementation."""

import logging
from typing import Any, Dict, Optional

from llama_index.core.tools import ToolMetadata
from neo4j import GraphDatabase
from tenacity import retry, stop_after_attempt, wait_exponential

from ..base import AclaraiBaseTool

logger = logging.getLogger(__name__)


class Neo4jQueryTool(AclaraiBaseTool):
    """Tool for executing Cypher queries against Neo4j."""

    def __init__(
        self,
        uri: str,
        auth: tuple[str, str],
        max_retries: int = 3,
        metadata: Optional[ToolMetadata] = None,
    ) -> None:
        """Initialize the Neo4j query tool.

        Args:
            uri: Neo4j connection URI
            auth: Tuple of (username, password) for authentication
            max_retries: Maximum number of retry attempts for failed queries
            metadata: Optional metadata for the tool
        """
        self._driver = GraphDatabase.driver(uri, auth=auth)
        self._max_retries = max_retries
        self._metadata = metadata
        super().__init__()

    @property
    def metadata(self) -> ToolMetadata:
        """Get tool metadata for agent use."""
        if self._metadata:
            return self._metadata
        # Fallback to default metadata if not provided
        return ToolMetadata(
            name="neo4j_query",
            description="Execute Cypher queries against the knowledge graph. Returns results as a formatted string.",
        )

    @metadata.setter
    def metadata(self, value: ToolMetadata) -> None:
        """Set tool metadata."""
        self._metadata = value

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def _execute_query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute a Cypher query with retry logic.

        Args:
            query: The Cypher query to execute
            params: Optional parameters for the query

        Returns:
            Formatted string representation of the query results

        Raises:
            Neo4jError: If the query fails after all retries
        """
        try:
            with self._driver.session() as session:
                result = session.run(query, parameters=params or {})
                records = result.data()

                if not records:
                    return "Query returned no results."

                # Format results as a readable string
                formatted = []
                for record in records:
                    formatted.append(str(record))

                return "\n".join(formatted)

        except Exception as e:
            logger.error(
                f"Error executing Neo4j query: {str(e)}",
                extra={"query": query, "params": params},
            )
            raise

    def __call__(self, input: Any) -> Any:
        """Executes the tool with the given arguments and keyword arguments."""
        if not isinstance(input, dict):
            raise ValueError("Input must be a dictionary")
        return self._call(**input)

    def _call(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Execute the provided Cypher query.

        Args:
            query: The Cypher query to execute
            params: Optional parameters for the query

        Returns:
            Formatted string containing query results or error message
        """
        try:
            return self._execute_query(query, params)
        except Exception as e:
            error_msg = f"Failed to execute Neo4j query: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def __del__(self) -> None:
        """Ensure Neo4j driver is closed on cleanup."""
        if hasattr(self, "_driver"):
            self._driver.close()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional["Neo4jQueryTool"]:
        """Create a Neo4j tool instance from configuration.

        Args:
            config: Configuration dictionary containing:
                - uri: Neo4j connection URI
                - username: Neo4j username
                - password: Neo4j password
                - retry_attempts: Optional number of retry attempts

        Returns:
            Configured Neo4jQueryTool instance or None if required config is missing
        """
        required = {"uri", "username", "password"}
        if not all(key in config for key in required):
            logger.error(
                "Missing required Neo4j configuration",
                extra={"missing": required - set(config.keys())},
            )
            return None

        return cls(
            uri=config["uri"],
            auth=(config["username"], config["password"]),
            max_retries=config.get("retry_attempts", 3),
        )
