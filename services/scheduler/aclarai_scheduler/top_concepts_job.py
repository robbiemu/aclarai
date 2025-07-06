"""
Top Concepts Job for aclarai scheduler.
This module implements the scheduled job for generating Top Concepts.md from
PageRank analysis of (:Concept) nodes in the Neo4j knowledge graph, following
the architecture from docs/arch/idea-neo4J-ineteraction.md and output format
from docs/arch/on-writing_vault_documents.md.
"""

import contextlib
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

from aclarai_shared import load_config
from aclarai_shared.config import TopConceptsJobConfig, aclaraiConfig
from aclarai_shared.graph.neo4j_manager import Neo4jGraphManager

logger = logging.getLogger(__name__)


class TopConceptsJobStats(TypedDict):
    """Type definition for job statistics."""

    success: bool
    concepts_analyzed: int
    top_concepts_selected: int
    file_written: bool
    pagerank_executed: bool
    duration: float
    error_details: List[str]


class TopConceptsJob:
    """
    Job for generating Top Concepts.md from PageRank analysis.

    This job:
    1. Connects to Neo4j and projects a graph of (:Concept) nodes with relationships
    2. Executes PageRank algorithm using Graph Data Science library
    3. Selects top N concepts by PageRank score (configurable count or percent)
    4. Generates markdown content following the "Trending Concepts Agent" format
    5. Writes the file atomically to the vault with aclarai:id tracking
    """

    def __init__(
        self,
        config: Optional[aclaraiConfig] = None,
        neo4j_manager: Optional[Neo4jGraphManager] = None,
    ):
        """Initialize top concepts job."""
        self.config = config or load_config(validate=True)
        self.neo4j_manager = neo4j_manager or Neo4jGraphManager(self.config)

        # Get job-specific configuration
        self.job_config: TopConceptsJobConfig = self.config.scheduler.jobs.top_concepts

        # Vault path from config
        self.vault_path = Path(self.config.vault_path)
        self.target_file_path = self.vault_path / self.job_config.target_file

    def _project_concept_graph(self) -> bool:
        """
        Project the concept graph into GDS memory for PageRank analysis.

        Returns:
            True if projection was successful, False otherwise
        """
        # First, drop existing graph if it exists
        drop_graph_query = """
        CALL gds.graph.exists('concept_graph')
        YIELD exists
        WITH exists
        WHERE exists
        CALL gds.graph.drop('concept_graph')
        YIELD graphName
        RETURN graphName
        """

        # Project the graph with concept nodes and their relationships
        project_query = """
        CALL gds.graph.project(
            'concept_graph',
            'Concept',
            {
                SUPPORTS_CONCEPT: {orientation: 'REVERSE'},
                MENTIONS_CONCEPT: {orientation: 'REVERSE'},
                RELATED_TO: {orientation: 'UNDIRECTED'}
            }
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """

        try:
            # Drop existing graph if it exists
            self.neo4j_manager.execute_query(drop_graph_query)

            # Project new graph
            result = self.neo4j_manager.execute_query(project_query)

            if result:
                record = result[0]
                logger.info(
                    f"top_concepts_job._project_concept_graph: Projected graph '{record['graphName']}' with {record['nodeCount']} nodes and {record['relationshipCount']} relationships",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "top_concepts_job._project_concept_graph",
                        "graph_name": record["graphName"],
                        "node_count": record["nodeCount"],
                        "relationship_count": record["relationshipCount"],
                    },
                )
                return True
            else:
                logger.error(
                    "top_concepts_job._project_concept_graph: No result returned from graph projection",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "top_concepts_job._project_concept_graph",
                    },
                )
                return False

        except Exception as e:
            logger.error(
                f"top_concepts_job._project_concept_graph: Failed to project concept graph: {e}",
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "top_concepts_job._project_concept_graph",
                    "error": str(e),
                },
            )
            return False

    def _execute_pagerank(self) -> bool:
        """
        Execute PageRank algorithm on the projected graph.

        Returns:
            True if PageRank execution was successful, False otherwise
        """
        pagerank_query = """
        CALL gds.pageRank.write('concept_graph', {
            writeProperty: 'pagerank_score',
            maxIterations: 20,
            dampingFactor: 0.85
        })
        YIELD nodePropertiesWritten, ranIterations
        RETURN nodePropertiesWritten, ranIterations
        """

        try:
            result = self.neo4j_manager.execute_query(pagerank_query)

            if result:
                record = result[0]
                logger.info(
                    f"top_concepts_job._execute_pagerank: PageRank completed with {record['nodePropertiesWritten']} properties written in {record['ranIterations']} iterations",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "top_concepts_job._execute_pagerank",
                        "nodes_updated": record["nodePropertiesWritten"],
                        "iterations": record["ranIterations"],
                    },
                )
                return True
            else:
                logger.error(
                    "top_concepts_job._execute_pagerank: No result returned from PageRank execution",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "top_concepts_job._execute_pagerank",
                    },
                )
                return False

        except Exception as e:
            logger.error(
                f"top_concepts_job._execute_pagerank: Failed to execute PageRank: {e}",
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "top_concepts_job._execute_pagerank",
                    "error": str(e),
                },
            )
            return False

    def _get_top_concepts(self) -> List[Tuple[str, float]]:
        """
        Retrieve top N concepts by PageRank score.

        Returns:
            List of tuples containing (concept_name, pagerank_score)
        """
        # Determine limit based on configuration
        if self.job_config.count is not None:
            limit = self.job_config.count
            limit_query = """
            MATCH (c:Concept)
            WHERE c.pagerank_score IS NOT NULL
            RETURN c.name as name, c.pagerank_score as score
            ORDER BY c.pagerank_score DESC
            LIMIT $limit
            """
            query_params = {"limit": limit}
        elif self.job_config.percent is not None:
            # First get total count, then calculate limit
            count_query = """
            MATCH (c:Concept)
            WHERE c.pagerank_score IS NOT NULL
            RETURN count(c) as total_count
            """

            try:
                count_result = self.neo4j_manager.execute_query(count_query)
                if count_result and count_result[0]["total_count"] > 0:
                    total_count = count_result[0]["total_count"]
                    limit = max(1, int(total_count * self.job_config.percent / 100))
                else:
                    logger.warning(
                        "top_concepts_job._get_top_concepts: No concepts found with PageRank scores",
                        extra={
                            "service": "aclarai-scheduler",
                            "filename.function_name": "top_concepts_job._get_top_concepts",
                        },
                    )
                    return []
            except Exception as e:
                logger.error(
                    f"top_concepts_job._get_top_concepts: Failed to get concept count: {e}",
                    exc_info=True,
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "top_concepts_job._get_top_concepts",
                        "error": str(e),
                    },
                )
                return []

            limit_query = """
            MATCH (c:Concept)
            WHERE c.pagerank_score IS NOT NULL
            RETURN c.name as name, c.pagerank_score as score
            ORDER BY c.pagerank_score DESC
            LIMIT $limit
            """
            query_params = {"limit": limit}
        else:
            # Default to 25 if neither count nor percent is specified
            limit = 25
            limit_query = """
            MATCH (c:Concept)
            WHERE c.pagerank_score IS NOT NULL
            RETURN c.name as name, c.pagerank_score as score
            ORDER BY c.pagerank_score DESC
            LIMIT $limit
            """
            query_params = {"limit": limit}

        try:
            result = self.neo4j_manager.execute_query(limit_query, query_params)

            if result:
                concepts = [(record["name"], record["score"]) for record in result]
                logger.info(
                    f"top_concepts_job._get_top_concepts: Retrieved {len(concepts)} top concepts",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "top_concepts_job._get_top_concepts",
                        "concepts_count": len(concepts),
                        "limit_used": limit,
                    },
                )
                return concepts
            else:
                logger.warning(
                    "top_concepts_job._get_top_concepts: No concepts returned from query",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "top_concepts_job._get_top_concepts",
                    },
                )
                return []

        except Exception as e:
            logger.error(
                f"top_concepts_job._get_top_concepts: Failed to retrieve top concepts: {e}",
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "top_concepts_job._get_top_concepts",
                    "error": str(e),
                },
            )
            return []

    def _generate_markdown_content(self, concepts: List[Tuple[str, float]]) -> str:
        """
        Generate markdown content following the "Trending Concepts Agent" format.

        Args:
            concepts: List of (concept_name, pagerank_score) tuples

        Returns:
            Formatted markdown content
        """
        if not concepts:
            content = """## Top Concepts

*No concepts found with PageRank scores.*

"""
        else:
            content = "## Top Concepts\n\n"

            for rank, (concept_name, score) in enumerate(concepts, 1):
                # Create wikilink and description
                description = f"Ranked #{rank} with PageRank score {score:.4f}"
                content += f"- [[{concept_name}]] — {description}\n"

            content += "\n"

        # Add aclarai:id metadata for vault synchronization
        # Use current timestamp to create a unique version
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        file_id = f"file_top_concepts_{timestamp}"
        content += f"<!-- aclarai:id={file_id} ver=1 -->\n"
        content += f"^{file_id}\n"

        return content

    def _write_file_atomically(self, content: str) -> bool:
        """
        Write content to target file atomically using temp file and rename.

        Args:
            content: The markdown content to write

        Returns:
            True if write was successful, False otherwise
        """
        try:
            # Ensure target directory exists
            self.target_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Create temporary file in same directory for atomic rename
            temp_dir = self.target_file_path.parent
            temp_fd, temp_path_str = tempfile.mkstemp(
                dir=temp_dir, prefix=self.target_file_path.name + ".tmp", suffix=".md"
            )
            temp_path = Path(temp_path_str)

            # Write content to temporary file
            with os.fdopen(temp_fd, "w", encoding="utf-8") as tmp_file:
                tmp_file.write(content)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())  # Ensure data is written to disk

            # Atomic rename to target file
            os.replace(temp_path, self.target_file_path)

            logger.info(
                f"top_concepts_job._write_file_atomically: Successfully wrote {self.target_file_path}",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "top_concepts_job._write_file_atomically",
                    "target_file": str(self.target_file_path),
                    "content_length": len(content),
                },
            )
            return True

        except Exception as e:
            logger.error(
                f"top_concepts_job._write_file_atomically: Failed to write {self.target_file_path}: {e}",
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "top_concepts_job._write_file_atomically",
                    "target_file": str(self.target_file_path),
                    "error": str(e),
                },
            )

            # Clean up temp file if it exists
            if "temp_path" in locals() and temp_path.exists():
                with contextlib.suppress(OSError):
                    temp_path.unlink()

            return False

    def run_job(self) -> TopConceptsJobStats:
        """
        Execute the complete top concepts job.

        Returns:
            Dictionary with job statistics and results
        """
        start_time = time.time()
        stats: TopConceptsJobStats = {
            "success": False,
            "concepts_analyzed": 0,
            "top_concepts_selected": 0,
            "file_written": False,
            "pagerank_executed": False,
            "duration": 0.0,
            "error_details": [],
        }

        logger.info(
            "top_concepts_job.run_job: Starting Top Concepts job",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "top_concepts_job.run_job",
                "target_file": self.job_config.target_file,
                "metric": self.job_config.metric,
                "count": self.job_config.count,
                "percent": self.job_config.percent,
            },
        )

        try:
            # Step 1: Project concept graph for GDS
            if not self._project_concept_graph():
                stats["error_details"].append("Failed to project concept graph")
                return stats

            # Step 2: Execute PageRank algorithm
            if not self._execute_pagerank():
                stats["error_details"].append("Failed to execute PageRank algorithm")
                return stats

            stats["pagerank_executed"] = True

            # Step 3: Get top concepts by PageRank score
            top_concepts = self._get_top_concepts()
            stats["concepts_analyzed"] = len(top_concepts)
            stats["top_concepts_selected"] = len(top_concepts)

            if not top_concepts:
                logger.warning(
                    "top_concepts_job.run_job: No concepts found for Top Concepts generation",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "top_concepts_job.run_job",
                    },
                )
                # Still generate file with empty content

            # Step 4: Generate markdown content
            markdown_content = self._generate_markdown_content(top_concepts)

            # Step 5: Write file atomically
            if self._write_file_atomically(markdown_content):
                stats["file_written"] = True
                stats["success"] = True
                logger.info(
                    f"top_concepts_job.run_job: Successfully completed Top Concepts job with {len(top_concepts)} concepts",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "top_concepts_job.run_job",
                        "concepts_processed": len(top_concepts),
                        "target_file": str(self.target_file_path),
                    },
                )
            else:
                stats["error_details"].append("Failed to write target file")

        except Exception as e:
            error_msg = f"Unexpected error in Top Concepts job: {e}"
            logger.error(
                f"top_concepts_job.run_job: {error_msg}",
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "top_concepts_job.run_job",
                    "error": str(e),
                },
            )
            stats["error_details"].append(error_msg)

        finally:
            # 1. Clean up the temporary property written to the persistent graph
            try:
                cleanup_prop_query = "MATCH (c:Concept) WHERE c.pagerank_score IS NOT NULL REMOVE c.pagerank_score"
                self.neo4j_manager.execute_query(cleanup_prop_query, allow_dangerous_operations=True)
                logger.info(
                    "top_concepts_job.run_job: Cleaned up pagerank_score property from nodes."
                )
            except Exception as e:
                logger.warning(
                    f"top_concepts_job.run_job: Failed to clean up pagerank_score property from nodes: {e}",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "top_concepts_job.run_job",
                        "cleanup_error": str(e),
                    },
                )

            # 2. Clean up the GDS in-memory graph projection
            try:
                cleanup_graph_query = "CALL gds.graph.drop('concept_graph') YIELD graphName RETURN graphName"
                self.neo4j_manager.execute_query(cleanup_graph_query)
                logger.info(
                    "top_concepts_job.run_job: Cleaned up GDS projected graph 'concept_graph'."
                )
            except Exception as e:
                # This error is common if the graph projection failed in the first place, so log as a warning.
                logger.warning(
                    f"top_concepts_job.run_job: Failed to clean up projected GDS graph 'concept_graph' (this may be expected if projection failed): {e}",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "top_concepts_job.run_job",
                        "cleanup_error": str(e),
                    },
                )

            stats["duration"] = time.time() - start_time

        return stats
