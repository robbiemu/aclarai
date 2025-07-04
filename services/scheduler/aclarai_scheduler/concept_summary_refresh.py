"""
Concept Summary Refresh Job

This job uses the ConceptSummaryAgent to generate detailed Markdown pages
for all canonical concepts in the knowledge graph.
"""

import logging
import time
from typing import List, Optional, TypedDict

from aclarai_shared.concept_summary_agent import ConceptSummaryAgent
from aclarai_shared.config import aclaraiConfig
from aclarai_shared.graph.neo4j_manager import Neo4jGraphManager

logger = logging.getLogger(__name__)


class ConceptSummaryRefreshJobStats(TypedDict):
    """Statistics for the concept summary refresh job."""

    success: bool
    concepts_processed: int
    concepts_updated: int
    concepts_skipped: int
    errors: int
    duration: float
    error_details: List[str]


class ConceptSummaryRefreshJob:
    """
    Job that generates concept summary pages for all canonical concepts.

    This job uses the ConceptSummaryAgent to generate detailed Markdown pages
    for each canonical concept, following the format specified in the
    documentation and ensuring atomic writes and vault synchronization support.
    """

    def __init__(self, config: Optional[aclaraiConfig] = None):
        """
        Initialize the concept summary refresh job.

        Args:
            config: aclarai configuration (loads default if None)
        """
        self.config = config or aclaraiConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize the concept summary agent
        self.concept_summary_agent = ConceptSummaryAgent(config)

        # Initialize Neo4j manager for querying concepts
        self.neo4j_manager = Neo4jGraphManager(config)

    def run_job(self) -> ConceptSummaryRefreshJobStats:
        """
        Execute the concept summary refresh job.

        This queries all canonical concepts from the knowledge graph and
        generates/updates their summary pages using the ConceptSummaryAgent.

        Returns:
            ConceptSummaryRefreshJobStats: Statistics from the job execution
        """
        start_time = time.time()
        job_stats: ConceptSummaryRefreshJobStats = {
            "success": True,
            "concepts_processed": 0,
            "concepts_updated": 0,
            "concepts_skipped": 0,
            "errors": 0,
            "duration": 0.0,
            "error_details": [],
        }

        self.logger.info(
            "concept_summary_refresh.run_job: Starting concept summary refresh job",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "concept_summary_refresh.run_job",
            },
        )

        try:
            # Query all canonical concepts from the knowledge graph
            concepts = self._get_canonical_concepts()

            self.logger.info(
                f"concept_summary_refresh.run_job: Found {len(concepts)} canonical concepts",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_summary_refresh.run_job",
                    "concept_count": len(concepts),
                },
            )

            # Process each concept
            for concept_name in concepts:
                try:
                    processed, updated = self._process_concept(concept_name)

                    if processed:
                        job_stats["concepts_processed"] += 1
                        if updated:
                            job_stats["concepts_updated"] += 1
                    else:
                        job_stats["concepts_skipped"] += 1

                except Exception as e:
                    self.logger.error(
                        f"concept_summary_refresh.run_job: Error processing concept '{concept_name}'",
                        extra={
                            "service": "aclarai-scheduler",
                            "filename.function_name": "concept_summary_refresh.run_job",
                            "concept_name": concept_name,
                            "error": str(e),
                        },
                    )
                    job_stats["errors"] += 1
                    job_stats["error_details"].append(
                        f"Concept '{concept_name}': {str(e)}"
                    )

            # Job succeeds if we processed at least one concept without catastrophic failure
            job_stats["success"] = (
                job_stats["concepts_processed"] > 0 or len(concepts) == 0
            )

        except Exception as e:
            self.logger.error(
                "concept_summary_refresh.run_job: Concept summary refresh job failed",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_summary_refresh.run_job",
                    "error": str(e),
                },
            )
            job_stats["success"] = False
            job_stats["error_details"].append(str(e))

        job_stats["duration"] = time.time() - start_time

        self.logger.info(
            "concept_summary_refresh.run_job: Concept summary refresh job completed",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "concept_summary_refresh.run_job",
                "success": job_stats["success"],
                "duration": job_stats["duration"],
                "concepts_processed": job_stats["concepts_processed"],
                "concepts_updated": job_stats["concepts_updated"],
                "concepts_skipped": job_stats["concepts_skipped"],
                "errors": job_stats["errors"],
            },
        )

        return job_stats

    def _get_canonical_concepts(self) -> List[str]:
        """
        Query all canonical concepts from the knowledge graph.

        Returns:
            List[str]: List of canonical concept names
        """
        query = """
        MATCH (c:Concept {is_canonical: true})
        RETURN c.name as name
        ORDER BY c.name
        """

        try:
            with self.neo4j_manager.driver.session() as session:
                result = session.run(query)
                concepts = [record["name"] for record in result]
                return concepts
        except Exception as e:
            self.logger.error(
                "concept_summary_refresh._get_canonical_concepts: Failed to query canonical concepts",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_summary_refresh._get_canonical_concepts",
                    "error": str(e),
                },
            )
            return []

    def _get_concept_details(self, concept_name: str) -> Optional[dict]:
        """
        Query concept details from the knowledge graph.

        Returns:
            dict: Concept details
        """
        query = """
        MATCH (c:Concept {name: $concept_name})
        RETURN c.id as id, c.name as text, c.aclarai_id as aclarai_id, c.version as version
        LIMIT 1
        """

        try:
            with self.neo4j_manager.driver.session() as session:
                result = session.run(query, concept_name=concept_name)
                record = result.single()
                return dict(record) if record else None
        except Exception as e:
            self.logger.error(
                f"concept_summary_refresh._get_concept_details: Failed to query concept details for '{concept_name}'",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_summary_refresh._get_concept_details",
                    "concept_name": concept_name,
                    "error": str(e),
                },
            )
            return None

    def _process_concept(self, concept_name: str) -> tuple[bool, bool]:
        """
        Process a single concept to generate its summary page.

        Args:
            concept_name: Name of the concept to process

        Returns:
            tuple[bool, bool]: (processed, updated) - whether the concept was processed and updated
        """
        try:
            # First, get the concept details from the graph
            concept_details = self._get_concept_details(concept_name)
            if not concept_details:
                self.logger.warning(
                    f"concept_summary_refresh._process_concept: Could not find details for concept '{concept_name}'",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_summary_refresh._process_concept",
                        "concept_name": concept_name,
                    },
                )
                return False, False

            # Check if concept should be skipped based on configuration
            if (
                self.config.concept_summaries.skip_if_no_claims
                and not self._concept_has_claims(concept_name)
            ):
                self.logger.info(
                    f"concept_summary_refresh._process_concept: Skipping concept '{concept_name}' - no claims",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_summary_refresh._process_concept",
                        "concept_name": concept_name,
                        "reason": "no_claims",
                    },
                )
                return False, False

            # Use the ConceptSummaryAgent to generate the summary page
            result = self.concept_summary_agent.generate_concept_page(concept_details)

            if result:
                self.logger.info(
                    f"concept_summary_refresh._process_concept: Successfully processed concept '{concept_name}'",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_summary_refresh._process_concept",
                        "concept_name": concept_name,
                    },
                )
                return True, True
            else:
                self.logger.warning(
                    f"concept_summary_refresh._process_concept: Failed to process concept '{concept_name}'",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_summary_refresh._process_concept",
                        "concept_name": concept_name,
                    },
                )
                return False, False

        except Exception as e:
            self.logger.error(
                f"concept_summary_refresh._process_concept: Error processing concept '{concept_name}'",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_summary_refresh._process_concept",
                    "concept_name": concept_name,
                    "error": str(e),
                },
            )
            raise

    def _concept_has_claims(self, concept_name: str) -> bool:
        """
        Check if a concept has any supporting claims.

        Args:
            concept_name: Name of the concept to check

        Returns:
            bool: True if the concept has claims, False otherwise
        """
        query = """
        MATCH (c:Concept {name: $concept_name, is_canonical: true})
        MATCH (c)-[:SUPPORTS_CONCEPT|MENTIONS_CONCEPT]-(claim:Claim)
        RETURN count(claim) > 0 as has_claims
        """

        try:
            with self.neo4j_manager.driver.session() as session:
                result = session.run(query, concept_name=concept_name)
                record = result.single()
                return record["has_claims"] if record else False
        except Exception as e:
            self.logger.error(
                f"concept_summary_refresh._concept_has_claims: Failed to check claims for concept '{concept_name}'",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_summary_refresh._concept_has_claims",
                    "concept_name": concept_name,
                    "error": str(e),
                },
            )
            # If we can't check, assume it has claims to be safe
            return True
