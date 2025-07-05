"""
Subject Summary Refresh Job

This job uses the SubjectSummaryAgent to generate [[Subject:XYZ]] Markdown pages
for all concept clusters identified by the concept clustering job.
"""

import logging
import time
from typing import List, Optional, TypedDict

from aclarai_shared.config import aclaraiConfig
from aclarai_shared.graph.neo4j_manager import Neo4jGraphManager
from aclarai_shared.subject_summary_agent import SubjectSummaryAgent
from aclarai_shared.tools.vector_store_manager import aclaraiVectorStoreManager

from .concept_clustering_job import ConceptClusteringJob

logger = logging.getLogger(__name__)


class SubjectSummaryRefreshJobStats(TypedDict):
    """Statistics for the subject summary refresh job."""

    success: bool
    clusters_processed: int
    subjects_generated: int
    subjects_skipped: int
    errors: int
    duration: float
    error_details: List[str]


class SubjectSummaryRefreshJob:
    """
    Job that generates subject summary pages for all concept clusters.

    This job uses the SubjectSummaryAgent to generate detailed [[Subject:XYZ]]
    Markdown pages for each cluster of related concepts, following the format
    specified in the documentation and ensuring atomic writes and vault
    synchronization support.
    """

    def __init__(
        self,
        config: Optional[aclaraiConfig] = None,
        neo4j_manager: Optional[Neo4jGraphManager] = None,
        vector_store_manager: Optional[aclaraiVectorStoreManager] = None,
        clustering_job: Optional[ConceptClusteringJob] = None,
    ):
        """
        Initialize the subject summary refresh job.

        Args:
            config: aclarai configuration (loads default if None)
            neo4j_manager: Neo4j graph manager (creates new if None)
            vector_store_manager: Vector store manager (creates new if None)
            clustering_job: Concept clustering job instance (creates new if None)
        """
        if config is None:
            from aclarai_shared.config import load_config

            config = load_config(validate=False)
        self.config = config

        self.neo4j_manager = neo4j_manager or Neo4jGraphManager(config)
        self.vector_store_manager = vector_store_manager or aclaraiVectorStoreManager(
            config
        )
        self.clustering_job = clustering_job or ConceptClusteringJob(
            config,
            self.neo4j_manager,
            # Pass the vector store from manager
            getattr(self.vector_store_manager, "_vector_store", None)
            if self.vector_store_manager
            else None,
        )

        # Initialize the subject summary agent
        self.subject_agent = SubjectSummaryAgent(
            config=config,
            neo4j_manager=self.neo4j_manager,
            vector_store_manager=self.vector_store_manager,
            clustering_job=self.clustering_job,
        )

        logger.debug(
            "Initialized SubjectSummaryRefreshJob",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "subject_summary_refresh.SubjectSummaryRefreshJob.__init__",
            },
        )

    def run_job(self) -> SubjectSummaryRefreshJobStats:
        """
        Execute the subject summary refresh job.

        Returns:
            Dictionary with job execution statistics
        """
        start_time = time.time()
        stats: SubjectSummaryRefreshJobStats = {
            "success": False,
            "clusters_processed": 0,
            "subjects_generated": 0,
            "subjects_skipped": 0,
            "errors": 0,
            "duration": 0.0,
            "error_details": [],
        }

        logger.info(
            "Starting subject summary refresh job",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "subject_summary_refresh.SubjectSummaryRefreshJob.run_job",
            },
        )

        try:
            # First, ensure we have fresh cluster assignments
            # Run the clustering job if no assignments are cached or if they're stale
            cluster_assignments = self.clustering_job.get_cluster_assignments()

            if cluster_assignments is None:
                logger.info(
                    "No cluster assignments available, running concept clustering job first",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "subject_summary_refresh.SubjectSummaryRefreshJob.run_job",
                    },
                )

                # Run the clustering job to generate fresh assignments
                clustering_stats = self.clustering_job.run_job()

                if not clustering_stats["success"]:
                    error_msg = "Concept clustering job failed"
                    stats["error_details"].append(error_msg)
                    logger.error(
                        error_msg,
                        extra={
                            "service": "aclarai-scheduler",
                            "filename.function_name": "subject_summary_refresh.SubjectSummaryRefreshJob.run_job",
                            "clustering_errors": clustering_stats.get(
                                "error_details", []
                            ),
                        },
                    )
                    return stats

                logger.info(
                    f"Concept clustering completed with {clustering_stats['clusters_formed']} clusters",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "subject_summary_refresh.SubjectSummaryRefreshJob.run_job",
                        "clusters_formed": clustering_stats["clusters_formed"],
                        "concepts_clustered": clustering_stats["concepts_clustered"],
                    },
                )

            # Run the subject summary agent
            agent_result = self.subject_agent.run_agent()

            # Update stats with agent results
            stats.update(
                {
                    "success": agent_result["success"],
                    "clusters_processed": agent_result["clusters_processed"],
                    "subjects_generated": agent_result["subjects_generated"],
                    "subjects_skipped": agent_result["subjects_skipped"],
                    "errors": agent_result["errors"],
                    "error_details": agent_result["error_details"],
                }
            )

            if agent_result["success"]:
                logger.info(
                    f"Subject summary refresh completed successfully: "
                    f"generated {stats['subjects_generated']} subjects, "
                    f"skipped {stats['subjects_skipped']}, "
                    f"errors {stats['errors']}",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "subject_summary_refresh.SubjectSummaryRefreshJob.run_job",
                        **stats,
                    },
                )
            else:
                logger.error(
                    f"Subject summary refresh failed with {stats['errors']} errors",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "subject_summary_refresh.SubjectSummaryRefreshJob.run_job",
                        **stats,
                    },
                )

        except Exception as e:
            error_msg = f"Unexpected error in subject summary refresh job: {e}"
            logger.error(
                error_msg,
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "subject_summary_refresh.SubjectSummaryRefreshJob.run_job",
                    "error": str(e),
                },
            )
            stats["errors"] = 1
            stats["error_details"].append(error_msg)

        finally:
            stats["duration"] = time.time() - start_time

        return stats
