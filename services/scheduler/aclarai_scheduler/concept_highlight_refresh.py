"""
Concept Highlight Refresh Job

This job combines the Top Concepts and Trending Topics jobs to generate
both highlight files in a single scheduled execution.
"""

import logging
import time
from typing import Dict, List, Optional, TypedDict

from aclarai_shared.config import aclaraiConfig

from .top_concepts_job import TopConceptsJob, TopConceptsJobStats
from .trending_topics_job import TrendingTopicsJob, TrendingTopicsJobStats

logger = logging.getLogger(__name__)


class ConceptHighlightRefreshJobStats(TypedDict):
    """Statistics for the concept highlight refresh job."""
    success: bool
    top_concepts_stats: TopConceptsJobStats
    trending_topics_stats: TrendingTopicsJobStats
    duration: float
    error_details: List[str]


class ConceptHighlightRefreshJob:
    """
    Job that combines Top Concepts and Trending Topics generation.
    
    This job runs both the top concepts analysis and trending topics analysis
    to generate both highlight files, ensuring they use atomic writes and
    support vault synchronization.
    """

    def __init__(self, config: Optional[aclaraiConfig] = None):
        """
        Initialize the concept highlight refresh job.
        
        Args:
            config: aclarai configuration (loads default if None)
        """
        self.config = config or aclaraiConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize the constituent jobs
        self.top_concepts_job = TopConceptsJob(config)
        self.trending_topics_job = TrendingTopicsJob(config)

    def run_job(self) -> ConceptHighlightRefreshJobStats:
        """
        Execute the concept highlight refresh job.
        
        This runs both the top concepts job and trending topics job,
        collecting their statistics and handling any errors.
        
        Returns:
            ConceptHighlightRefreshJobStats: Statistics from the job execution
        """
        start_time = time.time()
        job_stats: ConceptHighlightRefreshJobStats = {
            "success": True,
            "top_concepts_stats": {
                "success": False,
                "concepts_analyzed": 0,
                "top_concepts_selected": 0,
                "file_written": False,
                "pagerank_executed": False,
                "duration": 0.0,
                "error_details": [],
            },
            "trending_topics_stats": {
                "success": False,
                "concepts_analyzed": 0,
                "trending_concepts_selected": 0,
                "file_written": False,
                "window_start": "",
                "window_end": "",
                "duration": 0.0,
                "error_details": [],
            },
            "duration": 0.0,
            "error_details": [],
        }
        
        self.logger.info(
            "concept_highlight_refresh.run_job: Starting concept highlight refresh job",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "concept_highlight_refresh.run_job",
            },
        )
        
        try:
            # Run top concepts job
            self.logger.info(
                "concept_highlight_refresh.run_job: Running top concepts job",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_highlight_refresh.run_job",
                },
            )
            job_stats["top_concepts_stats"] = self.top_concepts_job.run_job()
            
            if not job_stats["top_concepts_stats"]["success"]:
                self.logger.warning(
                    "concept_highlight_refresh.run_job: Top concepts job failed",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_highlight_refresh.run_job",
                        "top_concepts_errors": job_stats["top_concepts_stats"]["error_details"],
                    },
                )
                job_stats["error_details"].extend(
                    [f"Top concepts: {error}" for error in job_stats["top_concepts_stats"]["error_details"]]
                )
            
            # Run trending topics job
            self.logger.info(
                "concept_highlight_refresh.run_job: Running trending topics job",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_highlight_refresh.run_job",
                },
            )
            job_stats["trending_topics_stats"] = self.trending_topics_job.run_job()
            
            if not job_stats["trending_topics_stats"]["success"]:
                self.logger.warning(
                    "concept_highlight_refresh.run_job: Trending topics job failed",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_highlight_refresh.run_job",
                        "trending_topics_errors": job_stats["trending_topics_stats"]["error_details"],
                    },
                )
                job_stats["error_details"].extend(
                    [f"Trending topics: {error}" for error in job_stats["trending_topics_stats"]["error_details"]]
                )
            
            # Overall success is true if at least one job succeeded
            job_stats["success"] = (
                job_stats["top_concepts_stats"]["success"] or 
                job_stats["trending_topics_stats"]["success"]
            )
            
        except Exception as e:
            self.logger.error(
                "concept_highlight_refresh.run_job: Concept highlight refresh job failed",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_highlight_refresh.run_job",
                    "error": str(e),
                },
            )
            job_stats["success"] = False
            job_stats["error_details"].append(str(e))
        
        job_stats["duration"] = time.time() - start_time
        
        self.logger.info(
            "concept_highlight_refresh.run_job: Concept highlight refresh job completed",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "concept_highlight_refresh.run_job",
                "success": job_stats["success"],
                "duration": job_stats["duration"],
                "top_concepts_success": job_stats["top_concepts_stats"]["success"],
                "trending_topics_success": job_stats["trending_topics_stats"]["success"],
                "error_count": len(job_stats["error_details"]),
            },
        )
        
        return job_stats