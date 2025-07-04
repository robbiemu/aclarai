"""
aclarai Scheduler Service
This service runs periodic jobs including:
- Concept hygiene
- Vault synchronization
- Reprocessing tasks
"""

import logging
import os
import signal
import sys
import time
from typing import Optional

from aclarai_shared import load_config
from aclarai_shared.automation.pause_controller import is_paused
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from .concept_clustering_job import ConceptClusteringJob, ConceptClusteringJobStats
from .concept_highlight_refresh import (
    ConceptHighlightRefreshJob,
    ConceptHighlightRefreshJobStats,
)
from .concept_refresh import ConceptEmbeddingRefreshJob, JobStatsTypedDict
from .concept_summary_refresh import (
    ConceptSummaryRefreshJob,
    ConceptSummaryRefreshJobStats,
)
from .top_concepts_job import TopConceptsJob, TopConceptsJobStats
from .trending_topics_job import TrendingTopicsJob, TrendingTopicsJobStats
from .vault_sync import VaultSyncJob


class SchedulerService:
    """
    Main scheduler service that manages periodic jobs using APScheduler.
    Follows configuration from settings/aclarai.config.yaml and supports
    pause/resume functionality via environment variables.
    """

    def __init__(self):
        """Initialize the scheduler service."""
        self.config = load_config(validate=True)
        self.logger = logging.getLogger(__name__)
        self.scheduler: Optional[BlockingScheduler] = None
        self.vault_sync_job = VaultSyncJob(self.config)
        self.concept_refresh_job = ConceptEmbeddingRefreshJob(self.config)
        self.top_concepts_job = TopConceptsJob(self.config)
        self.trending_topics_job = TrendingTopicsJob(self.config)
        self.concept_highlight_refresh_job = ConceptHighlightRefreshJob(self.config)
        self.concept_summary_refresh_job = ConceptSummaryRefreshJob(self.config)
        self.concept_clustering_job = ConceptClusteringJob(self.config)
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, _frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(
            f"scheduler.main._signal_handler: Received signal {signum}, shutting down...",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "scheduler.main._signal_handler",
                "signal": signum,
            },
        )
        self.shutdown()
        sys.exit(0)

    def _setup_scheduler(self):
        """Set up APScheduler with job configurations."""
        # Configure APScheduler
        executors = {
            "default": ThreadPoolExecutor(max_workers=2)  # Small pool for periodic jobs
        }
        job_defaults = {
            "coalesce": True,  # Combine multiple pending executions into one
            "max_instances": 1,  # Only one instance of each job at a time
            "misfire_grace_time": 300,  # 5 minutes grace period for missed jobs
        }
        self.scheduler = BlockingScheduler(
            executors=executors, job_defaults=job_defaults, timezone="UTC"
        )
        self.logger.info(
            "scheduler.main._setup_scheduler: APScheduler configured",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "scheduler.main._setup_scheduler",
            },
        )

    def _register_jobs(self):
        """Register all periodic jobs from configuration."""
        # Check if automation is paused
        if is_paused():
            self.logger.warning(
                "scheduler.main._register_jobs: Automation is paused, jobs will not be registered",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._register_jobs",
                    "automation_paused": True,
                },
            )
            return
        # Register vault sync job
        vault_sync_config = self.config.scheduler.jobs.vault_sync
        if vault_sync_config.enabled and not vault_sync_config.manual_only:
            assert self.scheduler is not None
            self.scheduler.add_job(
                func=self._run_vault_sync_job,
                trigger=CronTrigger.from_crontab(vault_sync_config.cron),
                id="vault_sync",
                name="Vault Synchronization Job",
                replace_existing=True,
            )
            self.logger.info(
                f"scheduler.main._register_jobs: Registered vault sync job with cron '{vault_sync_config.cron}'",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._register_jobs",
                    "job_id": "vault_sync",
                    "cron": vault_sync_config.cron,
                    "description": vault_sync_config.description,
                },
            )
        elif vault_sync_config.enabled and vault_sync_config.manual_only:
            self.logger.info(
                "scheduler.main._register_jobs: Vault sync job is enabled but set to manual_only, skipping automatic scheduling",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._register_jobs",
                    "job_id": "vault_sync",
                    "manual_only": True,
                    "description": vault_sync_config.description,
                },
            )
        # Register concept embedding refresh job (placeholder for future implementation)
        concept_refresh_config = self.config.scheduler.jobs.concept_embedding_refresh
        if concept_refresh_config.enabled and not concept_refresh_config.manual_only:
            # Environment variable override
            concept_refresh_enabled = (
                os.getenv("CONCEPT_EMBEDDING_REFRESH_ENABLED", "true").lower() == "true"
            )
            concept_refresh_cron = os.getenv(
                "CONCEPT_EMBEDDING_REFRESH_CRON", concept_refresh_config.cron
            )
            if concept_refresh_enabled:
                assert self.scheduler is not None
                self.scheduler.add_job(
                    func=self._run_concept_refresh_job,
                    trigger=CronTrigger.from_crontab(concept_refresh_cron),
                    id="concept_embedding_refresh",
                    name="Concept Embedding Refresh Job",
                    replace_existing=True,
                )
                self.logger.info(
                    f"scheduler.main._register_jobs: Registered concept refresh job with cron '{concept_refresh_cron}'",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "scheduler.main._register_jobs",
                        "job_id": "concept_embedding_refresh",
                        "cron": concept_refresh_cron,
                        "description": concept_refresh_config.description,
                    },
                )
        elif concept_refresh_config.enabled and concept_refresh_config.manual_only:
            self.logger.info(
                "scheduler.main._register_jobs: Concept refresh job is enabled but set to manual_only, skipping automatic scheduling",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._register_jobs",
                    "job_id": "concept_embedding_refresh",
                    "manual_only": True,
                    "description": concept_refresh_config.description,
                },
            )

        # Register top concepts job
        top_concepts_config = self.config.scheduler.jobs.top_concepts
        if top_concepts_config.enabled and not top_concepts_config.manual_only:
            # Environment variable override
            top_concepts_enabled = (
                os.getenv("TOP_CONCEPTS_ENABLED", "true").lower() == "true"
            )
            top_concepts_cron = os.getenv("TOP_CONCEPTS_CRON", top_concepts_config.cron)
            if top_concepts_enabled:
                assert self.scheduler is not None
                self.scheduler.add_job(
                    func=self._run_top_concepts_job,
                    trigger=CronTrigger.from_crontab(top_concepts_cron),
                    id="top_concepts",
                    name="Top Concepts Job",
                    replace_existing=True,
                )
                self.logger.info(
                    f"scheduler.main._register_jobs: Registered top concepts job with cron '{top_concepts_cron}'",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "scheduler.main._register_jobs",
                        "job_id": "top_concepts",
                        "cron": top_concepts_cron,
                        "description": top_concepts_config.description,
                    },
                )
        elif top_concepts_config.enabled and top_concepts_config.manual_only:
            self.logger.info(
                "scheduler.main._register_jobs: Top concepts job is enabled but set to manual_only, skipping automatic scheduling",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._register_jobs",
                    "job_id": "top_concepts",
                    "manual_only": True,
                    "description": top_concepts_config.description,
                },
            )

        # Register trending topics job
        trending_topics_config = self.config.scheduler.jobs.trending_topics
        if trending_topics_config.enabled and not trending_topics_config.manual_only:
            # Environment variable override
            trending_topics_enabled = (
                os.getenv("TRENDING_TOPICS_ENABLED", "true").lower() == "true"
            )
            trending_topics_cron = os.getenv(
                "TRENDING_TOPICS_CRON", trending_topics_config.cron
            )
            if trending_topics_enabled:
                assert self.scheduler is not None
                self.scheduler.add_job(
                    func=self._run_trending_topics_job,
                    trigger=CronTrigger.from_crontab(trending_topics_cron),
                    id="trending_topics",
                    name="Trending Topics Job",
                    replace_existing=True,
                )
                self.logger.info(
                    f"scheduler.main._register_jobs: Registered trending topics job with cron '{trending_topics_cron}'",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "scheduler.main._register_jobs",
                        "job_id": "trending_topics",
                        "cron": trending_topics_cron,
                        "description": trending_topics_config.description,
                    },
                )
        elif trending_topics_config.enabled and trending_topics_config.manual_only:
            self.logger.info(
                "scheduler.main._register_jobs: Trending topics job is enabled but set to manual_only, skipping automatic scheduling",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._register_jobs",
                    "job_id": "trending_topics",
                    "manual_only": True,
                    "description": trending_topics_config.description,
                },
            )

        # Register concept highlight refresh job
        concept_highlight_refresh_config = (
            self.config.scheduler.jobs.concept_highlight_refresh
        )
        if (
            concept_highlight_refresh_config.enabled
            and not concept_highlight_refresh_config.manual_only
        ):
            # Environment variable override
            concept_highlight_refresh_enabled = (
                os.getenv("CONCEPT_HIGHLIGHT_REFRESH_ENABLED", "true").lower() == "true"
            )
            concept_highlight_refresh_cron = os.getenv(
                "CONCEPT_HIGHLIGHT_REFRESH_CRON", concept_highlight_refresh_config.cron
            )
            if concept_highlight_refresh_enabled:
                assert self.scheduler is not None
                self.scheduler.add_job(
                    func=self._run_concept_highlight_refresh_job,
                    trigger=CronTrigger.from_crontab(concept_highlight_refresh_cron),
                    id="concept_highlight_refresh",
                    name="Concept Highlight Refresh Job",
                    replace_existing=True,
                )
                self.logger.info(
                    f"scheduler.main._register_jobs: Registered concept highlight refresh job with cron '{concept_highlight_refresh_cron}'",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "scheduler.main._register_jobs",
                        "job_id": "concept_highlight_refresh",
                        "cron": concept_highlight_refresh_cron,
                        "description": concept_highlight_refresh_config.description,
                    },
                )
        elif (
            concept_highlight_refresh_config.enabled
            and concept_highlight_refresh_config.manual_only
        ):
            self.logger.info(
                "scheduler.main._register_jobs: Concept highlight refresh job is enabled but set to manual_only, skipping automatic scheduling",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._register_jobs",
                    "job_id": "concept_highlight_refresh",
                    "manual_only": True,
                    "description": concept_highlight_refresh_config.description,
                },
            )

        # Register concept summary refresh job
        concept_summary_refresh_config = (
            self.config.scheduler.jobs.concept_summary_refresh
        )
        if (
            concept_summary_refresh_config.enabled
            and not concept_summary_refresh_config.manual_only
        ):
            # Environment variable override
            concept_summary_refresh_enabled = (
                os.getenv("CONCEPT_SUMMARY_REFRESH_ENABLED", "true").lower() == "true"
            )
            concept_summary_refresh_cron = os.getenv(
                "CONCEPT_SUMMARY_REFRESH_CRON", concept_summary_refresh_config.cron
            )
            if concept_summary_refresh_enabled:
                assert self.scheduler is not None
                self.scheduler.add_job(
                    func=self._run_concept_summary_refresh_job,
                    trigger=CronTrigger.from_crontab(concept_summary_refresh_cron),
                    id="concept_summary_refresh",
                    name="Concept Summary Refresh Job",
                    replace_existing=True,
                )
                self.logger.info(
                    f"scheduler.main._register_jobs: Registered concept summary refresh job with cron '{concept_summary_refresh_cron}'",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "scheduler.main._register_jobs",
                        "job_id": "concept_summary_refresh",
                        "cron": concept_summary_refresh_cron,
                        "description": concept_summary_refresh_config.description,
                    },
                )
        elif (
            concept_summary_refresh_config.enabled
            and concept_summary_refresh_config.manual_only
        ):
            self.logger.info(
                "scheduler.main._register_jobs: Concept summary refresh job is enabled but set to manual_only, skipping automatic scheduling",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._register_jobs",
                    "job_id": "concept_summary_refresh",
                    "manual_only": True,
                    "description": concept_summary_refresh_config.description,
                },
            )

        # Register concept clustering job
        concept_clustering_config = self.config.scheduler.jobs.concept_clustering
        if (
            concept_clustering_config.enabled
            and not concept_clustering_config.manual_only
        ):
            # Environment variable override
            concept_clustering_enabled = (
                os.getenv("CONCEPT_CLUSTERING_ENABLED", "true").lower() == "true"
            )
            concept_clustering_cron = os.getenv(
                "CONCEPT_CLUSTERING_CRON", concept_clustering_config.cron
            )
            if concept_clustering_enabled:
                assert self.scheduler is not None
                self.scheduler.add_job(
                    func=self._run_concept_clustering_job,
                    trigger=CronTrigger.from_crontab(concept_clustering_cron),
                    id="concept_clustering",
                    name="Concept Clustering Job",
                    replace_existing=True,
                )
                self.logger.info(
                    f"scheduler.main._register_jobs: Registered concept clustering job with cron '{concept_clustering_cron}'",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "scheduler.main._register_jobs",
                        "job_id": "concept_clustering",
                        "cron": concept_clustering_cron,
                        "description": concept_clustering_config.description,
                    },
                )
        elif (
            concept_clustering_config.enabled and concept_clustering_config.manual_only
        ):
            self.logger.info(
                "scheduler.main._register_jobs: Concept clustering job is enabled but set to manual_only, skipping automatic scheduling",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._register_jobs",
                    "job_id": "concept_clustering",
                    "manual_only": True,
                    "description": concept_clustering_config.description,
                },
            )

    def _run_vault_sync_job(self):
        """Execute the vault synchronization job."""
        # Check if automation is paused
        if is_paused():
            self.logger.info(
                "scheduler.main._run_vault_sync_job: Automation is paused. Skipping job.",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_vault_sync_job",
                },
            )
            return

        job_start_time = __import__("time").time()
        job_id = f"vault_sync_{int(job_start_time)}"
        self.logger.info(
            "scheduler.main._run_vault_sync_job: Starting vault synchronization job",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "scheduler.main._run_vault_sync_job",
                "job_id": job_id,
            },
        )
        try:
            # Run the sync job
            stats = self.vault_sync_job.run_sync()
            # Log completion with statistics
            self.logger.info(
                "scheduler.main._run_vault_sync_job: Vault synchronization job completed successfully",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_vault_sync_job",
                    "job_id": job_id,
                    "duration": stats.get("duration", 0),
                    "files_processed": stats.get("files_processed", 0),
                    "blocks_processed": stats.get("blocks_processed", 0),
                    "blocks_updated": stats.get("blocks_updated", 0),
                    "blocks_new": stats.get("blocks_new", 0),
                    "errors": stats.get("errors", 0),
                },
            )
        except Exception as e:
            self.logger.error(
                f"scheduler.main._run_vault_sync_job: Vault synchronization job failed: {e}",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_vault_sync_job",
                    "job_id": job_id,
                    "error": str(e),
                },
            )
            # Re-raise to let APScheduler handle the error
            raise

    def _run_concept_refresh_job(self) -> JobStatsTypedDict:
        """Execute the concept embedding refresh job."""
        # Check if automation is paused
        if is_paused():
            self.logger.info(
                "scheduler.main._run_concept_refresh_job: Automation is paused. Skipping job.",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_concept_refresh_job",
                },
            )
            return {
                "success": False,
                "concepts_processed": 0,
                "concepts_updated": 0,
                "concepts_skipped": 0,
                "errors": 0,
                "error_details": ["Automation is paused"],
                "duration": 0,
            }

        job_start_time = time.time()
        job_id = f"concept_refresh_{int(job_start_time)}"
        self.logger.info(
            "scheduler.main._run_concept_refresh_job: Starting concept embedding refresh job",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "scheduler.main._run_concept_refresh_job",
                "job_id": job_id,
            },
        )
        try:
            # Execute the concept refresh job
            job_stats = self.concept_refresh_job.run_job()
            # Log completion with statistics
            self.logger.info(
                "scheduler.main._run_concept_refresh_job: Concept embedding refresh job completed",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_concept_refresh_job",
                    "job_id": job_id,
                    "duration": job_stats["duration"],
                    "concepts_processed": job_stats["concepts_processed"],
                    "concepts_updated": job_stats["concepts_updated"],
                    "concepts_skipped": job_stats["concepts_skipped"],
                    "errors": job_stats["errors"],
                    "success": job_stats["success"],
                },
            )
            return job_stats
        except Exception as e:
            self.logger.error(
                "scheduler.main._run_concept_refresh_job: Concept embedding refresh job failed",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_concept_refresh_job",
                    "job_id": job_id,
                    "error": str(e),
                },
            )
            return {
                "success": False,
                "concepts_processed": 0,
                "concepts_updated": 0,
                "concepts_skipped": 0,
                "errors": 1,
                "error_details": [str(e)],
                "duration": time.time() - job_start_time,
            }

    def _run_top_concepts_job(self) -> TopConceptsJobStats:
        """Execute the top concepts job."""
        # Check if automation is paused
        if is_paused():
            self.logger.info(
                "scheduler.main._run_top_concepts_job: Automation is paused. Skipping job.",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_top_concepts_job",
                },
            )
            return {
                "success": True,
                "concepts_analyzed": 0,
                "top_concepts_selected": 0,
                "file_written": False,
                "pagerank_executed": False,
                "duration": 0.0,
                "error_details": ["Automation is paused"],
            }

        job_start_time = time.time()
        job_id = f"top_concepts_{int(job_start_time)}"
        self.logger.info(
            "scheduler.main._run_top_concepts_job: Starting top concepts job",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "scheduler.main._run_top_concepts_job",
                "job_id": job_id,
            },
        )
        try:
            # Run the top concepts job
            stats = self.top_concepts_job.run_job()
            # Log completion with statistics
            self.logger.info(
                "scheduler.main._run_top_concepts_job: Top concepts job completed",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_top_concepts_job",
                    "job_id": job_id,
                    "success": stats["success"],
                    "concepts_analyzed": stats["concepts_analyzed"],
                    "top_concepts_selected": stats["top_concepts_selected"],
                    "file_written": stats["file_written"],
                    "pagerank_executed": stats["pagerank_executed"],
                    "duration": stats["duration"],
                },
            )
            return stats
        except Exception as e:
            self.logger.error(
                "scheduler.main._run_top_concepts_job: Top concepts job failed",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_top_concepts_job",
                    "job_id": job_id,
                    "error": str(e),
                },
            )
            return {
                "success": False,
                "concepts_analyzed": 0,
                "top_concepts_selected": 0,
                "file_written": False,
                "pagerank_executed": False,
                "duration": time.time() - job_start_time,
                "error_details": [str(e)],
            }

    def _run_trending_topics_job(self) -> TrendingTopicsJobStats:
        """Execute the trending topics job."""
        # Check if automation is paused
        if is_paused():
            self.logger.info(
                "scheduler.main._run_trending_topics_job: Automation is paused. Skipping job.",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_trending_topics_job",
                },
            )
            return {
                "success": True,
                "concepts_analyzed": 0,
                "trending_concepts_selected": 0,
                "file_written": False,
                "window_start": "",
                "window_end": "",
                "duration": 0.0,
                "error_details": ["Automation is paused"],
            }

        job_start_time = time.time()
        job_id = f"trending_topics_{int(job_start_time)}"
        self.logger.info(
            "scheduler.main._run_trending_topics_job: Starting trending topics job",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "scheduler.main._run_trending_topics_job",
                "job_id": job_id,
            },
        )
        try:
            # Run the trending topics job
            stats = self.trending_topics_job.run_job()
            # Log completion with statistics
            self.logger.info(
                "scheduler.main._run_trending_topics_job: Trending topics job completed",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_trending_topics_job",
                    "job_id": job_id,
                    "success": stats["success"],
                    "concepts_analyzed": stats["concepts_analyzed"],
                    "trending_concepts_selected": stats["trending_concepts_selected"],
                    "file_written": stats["file_written"],
                    "window_start": stats["window_start"],
                    "window_end": stats["window_end"],
                    "duration": stats["duration"],
                },
            )
            return stats
        except Exception as e:
            self.logger.error(
                "scheduler.main._run_trending_topics_job: Trending topics job failed",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_trending_topics_job",
                    "job_id": job_id,
                    "error": str(e),
                },
            )
            return {
                "success": False,
                "concepts_analyzed": 0,
                "trending_concepts_selected": 0,
                "file_written": False,
                "window_start": "",
                "window_end": "",
                "duration": time.time() - job_start_time,
                "error_details": [str(e)],
            }

    def _run_concept_highlight_refresh_job(self) -> ConceptHighlightRefreshJobStats:
        """Execute the concept highlight refresh job."""
        # Check if automation is paused
        if is_paused():
            self.logger.info(
                "scheduler.main._run_concept_highlight_refresh_job: Automation is paused. Skipping job.",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_concept_highlight_refresh_job",
                },
            )
            return {
                "success": True,
                "top_concepts_stats": {
                    "success": False,
                    "concepts_analyzed": 0,
                    "top_concepts_selected": 0,
                    "file_written": False,
                    "pagerank_executed": False,
                    "duration": 0.0,
                    "error_details": ["Automation is paused"],
                },
                "trending_topics_stats": {
                    "success": False,
                    "concepts_analyzed": 0,
                    "trending_concepts_selected": 0,
                    "file_written": False,
                    "window_start": "",
                    "window_end": "",
                    "duration": 0.0,
                    "error_details": ["Automation is paused"],
                },
                "duration": 0.0,
                "error_details": ["Automation is paused"],
            }

        job_start_time = time.time()
        job_id = f"concept_highlight_refresh_{int(job_start_time)}"
        self.logger.info(
            "scheduler.main._run_concept_highlight_refresh_job: Starting concept highlight refresh job",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "scheduler.main._run_concept_highlight_refresh_job",
                "job_id": job_id,
            },
        )
        try:
            # Run the concept highlight refresh job
            stats = self.concept_highlight_refresh_job.run_job()
            # Log completion with statistics
            self.logger.info(
                "scheduler.main._run_concept_highlight_refresh_job: Concept highlight refresh job completed",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_concept_highlight_refresh_job",
                    "job_id": job_id,
                    "success": stats["success"],
                    "top_concepts_success": stats["top_concepts_stats"]["success"],
                    "trending_topics_success": stats["trending_topics_stats"][
                        "success"
                    ],
                    "duration": stats["duration"],
                },
            )
            return stats
        except Exception as e:
            self.logger.error(
                "scheduler.main._run_concept_highlight_refresh_job: Concept highlight refresh job failed",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_concept_highlight_refresh_job",
                    "job_id": job_id,
                    "error": str(e),
                },
            )
            return {
                "success": False,
                "top_concepts_stats": {
                    "success": False,
                    "concepts_analyzed": 0,
                    "top_concepts_selected": 0,
                    "file_written": False,
                    "pagerank_executed": False,
                    "duration": 0.0,
                    "error_details": [str(e)],
                },
                "trending_topics_stats": {
                    "success": False,
                    "concepts_analyzed": 0,
                    "trending_concepts_selected": 0,
                    "file_written": False,
                    "window_start": "",
                    "window_end": "",
                    "duration": 0.0,
                    "error_details": [str(e)],
                },
                "duration": time.time() - job_start_time,
                "error_details": [str(e)],
            }

    def _run_concept_summary_refresh_job(self) -> ConceptSummaryRefreshJobStats:
        """Execute the concept summary refresh job."""
        # Check if automation is paused
        if is_paused():
            self.logger.info(
                "scheduler.main._run_concept_summary_refresh_job: Automation is paused. Skipping job.",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_concept_summary_refresh_job",
                },
            )
            return {
                "success": True,
                "concepts_processed": 0,
                "concepts_updated": 0,
                "concepts_skipped": 0,
                "errors": 0,
                "duration": 0.0,
                "error_details": ["Automation is paused"],
            }

        job_start_time = time.time()
        job_id = f"concept_summary_refresh_{int(job_start_time)}"
        self.logger.info(
            "scheduler.main._run_concept_summary_refresh_job: Starting concept summary refresh job",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "scheduler.main._run_concept_summary_refresh_job",
                "job_id": job_id,
            },
        )
        try:
            # Run the concept summary refresh job
            stats = self.concept_summary_refresh_job.run_job()
            # Log completion with statistics
            self.logger.info(
                "scheduler.main._run_concept_summary_refresh_job: Concept summary refresh job completed",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_concept_summary_refresh_job",
                    "job_id": job_id,
                    "success": stats["success"],
                    "concepts_processed": stats["concepts_processed"],
                    "concepts_updated": stats["concepts_updated"],
                    "concepts_skipped": stats["concepts_skipped"],
                    "errors": stats["errors"],
                    "duration": stats["duration"],
                },
            )
            return stats
        except Exception as e:
            self.logger.error(
                "scheduler.main._run_concept_summary_refresh_job: Concept summary refresh job failed",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_concept_summary_refresh_job",
                    "job_id": job_id,
                    "error": str(e),
                },
            )
            return {
                "success": False,
                "concepts_processed": 0,
                "concepts_updated": 0,
                "concepts_skipped": 0,
                "errors": 1,
                "duration": time.time() - job_start_time,
                "error_details": [str(e)],
            }

    def _run_concept_clustering_job(self) -> ConceptClusteringJobStats:
        """Execute the concept clustering job."""
        # Check if automation is paused
        if is_paused():
            self.logger.info(
                "scheduler.main._run_concept_clustering_job: Automation is paused. Skipping job.",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_concept_clustering_job",
                },
            )
            return {
                "success": True,
                "concepts_processed": 0,
                "clusters_formed": 0,
                "concepts_clustered": 0,
                "concepts_outliers": 0,
                "cache_updated": False,
                "duration": 0.0,
                "error_details": ["Automation is paused"],
            }

        job_start_time = time.time()
        job_id = f"concept_clustering_{int(job_start_time)}"
        self.logger.info(
            "scheduler.main._run_concept_clustering_job: Starting concept clustering job",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "scheduler.main._run_concept_clustering_job",
                "job_id": job_id,
            },
        )
        try:
            # Run the concept clustering job
            stats = self.concept_clustering_job.run_job()
            # Log completion with statistics
            self.logger.info(
                "scheduler.main._run_concept_clustering_job: Concept clustering job completed",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_concept_clustering_job",
                    "job_id": job_id,
                    "success": stats["success"],
                    "concepts_processed": stats["concepts_processed"],
                    "clusters_formed": stats["clusters_formed"],
                    "concepts_clustered": stats["concepts_clustered"],
                    "concepts_outliers": stats["concepts_outliers"],
                    "cache_updated": stats["cache_updated"],
                    "duration": stats["duration"],
                },
            )
            return stats
        except Exception as e:
            self.logger.error(
                "scheduler.main._run_concept_clustering_job: Concept clustering job failed",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main._run_concept_clustering_job",
                    "job_id": job_id,
                    "error": str(e),
                },
            )
            return {
                "success": False,
                "concepts_processed": 0,
                "clusters_formed": 0,
                "concepts_clustered": 0,
                "concepts_outliers": 0,
                "cache_updated": False,
                "duration": time.time() - job_start_time,
                "error_details": [str(e)],
            }

    def run(self):
        """Start the scheduler service."""
        try:
            self.logger.info(
                "scheduler.main.run: Starting aclarai Scheduler service...",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main.run",
                },
            )
            # Log configuration details
            self.logger.info(
                f"scheduler.main.run: Vault path: {self.config.vault_path}",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main.run",
                    "vault_path": self.config.vault_path,
                },
            )
            # Set up and start scheduler
            self._setup_scheduler()
            self._register_jobs()
            assert self.scheduler is not None
            # Check if any jobs were registered
            job_count = len(self.scheduler.get_jobs())
            if job_count == 0:
                self.logger.warning(
                    "scheduler.main.run: No jobs registered, scheduler will run but do nothing",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "scheduler.main.run",
                        "job_count": job_count,
                    },
                )
            else:
                self.logger.info(
                    f"scheduler.main.run: Starting scheduler with {job_count} jobs",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "scheduler.main.run",
                        "job_count": job_count,
                    },
                )
            # Start the scheduler (blocking call)
            self.scheduler.start()
        except ValueError as e:
            # Configuration validation error
            self.logger.error(
                f"scheduler.main.run: Configuration error: {e}",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main.run",
                    "error": str(e),
                },
            )
            self.logger.error(
                "scheduler.main.run: Please check your .env file and ensure all required variables are set.",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main.run",
                },
            )
            raise
        except KeyboardInterrupt:
            self.logger.info(
                "scheduler.main.run: Received keyboard interrupt, shutting down...",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main.run",
                },
            )
            self.shutdown()
        except Exception as e:
            self.logger.error(
                f"scheduler.main.run: Error in Scheduler service: {e}",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "scheduler.main.run",
                    "error": str(e),
                },
            )
            raise

    def shutdown(self):
        """Shut down the scheduler service gracefully."""
        self.logger.info(
            "scheduler.main.shutdown: Shutting down Scheduler service...",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "scheduler.main.shutdown",
            },
        )
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=True)
        # Clean up vault sync job resources
        if hasattr(self, "vault_sync_job"):
            self.vault_sync_job.close()


def main():
    """Main entry point for the Scheduler service."""
    service = SchedulerService()
    service.run()


if __name__ == "__main__":
    main()
