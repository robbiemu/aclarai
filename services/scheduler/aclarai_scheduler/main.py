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
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from .concept_refresh import ConceptEmbeddingRefreshJob
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
        # Get automation pause state
        automation_pause = os.getenv("AUTOMATION_PAUSE", "false").lower() == "true"
        if automation_pause:
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

    def _run_vault_sync_job(self):
        """Execute the vault synchronization job."""
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

    def _run_concept_refresh_job(self):
        """Execute the concept embedding refresh job."""
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
                "error": str(e),
                "duration": time.time() - job_start_time,
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
