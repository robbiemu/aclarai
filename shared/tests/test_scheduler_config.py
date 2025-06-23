"""
Unit tests for scheduler configuration dataclasses.
"""

from aclarai_shared.config import JobConfig, SchedulerConfig, SchedulerJobsConfig


class TestSchedulerConfigClasses:
    """Test scheduler configuration dataclasses."""

    def test_job_config_defaults(self):
        """Test JobConfig default values."""
        job = JobConfig()

        assert job.enabled is True
        assert job.manual_only is False
        assert job.cron == "0 3 * * *"
        assert job.description == ""

    def test_job_config_with_values(self):
        """Test JobConfig with custom values."""
        job = JobConfig(
            enabled=False,
            manual_only=True,
            cron="*/15 * * * *",
            description="Custom job description",
        )

        assert job.enabled is False
        assert job.manual_only is True
        assert job.cron == "*/15 * * * *"
        assert job.description == "Custom job description"

    def test_scheduler_jobs_config_defaults(self):
        """Test SchedulerJobsConfig default values."""
        jobs = SchedulerJobsConfig()

        # Test concept_embedding_refresh defaults
        assert jobs.concept_embedding_refresh.enabled is True
        assert jobs.concept_embedding_refresh.manual_only is False
        assert jobs.concept_embedding_refresh.cron == "0 3 * * *"
        assert (
            "Refresh concept embeddings" in jobs.concept_embedding_refresh.description
        )

        # Test vault_sync defaults
        assert jobs.vault_sync.enabled is True
        assert jobs.vault_sync.manual_only is False
        assert jobs.vault_sync.cron == "*/30 * * * *"
        assert "Sync vault files" in jobs.vault_sync.description

    def test_scheduler_config_defaults(self):
        """Test SchedulerConfig default values."""
        scheduler = SchedulerConfig()

        assert isinstance(scheduler.jobs, SchedulerJobsConfig)
        assert scheduler.jobs.concept_embedding_refresh.enabled is True
        assert scheduler.jobs.vault_sync.enabled is True

    def test_job_config_manual_only_logic(self):
        """Test the logic for manual_only field."""
        # Job enabled and manual_only=False should be schedulable
        job1 = JobConfig(enabled=True, manual_only=False)
        assert job1.enabled and not job1.manual_only  # Should be scheduled

        # Job enabled and manual_only=True should not be schedulable
        job2 = JobConfig(enabled=True, manual_only=True)
        assert job2.enabled and job2.manual_only  # Should not be scheduled

        # Job disabled should not be schedulable regardless of manual_only
        job3 = JobConfig(enabled=False, manual_only=False)
        assert not job3.enabled  # Should not be scheduled

        job4 = JobConfig(enabled=False, manual_only=True)
        assert not job4.enabled  # Should not be scheduled

    def test_job_config_immutability_after_creation(self):
        """Test that job config values can be changed after creation."""
        job = JobConfig()

        # Test that we can modify values (dataclasses are mutable by default)
        job.enabled = False
        job.manual_only = True
        job.cron = "*/5 * * * *"
        job.description = "Modified description"

        assert job.enabled is False
        assert job.manual_only is True
        assert job.cron == "*/5 * * * *"
        assert job.description == "Modified description"
