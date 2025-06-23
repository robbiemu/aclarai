"""
Unit tests for job configuration functionality in the scheduler service.
"""

from unittest.mock import MagicMock, patch

from aclarai_scheduler.main import SchedulerService
from aclarai_shared import load_config


class TestJobConfiguration:
    """Test job configuration loading and processing."""

    def test_job_config_loading(self):
        """Test that job configuration is loaded correctly from YAML."""
        config = load_config(validate=False)

        # Test scheduler config exists
        assert hasattr(config, "scheduler")
        assert hasattr(config.scheduler, "jobs")

        # Test vault sync job config
        vault_sync = config.scheduler.jobs.vault_sync
        assert hasattr(vault_sync, "enabled")
        assert hasattr(vault_sync, "manual_only")
        assert hasattr(vault_sync, "cron")
        assert hasattr(vault_sync, "description")

        # Test concept refresh job config
        concept_refresh = config.scheduler.jobs.concept_embedding_refresh
        assert hasattr(concept_refresh, "enabled")
        assert hasattr(concept_refresh, "manual_only")
        assert hasattr(concept_refresh, "cron")
        assert hasattr(concept_refresh, "description")

    def test_default_job_config_values(self):
        """Test that job configurations have expected default values."""
        config = load_config(validate=False)

        # Test defaults for vault sync
        vault_sync = config.scheduler.jobs.vault_sync
        assert vault_sync.enabled is True
        assert vault_sync.manual_only is False
        assert vault_sync.cron == "*/30 * * * *"
        assert "Sync vault files" in vault_sync.description

        # Test defaults for concept refresh
        concept_refresh = config.scheduler.jobs.concept_embedding_refresh
        assert concept_refresh.enabled is True
        assert concept_refresh.manual_only is False
        assert concept_refresh.cron == "0 3 * * *"
        assert "Refresh concept embeddings" in concept_refresh.description

    @patch.dict("os.environ", {}, clear=True)
    def test_manual_only_job_not_scheduled(self):
        """Test that jobs with manual_only=True are not automatically scheduled."""
        # Mock configuration with manual_only job
        mock_config = MagicMock()
        mock_config.scheduler.jobs.vault_sync.enabled = True
        mock_config.scheduler.jobs.vault_sync.manual_only = True
        mock_config.scheduler.jobs.vault_sync.cron = "*/30 * * * *"
        mock_config.scheduler.jobs.vault_sync.description = "Test job"

        mock_config.scheduler.jobs.concept_embedding_refresh.enabled = True
        mock_config.scheduler.jobs.concept_embedding_refresh.manual_only = False
        mock_config.scheduler.jobs.concept_embedding_refresh.cron = "0 3 * * *"
        mock_config.scheduler.jobs.concept_embedding_refresh.description = "Test job"

        with patch("aclarai_scheduler.main.load_config", return_value=mock_config):
            service = SchedulerService()
            service._setup_scheduler()

            # Mock the jobs to track calls
            service.vault_sync_job = MagicMock()
            service.concept_refresh_job = MagicMock()

            # Register jobs
            service._register_jobs()

            # Check that only the non-manual_only job was registered
            registered_jobs = service.scheduler.get_jobs()
            job_ids = [job.id for job in registered_jobs]

            # Vault sync should not be registered (manual_only=True)
            assert "vault_sync" not in job_ids
            # Concept refresh should be registered (manual_only=False)
            assert "concept_embedding_refresh" in job_ids

    @patch.dict("os.environ", {}, clear=True)
    def test_disabled_job_not_scheduled(self):
        """Test that jobs with enabled=False are not scheduled."""
        # Mock configuration with disabled job
        mock_config = MagicMock()
        mock_config.scheduler.jobs.vault_sync.enabled = False
        mock_config.scheduler.jobs.vault_sync.manual_only = False
        mock_config.scheduler.jobs.vault_sync.cron = "*/30 * * * *"
        mock_config.scheduler.jobs.vault_sync.description = "Test job"

        mock_config.scheduler.jobs.concept_embedding_refresh.enabled = True
        mock_config.scheduler.jobs.concept_embedding_refresh.manual_only = False
        mock_config.scheduler.jobs.concept_embedding_refresh.cron = "0 3 * * *"
        mock_config.scheduler.jobs.concept_embedding_refresh.description = "Test job"

        with patch("aclarai_scheduler.main.load_config", return_value=mock_config):
            service = SchedulerService()
            service._setup_scheduler()

            # Mock the jobs to track calls
            service.vault_sync_job = MagicMock()
            service.concept_refresh_job = MagicMock()

            # Register jobs
            service._register_jobs()

            # Check that only the enabled job was registered
            registered_jobs = service.scheduler.get_jobs()
            job_ids = [job.id for job in registered_jobs]

            # Vault sync should not be registered (enabled=False)
            assert "vault_sync" not in job_ids
            # Concept refresh should be registered (enabled=True)
            assert "concept_embedding_refresh" in job_ids

    @patch.dict("os.environ", {}, clear=True)
    def test_normal_jobs_are_scheduled(self):
        """Test that jobs with enabled=True and manual_only=False are scheduled."""
        # Mock configuration with normal jobs
        mock_config = MagicMock()
        mock_config.scheduler.jobs.vault_sync.enabled = True
        mock_config.scheduler.jobs.vault_sync.manual_only = False
        mock_config.scheduler.jobs.vault_sync.cron = "*/30 * * * *"
        mock_config.scheduler.jobs.vault_sync.description = "Test job"

        mock_config.scheduler.jobs.concept_embedding_refresh.enabled = True
        mock_config.scheduler.jobs.concept_embedding_refresh.manual_only = False
        mock_config.scheduler.jobs.concept_embedding_refresh.cron = "0 3 * * *"
        mock_config.scheduler.jobs.concept_embedding_refresh.description = "Test job"

        with patch("aclarai_scheduler.main.load_config", return_value=mock_config):
            service = SchedulerService()
            service._setup_scheduler()

            # Mock the jobs to track calls
            service.vault_sync_job = MagicMock()
            service.concept_refresh_job = MagicMock()

            # Register jobs
            service._register_jobs()

            # Check that both jobs were registered
            registered_jobs = service.scheduler.get_jobs()
            job_ids = [job.id for job in registered_jobs]

            # Both jobs should be registered
            assert "vault_sync" in job_ids
            assert "concept_embedding_refresh" in job_ids

    @patch.dict("os.environ", {"AUTOMATION_PAUSE": "true"}, clear=True)
    def test_automation_pause_prevents_scheduling(self):
        """Test that AUTOMATION_PAUSE=true prevents all job scheduling."""
        config = load_config(validate=False)

        with patch("aclarai_scheduler.main.load_config", return_value=config):
            service = SchedulerService()
            service._setup_scheduler()

            # Mock the jobs to track calls
            service.vault_sync_job = MagicMock()
            service.concept_refresh_job = MagicMock()

            # Register jobs
            service._register_jobs()

            # Check that no jobs were registered due to automation pause
            registered_jobs = service.scheduler.get_jobs()
            assert len(registered_jobs) == 0
