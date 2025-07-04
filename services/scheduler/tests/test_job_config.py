import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from apscheduler.schedulers.blocking import BlockingScheduler

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aclarai_scheduler.main import SchedulerService

# This helper decorator mocks all job classes instantiated by SchedulerService.
# It prevents real jobs with external dependencies from being created during unit tests.
all_jobs_patch = patch.multiple(
    "aclarai_scheduler.main",
    # The conflicting `autospec=True` argument has been removed.
    VaultSyncJob=MagicMock(),
    ConceptEmbeddingRefreshJob=MagicMock(),
    TopConceptsJob=MagicMock(),
    TrendingTopicsJob=MagicMock(),
    ConceptHighlightRefreshJob=MagicMock(),
    ConceptSummaryRefreshJob=MagicMock(),
    ConceptClusteringJob=MagicMock(),
)


@all_jobs_patch
class TestJobConfiguration:
    """
    Tests for the scheduler's job configuration and registration logic.
    The `all_jobs_patch` decorator is applied to all tests in this class.
    """

    def _get_mock_config(self):
        """Helper to create a deeply nested mock config."""
        mock_config = MagicMock()
        # Set default values for all jobs to avoid AttributeError
        mock_config.scheduler.jobs.vault_sync.enabled = False
        mock_config.scheduler.jobs.concept_embedding_refresh.enabled = False
        mock_config.scheduler.jobs.top_concepts.enabled = False
        mock_config.scheduler.jobs.trending_topics.enabled = False
        mock_config.scheduler.jobs.concept_highlight_refresh.enabled = False
        mock_config.scheduler.jobs.concept_summary_refresh.enabled = False
        mock_config.scheduler.jobs.concept_clustering.enabled = False
        return mock_config

    @patch.dict("os.environ", {}, clear=True)
    def test_manual_only_job_not_scheduled(self, **_):
        """Test that jobs with manual_only=True are not automatically scheduled."""
        mock_config = self._get_mock_config()
        mock_config.scheduler.jobs.vault_sync.enabled = True
        mock_config.scheduler.jobs.vault_sync.manual_only = True
        mock_config.scheduler.jobs.vault_sync.cron = "*/30 * * * *"

        with patch("aclarai_scheduler.main.load_config", return_value=mock_config):
            service = SchedulerService()
            service.scheduler = MagicMock(spec=BlockingScheduler)
            service._register_jobs()

        # Assert that no jobs were scheduled because the only enabled one is manual
        service.scheduler.add_job.assert_not_called()

    @patch.dict("os.environ", {}, clear=True)
    def test_disabled_job_not_scheduled(self, **_):
        """Test that jobs with enabled=False are not scheduled."""
        mock_config = self._get_mock_config()
        mock_config.scheduler.jobs.vault_sync.enabled = False
        mock_config.scheduler.jobs.concept_embedding_refresh.enabled = True
        mock_config.scheduler.jobs.concept_embedding_refresh.manual_only = False
        mock_config.scheduler.jobs.concept_embedding_refresh.cron = "0 3 * * *"

        with patch("aclarai_scheduler.main.load_config", return_value=mock_config):
            service = SchedulerService()
            service.scheduler = MagicMock(spec=BlockingScheduler)
            service._register_jobs()

        # Assert that add_job was called exactly once for the enabled job
        service.scheduler.add_job.assert_called_once()
        call_args = service.scheduler.add_job.call_args
        assert call_args.kwargs["id"] == "concept_embedding_refresh"

    @patch.dict("os.environ", {}, clear=True)
    def test_normal_jobs_are_scheduled(self, **_):
        """Test that jobs with enabled=True and manual_only=False are scheduled."""
        mock_config = self._get_mock_config()
        mock_config.scheduler.jobs.vault_sync.enabled = True
        mock_config.scheduler.jobs.vault_sync.manual_only = False
        mock_config.scheduler.jobs.vault_sync.cron = "*/30 * * * *"
        mock_config.scheduler.jobs.concept_embedding_refresh.enabled = True
        mock_config.scheduler.jobs.concept_embedding_refresh.manual_only = False
        mock_config.scheduler.jobs.concept_embedding_refresh.cron = "0 3 * * *"

        with patch("aclarai_scheduler.main.load_config", return_value=mock_config):
            service = SchedulerService()
            service.scheduler = MagicMock(spec=BlockingScheduler)
            service._register_jobs()

        # Assert that add_job was called for both enabled jobs
        assert service.scheduler.add_job.call_count == 2

    @patch.dict("os.environ", {"AUTOMATION_PAUSE": "true"}, clear=True)
    def test_automation_pause_prevents_scheduling(self, **_):
        """
        Test that job registration proceeds even when automation is paused.
        The pause itself is handled at job execution time, not registration time.
        """
        mock_config = self._get_mock_config()
        mock_config.scheduler.jobs.vault_sync.enabled = True
        mock_config.scheduler.jobs.vault_sync.manual_only = False
        mock_config.scheduler.jobs.vault_sync.cron = "*/30 * * * *"

        with patch("aclarai_scheduler.main.load_config", return_value=mock_config):
            service = SchedulerService()
            service.scheduler = MagicMock(spec=BlockingScheduler)
            service._register_jobs()

        # The pause flag does not prevent registration.
        service.scheduler.add_job.assert_called_once()
