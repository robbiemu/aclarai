import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from apscheduler.schedulers.blocking import BlockingScheduler

# Add project root to path for imports to allow for absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


@pytest.fixture(autouse=True)
def mock_all_jobs(monkeypatch):
    """
    This fixture automatically mocks all job classes for every test in this file.
    """
    monkeypatch.setattr(
        "services.scheduler.aclarai_scheduler.main.VaultSyncJob", MagicMock()
    )
    monkeypatch.setattr(
        "services.scheduler.aclarai_scheduler.main.ConceptEmbeddingRefreshJob",
        MagicMock(),
    )
    monkeypatch.setattr(
        "services.scheduler.aclarai_scheduler.main.TopConceptsJob", MagicMock()
    )
    monkeypatch.setattr(
        "services.scheduler.aclarai_scheduler.main.TrendingTopicsJob", MagicMock()
    )
    monkeypatch.setattr(
        "services.scheduler.aclarai_scheduler.main.ConceptHighlightRefreshJob",
        MagicMock(),
    )
    monkeypatch.setattr(
        "services.scheduler.aclarai_scheduler.main.ConceptSummaryRefreshJob",
        MagicMock(),
    )
    monkeypatch.setattr(
        "services.scheduler.aclarai_scheduler.main.ConceptClusteringJob", MagicMock()
    )
    monkeypatch.setattr(
        "services.scheduler.aclarai_scheduler.main.SubjectSummaryRefreshJob",
        MagicMock(),
    )


class TestJobConfiguration:
    """
    Tests for the scheduler's job configuration and registration logic.
    """

    def _get_mock_config(self):
        """Helper to create a deeply nested mock config."""
        mock_config = MagicMock()
        mock_config.scheduler.jobs.vault_sync.enabled = False
        mock_config.scheduler.jobs.concept_embedding_refresh.enabled = False
        mock_config.scheduler.jobs.top_concepts.enabled = False
        mock_config.scheduler.jobs.trending_topics.enabled = False
        mock_config.scheduler.jobs.concept_highlight_refresh.enabled = False
        mock_config.scheduler.jobs.concept_summary_refresh.enabled = False
        mock_config.scheduler.jobs.concept_clustering.enabled = False
        mock_config.scheduler.jobs.subject_summary_refresh.enabled = False
        return mock_config

    def test_manual_only_job_not_scheduled(self, monkeypatch):
        """Test that jobs with manual_only=True are not automatically scheduled."""
        from services.scheduler.aclarai_scheduler.main import SchedulerService

        mock_config = self._get_mock_config()
        mock_config.scheduler.jobs.vault_sync.enabled = True
        mock_config.scheduler.jobs.vault_sync.manual_only = True
        mock_config.scheduler.jobs.vault_sync.cron = "*/30 * * * *"

        monkeypatch.setattr(
            "services.scheduler.aclarai_scheduler.main.load_config",
            lambda **_: mock_config,
        )
        monkeypatch.setattr(
            "services.scheduler.aclarai_scheduler.main.is_paused", lambda: False
        )

        service = SchedulerService()
        service.scheduler = MagicMock(spec=BlockingScheduler)
        service._register_jobs()

        service.scheduler.add_job.assert_not_called()

    def test_disabled_job_not_scheduled(self, monkeypatch):
        """Test that jobs with enabled=False are not scheduled."""
        from services.scheduler.aclarai_scheduler.main import SchedulerService

        mock_config = self._get_mock_config()
        mock_config.scheduler.jobs.vault_sync.enabled = False
        mock_config.scheduler.jobs.concept_embedding_refresh.enabled = True
        mock_config.scheduler.jobs.concept_embedding_refresh.manual_only = False
        mock_config.scheduler.jobs.concept_embedding_refresh.cron = "0 3 * * *"

        monkeypatch.setattr(
            "services.scheduler.aclarai_scheduler.main.load_config",
            lambda **_: mock_config,
        )
        monkeypatch.setattr(
            "services.scheduler.aclarai_scheduler.main.is_paused", lambda: False
        )
        monkeypatch.setattr("os.getenv", lambda _k, d: d)

        service = SchedulerService()
        service.scheduler = MagicMock(spec=BlockingScheduler)
        service._register_jobs()

        service.scheduler.add_job.assert_called_once()
        call_args = service.scheduler.add_job.call_args
        assert call_args.kwargs["id"] == "concept_embedding_refresh"

    def test_normal_jobs_are_scheduled(self, monkeypatch):
        """Test that jobs with enabled=True and manual_only=False are scheduled."""
        from services.scheduler.aclarai_scheduler.main import SchedulerService

        mock_config = self._get_mock_config()
        mock_config.scheduler.jobs.vault_sync.enabled = True
        mock_config.scheduler.jobs.vault_sync.manual_only = False
        mock_config.scheduler.jobs.vault_sync.cron = "*/30 * * * *"
        mock_config.scheduler.jobs.concept_embedding_refresh.enabled = True
        mock_config.scheduler.jobs.concept_embedding_refresh.manual_only = False
        mock_config.scheduler.jobs.concept_embedding_refresh.cron = "0 3 * * *"

        monkeypatch.setattr(
            "services.scheduler.aclarai_scheduler.main.load_config",
            lambda **_: mock_config,
        )
        monkeypatch.setattr(
            "services.scheduler.aclarai_scheduler.main.is_paused", lambda: False
        )
        monkeypatch.setattr("os.getenv", lambda _k, d: d)

        service = SchedulerService()
        service.scheduler = MagicMock(spec=BlockingScheduler)
        service._register_jobs()

        assert service.scheduler.add_job.call_count == 2

    def test_automation_pause_prevents_scheduling(self, monkeypatch):
        """Test that job registration is skipped entirely when automation is paused."""
        from services.scheduler.aclarai_scheduler.main import SchedulerService

        mock_config = self._get_mock_config()
        mock_config.scheduler.jobs.vault_sync.enabled = True
        mock_config.scheduler.jobs.vault_sync.manual_only = False
        mock_config.scheduler.jobs.vault_sync.cron = "*/30 * * * *"

        monkeypatch.setattr(
            "services.scheduler.aclarai_scheduler.main.load_config",
            lambda **_: mock_config,
        )
        monkeypatch.setattr(
            "services.scheduler.aclarai_scheduler.main.is_paused", lambda: True
        )

        service = SchedulerService()
        service.scheduler = MagicMock(spec=BlockingScheduler)
        service._register_jobs()

        service.scheduler.add_job.assert_not_called()
