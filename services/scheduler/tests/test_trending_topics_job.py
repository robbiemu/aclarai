"""
Tests for the TrendingTopicsJob.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from services.scheduler.aclarai_scheduler.trending_topics_job import TrendingTopicsJob
from shared.aclarai_shared.config import TrendingTopicsJobConfig


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock()
    config.vault_path = "/tmp/test_vault"

    # Create nested mock structure for scheduler.jobs.trending_topics
    trending_topics_config = TrendingTopicsJobConfig(
        enabled=True,
        manual_only=False,
        cron="0 5 * * *",
        description="Test trending topics job",
        window_days=7,
        count=None,
        percent=5,
        min_mentions=2,
        target_file="Trending Topics - {date}.md",
    )

    # Create the nested mock structure
    jobs_mock = Mock()
    jobs_mock.trending_topics = trending_topics_config

    scheduler_mock = Mock()
    scheduler_mock.jobs = jobs_mock

    config.scheduler = scheduler_mock

    return config


@pytest.fixture
def mock_neo4j_manager():
    """Create a mock Neo4j manager for testing."""
    mock_manager = Mock()
    # Mock successful query execution with sample data
    mock_manager.execute_query.return_value = [
        {
            "concept_name": "Python Debugging",
            "current_mentions": 10,
            "previous_mentions": 5,
            "growth_rate": 1.0,
        },
        {
            "concept_name": "Docker Containers",
            "current_mentions": 8,
            "previous_mentions": 0,
            "growth_rate": 1000.0,
        },
        {
            "concept_name": "Machine Learning",
            "current_mentions": 15,
            "previous_mentions": 12,
            "growth_rate": 0.25,
        },
    ]
    return mock_manager


def test_trending_topics_job_initialization(mock_config):
    """Test that TrendingTopicsJob initializes correctly."""
    with patch("aclarai_scheduler.trending_topics_job.Neo4jGraphManager"):
        job = TrendingTopicsJob(config=mock_config)
        assert job.config == mock_config
        assert job.job_config == mock_config.scheduler.jobs.trending_topics
        assert job.vault_path == Path("/tmp/test_vault")


def test_get_time_windows():
    """Test time window calculation."""
    job_config = TrendingTopicsJobConfig(window_days=7)

    with patch("aclarai_scheduler.trending_topics_job.Neo4jGraphManager"):
        with patch(
            "aclarai_scheduler.trending_topics_job.load_config"
        ) as mock_load_config:
            mock_config = Mock()
            mock_config.scheduler.jobs.trending_topics = job_config
            mock_config.vault_path = "/tmp/test"
            mock_load_config.return_value = mock_config

            job = TrendingTopicsJob(config=mock_config)

            with patch(
                "aclarai_scheduler.trending_topics_job.datetime"
            ) as mock_datetime:
                mock_now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
                mock_datetime.now.return_value = mock_now

                current_time, window_start, comparison_window_start = (
                    job._get_time_windows()
                )

                assert current_time == mock_now
                assert window_start == datetime(
                    2024, 1, 8, 12, 0, 0, tzinfo=timezone.utc
                )
                assert comparison_window_start == datetime(
                    2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc
                )


def test_select_trending_concepts():
    """Test concept selection logic."""
    job_config = TrendingTopicsJobConfig(percent=50)  # Select top 50%

    with patch("aclarai_scheduler.trending_topics_job.Neo4jGraphManager"):
        with patch(
            "aclarai_scheduler.trending_topics_job.load_config"
        ) as mock_load_config:
            mock_config = Mock()
            mock_config.scheduler.jobs.trending_topics = job_config
            mock_config.vault_path = "/tmp/test"
            mock_load_config.return_value = mock_config

            job = TrendingTopicsJob(config=mock_config)

            concept_deltas = [
                ("Concept A", 10, 5, 1.0),
                ("Concept B", 8, 4, 1.0),
                ("Concept C", 6, 3, 1.0),
                ("Concept D", 4, 2, 1.0),
            ]

            result = job._select_trending_concepts(concept_deltas)

            # Should select top 50% = 2 concepts
            assert len(result) == 2
            assert result[0][0] == "Concept A"
            assert result[1][0] == "Concept B"


def test_generate_markdown_content():
    """Test markdown content generation."""
    job_config = TrendingTopicsJobConfig()

    with patch("aclarai_scheduler.trending_topics_job.Neo4jGraphManager"):
        with patch(
            "aclarai_scheduler.trending_topics_job.load_config"
        ) as mock_load_config:
            mock_config = Mock()
            mock_config.scheduler.jobs.trending_topics = job_config
            mock_config.vault_path = "/tmp/test"
            mock_load_config.return_value = mock_config

            job = TrendingTopicsJob(config=mock_config)

            trending_concepts = [
                ("Python Debugging", 10, 5, 1.0),
                ("Docker Containers", 8, 0, 1000.0),
            ]

            with patch.object(job, "_get_time_windows") as mock_get_time_windows:
                mock_current = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
                mock_window_start = datetime(2024, 1, 8, 12, 0, 0, tzinfo=timezone.utc)
                mock_get_time_windows.return_value = (
                    mock_current,
                    mock_window_start,
                    None,
                )

                content = job._generate_markdown_content(trending_concepts)

                assert "## Trending This Week" in content
                assert "[[Python Debugging]]" in content
                assert "[[Docker Containers]]" in content
                assert "Mentions up 100%" in content
                assert "New concept with 8 mentions" in content
                assert "aclarai:id=" in content


def test_get_target_file_path():
    """Test target file path generation with date."""
    job_config = TrendingTopicsJobConfig(target_file="Trending Topics - {date}.md")

    with patch("aclarai_scheduler.trending_topics_job.Neo4jGraphManager"):
        with patch(
            "aclarai_scheduler.trending_topics_job.load_config"
        ) as mock_load_config:
            mock_config = Mock()
            mock_config.scheduler.jobs.trending_topics = job_config
            mock_config.vault_path = "/tmp/test"
            mock_load_config.return_value = mock_config

            job = TrendingTopicsJob(config=mock_config)

            with patch.object(job, "_get_time_windows") as mock_get_time_windows:
                mock_current = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
                mock_get_time_windows.return_value = (mock_current, None, None)

                target_path = job._get_target_file_path()

                expected_path = Path("/tmp/test/Trending Topics - 2024-01-15.md")
                assert target_path == expected_path


def test_write_file_atomically():
    """Test atomic file writing."""
    job_config = TrendingTopicsJobConfig()

    with patch("aclarai_scheduler.trending_topics_job.Neo4jGraphManager"):
        with patch(
            "aclarai_scheduler.trending_topics_job.load_config"
        ) as mock_load_config:
            mock_config = Mock()
            mock_config.scheduler.jobs.trending_topics = job_config
            mock_config.vault_path = "/tmp/test"
            mock_load_config.return_value = mock_config

            job = TrendingTopicsJob(config=mock_config)

            with tempfile.TemporaryDirectory() as temp_dir:
                target_path = Path(temp_dir) / "test_trending.md"
                content = "# Test Content\nThis is a test."

                result = job._write_file_atomically(content, target_path)

                assert result is True
                assert target_path.exists()
                assert target_path.read_text() == content


def test_run_job_success(mock_config, mock_neo4j_manager):
    """Test successful job execution."""
    with patch(
        "aclarai_scheduler.trending_topics_job.load_config", return_value=mock_config
    ):
        job = TrendingTopicsJob(config=mock_config, neo4j_manager=mock_neo4j_manager)

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_config.vault_path = temp_dir

            stats = job.run_job()

            assert stats["success"] is True
            assert stats["concepts_analyzed"] == 3
            assert stats["trending_concepts_selected"] > 0
            assert stats["file_written"] is True
            assert len(stats["error_details"]) == 0


def test_run_job_no_concepts(mock_config):
    """Test job execution when no concepts are found."""
    mock_neo4j_manager = Mock()
    mock_neo4j_manager.execute_query.return_value = []

    with patch(
        "aclarai_scheduler.trending_topics_job.load_config", return_value=mock_config
    ):
        job = TrendingTopicsJob(config=mock_config, neo4j_manager=mock_neo4j_manager)

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_config.vault_path = temp_dir

            stats = job.run_job()

            assert stats["success"] is True  # Should still succeed with empty content
            assert stats["concepts_analyzed"] == 0
            assert stats["trending_concepts_selected"] == 0
            assert stats["file_written"] is True
            assert len(stats["error_details"]) == 0


def test_run_job_database_error(mock_config):
    """Test job execution when database query fails."""
    mock_neo4j_manager = Mock()
    mock_neo4j_manager.execute_query.side_effect = Exception(
        "Database connection failed"
    )

    with patch(
        "aclarai_scheduler.trending_topics_job.load_config", return_value=mock_config
    ):
        job = TrendingTopicsJob(config=mock_config, neo4j_manager=mock_neo4j_manager)

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_config.vault_path = temp_dir

            stats = job.run_job()

            # Job should recover gracefully and write an empty file
            assert stats["success"] is True  # Job succeeds with graceful degradation
            assert stats["concepts_analyzed"] == 0
            assert stats["trending_concepts_selected"] == 0
            assert stats["file_written"] is True  # Empty file should still be written
            assert (
                len(stats["error_details"]) == 0
            )  # No errors in final stats (graceful handling)
