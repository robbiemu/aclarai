"""
Tests for the Concept Highlight Refresh Job.
"""

import pytest
from unittest.mock import Mock, patch

from services.scheduler.aclarai_scheduler.concept_highlight_refresh import (
    ConceptHighlightRefreshJob,
)


class TestConceptHighlightRefreshJob:
    """Test the ConceptHighlightRefreshJob class."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks for all tests in this class using a pytest fixture."""
        with (
            patch(
                "services.scheduler.aclarai_scheduler.concept_highlight_refresh.aclaraiConfig"
            ) as mock_config_class,
            patch(
                "services.scheduler.aclarai_scheduler.concept_highlight_refresh.TrendingTopicsJob"
            ) as mock_trending_topics_class,
            patch(
                "services.scheduler.aclarai_scheduler.concept_highlight_refresh.TopConceptsJob"
            ) as mock_top_concepts_class,
        ):
            # Mock the config to prevent it from loading real environment variables
            mock_config_class.return_value = Mock()

            # Store mocks on the instance for access in tests
            self.mock_top_concepts_job = Mock()
            self.mock_trending_topics_job = Mock()
            mock_top_concepts_class.return_value = self.mock_top_concepts_job
            mock_trending_topics_class.return_value = self.mock_trending_topics_job

            # Instantiate the job under test
            self.job = ConceptHighlightRefreshJob()

            yield  # This is where the test itself runs

    def test_init(self):
        """Test job initialization."""
        assert self.job.config is not None
        assert self.job.logger is not None
        assert self.job.top_concepts_job is self.mock_top_concepts_job
        assert self.job.trending_topics_job is self.mock_trending_topics_job

    def test_run_job_success(self):
        """Test successful job execution."""
        self.mock_top_concepts_job.run_job.return_value = {
            "success": True,
            "error_details": [],
        }
        self.mock_trending_topics_job.run_job.return_value = {
            "success": True,
            "error_details": [],
        }

        result = self.job.run_job()

        assert result["success"] is True
        assert result["top_concepts_stats"]["success"] is True
        assert result["trending_topics_stats"]["success"] is True
        assert len(result["error_details"]) == 0

    def test_run_job_partial_failure(self):
        """Test job execution with one component failing."""
        self.mock_top_concepts_job.run_job.return_value = {
            "success": False,
            "error_details": ["PageRank failed"],
        }
        self.mock_trending_topics_job.run_job.return_value = {
            "success": True,
            "error_details": [],
        }

        result = self.job.run_job()

        assert result["success"] is True
        assert result["top_concepts_stats"]["success"] is False
        assert result["trending_topics_stats"]["success"] is True
        assert len(result["error_details"]) == 1
        assert "Top concepts: PageRank failed" in result["error_details"]

    def test_run_job_complete_failure(self):
        """Test job execution with both components failing."""
        self.mock_top_concepts_job.run_job.return_value = {
            "success": False,
            "error_details": ["PageRank failed"],
        }
        self.mock_trending_topics_job.run_job.return_value = {
            "success": False,
            "error_details": ["Database error"],
        }

        result = self.job.run_job()

        assert result["success"] is False
        assert len(result["error_details"]) == 2
        assert "Top concepts: PageRank failed" in result["error_details"]
        assert "Trending topics: Database error" in result["error_details"]

    def test_run_job_exception_handling(self):
        """Test job execution with unexpected exception."""
        self.mock_top_concepts_job.run_job.side_effect = Exception("Unexpected error")

        result = self.job.run_job()

        assert result["success"] is False
        assert len(result["error_details"]) == 1
        assert "Unexpected error" in result["error_details"][0]
