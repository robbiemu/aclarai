"""
Tests for the Concept Highlight Refresh Job.
"""

from unittest.mock import Mock, patch

from services.scheduler.aclarai_scheduler.concept_highlight_refresh import (
    ConceptHighlightRefreshJob,
)


class TestConceptHighlightRefreshJob:
    """Test the ConceptHighlightRefreshJob class."""

    def test_init(self):
        """Test job initialization."""
        with patch(
            "services.scheduler.aclarai_scheduler.concept_highlight_refresh.TopConceptsJob"
        ) as mock_top_concepts:
            with patch(
                "services.scheduler.aclarai_scheduler.concept_highlight_refresh.TrendingTopicsJob"
            ) as mock_trending_topics:
                job = ConceptHighlightRefreshJob()

                assert job.config is not None
                assert job.logger is not None
                mock_top_concepts.assert_called_once()
                mock_trending_topics.assert_called_once()

    def test_run_job_success(self):
        """Test successful job execution."""
        with patch(
            "services.scheduler.aclarai_scheduler.concept_highlight_refresh.TopConceptsJob"
        ) as mock_top_concepts_class:
            with patch(
                "services.scheduler.aclarai_scheduler.trending_topics_job.TrendingTopicsJob"
            ) as mock_trending_topics_class:
                # Mock the job instances
                mock_top_concepts_job = Mock()
                mock_trending_topics_job = Mock()
                mock_top_concepts_class.return_value = mock_top_concepts_job
                mock_trending_topics_class.return_value = mock_trending_topics_job

                # Mock successful returns
                mock_top_concepts_job.run_job.return_value = {
                    "success": True,
                    "concepts_analyzed": 10,
                    "top_concepts_selected": 5,
                    "file_written": True,
                    "pagerank_executed": True,
                    "duration": 1.0,
                    "error_details": [],
                }

                mock_trending_topics_job.run_job.return_value = {
                    "success": True,
                    "concepts_analyzed": 8,
                    "trending_concepts_selected": 3,
                    "file_written": True,
                    "window_start": "2024-01-01",
                    "window_end": "2024-01-08",
                    "duration": 0.5,
                    "error_details": [],
                }

                job = ConceptHighlightRefreshJob()
                result = job.run_job()

                assert result["success"] is True
                assert result["top_concepts_stats"]["success"] is True
                assert result["trending_topics_stats"]["success"] is True
                assert len(result["error_details"]) == 0
                assert result["duration"] > 0

    def test_run_job_partial_failure(self):
        """Test job execution with one component failing."""
        with patch(
            "services.scheduler.aclarai_scheduler.concept_highlight_refresh.TopConceptsJob"
        ) as mock_top_concepts_class:
            with patch(
                "services.scheduler.aclarai_scheduler.trending_topics_job.TrendingTopicsJob"
            ) as mock_trending_topics_class:
                # Mock the job instances
                mock_top_concepts_job = Mock()
                mock_trending_topics_job = Mock()
                mock_top_concepts_class.return_value = mock_top_concepts_job
                mock_trending_topics_class.return_value = mock_trending_topics_job

                # Mock one failure, one success
                mock_top_concepts_job.run_job.return_value = {
                    "success": False,
                    "concepts_analyzed": 0,
                    "top_concepts_selected": 0,
                    "file_written": False,
                    "pagerank_executed": False,
                    "duration": 0.1,
                    "error_details": ["PageRank failed"],
                }

                mock_trending_topics_job.run_job.return_value = {
                    "success": True,
                    "concepts_analyzed": 8,
                    "trending_concepts_selected": 3,
                    "file_written": True,
                    "window_start": "2024-01-01",
                    "window_end": "2024-01-08",
                    "duration": 0.5,
                    "error_details": [],
                }

                job = ConceptHighlightRefreshJob()
                result = job.run_job()

                # Should still succeed if at least one component succeeds
                assert result["success"] is True
                assert result["top_concepts_stats"]["success"] is False
                assert result["trending_topics_stats"]["success"] is True
                assert len(result["error_details"]) == 1
                assert "Top concepts: PageRank failed" in result["error_details"]

    def test_run_job_complete_failure(self):
        """Test job execution with both components failing."""
        with patch(
            "services.scheduler.aclarai_scheduler.concept_highlight_refresh.TopConceptsJob"
        ) as mock_top_concepts_class:
            with patch(
                "services.scheduler.aclarai_scheduler.trending_topics_job.TrendingTopicsJob"
            ) as mock_trending_topics_class:
                # Mock the job instances
                mock_top_concepts_job = Mock()
                mock_trending_topics_job = Mock()
                mock_top_concepts_class.return_value = mock_top_concepts_job
                mock_trending_topics_class.return_value = mock_trending_topics_job

                # Mock both failures
                mock_top_concepts_job.run_job.return_value = {
                    "success": False,
                    "concepts_analyzed": 0,
                    "top_concepts_selected": 0,
                    "file_written": False,
                    "pagerank_executed": False,
                    "duration": 0.1,
                    "error_details": ["PageRank failed"],
                }

                mock_trending_topics_job.run_job.return_value = {
                    "success": False,
                    "concepts_analyzed": 0,
                    "trending_concepts_selected": 0,
                    "file_written": False,
                    "window_start": "",
                    "window_end": "",
                    "duration": 0.1,
                    "error_details": ["Database error"],
                }

                job = ConceptHighlightRefreshJob()
                result = job.run_job()

                # Should fail if both components fail
                assert result["success"] is False
                assert result["top_concepts_stats"]["success"] is False
                assert result["trending_topics_stats"]["success"] is False
                assert len(result["error_details"]) == 2
                assert "Top concepts: PageRank failed" in result["error_details"]
                assert "Trending topics: Database error" in result["error_details"]

    def test_run_job_exception_handling(self):
        """Test job execution with unexpected exception."""
        with patch(
            "services.scheduler.aclarai_scheduler.concept_highlight_refresh.TopConceptsJob"
        ) as mock_top_concepts_class:
            with patch(
                "services.scheduler.aclarai_scheduler.trending_topics_job.TrendingTopicsJob"
            ) as mock_trending_topics_class:
                # Mock the job instances
                mock_top_concepts_job = Mock()
                mock_trending_topics_job = Mock()
                mock_top_concepts_class.return_value = mock_top_concepts_job
                mock_trending_topics_class.return_value = mock_trending_topics_job

                # Mock exception during execution
                mock_top_concepts_job.run_job.side_effect = Exception(
                    "Unexpected error"
                )

                job = ConceptHighlightRefreshJob()
                result = job.run_job()

                assert result["success"] is False
                assert len(result["error_details"]) == 1
                assert "Unexpected error" in result["error_details"][0]
