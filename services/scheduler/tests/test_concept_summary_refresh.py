"""
Tests for the Concept Summary Refresh Job.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from services.scheduler.aclarai_scheduler.concept_summary_refresh import (
    ConceptSummaryRefreshJob,
)


class TestConceptSummaryRefreshJob:
    """Test the ConceptSummaryRefreshJob class."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks for all tests in this class using a pytest fixture."""
        with (
            patch(
                "services.scheduler.aclarai_scheduler.concept_summary_refresh.aclaraiConfig"
            ) as mock_config_class,
            patch(
                "services.scheduler.aclarai_scheduler.concept_summary_refresh.Neo4jGraphManager"
            ) as mock_neo4j_class,
            patch(
                "services.scheduler.aclarai_scheduler.concept_summary_refresh.ConceptSummaryAgent"
            ) as mock_agent_class,
        ):
            # Mock the config
            self.mock_config = Mock()
            mock_config_class.return_value = self.mock_config

            # Store mocks on the instance
            self.mock_agent = Mock()
            self.mock_neo4j = Mock()
            mock_agent_class.return_value = self.mock_agent
            mock_neo4j_class.return_value = self.mock_neo4j

            # Instantiate the job under test
            self.job = ConceptSummaryRefreshJob(config=self.mock_config)

            yield  # This is where the test itself runs

    def test_init(self):
        """Test job initialization."""
        assert self.job.config is not None
        assert self.job.concept_summary_agent is self.mock_agent
        assert self.job.neo4j_manager is self.mock_neo4j

    def test_run_job_success(self):
        """Test successful job execution."""
        self.job._get_canonical_concepts = Mock(return_value=["Concept A", "Concept B"])
        self.job._process_concept = Mock(return_value=(True, True))

        result = self.job.run_job()

        assert result["success"] is True
        assert result["concepts_processed"] == 2
        assert result["concepts_updated"] == 2
        assert result["errors"] == 0

    def test_run_job_with_skipped_concepts(self):
        """Test job execution with some concepts skipped."""
        self.job._get_canonical_concepts = Mock(return_value=["A", "B", "C"])

        def mock_process(name):
            return (True, True) if name == "A" else (False, False)

        self.job._process_concept = Mock(side_effect=mock_process)

        result = self.job.run_job()

        assert result["success"] is True
        assert result["concepts_processed"] == 1
        assert result["concepts_updated"] == 1
        assert result["concepts_skipped"] == 2

    def test_run_job_with_errors(self):
        """Test job execution with some concept processing errors."""
        self.job._get_canonical_concepts = Mock(return_value=["A", "B"])

        def mock_process(name):
            if name == "A":
                return (True, True)
            raise Exception("Processing error")

        self.job._process_concept = Mock(side_effect=mock_process)

        result = self.job.run_job()

        assert result["success"] is True
        assert result["concepts_processed"] == 1
        assert result["errors"] == 1
        assert "Concept 'B': Processing error" in result["error_details"]

    def test_run_job_no_concepts(self):
        """Test job execution with no concepts found."""
        self.job._get_canonical_concepts = Mock(return_value=[])
        result = self.job.run_job()
        assert result["success"] is True
        assert result["concepts_processed"] == 0

    def test_get_canonical_concepts_success(self):
        """Test successful retrieval of canonical concepts."""
        mock_session = Mock()

        # Use MagicMock to correctly mock the context manager
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_session
        self.mock_neo4j.driver.session.return_value = mock_context_manager

        mock_session.run.return_value = [{"name": "A"}, {"name": "B"}]

        assert self.job._get_canonical_concepts() == ["A", "B"]

    def test_get_canonical_concepts_failure(self):
        """Test failure in retrieving canonical concepts."""
        self.mock_neo4j.driver.session.side_effect = Exception("DB error")
        assert self.job._get_canonical_concepts() == []

    def test_process_concept_success(self):
        """Test successful concept processing."""
        self.job._get_concept_details = Mock(return_value={"name": "Test"})
        self.job._concept_has_claims = Mock(return_value=True)
        self.mock_agent.generate_concept_page.return_value = True
        processed, updated = self.job._process_concept("Test")
        assert processed and updated

    def test_process_concept_skip_no_claims(self):
        """Test concept processing skipped due to no claims."""
        self.mock_config.concept_summaries.skip_if_no_claims = True
        self.job._get_concept_details = Mock(return_value={"name": "Test"})
        self.job._concept_has_claims = Mock(return_value=False)
        processed, updated = self.job._process_concept("Test")
        assert not processed and not updated

    def test_concept_has_claims_success(self):
        """Test successful check for concept claims."""
        mock_session = Mock()

        # Use MagicMock to correctly mock the context manager
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_session
        self.mock_neo4j.driver.session.return_value = mock_context_manager

        mock_session.run.return_value.single.return_value = {"has_claims": True}

        assert self.job._concept_has_claims("Test") is True

    def test_concept_has_claims_failure(self):
        """Test failure in checking concept claims."""
        self.mock_neo4j.driver.session.side_effect = Exception("DB error")
        assert self.job._concept_has_claims("Test") is True
