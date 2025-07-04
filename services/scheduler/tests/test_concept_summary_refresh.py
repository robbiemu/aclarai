"""
Tests for the Concept Summary Refresh Job.
"""

from unittest.mock import Mock, patch

from services.scheduler.aclarai_scheduler.concept_summary_refresh import (
    ConceptSummaryRefreshJob,
)


class TestConceptSummaryRefreshJob:
    """Test the ConceptSummaryRefreshJob class."""

    def test_init(self):
        """Test job initialization."""
        with patch(
            "aclarai_scheduler.concept_summary_refresh.ConceptSummaryAgent"
        ) as mock_agent:
            with patch(
                "aclarai_scheduler.concept_summary_refresh.Neo4jGraphManager"
            ) as mock_neo4j:
                job = ConceptSummaryRefreshJob()

                assert job.config is not None
                assert job.logger is not None
                mock_agent.assert_called_once()
                mock_neo4j.assert_called_once()

    def test_run_job_success(self):
        """Test successful job execution."""
        with patch(
            "aclarai_scheduler.concept_summary_refresh.ConceptSummaryAgent"
        ) as mock_agent_class:
            with patch(
                "aclarai_scheduler.concept_summary_refresh.Neo4jGraphManager"
            ) as mock_neo4j_class:
                # Mock the agent and Neo4j manager
                mock_agent = Mock()
                mock_neo4j = Mock()
                mock_agent_class.return_value = mock_agent
                mock_neo4j_class.return_value = mock_neo4j

                job = ConceptSummaryRefreshJob()

                # Mock the methods
                job._get_canonical_concepts = Mock(
                    return_value=["Concept A", "Concept B"]
                )
                job._process_concept = Mock(return_value=(True, True))

                result = job.run_job()

                assert result["success"] is True
                assert result["concepts_processed"] == 2
                assert result["concepts_updated"] == 2
                assert result["concepts_skipped"] == 0
                assert result["errors"] == 0
                assert len(result["error_details"]) == 0

    def test_run_job_with_skipped_concepts(self):
        """Test job execution with some concepts skipped."""
        with patch(
            "aclarai_scheduler.concept_summary_refresh.ConceptSummaryAgent"
        ) as mock_agent_class:
            with patch(
                "aclarai_scheduler.concept_summary_refresh.Neo4jGraphManager"
            ) as mock_neo4j_class:
                # Mock the agent and Neo4j manager
                mock_agent = Mock()
                mock_neo4j = Mock()
                mock_agent_class.return_value = mock_agent
                mock_neo4j_class.return_value = mock_neo4j

                job = ConceptSummaryRefreshJob()

                # Mock the methods - some concepts processed, some skipped
                job._get_canonical_concepts = Mock(
                    return_value=["Concept A", "Concept B", "Concept C"]
                )

                def mock_process_concept(name):
                    if name == "Concept A":
                        return (True, True)  # Processed and updated
                    elif name == "Concept B":
                        return (False, False)  # Skipped
                    else:
                        return (True, False)  # Processed but not updated

                job._process_concept = Mock(side_effect=mock_process_concept)

                result = job.run_job()

                assert result["success"] is True
                assert result["concepts_processed"] == 2
                assert result["concepts_updated"] == 1
                assert result["concepts_skipped"] == 1
                assert result["errors"] == 0

    def test_run_job_with_errors(self):
        """Test job execution with some concept processing errors."""
        with patch(
            "aclarai_scheduler.concept_summary_refresh.ConceptSummaryAgent"
        ) as mock_agent_class:
            with patch(
                "aclarai_scheduler.concept_summary_refresh.Neo4jGraphManager"
            ) as mock_neo4j_class:
                # Mock the agent and Neo4j manager
                mock_agent = Mock()
                mock_neo4j = Mock()
                mock_agent_class.return_value = mock_agent
                mock_neo4j_class.return_value = mock_neo4j

                job = ConceptSummaryRefreshJob()

                # Mock the methods
                job._get_canonical_concepts = Mock(
                    return_value=["Concept A", "Concept B"]
                )

                def mock_process_concept(name):
                    if name == "Concept A":
                        return (True, True)  # Success
                    else:
                        raise Exception("Processing error")

                job._process_concept = Mock(side_effect=mock_process_concept)

                result = job.run_job()

                assert (
                    result["success"] is True
                )  # Still succeeds if some concepts are processed
                assert result["concepts_processed"] == 1
                assert result["concepts_updated"] == 1
                assert result["concepts_skipped"] == 0
                assert result["errors"] == 1
                assert len(result["error_details"]) == 1
                assert (
                    "Concept 'Concept B': Processing error" in result["error_details"]
                )

    def test_run_job_no_concepts(self):
        """Test job execution with no concepts found."""
        with patch(
            "aclarai_scheduler.concept_summary_refresh.ConceptSummaryAgent"
        ) as mock_agent_class:
            with patch(
                "aclarai_scheduler.concept_summary_refresh.Neo4jGraphManager"
            ) as mock_neo4j_class:
                # Mock the agent and Neo4j manager
                mock_agent = Mock()
                mock_neo4j = Mock()
                mock_agent_class.return_value = mock_agent
                mock_neo4j_class.return_value = mock_neo4j

                job = ConceptSummaryRefreshJob()

                # Mock empty concepts list
                job._get_canonical_concepts = Mock(return_value=[])

                result = job.run_job()

                assert (
                    result["success"] is True
                )  # Should still succeed with no concepts
                assert result["concepts_processed"] == 0
                assert result["concepts_updated"] == 0
                assert result["concepts_skipped"] == 0
                assert result["errors"] == 0

    def test_get_canonical_concepts_success(self):
        """Test successful retrieval of canonical concepts."""
        with patch(
            "services.scheduler.aclarai_scheduler.concept_summary_refresh.ConceptSummaryAgent"
        ):
            with patch(
                "aclarai_scheduler.concept_summary_refresh.Neo4jGraphManager"
            ) as mock_neo4j_class:
                # Mock the Neo4j manager
                mock_driver = Mock()
                mock_session = Mock()
                mock_result = Mock()
                mock_neo4j = Mock()

                mock_neo4j_class.return_value = mock_neo4j
                mock_neo4j.get_driver.return_value.__enter__.return_value = mock_driver
                mock_driver.session.return_value.__enter__.return_value = mock_session
                mock_session.run.return_value = mock_result

                # Mock result records
                mock_result.__iter__.return_value = [
                    {"name": "Concept A"},
                    {"name": "Concept B"},
                ]

                job = ConceptSummaryRefreshJob()
                concepts = job._get_canonical_concepts()

                assert concepts == ["Concept A", "Concept B"]

    def test_get_canonical_concepts_failure(self):
        """Test failure in retrieving canonical concepts."""
        with patch(
            "services.scheduler.aclarai_scheduler.concept_summary_refresh.ConceptSummaryAgent"
        ):
            with patch(
                "aclarai_scheduler.concept_summary_refresh.Neo4jGraphManager"
            ) as mock_neo4j_class:
                # Mock the Neo4j manager to raise an exception
                mock_neo4j = Mock()
                mock_neo4j_class.return_value = mock_neo4j
                mock_neo4j.get_driver.side_effect = Exception(
                    "Database connection failed"
                )

                job = ConceptSummaryRefreshJob()
                concepts = job._get_canonical_concepts()

                assert concepts == []

    def test_process_concept_success(self):
        """Test successful concept processing."""
        with patch(
            "aclarai_scheduler.concept_summary_refresh.ConceptSummaryAgent"
        ) as mock_agent_class:
            with patch("aclarai_scheduler.concept_summary_refresh.Neo4jGraphManager"):
                # Mock the agent
                mock_agent = Mock()
                mock_agent_class.return_value = mock_agent
                mock_agent.process_concept.return_value = True

                job = ConceptSummaryRefreshJob()
                job._concept_has_claims = Mock(return_value=True)

                processed, updated = job._process_concept("Test Concept")

                assert processed is True
                assert updated is True
                mock_agent.process_concept.assert_called_once_with("Test Concept")

    def test_process_concept_skip_no_claims(self):
        """Test concept processing skipped due to no claims."""
        with patch(
            "aclarai_scheduler.concept_summary_refresh.ConceptSummaryAgent"
        ) as mock_agent_class:
            with patch("aclarai_scheduler.concept_summary_refresh.Neo4jGraphManager"):
                # Mock config to skip if no claims
                mock_config = Mock()
                mock_config.concept_summaries.skip_if_no_claims = True

                job = ConceptSummaryRefreshJob(config=mock_config)
                job._concept_has_claims = Mock(return_value=False)

                processed, updated = job._process_concept("Test Concept")

                assert processed is False
                assert updated is False

    def test_concept_has_claims_success(self):
        """Test successful check for concept claims."""
        with patch(
            "services.scheduler.aclarai_scheduler.concept_summary_refresh.ConceptSummaryAgent"
        ):
            with patch(
                "aclarai_scheduler.concept_summary_refresh.Neo4jGraphManager"
            ) as mock_neo4j_class:
                # Mock the Neo4j manager
                mock_driver = Mock()
                mock_session = Mock()
                mock_result = Mock()
                mock_neo4j = Mock()

                mock_neo4j_class.return_value = mock_neo4j
                mock_neo4j.get_driver.return_value.__enter__.return_value = mock_driver
                mock_driver.session.return_value.__enter__.return_value = mock_session
                mock_session.run.return_value = mock_result
                mock_result.single.return_value = {"has_claims": True}

                job = ConceptSummaryRefreshJob()
                has_claims = job._concept_has_claims("Test Concept")

                assert has_claims is True

    def test_concept_has_claims_failure(self):
        """Test failure in checking concept claims."""
        with patch(
            "services.scheduler.aclarai_scheduler.concept_summary_refresh.ConceptSummaryAgent"
        ):
            with patch(
                "aclarai_scheduler.concept_summary_refresh.Neo4jGraphManager"
            ) as mock_neo4j_class:
                # Mock the Neo4j manager to raise an exception
                mock_neo4j = Mock()
                mock_neo4j_class.return_value = mock_neo4j
                mock_neo4j.get_driver.side_effect = Exception("Database error")

                job = ConceptSummaryRefreshJob()
                has_claims = job._concept_has_claims("Test Concept")

                # Should default to True if we can't check
                assert has_claims is True
