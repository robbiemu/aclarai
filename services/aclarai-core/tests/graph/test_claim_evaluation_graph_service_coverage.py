from unittest.mock import MagicMock, patch

import pytest
from aclarai_core.graph.claim_evaluation_graph_service import (
    ClaimEvaluationGraphService,
)
from aclarai_shared.config import aclaraiConfig
from neo4j import Driver


@pytest.fixture
def mock_driver():
    """Mock Neo4j driver for testing."""
    driver = MagicMock(spec=Driver)
    return driver


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = MagicMock(spec=aclaraiConfig)
    return config


@pytest.fixture
def graph_service(mock_driver, mock_config):
    """Create a ClaimEvaluationGraphService instance for testing."""
    return ClaimEvaluationGraphService(mock_driver, mock_config)


class TestClaimEvaluationGraphServiceElementsAndOmits:
    """Test the Element nodes and OMITS relationships functionality."""

    def test_create_element_nodes_and_omits_relationships_success(
        self, graph_service, mock_driver
    ):
        """Test successful creation of Element nodes and OMITS relationships."""
        # Mock session and transaction
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None

        # Mock query result
        mock_result = MagicMock()
        mock_record = MagicMock()
        mock_record.__getitem__.return_value = 2  # created_elements count
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result

        omitted_elements = [
            {"text": "European Commission", "significance": "Key regulatory body"},
            {"text": "2023", "significance": "Specific timeframe"},
        ]

        result = graph_service.create_element_nodes_and_omits_relationships(
            "claim123", omitted_elements
        )

        assert result is True
        mock_session.run.assert_called_once()

        # Verify the query structure
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        parameters = call_args[0][1]

        assert "MATCH (c:Claim {id: $claim_id})" in query
        assert "UNWIND $elements AS element" in query
        assert (
            "CREATE (e:Element {text: element.text, significance: element.significance})"
            in query
        )
        assert "CREATE (c)-[:OMITS]->(e)" in query
        assert parameters["claim_id"] == "claim123"
        assert parameters["elements"] == omitted_elements

    def test_create_element_nodes_and_omits_relationships_empty_elements(
        self, graph_service, mock_driver
    ):
        """Test handling of empty omitted elements list."""
        result = graph_service.create_element_nodes_and_omits_relationships(
            "claim123", []
        )

        assert result is True
        # Should not attempt any database operations
        mock_driver.session.assert_not_called()

    def test_create_element_nodes_and_omits_relationships_partial_creation(
        self, graph_service, mock_driver
    ):
        """Test handling when fewer elements are created than expected."""
        # Mock session and transaction
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None

        # Mock query result - only 1 element created instead of 2
        mock_result = MagicMock()
        mock_record = MagicMock()
        mock_record.__getitem__.return_value = 1  # created_elements count
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result

        omitted_elements = [
            {"text": "European Commission", "significance": "Key regulatory body"},
            {"text": "2023", "significance": "Specific timeframe"},
        ]

        result = graph_service.create_element_nodes_and_omits_relationships(
            "claim123", omitted_elements
        )

        assert result is False  # Should return False when not all elements created

    def test_create_element_nodes_and_omits_relationships_database_error(
        self, graph_service, mock_driver
    ):
        """Test handling of database errors during element creation."""
        # Mock session to raise an exception
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        mock_session.run.side_effect = Exception("Database connection error")

        omitted_elements = [
            {"text": "European Commission", "significance": "Key regulatory body"}
        ]

        result = graph_service.create_element_nodes_and_omits_relationships(
            "claim123", omitted_elements
        )

        assert result is False

    def test_update_coverage_score_delegates_to_update_relationship_score(
        self, graph_service
    ):
        """Test that update_coverage_score properly delegates to update_relationship_score."""
        with patch.object(graph_service, "update_relationship_score") as mock_update:
            mock_update.return_value = True

            result = graph_service.update_coverage_score("claim123", "block456", 0.85)

            assert result is True
            mock_update.assert_called_once_with(
                "claim123", "block456", "coverage_score", 0.85
            )

    def test_update_coverage_score_with_null_score(self, graph_service):
        """Test updating coverage score with null value."""
        with patch.object(graph_service, "update_relationship_score") as mock_update:
            mock_update.return_value = True

            result = graph_service.update_coverage_score("claim123", "block456", None)

            assert result is True
            mock_update.assert_called_once_with(
                "claim123", "block456", "coverage_score", None
            )


class TestClaimEvaluationGraphServiceValidation:
    """Test validation in the ClaimEvaluationGraphService."""

    def test_valid_score_names_includes_coverage_score(self, graph_service):
        """Test that coverage_score is included in valid score names."""
        assert "coverage_score" in graph_service.VALID_SCORE_NAMES
        assert "entailed_score" in graph_service.VALID_SCORE_NAMES
        assert "decontextualization_score" in graph_service.VALID_SCORE_NAMES
