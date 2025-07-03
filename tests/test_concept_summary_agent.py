"""
Tests for the Concept Summary Agent.
This module tests the functionality of the ConceptSummaryAgent that generates
detailed Markdown pages for canonical concepts.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

from aclarai_shared.concept_summary_agent import ConceptSummaryAgent


class MockNeo4jManager:
    """Mock Neo4j manager for testing."""

    def __init__(self):
        self.execute_query = Mock()

    def close(self):
        pass


def test_concept_summary_agent_initialization():
    """Test ConceptSummaryAgent initialization with default config."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_config.llm.provider = "openai"
    mock_config.llm.model = "gpt-4"
    mock_config.llm.api_key = "test-key"
    mock_config.model_dump.return_value = {}

    mock_neo4j = MockNeo4jManager()

    agent = ConceptSummaryAgent(config=mock_config, neo4j_manager=mock_neo4j)

    assert agent.config == mock_config
    assert agent.neo4j_manager == mock_neo4j
    assert agent.model_name == "gpt-4"
    assert agent.max_examples == 5
    assert agent.skip_if_no_claims is True
    assert agent.include_see_also is True


def test_generate_concept_slug():
    """Test concept slug generation."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_config.llm.provider = "unsupported"  # This will trigger fallback
    mock_config.model_dump.return_value = {}
    mock_neo4j = MockNeo4jManager()

    agent = ConceptSummaryAgent(config=mock_config, neo4j_manager=mock_neo4j)

    # Test normal concept
    assert agent.generate_concept_slug("machine learning") == "machine_learning"

    # Test concept with special characters
    assert agent.generate_concept_slug("API/REST endpoints") == "api_rest_endpoints"

    # Test concept with numbers
    assert agent.generate_concept_slug("HTTP 404 error") == "http_404_error"

    # Test empty concept
    assert agent.generate_concept_slug("") == "unnamed_concept"

    # Test concept with only special characters
    assert agent.generate_concept_slug("@#$%") == "unnamed_concept"


def test_generate_concept_filename():
    """Test concept filename generation."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_neo4j = MockNeo4jManager()

    agent = ConceptSummaryAgent(config=mock_config, neo4j_manager=mock_neo4j)

    # Test normal concept
    assert agent.generate_concept_filename("machine learning") == "machine_learning.md"

    # Test concept with special characters
    assert (
        agent.generate_concept_filename("API/REST endpoints") == "API_REST_endpoints.md"
    )

    # Test very long concept name
    long_name = "a" * 250
    filename = agent.generate_concept_filename(long_name)
    assert len(filename) <= 203  # 200 + ".md"
    assert filename.endswith(".md")


def test_get_canonical_concepts_empty():
    """Test getting canonical concepts when none exist."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_neo4j = MockNeo4jManager()
    mock_neo4j.execute_query.return_value = []

    agent = ConceptSummaryAgent(config=mock_config, neo4j_manager=mock_neo4j)

    concepts = agent.get_canonical_concepts()
    assert concepts == []
    mock_neo4j.execute_query.assert_called_once()


def test_get_canonical_concepts_with_data():
    """Test getting canonical concepts with sample data."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_neo4j = MockNeo4jManager()

    # Mock Neo4j response
    mock_records = [
        {
            "id": "concept_1",
            "text": "machine learning",
            "aclarai_id": "concept_machine_learning",
            "version": 1,
            "timestamp": "2024-01-01T00:00:00Z",
        },
        {
            "id": "concept_2",
            "text": "deep learning",
            "aclarai_id": "concept_deep_learning",
            "version": 1,
            "timestamp": "2024-01-01T01:00:00Z",
        },
    ]
    mock_neo4j.execute_query.return_value = mock_records

    agent = ConceptSummaryAgent(config=mock_config, neo4j_manager=mock_neo4j)

    concepts = agent.get_canonical_concepts()
    assert len(concepts) == 2
    assert concepts[0]["text"] == "machine learning"
    assert concepts[1]["text"] == "deep learning"
    mock_neo4j.execute_query.assert_called_once()


def test_should_skip_concept():
    """Test concept skipping logic."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_neo4j = MockNeo4jManager()

    # Test with skip_if_no_claims = True (default)
    agent = ConceptSummaryAgent(config=mock_config, neo4j_manager=mock_neo4j)

    concept = {"id": "concept_1", "text": "test concept"}

    # Should skip when no claims
    context_no_claims = {"claims": [], "summaries": []}
    assert agent.should_skip_concept(concept, context_no_claims) is True

    # Should not skip when claims exist
    context_with_claims = {"claims": [{"text": "test claim"}], "summaries": []}
    assert agent.should_skip_concept(concept, context_with_claims) is False

    # Test with skip_if_no_claims = False
    agent.skip_if_no_claims = False
    assert agent.should_skip_concept(concept, context_no_claims) is False


def test_generate_concept_content():
    """Test concept content generation."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_neo4j = MockNeo4jManager()

    agent = ConceptSummaryAgent(config=mock_config, neo4j_manager=mock_neo4j)

    concept = {
        "id": "concept_1",
        "text": "machine learning",
        "aclarai_id": "concept_machine_learning",
        "version": 1,
    }

    context = {
        "claims": [
            {
                "text": "ML is a subset of AI",
                "aclarai_id": "claim_ml_subset",
            }
        ],
        "summaries": [
            {
                "text": "Summary of ML concepts",
                "aclarai_id": "summary_ml_concepts",
            }
        ],
        "related_concepts": ["artificial intelligence", "deep learning"],
    }

    content = agent.generate_concept_content(concept, context)

    # Check required sections
    assert "## Concept: machine learning" in content
    assert "### Examples" in content
    assert "### See Also" in content
    assert "ML is a subset of AI ^claim_ml_subset" in content
    assert "Summary of ML concepts ^summary_ml_concepts" in content
    assert "- [[artificial intelligence]]" in content
    assert "- [[deep learning]]" in content
    assert "<!-- aclarai:id=concept_machine_learning ver=1 -->" in content
    assert "^concept_machine_learning" in content


def test_generate_concept_page_with_atomic_write():
    """Test concept page generation with atomic file writing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)
        concepts_path = vault_path / "concepts"

        mock_config = MagicMock()
        mock_config.paths.vault = str(vault_path)
        mock_config.paths.tier3 = "concepts"
        mock_neo4j = MockNeo4jManager()

        agent = ConceptSummaryAgent(config=mock_config, neo4j_manager=mock_neo4j)

        # Mock the retrieval methods to return test data
        agent.claim_retrieval_agent = Mock()
        agent.claim_retrieval_agent.chat.return_value = Mock(
            response=[{"text": "Test claim", "aclarai_id": "test_claim"}]
        )
        agent.related_concepts_agent = Mock()
        agent.related_concepts_agent.chat.return_value = Mock(response=[])

        concept = {
            "id": "concept_1",
            "text": "test concept",
            "aclarai_id": "concept_test",
            "version": 1,
        }

        # Generate the concept page
        result = agent.generate_concept_page(concept)

        assert result is True

        # Check that file was created
        expected_file = concepts_path / "test_concept.md"
        assert expected_file.exists()

        # Check file content
        content = expected_file.read_text()
        assert "## Concept: test concept" in content
        assert "Test claim ^test_claim" in content


def test_generate_concept_page_skip_no_claims():
    """Test concept page generation when concept should be skipped."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)

        mock_config = MagicMock()
        mock_config.paths.vault = str(vault_path)
        mock_config.paths.tier3 = "concepts"
        mock_neo4j = MockNeo4jManager()

        agent = ConceptSummaryAgent(config=mock_config, neo4j_manager=mock_neo4j)
        agent.skip_if_no_claims = True

        # Mock the retrieval methods to return no data
        agent.get_concept_claims = Mock(return_value=[])
        agent.get_concept_summaries = Mock(return_value=[])
        agent.get_related_concepts = Mock(return_value=[])

        concept = {
            "id": "concept_1",
            "text": "test concept",
            "aclarai_id": "concept_test",
            "version": 1,
        }

        # Generate the concept page - should be skipped
        result = agent.generate_concept_page(concept)

        assert result is False

        # Check that no file was created
        concepts_path = Path(vault_path) / "concepts"
        if concepts_path.exists():
            assert len(list(concepts_path.glob("*.md"))) == 0


def test_run_agent():
    """Test running the full agent workflow."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)

        mock_config = MagicMock()
        mock_config.paths.vault = str(vault_path)
        mock_config.paths.tier3 = "concepts"
        mock_neo4j = MockNeo4jManager()

        agent = ConceptSummaryAgent(config=mock_config, neo4j_manager=mock_neo4j)

        # Mock the get_canonical_concepts method
        test_concepts = [
            {
                "id": "concept_1",
                "text": "machine learning",
                "aclarai_id": "concept_ml",
                "version": 1,
            },
            {
                "id": "concept_2",
                "text": "deep learning",
                "aclarai_id": "concept_dl",
                "version": 1,
            },
        ]
        agent.get_canonical_concepts = Mock(return_value=test_concepts)

        # Mock the generate_concept_page method
        agent.generate_concept_page = Mock(side_effect=[True, True])

        # Run the agent
        result = agent.run_agent()

        assert result["success"] is True
        assert result["concepts_processed"] == 2
        assert result["concepts_generated"] == 2
        assert result["concepts_skipped"] == 0
        assert result["errors"] == 0
        assert result["error_details"] == []

        # Verify methods were called
        agent.get_canonical_concepts.assert_called_once()
        assert agent.generate_concept_page.call_count == 2


def test_run_agent_no_concepts():
    """Test running agent when no concepts exist."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_neo4j = MockNeo4jManager()

    agent = ConceptSummaryAgent(config=mock_config, neo4j_manager=mock_neo4j)

    # Mock no concepts
    agent.get_canonical_concepts = Mock(return_value=[])

    result = agent.run_agent()

    assert result["success"] is True
    assert result["concepts_processed"] == 0
    assert result["concepts_generated"] == 0
    assert result["concepts_skipped"] == 0
    assert result["errors"] == 0


def test_run_agent_with_errors():
    """Test running agent when errors occur during processing."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_neo4j = MockNeo4jManager()

    agent = ConceptSummaryAgent(config=mock_config, neo4j_manager=mock_neo4j)

    # Mock concepts
    test_concepts = [
        {
            "id": "concept_1",
            "text": "test concept",
            "aclarai_id": "concept_test",
            "version": 1,
        },
    ]
    agent.get_canonical_concepts = Mock(return_value=test_concepts)

    # Mock generate_concept_page to raise an error
    agent.generate_concept_page = Mock(side_effect=Exception("Test error"))

    result = agent.run_agent()

    assert result["success"] is True  # Agent continues despite errors
    assert result["concepts_processed"] == 1
    assert result["concepts_generated"] == 0
    assert result["concepts_skipped"] == 0
    assert result["errors"] == 1
    assert len(result["error_details"]) == 1
    assert "Test error" in result["error_details"][0]


if __name__ == "__main__":
    # Run basic tests
    print("Running concept summary agent tests...")
    try:
        test_concept_summary_agent_initialization()
        test_generate_concept_slug()
        test_generate_concept_filename()
        test_get_canonical_concepts_empty()
        test_get_canonical_concepts_with_data()
        test_should_skip_concept()
        test_generate_concept_content()
        test_generate_concept_page_with_atomic_write()
        test_generate_concept_page_skip_no_claims()
        test_run_agent()
        test_run_agent_no_concepts()
        test_run_agent_with_errors()
        print("✓ All tests passed")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        exit(1)
    print("All tests completed successfully!")
