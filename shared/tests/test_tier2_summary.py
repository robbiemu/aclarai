"""
Tests for the Tier 2 Summary Agent module.
This test suite validates the functionality of the Tier 2 Summary Agent including
data models, summary generation, and file writing operations.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from aclarai_shared.config import aclaraiConfig
from aclarai_shared.tier2_summary import (
    SummaryBlock,
    SummaryInput,
    SummaryResult,
    Tier2SummaryAgent,
    generate_summary_id,
)


class TestSummaryDataModels:
    """Test the data models for Tier 2 Summary."""

    def test_summary_input_creation(self):
        """Test SummaryInput object creation and properties."""
        claims = [
            {
                "id": "claim1",
                "text": "First claim",
                "node_type": "claim",
                "source_block_id": "blk_001",
            },
            {
                "id": "claim2",
                "text": "Second claim",
                "node_type": "claim",
                "source_block_id": "blk_002",
            },
        ]
        sentences = [
            {
                "id": "sent1",
                "text": "First sentence",
                "node_type": "sentence",
                "source_block_id": "blk_003",
            },
        ]
        summary_input = SummaryInput(
            claims=claims, sentences=sentences, group_context="Test context"
        )
        assert len(summary_input.claims) == 2
        assert len(summary_input.sentences) == 1
        assert summary_input.group_context == "Test context"
        # Test all_texts property
        all_texts = summary_input.all_texts
        assert len(all_texts) == 3
        assert "First claim" in all_texts
        assert "Second claim" in all_texts
        assert "First sentence" in all_texts
        # Test source_block_ids property
        block_ids = summary_input.source_block_ids
        assert len(block_ids) == 3
        assert "blk_001" in block_ids
        assert "blk_002" in block_ids
        assert "blk_003" in block_ids

    def test_summary_block_creation(self):
        """Test SummaryBlock object creation and markdown generation."""
        summary_block = SummaryBlock(
            summary_text="This is a test summary\nWith multiple lines",
            aclarai_id="clm_test123",
            version=2,
            source_block_ids=["blk_001", "blk_002"],
        )
        assert (
            summary_block.summary_text == "This is a test summary\nWith multiple lines"
        )
        assert summary_block.aclarai_id == "clm_test123"
        assert summary_block.version == 2
        assert summary_block.source_block_ids == ["blk_001", "blk_002"]
        assert summary_block.timestamp is not None

    def test_summary_block_markdown_generation(self):
        """Test markdown generation from SummaryBlock."""
        summary_block = SummaryBlock(
            summary_text="First point\nSecond point",
            aclarai_id="clm_abc123",
            version=1,
        )
        markdown = summary_block.to_markdown()
        # Check structure
        assert "- First point" in markdown
        assert "- Second point ^clm_abc123" in markdown  # Last line gets anchor
        assert "<!-- aclarai:id=clm_abc123 ver=1 -->" in markdown
        assert "^clm_abc123" in markdown
        # Check it ends with anchor
        lines = markdown.strip().split("\n")
        assert lines[-1] == "^clm_abc123"

    def test_summary_block_markdown_with_concepts(self):
        """Test markdown generation from SummaryBlock with linked concepts."""
        summary_block = SummaryBlock(
            summary_text="Point about machine learning\nAnother point about AI",
            aclarai_id="clm_def456",
            version=1,
            linked_concepts=["Machine Learning", "Artificial Intelligence"],
        )
        markdown = summary_block.to_markdown()
        # Check structure
        assert "- Point about machine learning" in markdown
        assert (
            "- Another point about AI ^clm_def456" in markdown
        )  # Last line gets anchor
        assert (
            "Related concepts: [[Machine Learning]], [[Artificial Intelligence]]"
            in markdown
        )
        assert "<!-- aclarai:id=clm_def456 ver=1 -->" in markdown
        assert "^clm_def456" in markdown
        # Check it ends with anchor
        lines = markdown.strip().split("\n")
        assert lines[-1] == "^clm_def456"

    def test_summary_block_markdown_without_concepts(self):
        """Test markdown generation from SummaryBlock without linked concepts."""
        summary_block = SummaryBlock(
            summary_text="Basic summary point",
            aclarai_id="clm_ghi789",
            version=1,
            linked_concepts=[],  # Empty concepts
        )
        markdown = summary_block.to_markdown()
        # Should not contain concept section when list is empty
        assert "Related concepts:" not in markdown
        assert "[[" not in markdown
        assert "- Basic summary point ^clm_ghi789" in markdown
        assert "<!-- aclarai:id=clm_ghi789 ver=1 -->" in markdown

    def test_summary_result_creation(self):
        """Test SummaryResult object creation and properties."""
        block1 = SummaryBlock(summary_text="First summary", aclarai_id="clm_001")
        block2 = SummaryBlock(summary_text="Second summary", aclarai_id="clm_002")
        result = SummaryResult(
            summary_blocks=[block1, block2],
            source_file_context="Test conversation",
            processing_time=1.5,
            model_used="gpt-3.5-turbo",
        )
        assert len(result.summary_blocks) == 2
        assert result.source_file_context == "Test conversation"
        assert result.processing_time == 1.5
        assert result.model_used == "gpt-3.5-turbo"
        assert result.is_successful is True
        assert result.error is None

    def test_summary_result_error_case(self):
        """Test SummaryResult with error."""
        result = SummaryResult(error="Test error message", processing_time=0.5)
        assert result.is_successful is False
        assert result.error == "Test error message"
        assert len(result.summary_blocks) == 0

    def test_summary_result_markdown_generation(self):
        """Test full markdown file generation from SummaryResult."""
        block1 = SummaryBlock(summary_text="First summary point", aclarai_id="clm_001")
        block2 = SummaryBlock(summary_text="Second summary point", aclarai_id="clm_002")
        result = SummaryResult(
            summary_blocks=[block1, block2], source_file_context="test_context"
        )
        markdown = result.to_markdown(title="Test Summary")
        # Check title
        assert "# Test Summary" in markdown
        # Check context metadata
        assert "<!-- aclarai:source_context=test_context -->" in markdown
        # Check both blocks are present
        assert "clm_001" in markdown
        assert "clm_002" in markdown
        # Check structure - blocks should be separated by empty lines
        assert "clm_001\n\n- Second summary point" in markdown

    def test_generate_summary_id(self):
        """Test summary ID generation."""
        id1 = generate_summary_id()
        id2 = generate_summary_id()
        # Should start with clm_
        assert id1.startswith("clm_")
        assert id2.startswith("clm_")
        # Should be unique
        assert id1 != id2
        # Should have correct length (clm_ + 8 hex chars = 12 total)
        assert len(id1) == 12
        assert len(id2) == 12


class TestTier2SummaryAgent:
    """Test the main Tier2SummaryAgent class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = Mock(spec=aclaraiConfig)
        # Create nested mock structure for llm and paths
        config.llm = Mock()
        config.llm.model = "gpt-3.5-turbo"
        config.llm.model_params = {"temperature": 0.1, "max_tokens": 1000}
        config.paths = Mock()
        config.paths.vault = "/test/vault"
        config.paths.tier2 = "summaries"
        # Add features configuration
        config.features = {"tier2_generation": True}
        # Add threshold configuration
        config.threshold = Mock()
        config.threshold.summary_grouping_similarity = 0.80
        # Add processing configuration for retries
        config.processing = {
            "retries": {"max_attempts": 3, "backoff_factor": 2, "max_wait_time": 60}
        }
        return config

    @pytest.fixture
    def mock_neo4j_manager(self):
        """Create a mock Neo4j manager for testing."""
        manager = Mock()
        # Mock execute_query to return test data
        manager.execute_query.return_value = [
            {
                "id": "claim1",
                "text": "Test claim 1",
                "entailed_score": 0.8,
                "coverage_score": 0.9,
                "decontextualization_score": 0.7,
                "version": 1,
                "timestamp": "2024-01-01T10:00:00Z",
            },
            {
                "id": "claim2",
                "text": "Test claim 2",
                "entailed_score": 0.75,
                "coverage_score": 0.85,
                "decontextualization_score": 0.8,
                "version": 1,
                "timestamp": "2024-01-01T11:00:00Z",
            },
        ]
        return manager

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock()
        llm.model = "gpt-3.5-turbo"
        # Mock complete method
        response = Mock()
        response.text = "- Key finding from analysis\n- Important claim identified\n- Summary of main points"
        llm.complete.return_value = response
        return llm

    def test_agent_initialization(self, mock_config, mock_neo4j_manager, mock_llm):
        """Test agent initialization with different configurations."""
        # Test with all parameters provided
        agent = Tier2SummaryAgent(
            config=mock_config, neo4j_manager=mock_neo4j_manager, llm=mock_llm
        )
        assert agent.config == mock_config
        assert agent.neo4j_manager == mock_neo4j_manager
        assert agent.llm == mock_llm

    def test_agent_initialization_defaults(self):
        """Test agent initialization with defaults."""
        # Mock aclaraiConfig to avoid loading actual config file
        with patch(
            "aclarai_shared.tier2_summary.agent.aclaraiConfig"
        ) as mock_config_class:
            mock_config = Mock()
            mock_config.llm.model = "gpt-3.5-turbo"
            mock_config.llm.model_params = {"temperature": 0.1, "max_tokens": 1000}
            mock_config_class.return_value = mock_config
            with patch("aclarai_shared.tier2_summary.agent.OpenAI") as mock_openai:
                mock_llm = Mock()
                mock_openai.return_value = mock_llm
                agent = Tier2SummaryAgent()
                assert agent.config == mock_config
                print(mock_openai.call_args)
                mock_openai.assert_called_once_with(
                    model="gpt-3.5-turbo", temperature=0.1, max_tokens=1000
                )

    def test_get_high_quality_claims(self, mock_config, mock_neo4j_manager, mock_llm):
        """Test retrieval of high-quality claims from Neo4j."""
        # Mock high-quality claims response
        mock_neo4j_manager.execute_query.return_value = [
            {
                "id": "claim1",
                "text": "High quality claim 1",
                "entailed_score": 0.8,
                "coverage_score": 0.9,
                "decontextualization_score": 0.85,
                "version": 1,
                "timestamp": "2024-01-01T10:00:00Z",
                "source_block_id": "blk_001",
                "source_block_text": "Original utterance text 1",
            },
            {
                "id": "claim2",
                "text": "High quality claim 2",
                "entailed_score": 0.75,
                "coverage_score": 0.8,
                "decontextualization_score": 0.78,
                "version": 1,
                "timestamp": "2024-01-01T11:00:00Z",
                "source_block_id": "blk_002",
                "source_block_text": "Original utterance text 2",
            },
        ]
        agent = Tier2SummaryAgent(
            config=mock_config, neo4j_manager=mock_neo4j_manager, llm=mock_llm
        )
        claims = agent._get_high_quality_claims()
        assert len(claims) == 2
        assert claims[0]["id"] == "claim1"
        assert claims[0]["text"] == "High quality claim 1"
        assert claims[0]["source_block_id"] == "blk_001"
        assert claims[0]["node_type"] == "claim"
        assert claims[1]["id"] == "claim2"
        # Verify Neo4j query was called
        mock_neo4j_manager.execute_query.assert_called_once()
        call_args = mock_neo4j_manager.execute_query.call_args[0]
        assert "MATCH (c:Claim)-[:REFERENCES]->(b:Block)" in call_args[0]
        assert "entailed_score > 0.7" in call_args[0]

    def test_get_high_quality_claims_error_handling(self, mock_config, mock_llm):
        """Test error handling in high-quality claim retrieval."""
        mock_neo4j_manager = Mock()
        mock_neo4j_manager.execute_query.side_effect = Exception("Database error")
        agent = Tier2SummaryAgent(
            config=mock_config, neo4j_manager=mock_neo4j_manager, llm=mock_llm
        )
        claims = agent._get_high_quality_claims()
        # Should return empty list on error
        assert claims == []

    def test_generate_summary(self, mock_config, mock_neo4j_manager, mock_llm):
        """Test summary generation from SummaryInput."""
        agent = Tier2SummaryAgent(
            config=mock_config, neo4j_manager=mock_neo4j_manager, llm=mock_llm
        )
        summary_input = SummaryInput(
            claims=[{"text": "Test claim", "node_type": "claim"}],
            sentences=[{"text": "Test sentence", "node_type": "sentence"}],
            group_context="Test context",
        )
        result = agent.generate_summary(summary_input)
        assert result.is_successful
        assert len(result.summary_blocks) == 1
        assert result.error is None
        assert result.processing_time is not None
        assert result.model_used == "gpt-3.5-turbo"
        # Check summary block content
        block = result.summary_blocks[0]
        assert "Key finding from analysis" in block.summary_text
        assert block.aclarai_id.startswith("clm_")
        # Verify LLM was called
        mock_llm.complete.assert_called_once()

    def test_generate_summary_empty_input(
        self, mock_config, mock_neo4j_manager, mock_llm
    ):
        """Test summary generation with empty input."""
        agent = Tier2SummaryAgent(
            config=mock_config, neo4j_manager=mock_neo4j_manager, llm=mock_llm
        )
        summary_input = SummaryInput()  # Empty input
        result = agent.generate_summary(summary_input)
        assert not result.is_successful
        assert result.error == "No content to summarize"
        assert len(result.summary_blocks) == 0

    def test_generate_summary_llm_error(self, mock_config, mock_neo4j_manager):
        """Test summary generation with LLM error."""
        mock_llm = Mock()
        mock_llm.complete.side_effect = Exception("LLM error")
        agent = Tier2SummaryAgent(
            config=mock_config, neo4j_manager=mock_neo4j_manager, llm=mock_llm
        )
        summary_input = SummaryInput(
            claims=[{"text": "Test claim", "node_type": "claim"}]
        )
        result = agent.generate_summary(summary_input)
        assert not result.is_successful
        assert "LLM error" in result.error
        assert len(result.summary_blocks) == 0

    def test_create_summary_prompt(self, mock_config, mock_neo4j_manager, mock_llm):
        """Test prompt creation for LLM."""
        agent = Tier2SummaryAgent(
            config=mock_config, neo4j_manager=mock_neo4j_manager, llm=mock_llm
        )
        summary_input = SummaryInput(
            claims=[{"text": "First claim", "node_type": "claim"}],
            sentences=[{"text": "Second sentence", "node_type": "sentence"}],
        )
        prompt = agent._create_summary_prompt(summary_input)
        # Check prompt structure
        assert "summarization agent" in prompt
        assert "bullet points" in prompt
        assert "First claim" in prompt
        assert "Second sentence" in prompt
        assert "1. First claim" in prompt
        assert "2. Second sentence" in prompt

    @patch("aclarai_shared.tier2_summary.agent.write_file_atomically")
    def test_write_summary_file_success(
        self, mock_write, mock_config, mock_neo4j_manager, mock_llm
    ):
        """Test successful summary file writing."""
        agent = Tier2SummaryAgent(
            config=mock_config, neo4j_manager=mock_neo4j_manager, llm=mock_llm
        )
        summary_block = SummaryBlock(
            summary_text="Test summary", aclarai_id="clm_test123"
        )
        result = SummaryResult(summary_blocks=[summary_block])
        success = agent.write_summary_file(result, "/test/path.md", title="Test Title")
        assert success is True
        mock_write.assert_called_once()
        # Check the arguments passed to write_file_atomically
        call_args = mock_write.call_args
        path_arg = call_args[0][0]
        content_arg = call_args[0][1]
        assert str(path_arg) == "/test/path.md"
        assert "# Test Title" in content_arg
        assert "clm_test123" in content_arg

    @patch("aclarai_shared.tier2_summary.agent.write_file_atomically")
    def test_write_summary_file_failed_result(
        self, mock_write, mock_config, mock_neo4j_manager, mock_llm
    ):
        """Test writing summary file with failed result."""
        agent = Tier2SummaryAgent(
            config=mock_config, neo4j_manager=mock_neo4j_manager, llm=mock_llm
        )
        result = SummaryResult(error="Generation failed")
        success = agent.write_summary_file(result, "/test/path.md")
        assert success is False
        mock_write.assert_not_called()

    @patch("aclarai_shared.tier2_summary.agent.write_file_atomically")
    def test_write_summary_file_write_error(
        self, mock_write, mock_config, mock_neo4j_manager, mock_llm
    ):
        """Test handling of write errors."""
        mock_write.side_effect = Exception("Write error")
        agent = Tier2SummaryAgent(
            config=mock_config, neo4j_manager=mock_neo4j_manager, llm=mock_llm
        )
        summary_block = SummaryBlock(
            summary_text="Test summary", aclarai_id="clm_test123"
        )
        result = SummaryResult(summary_blocks=[summary_block])
        success = agent.write_summary_file(result, "/test/path.md")
        assert success is False

    def test_retrieve_grouped_content_no_manager(self, mock_config, mock_llm):
        """Test content retrieval without Neo4j manager."""
        agent = Tier2SummaryAgent(
            config=mock_config,
            neo4j_manager=None,  # No manager
            llm=mock_llm,
        )
        groups = agent.retrieve_grouped_content()
        assert groups == []

    def test_retrieve_grouped_content_no_embedding_storage(
        self, mock_config, mock_neo4j_manager, mock_llm
    ):
        """Test content retrieval without embedding storage."""
        agent = Tier2SummaryAgent(
            config=mock_config,
            neo4j_manager=mock_neo4j_manager,
            embedding_storage=None,  # No embedding storage
            llm=mock_llm,
        )
        groups = agent.retrieve_grouped_content()
        assert groups == []

    def test_retrieve_grouped_content_no_seeds(
        self, mock_config, mock_neo4j_manager, mock_llm
    ):
        """Test content retrieval when no high-quality claims are found."""
        # Mock empty high-quality claims response
        mock_neo4j_manager.execute_query.return_value = []
        mock_embedding_storage = Mock()
        agent = Tier2SummaryAgent(
            config=mock_config,
            neo4j_manager=mock_neo4j_manager,
            embedding_storage=mock_embedding_storage,
            llm=mock_llm,
        )
        groups = agent.retrieve_grouped_content()
        assert groups == []
        # Should call high-quality claims but not similarity search
        mock_neo4j_manager.execute_query.assert_called_once()
        mock_embedding_storage.similarity_search.assert_not_called()

    @patch(
        "aclarai_shared.tier2_summary.agent.Tier2SummaryAgent._build_semantic_neighborhoods"
    )
    def test_retrieve_grouped_content_success(
        self, mock_build_neighborhoods, mock_config, mock_neo4j_manager, mock_llm
    ):
        """Test successful content retrieval with vector similarity."""
        # Mock high-quality claims
        mock_neo4j_manager.execute_query.return_value = [
            {
                "id": "claim1",
                "text": "High quality claim",
                "entailed_score": 0.8,
                "coverage_score": 0.9,
                "decontextualization_score": 0.85,
                "version": 1,
                "timestamp": "2024-01-01T10:00:00Z",
                "source_block_id": "blk_001",
                "source_block_text": "Original utterance text",
            }
        ]
        # Mock semantic neighborhoods result
        mock_summary_input = SummaryInput(
            claims=[{"text": "Test claim", "source_block_id": "blk_001"}],
            sentences=[{"text": "Test sentence", "source_block_id": "blk_002"}],
            group_context="Semantic neighborhood test",
        )
        mock_build_neighborhoods.return_value = [mock_summary_input]
        mock_embedding_storage = Mock()
        agent = Tier2SummaryAgent(
            config=mock_config,
            neo4j_manager=mock_neo4j_manager,
            embedding_storage=mock_embedding_storage,
            llm=mock_llm,
        )
        groups = agent.retrieve_grouped_content()
        assert len(groups) == 1
        assert groups[0].group_context == "Semantic neighborhood test"
        # Verify the semantic neighborhoods builder was called with correct params
        mock_build_neighborhoods.assert_called_once()
        call_args = mock_build_neighborhoods.call_args[0]
        assert len(call_args[0]) == 1  # seed_claims
        assert call_args[1] == 0.80  # similarity_threshold (from config)

    def test_get_linked_concepts_for_summary(self):
        """Test retrieving linked concepts for a summary."""
        # Create test claims and concepts data
        claims = [
            {
                "id": "claim1",
                "text": "Machine learning is advancing",
                "node_type": "claim",
            },
            {"id": "claim2", "text": "AI improves efficiency", "node_type": "claim"},
        ]
        summary_input = SummaryInput(claims=claims)
        # Mock the claim-concept manager
        mock_concepts_mapping = {
            "claim1": [
                {
                    "concept_text": "Machine Learning",
                    "relationship_type": "SUPPORTS_CONCEPT",
                    "strength": 0.9,
                },
                {
                    "concept_text": "Technology",
                    "relationship_type": "MENTIONS_CONCEPT",
                    "strength": 0.7,
                },
            ],
            "claim2": [
                {
                    "concept_text": "Artificial Intelligence",
                    "relationship_type": "SUPPORTS_CONCEPT",
                    "strength": 0.95,
                },
                {
                    "concept_text": "Efficiency",
                    "relationship_type": "MENTIONS_CONCEPT",
                    "strength": 0.6,
                },
            ],
        }
        with patch(
            "aclarai_shared.tier2_summary.agent.aclaraiConfig"
        ) as mock_config_class:
            mock_config = Mock()
            mock_config.llm.model = "gpt-3.5-turbo"
            mock_config.llm.model_params = {}
            mock_config.features = {"tier2_generation": True}
            mock_config_class.return_value = mock_config
            with patch("aclarai_shared.tier2_summary.agent.OpenAI") as mock_openai:
                mock_llm = Mock()
                mock_openai.return_value = mock_llm
                agent = Tier2SummaryAgent()
                # Mock the claim-concept manager's method
                agent.claim_concept_manager.get_concepts_for_claims = Mock(
                    return_value=mock_concepts_mapping
                )
                # Test the method
                linked_concepts = agent._get_linked_concepts_for_summary(summary_input)
                # Should return unique concept names
                assert len(linked_concepts) == 4
                assert "Machine Learning" in linked_concepts
                assert "Artificial Intelligence" in linked_concepts
                assert "Technology" in linked_concepts
                assert "Efficiency" in linked_concepts
                # Verify the claim-concept manager was called correctly
                agent.claim_concept_manager.get_concepts_for_claims.assert_called_once_with(
                    ["claim1", "claim2"]
                )

    def test_get_linked_concepts_empty_claims(self):
        """Test retrieving linked concepts when no claims are present."""
        summary_input = SummaryInput(claims=[])
        with patch(
            "aclarai_shared.tier2_summary.agent.aclaraiConfig"
        ) as mock_config_class:
            mock_config = Mock()
            mock_config.llm.model = "gpt-3.5-turbo"
            mock_config.llm.model_params = {}
            mock_config.features = {"tier2_generation": True}
            mock_config_class.return_value = mock_config
            with patch("aclarai_shared.tier2_summary.agent.OpenAI") as mock_openai:
                mock_llm = Mock()
                mock_openai.return_value = mock_llm
                agent = Tier2SummaryAgent()
                # Test with empty claims
                linked_concepts = agent._get_linked_concepts_for_summary(summary_input)
                # Should return empty list
                assert linked_concepts == []


class TestAtomicFileWriting:
    """Test the atomic file writing functionality."""

    def test_atomic_write_integration(self):
        """Test that atomic write is used correctly."""
        # This test verifies the integration with the existing atomic write function
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_summary.md"
            summary_block = SummaryBlock(
                summary_text="Test summary content", aclarai_id="clm_integration"
            )
            result = SummaryResult(summary_blocks=[summary_block])
            # Create agent with minimal mocking
            with patch(
                "aclarai_shared.tier2_summary.agent.aclaraiConfig"
            ) as mock_config_class:
                mock_config = Mock()
                mock_config.llm.model = "gpt-3.5-turbo"
                mock_config.llm.model_params = {}
                mock_config_class.return_value = mock_config
                with patch("aclarai_shared.tier2_summary.agent.OpenAI") as mock_openai:
                    mock_llm = Mock()
                    mock_openai.return_value = mock_llm
                    agent = Tier2SummaryAgent()
                    # Write the file
                    success = agent.write_summary_file(
                        result, test_file, title="Integration Test"
                    )
                    assert success is True
                    assert test_file.exists()
                    # Verify content
                    content = test_file.read_text()
                    assert "# Integration Test" in content
                    assert "Test summary content" in content
                    assert "clm_integration" in content
                    assert "<!-- aclarai:id=clm_integration ver=1 -->" in content
                    assert "^clm_integration" in content


if __name__ == "__main__":
    pytest.main([__file__])
