"""
Test suite for noun phrase extraction functionality.
Tests the core noun phrase extraction logic including:
- spaCy integration and noun phrase extraction
- Phrase normalization
- Data model creation and validation
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add the parent directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aclarai_shared.config import load_config
from aclarai_shared.noun_phrase_extraction.extractor import NounPhraseExtractor
from aclarai_shared.noun_phrase_extraction.models import (
    ExtractionResult,
    NounPhraseCandidate,
)


class TestNounPhraseModels:
    """Test the data models for noun phrase extraction."""

    def test_noun_phrase_candidate_creation(self):
        """Test creating a NounPhraseCandidate with all required fields."""
        candidate = NounPhraseCandidate(
            text="natural language processing",
            normalized_text="natural language processing",
            source_node_id="claim_123",
            source_node_type="claim",
            aclarai_id="aclarai_123",
        )
        assert candidate.text == "natural language processing"
        assert candidate.normalized_text == "natural language processing"
        assert candidate.source_node_id == "claim_123"
        assert candidate.source_node_type == "claim"
        assert candidate.aclarai_id == "aclarai_123"
        assert candidate.status == "pending"
        assert candidate.timestamp is not None

    def test_extraction_result_success_rate(self):
        """Test calculating success rate for ExtractionResult."""
        result = ExtractionResult(
            total_nodes_processed=10, successful_extractions=8, failed_extractions=2
        )
        assert result.success_rate == 0.8
        assert not result.is_successful  # Due to failed extractions

    def test_extraction_result_empty(self):
        """Test ExtractionResult with no data."""
        result = ExtractionResult()
        assert result.success_rate == 0.0
        assert result.is_successful  # No errors or failures


class TestNounPhraseExtractor:
    """Test the main NounPhraseExtractor functionality."""

    @patch("aclarai_shared.noun_phrase_extraction.extractor.spacy")
    @patch("aclarai_shared.noun_phrase_extraction.extractor.Neo4jGraphManager")
    @patch("aclarai_shared.noun_phrase_extraction.extractor.EmbeddingGenerator")
    @patch(
        "aclarai_shared.noun_phrase_extraction.extractor.ConceptCandidatesVectorStore"
    )
    @patch("aclarai_shared.noun_phrase_extraction.extractor.load_config")
    def test_extractor_initialization(
        self,
        mock_load_config,
        _mock_vector_store,
        _mock_embedding_gen,
        _mock_neo4j,
        mock_spacy,
    ):
        """Test that NounPhraseExtractor initializes properly."""
        # Mock spaCy model
        mock_nlp = Mock()
        mock_spacy.load.return_value = mock_nlp
        # Mock config
        mock_config = Mock()
        mock_load_config.return_value = mock_config
        extractor = NounPhraseExtractor()
        assert extractor.config == mock_config
        assert extractor.spacy_model_name == "en_core_web_sm"
        assert extractor._nlp == mock_nlp
        mock_spacy.load.assert_called_once_with("en_core_web_sm")

    @patch("aclarai_shared.noun_phrase_extraction.extractor.spacy")
    @patch("aclarai_shared.noun_phrase_extraction.extractor.Neo4jGraphManager")
    @patch("aclarai_shared.noun_phrase_extraction.extractor.EmbeddingGenerator")
    @patch(
        "aclarai_shared.noun_phrase_extraction.extractor.ConceptCandidatesVectorStore"
    )
    @patch("aclarai_shared.noun_phrase_extraction.extractor.load_config")
    def test_extract_noun_phrases(
        self,
        mock_load_config,
        _mock_vector_store,
        _mock_embedding_gen,
        _mock_neo4j,
        mock_spacy,
    ):
        """Test extracting noun phrases from text using spaCy."""
        # Mock spaCy components
        mock_doc = Mock()
        mock_chunk1 = Mock()
        mock_chunk1.text = "  natural language processing  "
        mock_chunk2 = Mock()
        mock_chunk2.text = "machine learning algorithms"
        mock_chunk3 = Mock()
        mock_chunk3.text = "5"  # Should be filtered out
        mock_doc.noun_chunks = [mock_chunk1, mock_chunk2, mock_chunk3]
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_spacy.load.return_value = mock_nlp
        # Mock config
        mock_config = Mock()
        mock_load_config.return_value = mock_config
        extractor = NounPhraseExtractor()
        # Test extraction
        phrases = extractor._extract_noun_phrases("Some text with noun phrases")
        # Should exclude the digit-only phrase "5"
        expected_phrases = [
            "natural language processing",
            "machine learning algorithms",
        ]
        assert phrases == expected_phrases

    @patch("aclarai_shared.noun_phrase_extraction.extractor.spacy")
    @patch("aclarai_shared.noun_phrase_extraction.extractor.Neo4jGraphManager")
    @patch("aclarai_shared.noun_phrase_extraction.extractor.EmbeddingGenerator")
    @patch(
        "aclarai_shared.noun_phrase_extraction.extractor.ConceptCandidatesVectorStore"
    )
    @patch("aclarai_shared.noun_phrase_extraction.extractor.load_config")
    def test_normalize_phrase(
        self,
        mock_load_config,
        _mock_vector_store,
        _mock_embedding_gen,
        _mock_neo4j,
        mock_spacy,
    ):
        """Test phrase normalization (lowercase, lemmatize, strip punctuation)."""
        # Mock spaCy token processing
        mock_token1 = Mock()
        mock_token1.lemma_ = "natural"
        mock_token1.is_punct = False
        mock_token1.is_space = False
        mock_token2 = Mock()
        mock_token2.lemma_ = "language"
        mock_token2.is_punct = False
        mock_token2.is_space = False
        mock_token3 = Mock()
        mock_token3.lemma_ = "processing"
        mock_token3.is_punct = False
        mock_token3.is_space = False
        mock_punct = Mock()
        mock_punct.is_punct = True
        mock_punct.is_space = False
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(
            return_value=iter([mock_token1, mock_token2, mock_token3, mock_punct])
        )
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_spacy.load.return_value = mock_nlp
        # Mock config
        mock_config = Mock()
        mock_load_config.return_value = mock_config
        extractor = NounPhraseExtractor()
        # Test normalization
        normalized = extractor._normalize_phrase("Natural Language Processing!")
        assert normalized == "natural language processing"

    @patch("aclarai_shared.noun_phrase_extraction.extractor.spacy")
    @patch("aclarai_shared.noun_phrase_extraction.extractor.Neo4jGraphManager")
    @patch("aclarai_shared.noun_phrase_extraction.extractor.EmbeddingGenerator")
    @patch(
        "aclarai_shared.noun_phrase_extraction.extractor.ConceptCandidatesVectorStore"
    )
    @patch("aclarai_shared.noun_phrase_extraction.extractor.load_config")
    def test_extract_from_node(
        self,
        mock_load_config,
        _mock_vector_store,
        _mock_embedding_gen,
        _mock_neo4j,
        mock_spacy,
    ):
        """Test extracting noun phrases from a single node."""
        # Mock spaCy processing for extraction
        mock_chunk = Mock()
        mock_chunk.text = "machine learning"
        mock_doc_extract = Mock()
        mock_doc_extract.noun_chunks = [mock_chunk]
        # Mock spaCy processing for normalization
        mock_token = Mock()
        mock_token.lemma_ = "machine"
        mock_token.is_punct = False
        mock_token.is_space = False
        mock_token2 = Mock()
        mock_token2.lemma_ = "learning"
        mock_token2.is_punct = False
        mock_token2.is_space = False
        mock_doc_normalize = Mock()
        mock_doc_normalize.__iter__ = Mock(return_value=iter([mock_token, mock_token2]))
        mock_nlp = Mock()
        # Return different mocks for extraction vs normalization calls
        mock_nlp.side_effect = [mock_doc_extract, mock_doc_normalize]
        mock_spacy.load.return_value = mock_nlp
        # Mock config
        mock_config = Mock()
        mock_load_config.return_value = mock_config
        extractor = NounPhraseExtractor()
        # Test node data
        node = {
            "id": "claim_123",
            "text": "This claim discusses machine learning techniques.",
            "node_type": "claim",
        }
        extraction_result = extractor._extract_from_node(
            text=node["text"],
            source_node_id=node["id"],
            source_node_type=node["node_type"],
            aclarai_id=node[
                "id"
            ],  # Assuming aclarai_id is the same as node id for this test
        )
        assert len(extraction_result.candidates) == 1
        candidate = extraction_result.candidates[0]
        assert candidate.text == "machine learning"
        assert candidate.normalized_text == "machine learning"
        assert candidate.source_node_id == "claim_123"
        assert candidate.source_node_type == "claim"
        assert candidate.status == "pending"


@pytest.mark.integration
class TestNounPhraseExtractionRealIntegration:
    """Integration tests requiring actual services to be running."""

    @pytest.fixture(autouse=True)
    def setup_integration_environment(self):
        """Set up integration test environment."""
        try:
            config = load_config()

            postgres_url = config.postgres.get_connection_url()
            neo4j_url = config.neo4j.get_neo4j_bolt_url()

            if not postgres_url or not (config.postgres.host and config.postgres.port):
                pytest.skip("Integration test requires valid PostgreSQL configuration")

            if not config.neo4j.host or not config.neo4j.port:
                pytest.skip("Integration test requires valid Neo4j configuration")

        except Exception as e:
            pytest.skip(f"Integration test requires valid configuration: {e}")

    def test_extractor_with_real_spacy_model(self):
        """Test that the extractor can work with a real spaCy model."""
        try:
            from aclarai_shared.config import load_config

            config = load_config()
            # This will load the actual spaCy model
            extractor = NounPhraseExtractor(config)
            # Test phrase extraction with real spaCy
            test_text = (
                "Machine learning algorithms process natural language efficiently."
            )
            phrases = extractor._extract_noun_phrases(test_text)
            # Should extract some phrases
            assert isinstance(phrases, list)
            # Verify structure if phrases found
            for phrase in phrases:
                assert isinstance(phrase, str)
                assert len(phrase.strip()) > 0
        except Exception as e:
            pytest.fail(f"Real spaCy integration test failed: {e}")

    def test_extractor_with_real_databases(self):
        """Test extractor initialization with real database connections."""
        try:
            from aclarai_shared.config import load_config

            config = load_config()
            # This will attempt real database connections
            extractor = NounPhraseExtractor(config)
            # Verify components initialized
            assert extractor.neo4j_manager is not None
            assert extractor.vector_store is not None
            # Test basic operations don't crash
            result = extractor.extract_from_all_nodes()
            assert hasattr(result, "total_nodes_processed")
            assert hasattr(result, "total_phrases_extracted")
        except Exception as e:
            pytest.fail(f"Real database integration test failed: {e}")

    def test_normalization_with_real_spacy(self):
        """Test phrase normalization with real spaCy model."""
        try:
            from aclarai_shared.config import load_config

            config = load_config()
            extractor = NounPhraseExtractor(config)
            # Test various normalization cases
            test_cases = [
                ("Machine Learning Algorithms", "machine learning algorithm"),
                ("Natural Language Processing!", "natural language processing"),
                ("Deep Neural Networks.", "deep neural network"),
            ]
            for original, _expected_pattern in test_cases:
                normalized = extractor._normalize_phrase(original)
                # Normalized should be lowercase and without punctuation
                assert normalized.islower()
                assert not any(c in normalized for c in "!.,;:")
                assert len(normalized.strip()) > 0
        except Exception as e:
            pytest.fail(f"Real normalization integration test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
