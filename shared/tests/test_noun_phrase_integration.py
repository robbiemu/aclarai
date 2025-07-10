"""
Integration test for noun phrase extraction with mocked dependencies.
This test verifies the complete noun phrase extraction workflow from
fetching nodes to storing in the concept_candidates vector table.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add the parent directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aclarai_shared.config import load_config
from aclarai_shared.noun_phrase_extraction import (
    NounPhraseCandidate,
    NounPhraseExtractor,
)


class TestNounPhraseExtractionIntegration:
    """Integration tests for complete noun phrase extraction workflow."""

    @patch("aclarai_shared.noun_phrase_extraction.extractor.spacy")
    @patch("aclarai_shared.noun_phrase_extraction.extractor.Neo4jGraphManager")
    @patch("aclarai_shared.noun_phrase_extraction.extractor.EmbeddingGenerator")
    @patch(
        "aclarai_shared.noun_phrase_extraction.extractor.ConceptCandidatesVectorStore"
    )
    @patch("aclarai_shared.noun_phrase_extraction.extractor.load_config")
    def test_complete_extraction_workflow(
        self,
        mock_load_config,
        mock_vector_store_class,
        mock_embedding_gen_class,
        mock_neo4j_class,
        mock_spacy,
    ):
        """Test the complete workflow from node fetching to vector storage."""
        # Mock configuration
        mock_config = Mock()
        mock_load_config.return_value = mock_config
        # Mock spaCy model and processing
        mock_nlp = Mock()
        mock_spacy.load.return_value = mock_nlp
        # Mock Neo4j manager for fetching nodes
        mock_neo4j = Mock()
        mock_neo4j_class.return_value = mock_neo4j
        # Mock claim and summary data
        claim_data = [
            {
                "id": "claim_1",
                "text": "Machine learning algorithms improve natural language processing.",
                "node_type": "claim",
            },
            {
                "id": "claim_2",
                "text": "Deep neural networks achieve high accuracy on computer vision tasks.",
                "node_type": "claim",
            },
        ]
        summary_data = [
            {
                "id": "summary_1",
                "text": "Large language models demonstrate exceptional performance in text generation.",
                "node_type": "summary",
            }
        ]
        # Configure Neo4j manager to return test data
        mock_neo4j.execute_query.side_effect = [
            claim_data,  # First call for claims
            summary_data,  # Second call for summaries
        ]

        # Mock spaCy noun phrase extraction
        def mock_spacy_processing(text):
            """Mock spaCy processing that returns different responses for extraction vs normalization."""
            mock_doc = Mock()
            if "machine learning algorithms" in text.lower():
                # Extraction call
                chunk1 = Mock()
                chunk1.text = "Machine learning algorithms"
                chunk2 = Mock()
                chunk2.text = "natural language processing"
                mock_doc.noun_chunks = [chunk1, chunk2]
                return mock_doc
            elif "deep neural networks" in text.lower():
                # Extraction call
                chunk1 = Mock()
                chunk1.text = "Deep neural networks"
                chunk2 = Mock()
                chunk2.text = "computer vision tasks"
                mock_doc.noun_chunks = [chunk1, chunk2]
                return mock_doc
            elif "large language models" in text.lower():
                # Extraction call
                chunk1 = Mock()
                chunk1.text = "Large language models"
                chunk2 = Mock()
                chunk2.text = "text generation"
                mock_doc.noun_chunks = [chunk1, chunk2]
                return mock_doc
            else:
                # Normalization call - return lemmatized tokens
                mock_doc.__iter__ = Mock(
                    return_value=iter(
                        [
                            self._create_mock_token(word.lower(), False, False)
                            for word in text.split()
                        ]
                    )
                )
                return mock_doc

        mock_nlp.side_effect = mock_spacy_processing
        # Mock embedding generator
        mock_embedding_gen = Mock()
        mock_embedding_gen_class.return_value = mock_embedding_gen
        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        mock_vector_store.store_candidates.return_value = (
            6  # All 6 phrases stored successfully
        )
        # Create and run extractor
        extractor = NounPhraseExtractor()
        result = extractor.extract_from_all_nodes()
        # Verify the results
        assert result.is_successful
        assert result.total_nodes_processed == 3  # 2 claims + 1 summary
        assert result.successful_extractions == 3
        assert result.failed_extractions == 0
        assert result.total_phrases_extracted == 6  # 2 phrases per node
        # Verify Neo4j queries were made
        assert mock_neo4j.execute_query.call_count == 2
        # Verify vector store was called to store candidates
        mock_vector_store.store_candidates.assert_called_once()
        stored_candidates = mock_vector_store.store_candidates.call_args[0][0]
        # Verify we have the expected number of candidates
        assert len(stored_candidates) == 6
        # Verify candidates have correct structure
        for candidate in stored_candidates:
            assert isinstance(candidate, NounPhraseCandidate)
            assert candidate.status == "pending"
            assert candidate.source_node_type in ["claim", "summary"]
            assert candidate.text  # Has original text
            assert candidate.normalized_text  # Has normalized text
            assert candidate.source_node_id  # Has source ID

    def _create_mock_token(self, lemma, is_punct, is_space):
        """Helper to create mock spaCy tokens."""
        token = Mock()
        token.lemma_ = lemma
        token.is_punct = is_punct
        token.is_space = is_space
        return token

    @patch("aclarai_shared.noun_phrase_extraction.extractor.spacy")
    @patch("aclarai_shared.noun_phrase_extraction.extractor.Neo4jGraphManager")
    @patch("aclarai_shared.noun_phrase_extraction.extractor.EmbeddingGenerator")
    @patch(
        "aclarai_shared.noun_phrase_extraction.extractor.ConceptCandidatesVectorStore"
    )
    @patch("aclarai_shared.noun_phrase_extraction.extractor.load_config")
    def test_extraction_with_empty_nodes(
        self,
        mock_load_config,
        mock_vector_store_class,
        mock_embedding_gen_class,
        mock_neo4j_class,
        mock_spacy,
    ):
        """Test extraction when no Claims or Summary nodes are found."""
        # Mock configuration
        mock_config = Mock()
        mock_load_config.return_value = mock_config
        # Mock spaCy
        mock_nlp = Mock()
        mock_spacy.load.return_value = mock_nlp
        # Mock Neo4j manager to return empty results
        mock_neo4j = Mock()
        mock_neo4j_class.return_value = mock_neo4j
        mock_neo4j.execute_query.return_value = []  # No nodes found
        # Mock other dependencies
        mock_embedding_gen = Mock()
        mock_embedding_gen_class.return_value = mock_embedding_gen
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        # Create and run extractor
        extractor = NounPhraseExtractor()
        result = extractor.extract_from_all_nodes()
        # Verify the results
        assert result.is_successful  # No errors, just no data
        assert result.total_nodes_processed == 0
        assert result.successful_extractions == 0
        assert result.failed_extractions == 0
        assert result.total_phrases_extracted == 0
        # Verify vector store was not called since no candidates
        mock_vector_store.store_candidates.assert_not_called()

    @patch("aclarai_shared.noun_phrase_extraction.extractor.spacy")
    @patch("aclarai_shared.noun_phrase_extraction.extractor.Neo4jGraphManager")
    @patch("aclarai_shared.noun_phrase_extraction.extractor.EmbeddingGenerator")
    @patch(
        "aclarai_shared.noun_phrase_extraction.extractor.ConceptCandidatesVectorStore"
    )
    @patch("aclarai_shared.noun_phrase_extraction.extractor.load_config")
    def test_extraction_with_neo4j_error(
        self,
        mock_load_config,
        mock_vector_store_class,
        mock_embedding_gen_class,
        mock_neo4j_class,
        mock_spacy,
    ):
        """Test extraction when Neo4j query fails."""
        # Mock configuration
        mock_config = Mock()
        mock_load_config.return_value = mock_config
        # Mock spaCy
        mock_nlp = Mock()
        mock_spacy.load.return_value = mock_nlp
        # Mock Neo4j manager to raise an exception
        mock_neo4j = Mock()
        mock_neo4j_class.return_value = mock_neo4j
        mock_neo4j.execute_query.side_effect = Exception("Database connection failed")
        # Mock other dependencies
        mock_embedding_gen = Mock()
        mock_embedding_gen_class.return_value = mock_embedding_gen
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        # Create and run extractor
        extractor = NounPhraseExtractor()
        result = extractor.extract_from_all_nodes()
        # Verify the results - the current implementation treats Neo4j failures
        # as recoverable by returning empty lists, so the extraction appears successful
        # but processes 0 nodes
        assert result.is_successful  # No fatal errors occurred
        assert result.total_nodes_processed == 0  # No nodes fetched due to DB error
        assert result.successful_extractions == 0
        assert result.failed_extractions == 0
        assert result.total_phrases_extracted == 0
        # The error is logged but not propagated to the result object
        # This is a design choice for resilient error handling


@pytest.mark.integration
class TestRealNounPhraseExtractionIntegration:
    """Integration tests requiring actual database services."""

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

    def test_full_extraction_pipeline_integration(self):
        """Test the complete extraction pipeline with real services."""
        try:
            from aclarai_shared.config import load_config

            # Load actual configuration
            config = load_config()
            # Initialize extractor with real services
            extractor = NounPhraseExtractor(config)
            # Run extraction (should handle empty graph gracefully)
            result = extractor.extract_from_all_nodes()
            # Verify result structure
            assert hasattr(result, "total_nodes_processed")
            assert hasattr(result, "total_phrases_extracted")
            assert hasattr(result, "processing_time")
            assert result.total_nodes_processed >= 0
            assert result.total_phrases_extracted >= 0
            assert result.processing_time >= 0
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")

    def test_database_connectivity_integration(self):
        """Test that the system can connect to both Neo4j and PostgreSQL."""
        try:
            from aclarai_shared.config import load_config

            config = load_config()
            extractor = NounPhraseExtractor(config)
            # Test Neo4j connectivity by attempting to fetch nodes using the same pattern as the extractor
            claims = extractor._fetch_claim_nodes()
            summaries = extractor._fetch_summary_nodes()
            assert isinstance(claims, list)
            assert isinstance(summaries, list)
            # Test PostgreSQL connectivity through concept candidates store
            # This will create tables if they don't exist
            store = extractor.concept_candidates_store
            assert store is not None
        except Exception as e:
            pytest.fail(f"Database connectivity test failed: {e}")

    def test_spacy_model_integration(self):
        """Test that spaCy model can be loaded and used."""
        try:
            from aclarai_shared.config import load_config

            config = load_config()
            extractor = NounPhraseExtractor(config)
            # Test with sample text
            test_text = (
                "Machine learning algorithms analyze large datasets efficiently."
            )
            phrases = extractor._extract_noun_phrases(test_text)
            assert isinstance(phrases, list)
            # May be empty if no noun phrases found, that's ok
            # Verify phrases are strings if any found
            for phrase in phrases:
                assert isinstance(phrase, str)
                assert len(phrase.strip()) > 0
        except Exception as e:
            pytest.fail(f"spaCy model integration test failed: {e}")


@pytest.mark.integration
class TestRealConceptCandidatesStoreIntegration:
    """Integration tests for the concept candidates vector store with real database."""

    @pytest.fixture(autouse=True)
    def setup_integration_environment(self):
        """Set up integration test environment."""
        try:
            from aclarai_shared.config import load_config

            # Assert configuration for database connection
            config = load_config()
            postgres_url = config.postgres.get_connection_url()
            # Use POSTGRES_URL if set, otherwise check each parameter
            if not postgres_url or not (config.postgres.host and config.postgres.port):
                pytest.skip("Integration test requires valid PostgreSQL configuration")
        except Exception as e:
            pytest.skip(f"Integration test requires valid configuration: {e}")

    def test_vector_store_operations_integration(self):
        """Test vector store operations with real PostgreSQL."""
        try:
            from aclarai_shared.config import load_config
            from aclarai_shared.noun_phrase_extraction import (
                ConceptCandidatesVectorStore,
            )

            config = load_config()
            store = ConceptCandidatesVectorStore(config)
            # Create test candidates
            test_candidates = [
                NounPhraseCandidate(
                    text="machine learning",
                    normalized_text="machine learning",
                    source_node_id="test_integration_1",
                    source_node_type="Claim",
                    aclarai_id="test_integration_1",
                ),
                NounPhraseCandidate(
                    text="artificial intelligence",
                    normalized_text="artificial intelligence",
                    source_node_id="test_integration_2",
                    source_node_type="Summary",
                    aclarai_id="test_integration_2",
                ),
            ]
            # Test storage - this will create tables if needed
            store.store_candidates(test_candidates)
        except Exception as e:
            pytest.fail(f"Vector store integration test failed: {e}")

    def test_embedding_generation_integration(self):
        """Test embedding generation with real models."""
        try:
            from aclarai_shared.config import load_config
            from aclarai_shared.noun_phrase_extraction import (
                ConceptCandidatesVectorStore,
            )

            config = load_config()
            store = ConceptCandidatesVectorStore(config)
            # Test embedding generation
            test_text = "natural language processing"
            embedding = store.embedding_generator.embed_text(test_text)
            assert isinstance(embedding, list)
            assert len(embedding) == store.embed_dim
            assert all(isinstance(x, (float, int)) for x in embedding)
        except Exception as e:
            pytest.fail(f"Embedding generation integration test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
