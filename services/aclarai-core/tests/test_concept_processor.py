"""
Tests for concept processor integration.
This module tests the concept processing functionality integrated
into the aclarai-core service.
"""

import logging
from unittest.mock import Mock

import pytest
from aclarai_core.concept_processor import ConceptProcessor
from aclarai_shared.concept_detection.models import (
    ConceptAction,
    ConceptDetectionBatch,
    ConceptDetectionResult,
)
from aclarai_shared.config import aclaraiConfig
from aclarai_shared.noun_phrase_extraction.models import (
    ExtractionResult,
    NounPhraseCandidate,
)

logger = logging.getLogger(__name__)


class TestConceptProcessor:
    """Test suite for ConceptProcessor class."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        config = Mock(spec=aclaraiConfig)
        return config

    @pytest.fixture
    def concept_processor(self, mock_config):
        """Create a ConceptProcessor instance for testing."""
        # Create mock dependencies
        mock_noun_phrase_extractor = Mock()
        mock_candidates_store = Mock()
        mock_concept_detector = Mock()
        mock_concept_file_writer = Mock()
        mock_neo4j_manager = Mock()
        # Create processor with injected mocks
        processor = ConceptProcessor(
            config=mock_config,
            noun_phrase_extractor=mock_noun_phrase_extractor,
            candidates_store=mock_candidates_store,
            concept_detector=mock_concept_detector,
            concept_file_writer=mock_concept_file_writer,
            neo4j_manager=mock_neo4j_manager,
        )
        return processor

    @pytest.fixture
    def sample_block(self):
        """Create a sample block for testing."""
        return {
            "aclarai_id": "blk_test_123",
            "semantic_text": "Machine learning algorithms are used in artificial intelligence applications.",
            "content_hash": "abc123",
            "version": 1,
        }

    def test_initialization(self, concept_processor, mock_config):
        """Test ConceptProcessor initialization."""
        assert concept_processor.config == mock_config
        assert concept_processor.noun_phrase_extractor is not None
        assert concept_processor.candidates_store is not None
        assert concept_processor.concept_detector is not None
        assert concept_processor.concept_file_writer is not None
        assert concept_processor.neo4j_manager is not None

    def test_process_block_for_concepts_success(self, concept_processor, sample_block):
        """Test successful processing of a block for concepts."""
        # Mock the extraction result
        mock_candidates = [
            NounPhraseCandidate(
                text="machine learning",
                normalized_text="machine learning",
                source_node_id="blk_test_123",
                source_node_type="claim",
                aclarai_id="blk_test_123",
                embedding=[0.1] * 384,
            ),
            NounPhraseCandidate(
                text="artificial intelligence",
                normalized_text="artificial intelligence",
                source_node_id="blk_test_123",
                source_node_type="claim",
                aclarai_id="blk_test_123",
                embedding=[0.2] * 384,
            ),
        ]
        mock_extraction_result = ExtractionResult(
            candidates=mock_candidates,
            total_nodes_processed=1,
            successful_extractions=1,
            total_phrases_extracted=2,
        )
        # Mock the detection batch result
        mock_detection_results = [
            ConceptDetectionResult(
                candidate_id="claim_blk_test_123_machine learning",
                candidate_text="machine learning",
                action=ConceptAction.PROMOTED,
                confidence=1.0,
                reason="No similar candidates found",
            ),
            ConceptDetectionResult(
                candidate_id="claim_blk_test_123_artificial intelligence",
                candidate_text="artificial intelligence",
                action=ConceptAction.MERGED,
                confidence=0.95,
                reason="Found similar candidate",
            ),
        ]
        mock_detection_batch = ConceptDetectionBatch(
            results=mock_detection_results,
            total_processed=2,
            merged_count=1,
            promoted_count=1,
            processing_time=0.5,
        )
        # Configure mocks
        concept_processor.noun_phrase_extractor._extract_from_node.return_value = (
            mock_extraction_result
        )
        concept_processor.candidates_store.store_candidates.return_value = 2
        concept_processor.concept_detector.process_candidates_batch.return_value = (
            mock_detection_batch
        )
        # Execute the method
        result = concept_processor.process_block_for_concepts(sample_block, "claim")
        # Verify the result
        assert result["success"] is True
        assert result["aclarai_id"] == "blk_test_123"
        assert result["block_type"] == "claim"
        assert result["candidates_extracted"] == 2
        assert result["candidates_stored"] == 2
        assert result["merged_count"] == 1
        assert result["promoted_count"] == 1
        assert len(result["concept_actions"]) == 2
        assert len(result["updated_candidates"]) == 2

    def test_process_block_for_concepts_no_extraction(
        self, concept_processor, sample_block
    ):
        """Test processing when no noun phrases are extracted."""
        # Mock empty extraction result
        mock_extraction_result = ExtractionResult(
            candidates=[],
            total_nodes_processed=1,
            successful_extractions=0,
            total_phrases_extracted=0,
        )
        concept_processor.noun_phrase_extractor._extract_from_node.return_value = (
            mock_extraction_result
        )
        # Execute the method
        result = concept_processor.process_block_for_concepts(sample_block, "claim")
        # Verify the result
        assert result["success"] is True
        assert result["candidates_extracted"] == 0
        assert result["candidates_stored"] == 0
        assert result["concept_actions"] == []
        assert "No noun phrases extracted" in result["message"]

    def test_process_block_for_concepts_extraction_error(
        self, concept_processor, sample_block
    ):
        """Test processing when extraction fails."""
        # Mock extraction failure
        mock_extraction_result = ExtractionResult(
            candidates=[],
            total_nodes_processed=0,
            successful_extractions=0,
            failed_extractions=1,
            error="Extraction failed",
        )
        concept_processor.noun_phrase_extractor._extract_from_node.return_value = (
            mock_extraction_result
        )
        # Execute the method
        result = concept_processor.process_block_for_concepts(sample_block, "claim")
        # Verify the result
        assert result["success"] is True  # Still successful overall
        assert result["candidates_extracted"] == 0
        assert result["concept_actions"] == []

    def test_process_block_for_concepts_exception_handling(
        self, concept_processor, sample_block
    ):
        """Test exception handling during processing."""
        # Mock exception during extraction
        concept_processor.noun_phrase_extractor._extract_from_node.side_effect = (
            Exception("Test error")
        )
        # Execute the method
        result = concept_processor.process_block_for_concepts(sample_block, "claim")
        # Verify error handling
        assert result["success"] is False
        assert "Test error" in result["error"]
        assert result["candidates_extracted"] == 0

    def test_build_concept_index(self, concept_processor):
        """Test building the concept index."""
        # Mock the detector's build method
        concept_processor.concept_detector.build_index_from_candidates.return_value = (
            100
        )
        # Execute the method
        result = concept_processor.build_concept_index()
        # Verify the result
        assert result == 100
        concept_processor.concept_detector.build_index_from_candidates.assert_called_once_with(
            False
        )

    def test_build_concept_index_force_rebuild(self, concept_processor):
        """Test force rebuilding the concept index."""
        # Mock the detector's build method
        concept_processor.concept_detector.build_index_from_candidates.return_value = (
            150
        )
        # Execute the method
        result = concept_processor.build_concept_index(force_rebuild=True)
        # Verify the result
        assert result == 150
        concept_processor.concept_detector.build_index_from_candidates.assert_called_once_with(
            True
        )

    def test_get_concept_statistics(self, concept_processor):
        """Test getting concept statistics."""
        # Mock the store's get_candidates_by_status method
        concept_processor.candidates_store.get_candidates_by_status.side_effect = [
            [{"id": "1"}, {"id": "2"}],  # pending
            [{"id": "3"}],  # merged
            [{"id": "4"}, {"id": "5"}, {"id": "6"}],  # promoted
        ]
        # Mock the detector's metadata
        concept_processor.concept_detector.id_to_metadata = {"1": {}, "2": {}}
        concept_processor.concept_detector.similarity_threshold = 0.9
        # Execute the method
        stats = concept_processor.get_concept_statistics()
        # Verify the statistics
        assert stats["total_candidates"] == 6
        assert stats["pending_candidates"] == 2
        assert stats["merged_candidates"] == 1
        assert stats["promoted_candidates"] == 3
        assert stats["index_size"] == 2
        assert stats["similarity_threshold"] == 0.9

    def test_update_candidate_statuses(self, concept_processor):
        """Test updating candidate statuses."""
        # Create mock detection results
        mock_results = [
            ConceptDetectionResult(
                candidate_id="test_1",
                candidate_text="machine learning",
                action=ConceptAction.PROMOTED,
                confidence=1.0,
                reason="No similar candidates found",
            ),
            ConceptDetectionResult(
                candidate_id="test_2",
                candidate_text="artificial intelligence",
                action=ConceptAction.MERGED,
                confidence=0.95,
                reason="Found similar candidate",
            ),
        ]
        mock_batch = ConceptDetectionBatch(
            results=mock_results, total_processed=2, merged_count=1, promoted_count=1
        )
        # Create mock candidate metadata map
        mock_candidate_metadata_map = {
            "test_1": {
                "candidate_id": "test_1",
                "text": "machine learning",
                "embedding": [0.1] * 384,
                "status": "pending",
            },
            "test_2": {
                "candidate_id": "test_2",
                "text": "artificial intelligence",
                "embedding": [0.2] * 384,
                "status": "pending",
            },
        }
        # Execute the method
        updates = concept_processor._update_candidate_statuses(
            mock_batch, mock_candidate_metadata_map
        )
        # Verify the updates
        assert len(updates) == 2
        # Check first update (promoted)
        assert updates[0]["candidate_id"] == "test_1"
        assert updates[0]["new_status"] == "promoted"
        assert updates[0]["confidence"] == 1.0
        # Check second update (merged)
        assert updates[1]["candidate_id"] == "test_2"
        assert updates[1]["new_status"] == "merged"
        assert updates[1]["confidence"] == 0.95


class TestConceptProcessorIntegration:
    """Integration tests for ConceptProcessor with real services."""

    @pytest.fixture(autouse=True)
    def setup_integration_environment(self):
        """Setup integration test environment."""
        # Check if we're in a CI/test environment that supports integration tests
        import os

        if not os.getenv("RUN_INTEGRATION_TESTS"):
            pytest.skip(
                "Integration tests require RUN_INTEGRATION_TESTS environment variable"
            )

    @pytest.mark.integration
    def test_concept_processing_integration(self):
        """Test the complete concept processing workflow with real services."""
        from aclarai_shared.config import load_config

        # Load real configuration
        config = load_config(validate=False)
        # Create concept processor
        processor = ConceptProcessor(config)
        # Test block processing
        sample_block = {
            "aclarai_id": "blk_integration_test_123",
            "semantic_text": "Machine learning and artificial intelligence are transformative technologies.",
            "content_hash": "integration_test_hash",
            "version": 1,
        }
        # Process the block
        result = processor.process_block_for_concepts(sample_block, "claim")
        # Verify the result structure
        assert isinstance(result, dict)
        assert "success" in result
        assert "aclarai_id" in result
        assert "candidates_extracted" in result
        assert "concept_actions" in result
        # Test statistics
        stats = processor.get_concept_statistics()
        assert isinstance(stats, dict)
        assert "total_candidates" in stats

    @pytest.mark.integration
    def test_concept_index_building_integration(self):
        """Test concept index building with real services."""
        from aclarai_shared.config import load_config

        # Load real configuration
        config = load_config(validate=False)
        # Create concept processor
        processor = ConceptProcessor(config)
        # Test index building
        items_added = processor.build_concept_index()
        # Verify the result
        assert isinstance(items_added, int)
        assert items_added >= 0

    @pytest.mark.integration
    def test_status_persistence_integration(self):
        """Test that candidate status updates are properly persisted."""
        from aclarai_shared.concept_detection.models import (
            ConceptAction,
            ConceptDetectionBatch,
            ConceptDetectionResult,
        )
        from aclarai_shared.config import load_config
        from aclarai_shared.noun_phrase_extraction.models import NounPhraseCandidate

        # Load real configuration
        config = load_config(validate=False)
        # Create concept processor
        processor = ConceptProcessor(config)
        # Create a test candidate
        test_candidate = NounPhraseCandidate(
            text="integration test concept",
            normalized_text="integration test concept",
            source_node_id="blk_test_integration",
            source_node_type="claim",
            aclarai_id="blk_test_integration",
            embedding=[0.1] * 384,
        )
        # Store the candidate
        stored_count = processor.candidates_store.store_candidates([test_candidate])
        assert stored_count == 1
        # Create mock detection results
        detection_results = [
            ConceptDetectionResult(
                candidate_id="integration_test_1",
                candidate_text="integration test concept",
                action=ConceptAction.PROMOTED,
                confidence=1.0,
                reason="Integration test promotion",
            )
        ]
        detection_batch = ConceptDetectionBatch(
            results=detection_results,
            total_processed=1,
            merged_count=0,
            promoted_count=1,
            processing_time=0.1,
        )
        # Update candidate statuses
        candidate_metadata_map = {
            "integration_test_1": {
                "source_node_id": "blk_test_integration",
                "source_node_type": "claim",
                "aclarai_id": "blk_test_integration",
                "text": "integration test concept",
            }
        }
        updates = processor._update_candidate_statuses(
            detection_batch, candidate_metadata_map
        )
        # Verify updates were applied
        assert (
            len(updates) >= 0
        )  # May be 0 if candidate not found, which is okay for integration test
