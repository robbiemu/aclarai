"""
Integration tests for ConceptProcessor with Tier 3 file generation.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from aclarai_shared.concept_detection.models import (
    ConceptAction,
    ConceptDetectionBatch,
    ConceptDetectionResult,
)
from aclarai_shared.config import PathsConfig, aclaraiConfig
from aclarai_shared.graph.models import Concept
from aclarai_shared.tier3_concept import ConceptFileWriter


# Skip spaCy model loading for these tests
@pytest.fixture(autouse=True)
def mock_spacy_model():
    """Mock spaCy model loading to avoid network dependencies."""
    with patch("spacy.load") as mock_load:
        mock_nlp = Mock()
        mock_load.return_value = mock_nlp
        yield mock_nlp


# Skip embedding model loading for these tests
@pytest.fixture(autouse=True)
def mock_embedding_models():
    """Mock embedding model loading to avoid network dependencies."""
    with patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding") as mock_hf:
        mock_embedding = Mock()
        mock_hf.return_value = mock_embedding
        yield mock_embedding


class TestConceptProcessorTier3Integration:
    """
    Test ConceptProcessor integration with Tier 3 file creation.
    Note: These tests mock dependencies to avoid network access requirements.
    """

    def test_promoted_concepts_create_tier3_files(self):
        """Test that promoted concepts create both Neo4j nodes and Tier 3 files."""
        # Try importing ConceptProcessor
        import sys

        sys.path.append("/home/runner/work/aclarai/aclarai/services/aclarai-core")
        from aclarai_core.concept_processor import ConceptProcessor

        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup config with temp directory
            config = aclaraiConfig(
                vault_path=temp_dir, paths=PathsConfig(tier3="concepts")
            )
            # Setup mocks
            mock_store_instance = Mock()
            mock_store_instance.update_candidate_status.return_value = True
            mock_extractor_instance = Mock()
            mock_detector_instance = Mock()
            # Setup ConceptFileWriter mock
            mock_concept_file_writer_instance = Mock()
            mock_concept_file_writer_instance.write_concept_file.return_value = True
            # Mock concept detection results with a promoted concept
            promoted_result = ConceptDetectionResult(
                candidate_id="test_candidate_1",
                candidate_text="machine learning",
                action=ConceptAction.PROMOTED,
                confidence=0.95,
                reason="No similar concepts found",
            )
            detection_batch = ConceptDetectionBatch(
                results=[promoted_result],
                total_processed=1,
                merged_count=0,
                promoted_count=1,
                processing_time=0.1,
                error=None,
            )
            mock_detector_instance.process_candidates_batch.return_value = (
                detection_batch
            )
            # Mock Neo4j manager to return created concepts
            mock_neo4j_instance = Mock()
            created_concept = Concept(
                concept_id="concept_test123",
                text="machine learning",
                source_candidate_id="test_candidate_1",
                source_node_id="claim_456",
                source_node_type="claim",
                aclarai_id="doc_789",
                version=1,
                timestamp=datetime.now(timezone.utc),
            )
            mock_neo4j_instance.create_concepts.return_value = [created_concept]
            # Initialize ConceptProcessor with mocked dependencies
            processor = ConceptProcessor(
                config=config,
                noun_phrase_extractor=mock_extractor_instance,
                candidates_store=mock_store_instance,
                concept_detector=mock_detector_instance,
                concept_file_writer=mock_concept_file_writer_instance,
                neo4j_manager=mock_neo4j_instance,
            )
            # Create candidate metadata map
            candidate_metadata_map = {
                "test_candidate_1": {
                    "source_node_id": "claim_456",
                    "source_node_type": "claim",
                    "aclarai_id": "doc_789",
                    "text": "machine learning",
                }
            }
            # Call the method that updates candidate statuses
            updates = processor._update_candidate_statuses(
                detection_batch, candidate_metadata_map
            )
            # Verify Neo4j concept creation was called
            mock_neo4j_instance.create_concepts.assert_called_once()
            created_concepts_arg = mock_neo4j_instance.create_concepts.call_args[0][0]
            assert len(created_concepts_arg) == 1
            assert created_concepts_arg[0].text == "machine learning"
            # Verify ConceptFileWriter.write_concept_file was called with the created concept
            mock_concept_file_writer_instance.write_concept_file.assert_called_once()
            written_concept = (
                mock_concept_file_writer_instance.write_concept_file.call_args[0][0]
            )
            assert written_concept.concept_id == "concept_test123"
            assert written_concept.text == "machine learning"
            # Verify updates include concept_id
            assert len(updates) == 1
            assert updates[0]["concept_id"] == "concept_test123"

    def test_no_tier3_files_for_merged_concepts(self):
        """Test that merged concepts don't create Tier 3 files."""
        # Try importing ConceptProcessor
        import sys

        sys.path.append("/home/runner/work/aclarai/aclarai/services/aclarai-core")
        from aclarai_core.concept_processor import ConceptProcessor

        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup config with temp directory
            config = aclaraiConfig(
                vault_path=temp_dir, paths=PathsConfig(tier3="concepts")
            )
            # Setup mocks
            mock_store_instance = Mock()
            mock_store_instance.update_candidate_status.return_value = True
            mock_extractor_instance = Mock()
            mock_detector_instance = Mock()
            # Setup ConceptFileWriter mock
            mock_concept_file_writer_instance = Mock()
            mock_concept_file_writer_instance.write_concept_file.return_value = True
            # Mock concept detection results with a merged concept (no promotion)
            from aclarai_shared.concept_detection.models import SimilarityMatch

            best_match = SimilarityMatch(
                candidate_id="test_candidate_1",
                matched_concept_id="existing_concept",
                matched_candidate_id="existing_concept_candidate",
                similarity_score=0.95,
                matched_text="machine learning",
            )
            merged_result = ConceptDetectionResult(
                candidate_id="test_candidate_1",
                candidate_text="machine learning",
                action=ConceptAction.MERGED,
                similarity_matches=[best_match],
                confidence=0.95,
                reason="Similar concept found",
            )
            detection_batch = ConceptDetectionBatch(
                results=[merged_result],
                total_processed=1,
                merged_count=1,
                promoted_count=0,
                processing_time=0.1,
                error=None,
            )
            mock_detector_instance.process_candidates_batch.return_value = (
                detection_batch
            )
            # Mock Neo4j manager (should not be called)
            mock_neo4j_instance = Mock()
            # Initialize ConceptProcessor with mocked dependencies
            processor = ConceptProcessor(
                config=config,
                noun_phrase_extractor=mock_extractor_instance,
                candidates_store=mock_store_instance,
                concept_detector=mock_detector_instance,
                concept_file_writer=mock_concept_file_writer_instance,
                neo4j_manager=mock_neo4j_instance,
            )
            # Create candidate metadata map
            candidate_metadata_map = {
                "test_candidate_1": {
                    "source_node_id": "claim_456",
                    "source_node_type": "claim",
                    "aclarai_id": "doc_789",
                    "text": "machine learning",
                }
            }
            # Call the method that updates candidate statuses
            updates = processor._update_candidate_statuses(
                detection_batch, candidate_metadata_map
            )
            # Verify Neo4j concept creation was NOT called (no promoted concepts)
            mock_neo4j_instance.create_concepts.assert_not_called()
            # Verify ConceptFileWriter was NOT called (no promoted concepts)
            mock_concept_file_writer_instance.write_concept_file.assert_not_called()
            # Verify updates don't include concept_id for merged concepts
            assert len(updates) == 1
            assert updates[0]["concept_id"] is None
            assert updates[0]["new_status"] == "merged"


@pytest.mark.integration
class TestConceptFileWriterIntegration:
    """Additional integration tests specifically for ConceptFileWriter."""

    def test_concept_file_writer_integration(self):
        """Test that ConceptFileWriter correctly creates Tier 3 files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup config with temp directory
            config = aclaraiConfig(
                vault_path=temp_dir, paths=PathsConfig(tier3="concepts")
            )
            # Initialize ConceptFileWriter
            writer = ConceptFileWriter(config)
            # Create a test concept
            concept = Concept(
                concept_id="concept_test123",
                text="machine learning",
                source_candidate_id="test_candidate_1",
                source_node_id="claim_456",
                source_node_type="claim",
                aclarai_id="doc_789",
                version=1,
                timestamp=datetime.now(timezone.utc),
            )
            # Write the concept file
            success = writer.write_concept_file(concept)
            assert success, "ConceptFileWriter should return True for successful write"
            # Verify Tier 3 file was created
            concepts_dir = Path(temp_dir) / "concepts"
            expected_file = concepts_dir / "machine_learning.md"
            assert expected_file.exists(), (
                f"Expected file {expected_file} was not created"
            )
            # Verify file content
            content = expected_file.read_text()
            assert "## Concept: machine learning" in content
            assert "<!-- aclarai:id=concept_test123 ver=1 -->" in content
            assert "^concept_test123" in content
            assert "This concept was automatically extracted" in content

    def test_concept_file_writer_error_handling(self):
        """Test that ConceptFileWriter handles errors gracefully."""
        # Use a non-existent directory to simulate write errors
        config = aclaraiConfig(
            vault_path="/nonexistent/path", paths=PathsConfig(tier3="concepts")
        )
        writer = ConceptFileWriter(config)
        concept = Concept(
            concept_id="concept_test123",
            text="machine learning",
            source_candidate_id="test_candidate_1",
            source_node_id="claim_456",
            source_node_type="claim",
            aclarai_id="doc_789",
            version=1,
            timestamp=datetime.now(timezone.utc),
        )
        # Write should fail gracefully and return False
        success = writer.write_concept_file(concept)
        assert not success, "ConceptFileWriter should return False for failed write"

    def test_multiple_concepts_integration(self):
        """Test writing multiple concept files in sequence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = aclaraiConfig(
                vault_path=temp_dir, paths=PathsConfig(tier3="concepts")
            )
            writer = ConceptFileWriter(config)
            # Create multiple test concepts
            concepts = [
                Concept(
                    concept_id="concept_ml123",
                    text="machine learning",
                    source_candidate_id="candidate_1",
                    source_node_id="claim_456",
                    source_node_type="claim",
                    aclarai_id="doc_789",
                    version=1,
                    timestamp=datetime.now(timezone.utc),
                ),
                Concept(
                    concept_id="concept_ai456",
                    text="artificial intelligence",
                    source_candidate_id="candidate_2",
                    source_node_id="claim_789",
                    source_node_type="claim",
                    aclarai_id="doc_012",
                    version=1,
                    timestamp=datetime.now(timezone.utc),
                ),
            ]
            # Write all concepts
            for concept in concepts:
                success = writer.write_concept_file(concept)
                assert success, f"Failed to write concept file for {concept.text}"
            # Verify all files were created
            concepts_dir = Path(temp_dir) / "concepts"
            assert (concepts_dir / "machine_learning.md").exists()
            assert (concepts_dir / "artificial_intelligence.md").exists()
            # Verify each file has correct content
            ml_content = (concepts_dir / "machine_learning.md").read_text()
            assert "## Concept: machine learning" in ml_content
            assert "concept_ml123" in ml_content
            ai_content = (concepts_dir / "artificial_intelligence.md").read_text()
            assert "## Concept: artificial intelligence" in ai_content
            assert "concept_ai456" in ai_content
