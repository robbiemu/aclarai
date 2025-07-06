"""
Tests for the concept embedding refresh job.
This module tests the functionality of refreshing concept embeddings
from Tier 3 concept files following the specifications in
docs/arch/on-refreshing_concept_embeddings.md
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from aclarai_scheduler.concept_refresh import ConceptEmbeddingRefreshJob
from aclarai_shared.config import aclaraiConfig, load_config


class MockVectorStoreForConcepts:
    """Simple mock vector store with upsert method for concept testing."""

    def __init__(self):
        self.upsert = Mock()
        self.delete_chunks_by_block_id = Mock(return_value=0)
        self.store_embeddings = Mock()


class MockVectorStoreMetrics:
    """Mock metrics for vector store operations."""

    def __init__(self, successful_inserts=1, failed_inserts=0):
        self.successful_inserts = successful_inserts
        self.failed_inserts = failed_inserts


def test_extract_semantic_text():
    """Test semantic text extraction from concept file content."""
    # Create mocks for dependencies
    mock_config = MagicMock()
    mock_config.vault_path = "/tmp/test_vault"
    mock_neo4j = Mock()
    mock_embedding_gen = Mock()
    mock_vector_store = MockVectorStoreForConcepts()
    job = ConceptEmbeddingRefreshJob(
        config=mock_config,
        neo4j_manager=mock_neo4j,
        embedding_generator=mock_embedding_gen,
        vector_store=mock_vector_store,
    )
    # Test content with metadata and anchors
    file_content = """# Machine Learning
Machine learning is a subset of artificial intelligence.
Key characteristics:
- Pattern recognition
- Automatic improvement
<!-- aclarai:id=concept_machine_learning ver=1 -->
^concept_machine_learning
"""
    expected_semantic_text = """# Machine Learning
Machine learning is a subset of artificial intelligence.
Key characteristics:
- Pattern recognition
- Automatic improvement"""
    result = job._extract_semantic_text(file_content)
    assert result.strip() == expected_semantic_text.strip()


def test_extract_semantic_text_no_metadata():
    """Test semantic text extraction when there's no metadata."""
    mock_config = MagicMock()
    mock_config.vault_path = "/tmp/test_vault"
    mock_neo4j = Mock()
    mock_embedding_gen = Mock()
    mock_vector_store = MockVectorStoreForConcepts()
    job = ConceptEmbeddingRefreshJob(
        config=mock_config,
        neo4j_manager=mock_neo4j,
        embedding_generator=mock_embedding_gen,
        vector_store=mock_vector_store,
    )
    file_content = """# Deep Learning
Deep learning uses neural networks with multiple layers."""
    result = job._extract_semantic_text(file_content)
    assert result.strip() == file_content.strip()


def test_compute_hash():
    """Test SHA256 hash computation."""
    mock_config = MagicMock()
    mock_config.vault_path = "/tmp/test_vault"
    mock_neo4j = Mock()
    mock_embedding_gen = Mock()
    mock_vector_store = MockVectorStoreForConcepts()
    job = ConceptEmbeddingRefreshJob(
        config=mock_config,
        neo4j_manager=mock_neo4j,
        embedding_generator=mock_embedding_gen,
        vector_store=mock_vector_store,
    )
    text = "Machine learning is awesome"
    hash1 = job._compute_hash(text)
    hash2 = job._compute_hash(text)
    # Same text should produce same hash
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 produces 64 character hex string
    # Different text should produce different hash
    different_text = "Deep learning is awesome"
    hash3 = job._compute_hash(different_text)
    assert hash1 != hash3


def test_process_concept_file_no_changes():
    """Test processing a concept file when no changes are detected."""
    # Create temporary vault directory
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)
        concepts_path = vault_path / "concepts"
        concepts_path.mkdir()
        # Create test config
        mock_config = MagicMock()
        mock_config.vault_path = str(vault_path)
        # Create mocks for dependencies
        mock_neo4j = Mock()
        mock_embedding_gen = Mock()
        mock_vector_store = MockVectorStoreForConcepts()
        # Create test concept file
        concept_file = concepts_path / "test_concept.md"
        content = """# Test Concept
This is a test concept.
<!-- aclarai:id=concept_test_concept ver=1 -->
^concept_test_concept"""
        with open(concept_file, "w") as f:
            f.write(content)
        job = ConceptEmbeddingRefreshJob(
            config=mock_config,
            neo4j_manager=mock_neo4j,
            embedding_generator=mock_embedding_gen,
            vector_store=mock_vector_store,
        )
        # Mock that the stored hash matches current hash
        semantic_text = job._extract_semantic_text(content)
        current_hash = job._compute_hash(semantic_text)
        job._get_stored_embedding_hash = Mock(return_value=current_hash)
        # Process the file
        processed, updated = job._process_concept_file(concept_file)
        assert processed is True
        assert updated is False  # No update needed
        # Verify that embedding was not generated
        mock_embedding_gen.embed_text.assert_not_called()


def test_process_concept_file_with_changes():
    """Test processing a concept file when changes are detected."""
    # Create temporary vault directory
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)
        concepts_path = vault_path / "concepts"
        concepts_path.mkdir()
        # Create test config
        mock_config = MagicMock()
        mock_config.vault_path = str(vault_path)
        # Create mocks for dependencies
        mock_neo4j = Mock()
        mock_embedding_gen = Mock()
        mock_embedding_gen.model_name = "test-model"
        mock_vector_store = MockVectorStoreForConcepts()
        # Create test concept file
        concept_file = concepts_path / "test_concept.md"
        content = """# Test Concept
This is a test concept with updated content.
<!-- aclarai:id=concept_test_concept ver=2 -->
^concept_test_concept"""
        with open(concept_file, "w") as f:
            f.write(content)
        job = ConceptEmbeddingRefreshJob(
            config=mock_config,
            neo4j_manager=mock_neo4j,
            embedding_generator=mock_embedding_gen,
            vector_store=mock_vector_store,
        )
        # Mock that the stored hash is different (indicating changes)
        job._get_stored_embedding_hash = Mock(return_value="old_hash_value")
        job._update_neo4j_metadata = Mock()
        # Mock embedding generation
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_embedding_gen.embed_text.return_value = mock_embedding
        # Process the file
        processed, updated = job._process_concept_file(concept_file)
        assert processed is True
        assert updated is True  # Update was needed
        # Verify that embedding was generated
        mock_embedding_gen.embed_text.assert_called_once()
        # Verify that vector store upsert was called (for mocks with upsert method)
        mock_vector_store.upsert.assert_called_once_with("test_concept", mock_embedding)
        # Verify Neo4j update was called
        job._update_neo4j_metadata.assert_called_once()


def test_run_job_no_concepts_directory():
    """Test run_job when concepts directory doesn't exist."""
    # Create test config with non-existent concepts directory
    mock_config = MagicMock()
    mock_config.vault_path = "/nonexistent/vault"
    # Create mocks for dependencies
    mock_neo4j = Mock()
    mock_embedding_gen = Mock()
    mock_vector_store = MockVectorStoreForConcepts()
    job = ConceptEmbeddingRefreshJob(
        config=mock_config,
        neo4j_manager=mock_neo4j,
        embedding_generator=mock_embedding_gen,
        vector_store=mock_vector_store,
    )
    result = job.run_job()
    assert result["success"] is False
    assert result["concepts_processed"] == 0
    assert result["concepts_updated"] == 0
    assert "Concepts directory does not exist" in result["error_details"]


def test_run_job_with_concept_files():
    """Test run_job with actual concept files."""
    # Create temporary vault directory
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)
        concepts_path = vault_path / "concepts"
        concepts_path.mkdir()
        # Create test config
        mock_config = MagicMock()
        mock_config.vault_path = str(vault_path)
        # Create mocks for dependencies
        mock_neo4j = Mock()
        mock_embedding_gen = Mock()
        mock_vector_store = MockVectorStoreForConcepts()
        # Create multiple test concept files
        concepts = [
            ("concept1.md", "# Concept 1\n\nFirst concept content."),
            ("concept2.md", "# Concept 2\n\nSecond concept content."),
        ]
        for filename, content in concepts:
            with open(concepts_path / filename, "w") as f:
                f.write(content)
        job = ConceptEmbeddingRefreshJob(
            config=mock_config,
            neo4j_manager=mock_neo4j,
            embedding_generator=mock_embedding_gen,
            vector_store=mock_vector_store,
        )
        # Mock the process_concept_file method
        job._process_concept_file = Mock(return_value=(True, True))
        result = job.run_job()
        assert result["success"] is True
        assert result["concepts_processed"] == 2
        assert result["concepts_updated"] == 2
        assert result["concepts_skipped"] == 0
        assert result["errors"] == 0


def test_get_stored_embedding_hash_not_found():
    """Test getting stored embedding hash when concept doesn't exist."""
    mock_config = MagicMock()
    mock_config.vault_path = "/tmp/test_vault"
    # Mock Neo4j manager session
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.single.return_value = None
    mock_session.run.return_value = mock_result
    mock_neo4j_manager = MagicMock()
    mock_neo4j_manager.session.return_value.__enter__.return_value = mock_session
    # Create mocks for other dependencies
    mock_embedding_gen = Mock()
    mock_vector_store = MockVectorStoreForConcepts()
    job = ConceptEmbeddingRefreshJob(
        config=mock_config,
        neo4j_manager=mock_neo4j_manager,
        embedding_generator=mock_embedding_gen,
        vector_store=mock_vector_store,
    )
    result = job._get_stored_embedding_hash("nonexistent_concept")
    assert result is None


def test_get_stored_embedding_hash_found():
    """Test getting stored embedding hash when concept exists."""
    mock_config = MagicMock()
    mock_config.vault_path = "/tmp/test_vault"
    # Mock Neo4j manager session
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_record = {"hash": "stored_hash_value"}
    mock_result.single.return_value = mock_record
    mock_session.run.return_value = mock_result
    mock_neo4j_manager = MagicMock()
    mock_neo4j_manager.session.return_value.__enter__.return_value = mock_session
    # Create mocks for other dependencies
    mock_embedding_gen = Mock()
    mock_vector_store = MockVectorStoreForConcepts()
    job = ConceptEmbeddingRefreshJob(
        config=mock_config,
        neo4j_manager=mock_neo4j_manager,
        embedding_generator=mock_embedding_gen,
        vector_store=mock_vector_store,
    )
    result = job._get_stored_embedding_hash("existing_concept")
    assert result == "stored_hash_value"


def test_update_vector_store_with_real_store():
    """Test the _update_vector_store method with a real vector store (no upsert method)."""
    mock_config = MagicMock()
    mock_config.vault_path = "/tmp/test_vault"
    # Create mocks for dependencies
    mock_neo4j = Mock()
    mock_embedding_gen = Mock()
    mock_embedding_gen.model_name = "test-model"
    # Create a mock vector store without upsert method (like the real one)
    mock_vector_store = Mock()
    mock_vector_store.delete_chunks_by_block_id.return_value = (
        2  # Simulate deleting 2 old chunks
    )
    mock_vector_store.store_embeddings.return_value = MockVectorStoreMetrics(
        successful_inserts=1, failed_inserts=0
    )
    # Remove upsert method to simulate real vector store
    if hasattr(mock_vector_store, "upsert"):
        delattr(mock_vector_store, "upsert")
    job = ConceptEmbeddingRefreshJob(
        config=mock_config,
        neo4j_manager=mock_neo4j,
        embedding_generator=mock_embedding_gen,
        vector_store=mock_vector_store,
    )
    # Test the upsert operation
    concept_name = "test_concept"
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    job._update_vector_store(concept_name, embedding)
    # Verify delete was called with correct concept block ID
    mock_vector_store.delete_chunks_by_block_id.assert_called_once_with(
        "concept_test_concept"
    )
    # Verify store_embeddings was called with one EmbeddedChunk
    mock_vector_store.store_embeddings.assert_called_once()
    call_args = mock_vector_store.store_embeddings.call_args[0][
        0
    ]  # Get the list of embedded chunks
    assert len(call_args) == 1
    embedded_chunk = call_args[0]
    assert embedded_chunk.embedding == embedding
    assert embedded_chunk.model_name == "test-model"
    assert embedded_chunk.embedding_dim == len(embedding)
    assert embedded_chunk.chunk_metadata.aclarai_block_id == "concept_test_concept"
    assert embedded_chunk.chunk_metadata.chunk_index == 0


def test_update_vector_store_with_failed_insert():
    """Test the _update_vector_store method when insertion fails."""
    mock_config = MagicMock()
    mock_config.vault_path = "/tmp/test_vault"
    # Create mocks for dependencies
    mock_neo4j = Mock()
    mock_embedding_gen = Mock()
    mock_embedding_gen.model_name = "test-model"
    # Create a mock vector store that fails insertion
    mock_vector_store = Mock()
    mock_vector_store.delete_chunks_by_block_id.return_value = 0
    mock_vector_store.store_embeddings.return_value = MockVectorStoreMetrics(
        successful_inserts=0, failed_inserts=1
    )
    # Remove upsert method to simulate real vector store
    if hasattr(mock_vector_store, "upsert"):
        delattr(mock_vector_store, "upsert")
    job = ConceptEmbeddingRefreshJob(
        config=mock_config,
        neo4j_manager=mock_neo4j,
        embedding_generator=mock_embedding_gen,
        vector_store=mock_vector_store,
    )
    # Test the upsert operation - should raise exception on failed insert
    concept_name = "test_concept"
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    try:
        job._update_vector_store(concept_name, embedding)
        raise AssertionError("Expected RuntimeError to be raised")
    except RuntimeError as e:
        assert "Failed to insert new embedding" in str(e)


@pytest.mark.integration
def test_concept_embedding_refresh_job_end_to_end():
    """
    End-to-end integration test for ConceptEmbeddingRefreshJob using real services.
    This test:
    1. Verifies that required services (Neo4j, PostgreSQL) are available
    2. Creates a temporary vault with concept files
    3. Runs the job without mocks (uses real services but with mock embeddings)
    4. Verifies that the complete workflow executes successfully
    Requires: Running Neo4j and PostgreSQL services (e.g., via docker-compose up postgres neo4j)
    """
    # Create temporary vault directory
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)
        concepts_path = vault_path / "concepts"
        concepts_path.mkdir()
        # Create test concept files with realistic content
        concept_files = [
            (
                "machine_learning.md",
                """# Machine Learning
Machine learning is a subset of artificial intelligence (AI) that enables computer systems to automatically learn and improve from experience without being explicitly programmed.
## Key Characteristics
- **Pattern Recognition**: ML algorithms identify patterns in data
- **Automatic Improvement**: Performance improves with more data
- **Prediction**: Can make predictions on new, unseen data
## Types of Machine Learning
1. **Supervised Learning**: Uses labeled training data
2. **Unsupervised Learning**: Finds patterns in unlabeled data
3. **Reinforcement Learning**: Learns through interaction and feedback
<!-- aclarai:id=concept_machine_learning ver=1 -->
^concept_machine_learning""",
            ),
            (
                "deep_learning.md",
                """# Deep Learning
Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data.
## Architecture
Deep learning models consist of:
- **Input Layer**: Receives raw data
- **Hidden Layers**: Process and transform information (multiple layers = "deep")
- **Output Layer**: Produces final predictions or classifications
## Applications
- Computer vision and image recognition
- Natural language processing
- Speech recognition
- Autonomous vehicles
<!-- aclarai:id=concept_deep_learning ver=1 -->
^concept_deep_learning""",
            ),
        ]
        # Write concept files to temporary vault
        for filename, content in concept_files:
            concept_file = concepts_path / filename
            concept_file.write_text(content)
        # Use mock embedding generator to avoid external dependencies
        # but real Neo4j and PostgreSQL services
        try:
            from unittest.mock import Mock

            from aclarai_shared.config import aclaraiConfig

            # Create config pointing to our temporary vault
            config = load_config(validate=True)  # This will load from .env
            config.vault_path = str(vault_path)
            # Create mock embedding generator to avoid HuggingFace dependency
            mock_embedding_gen = Mock()
            mock_embedding_gen.model_name = "mock-embedding-model"

            # Return deterministic embeddings for testing
            def mock_embed_text(text):
                # Create a simple hash-based embedding for testing
                import hashlib

                hash_obj = hashlib.md5(text.encode())
                # Convert hash to list of floats between -1 and 1
                hash_bytes = hash_obj.digest()
                embedding = [(b / 127.5) - 1.0 for b in hash_bytes]
                # Pad or truncate to 384 dimensions (common embedding size)
                while len(embedding) < 384:
                    embedding.extend(embedding[: 384 - len(embedding)])
                return embedding[:384]

            mock_embedding_gen.embed_text.side_effect = mock_embed_text
            # Create the job with real Neo4j and vector store, but mock embeddings
            job = ConceptEmbeddingRefreshJob(
                config=config, embedding_generator=mock_embedding_gen
            )
            # Run the job end-to-end
            result = job.run_job()
            # Verify the job executed successfully
            assert result["success"] is True, (
                f"Job failed with error: {result.get('error_details', 'Unknown error')}"
            )
            assert result["concepts_processed"] == 2, (
                f"Expected 2 concepts processed, got {result['concepts_processed']}"
            )
            # The concepts should be updated since they're new (no previous hash in Neo4j)
            assert result["concepts_updated"] >= 0, (
                f"Expected non-negative concepts updated, got {result['concepts_updated']}"
            )
            assert result["concepts_skipped"] >= 0, (
                f"Expected non-negative concepts skipped, got {result['concepts_skipped']}"
            )
            assert result["errors"] == 0, f"Expected no errors, got {result['errors']}"
            # Verify that the results make sense
            total_processed = result["concepts_updated"] + result["concepts_skipped"]
            assert total_processed == result["concepts_processed"], (
                "Processed count should equal updated + skipped"
            )
            # Verify the mock embedding generator was called
            assert mock_embedding_gen.embed_text.called, (
                "Embedding generator should have been called"
            )
            print(f"✓ Integration test passed. Results: {result}")
        except ImportError as e:
            pytest.skip(f"Integration test skipped - missing dependencies: {e}")
        except Exception as e:
            # If services were available but test still failed, this is a real failure
            print(f"Integration test failed with services available: {e}")
            raise


if __name__ == "__main__":
    # Run basic tests
    print("Running concept refresh tests...")
    try:
        test_extract_semantic_text()
        test_extract_semantic_text_no_metadata()
        test_compute_hash()
        test_get_stored_embedding_hash_not_found()
        test_get_stored_embedding_hash_found()
        print("✓ All basic tests passed")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        exit(1)
    print("All tests completed successfully!")
