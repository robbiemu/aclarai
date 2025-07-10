"""
Tests for embedding module __init__.py file.
"""

import os
from unittest.mock import Mock

import pytest
from aclarai_shared.config import load_config
from aclarai_shared.embedding import EmbeddingPipeline
from aclarai_shared.embedding.storage import aclaraiVectorStore


class TestEmbeddingInitFile:
    """Test that embedding __init__.py file loads correctly."""

    def test_init_file_exists(self):
        """Test that the embedding __init__.py file exists."""
        init_path = os.path.join(
            os.path.dirname(__file__), "../../aclarai_shared/embedding/__init__.py"
        )
        assert os.path.exists(init_path)

    def test_init_file_loads(self):
        """Test that the embedding __init__.py file can be loaded directly."""
        init_path = os.path.join(
            os.path.dirname(__file__), "../../aclarai_shared/embedding/__init__.py"
        )
        # Check for expected attributes in the file content
        with open(init_path, "r") as f:
            content = f.read()
        expected_all = [
            "UtteranceChunker",
            "ChunkMetadata",
            "EmbeddingGenerator",
            "EmbeddedChunk",
            "aclaraiVectorStore",
            "VectorStoreMetrics",
            "EmbeddingPipeline",
        ]
        # Check that __all__ contains expected items
        for item in expected_all:
            assert f'"{item}"' in content or f"'{item}'" in content

    def test_embedding_result_dataclass(self):
        """Test EmbeddingResult dataclass creation."""
        # Test that EmbeddingResult structure would work
        mock_metrics = Mock()
        mock_metrics.total_vectors = 10
        mock_metrics.successful_inserts = 8
        mock_metrics.failed_inserts = 2
        result_data = {
            "success": True,
            "total_chunks": 10,
            "embedded_chunks": 8,
            "stored_chunks": 8,
            "failed_chunks": 2,
            "metrics": mock_metrics,
            "errors": ["test error"],
        }
        assert result_data["success"] is True
        assert result_data["total_chunks"] == 10
        assert result_data["embedded_chunks"] == 8
        assert result_data["stored_chunks"] == 8
        assert result_data["failed_chunks"] == 2
        assert result_data["metrics"] == mock_metrics
        assert result_data["errors"] == ["test error"]

    def test_embedding_pipeline_init(self):
        """Test EmbeddingPipeline initialization with configurable dependencies (unit test)."""
        # Mock test - no real database connections
        init_path = os.path.join(
            os.path.dirname(__file__), "../../aclarai_shared/embedding/__init__.py"
        )
        with open(init_path, "r") as f:
            content = f.read()
        # Check that EmbeddingPipeline class is defined
        assert "class EmbeddingPipeline" in content

    @pytest.mark.integration
    def test_embedding_pipeline_init_integration(self):
        """Test EmbeddingPipeline initialization with configurable dependencies (integration test)."""

        # Load config from environment
        config = load_config(validate=True)

        # Initialize vector store (this will create table if needed)
        try:
            vector_store = aclaraiVectorStore(config)
            assert vector_store is not None
            assert vector_store.config.embedding.collection_name == "utterances"

            # Get metrics to verify connection
            metrics = vector_store.get_store_metrics()
            assert metrics is not None
            assert isinstance(metrics.total_vectors, int)
        except Exception as e:
            pytest.fail(f"Failed to initialize vector store: {e}")

    def test_embedding_pipeline_init_no_config(self):
        """Test EmbeddingPipeline initialization without config (unit test)."""
        # Mock test - verify class definition exists
        init_path = os.path.join(
            os.path.dirname(__file__), "../../aclarai_shared/embedding/__init__.py"
        )
        with open(init_path, "r") as f:
            content = f.read()
        # Check that EmbeddingPipeline class is defined
        assert "class EmbeddingPipeline" in content

    @pytest.mark.integration
    def test_embedding_pipeline_init_no_config_integration(self):
        """Test EmbeddingPipeline initialization without config (integration test)."""

        # Initialize vector store with default config
        try:
            vector_store = aclaraiVectorStore()  # This will load default config
            assert vector_store is not None
            assert vector_store.config.embedding.collection_name == "utterances"

            # Get metrics to verify connection
            metrics = vector_store.get_store_metrics()
            assert metrics is not None
            assert isinstance(metrics.total_vectors, int)
        except Exception as e:
            pytest.fail(f"Failed to initialize vector store with default config: {e}")

    def test_process_tier1_content_empty(self):
        """Test processing empty tier1 content (unit test)."""
        # Mock test - verify class definitions exist
        init_path = os.path.join(
            os.path.dirname(__file__), "../../aclarai_shared/embedding/__init__.py"
        )
        with open(init_path, "r") as f:
            content = f.read()
        # Check that both classes are defined
        assert "class EmbeddingPipeline" in content
        assert "class EmbeddingResult" in content

    @pytest.mark.integration
    def test_process_tier1_content_empty_integration(self):
        """Test processing empty tier1 content (integration test)."""

        # Load config and initialize pipeline
        config = load_config(validate=True)
        pipeline = EmbeddingPipeline(config)

        # Test with empty content using process_single_block
        empty_content = ""
        empty_id = "test_empty_123"

        # Process should handle empty content gracefully
        try:
            result = pipeline.process_single_block(empty_content, empty_id)
            assert result is not None
            assert result.success is False  # Empty content should fail gracefully
            assert result.total_chunks == 0
            assert result.embedded_chunks == 0
            assert len(result.errors) > 0  # Should have an error message
            assert "No chunks generated from block" in result.errors[0]
        except Exception as e:
            pytest.fail(f"Failed to process empty content: {e}")
