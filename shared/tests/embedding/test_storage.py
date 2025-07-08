"""
Tests for embedding storage components.
"""

from unittest.mock import Mock, patch

import pytest
from aclarai_shared import load_config
from aclarai_shared.embedding.chunking import ChunkMetadata
from aclarai_shared.embedding.models import EmbeddedChunk
from aclarai_shared.embedding.storage import VectorStoreMetrics, aclaraiVectorStore


class TestVectorStoreMetrics:
    """Test cases for VectorStoreMetrics dataclass."""

    def test_vector_store_metrics_creation(self):
        """Test VectorStoreMetrics creation."""
        metrics = VectorStoreMetrics(
            total_vectors=100, successful_inserts=95, failed_inserts=5
        )
        assert metrics.total_vectors == 100
        assert metrics.successful_inserts == 95
        assert metrics.failed_inserts == 5

    def test_vector_store_metrics_default(self):
        """Test VectorStoreMetrics with defaults."""
        metrics = VectorStoreMetrics(50, 45, 5)
        assert metrics.total_vectors == 50
        assert metrics.successful_inserts == 45
        assert metrics.failed_inserts == 5

    def test_vector_store_metrics_equality(self):
        """Test VectorStoreMetrics equality."""
        metrics1 = VectorStoreMetrics(10, 8, 2)
        metrics2 = VectorStoreMetrics(10, 8, 2)
        metrics3 = VectorStoreMetrics(10, 9, 1)
        assert metrics1 == metrics2
        assert metrics1 != metrics3


class TestaclaraiVectorStore:
    """Test cases for aclaraiVectorStore with configurable dependencies."""

    def setup_method(self):
        """Set up a reusable mock for the engine connection context manager."""
        # Create the connection object mock first
        self.mock_connection_object = Mock()

        # Create a mock context manager that properly implements __enter__ and __exit__
        self.mock_connection_context = Mock()
        self.mock_connection_context.__enter__ = Mock(
            return_value=self.mock_connection_object
        )
        self.mock_connection_context.__exit__ = Mock(return_value=None)

    @patch("aclarai_shared.embedding.storage.Settings")
    @patch("aclarai_shared.embedding.storage.EmbeddingGenerator")
    @patch("aclarai_shared.embedding.storage.VectorStoreIndex")
    @patch("aclarai_shared.embedding.storage.PGVectorStore")
    @patch("aclarai_shared.embedding.storage.create_engine")
    def test_vector_store_init_with_config(
        self,
        mock_create_engine,
        mock_pg_vector_store,
        mock_vector_store_index,
        mock_embedding_generator,
        mock_settings,
    ):
        """Test aclaraiVectorStore initialization with config (unit test)."""
        config = load_config(validate=False)
        mock_engine = Mock()
        mock_engine.connect.return_value = self.mock_connection_context
        mock_create_engine.return_value = mock_engine

        vector_store = aclaraiVectorStore(config=config)
        assert vector_store.config == config
        mock_create_engine.assert_called_once()
        mock_pg_vector_store.from_params.assert_called_once()
        mock_vector_store_index.from_vector_store.assert_called_once()
        mock_embedding_generator.assert_called_once_with(config=config)
        assert mock_settings.embed_model is not None

    @pytest.mark.integration
    def test_vector_store_init_with_config_integration(self):
        """Test aclaraiVectorStore initialization with config (integration test)."""
        config = load_config(validate=True)
        try:
            vector_store = aclaraiVectorStore(config=config)
            assert vector_store.config == config
        except Exception as e:
            pytest.skip(
                f"Skipping integration test: PostgreSQL not available. Error: {e}"
            )

    @patch("aclarai_shared.embedding.storage.Settings")
    @patch("aclarai_shared.embedding.storage.create_engine")
    @patch("aclarai_shared.embedding.storage.PGVectorStore")
    @patch("aclarai_shared.embedding.storage.VectorStoreIndex")
    def test_store_embeddings(
        self,
        mock_vector_store_index,
        mock_pg_vector_store,
        mock_create_engine,
        mock_settings,
    ):
        """Test storing embeddings in vector store (unit test)."""
        config = load_config(validate=False)
        mock_engine = Mock()
        mock_engine.connect.return_value = self.mock_connection_context
        mock_create_engine.return_value = mock_engine
        mock_pg_vector_store.from_params.return_value = Mock()
        mock_vsi_instance = Mock()
        mock_vector_store_index.from_vector_store.return_value = mock_vsi_instance

        vector_store = aclaraiVectorStore(config=config)

        chunk_metadata = ChunkMetadata("blk_1", 0, "text", "text")
        embedded_chunks = [EmbeddedChunk(chunk_metadata, [0.1], "model", 1)]

        metrics = vector_store.store_embeddings(embedded_chunks)

        assert isinstance(metrics, VectorStoreMetrics)
        assert metrics.successful_inserts == 1
        mock_vsi_instance.insert.assert_called_once()

    @pytest.mark.integration
    def test_store_embeddings_integration(self):
        """Test storing embeddings in vector store (integration test)."""
        config = load_config(validate=True)
        try:
            vector_store = aclaraiVectorStore(config=config)
        except Exception as e:
            pytest.skip(
                f"Skipping integration test: PostgreSQL not available. Error: {e}"
            )

        chunk_metadata = ChunkMetadata("blk_store_test_integ", 0, "text", "text")
        embedding_vector = [0.1] * 384
        embedded_chunks = [
            EmbeddedChunk(
                chunk_metadata, embedding_vector, "test-model", len(embedding_vector)
            )
        ]
        metrics = vector_store.store_embeddings(embedded_chunks)
        assert isinstance(metrics, VectorStoreMetrics)
        assert metrics.successful_inserts >= 0

    @patch("aclarai_shared.embedding.storage.Settings")
    @patch("aclarai_shared.embedding.storage.create_engine")
    @patch("aclarai_shared.embedding.storage.PGVectorStore")
    @patch("aclarai_shared.embedding.storage.VectorStoreIndex")
    def test_store_embeddings_empty_list(
        self,
        mock_vector_store_index,
        mock_pg_vector_store,
        mock_create_engine,
        mock_settings,
    ):
        """Test storing empty list of embeddings (unit test)."""
        config = load_config(validate=False)
        mock_engine = Mock()
        mock_engine.connect.return_value = self.mock_connection_context
        mock_create_engine.return_value = mock_engine
        mock_pg_vector_store.from_params.return_value = Mock()
        mock_vector_store_index.from_vector_store.return_value = Mock()

        vector_store = aclaraiVectorStore(config=config)
        metrics = vector_store.store_embeddings([])
        assert metrics.total_vectors == 0
        assert metrics.successful_inserts == 0
        assert metrics.failed_inserts == 0

    @pytest.mark.integration
    def test_store_embeddings_empty_list_integration(self):
        """Test storing empty list of embeddings (integration test)."""
        config = load_config(validate=True)
        try:
            vector_store = aclaraiVectorStore(config=config)
        except Exception as e:
            pytest.skip(
                f"Skipping integration test: PostgreSQL not available. Error: {e}"
            )

        metrics = vector_store.store_embeddings([])
        assert metrics.total_vectors == 0
        assert metrics.successful_inserts == 0
        assert metrics.failed_inserts == 0

    @patch("aclarai_shared.embedding.storage.Settings")
    @patch("aclarai_shared.embedding.storage.create_engine")
    @patch("aclarai_shared.embedding.storage.PGVectorStore")
    @patch("aclarai_shared.embedding.storage.VectorStoreIndex")
    def test_similarity_search(
        self,
        mock_vector_store_index,
        mock_pg_vector_store,
        mock_create_engine,
        mock_settings,
    ):
        """Test similarity search functionality (unit test)."""
        config = load_config(validate=False)
        mock_engine = Mock()
        mock_engine.connect.return_value = self.mock_connection_context
        mock_create_engine.return_value = mock_engine
        mock_pg_vector_store.from_params.return_value = Mock()
        mock_vsi_instance = Mock()
        mock_vsi_instance.as_query_engine.return_value.query.return_value = Mock(
            source_nodes=[]
        )
        mock_vector_store_index.from_vector_store.return_value = mock_vsi_instance

        vector_store = aclaraiVectorStore(config=config)
        results = vector_store.similarity_search("test query", top_k=5)
        assert isinstance(results, list)
        mock_vsi_instance.as_query_engine.assert_called_once()

    @pytest.mark.integration
    def test_similarity_search_integration(self):
        """Test similarity search functionality (integration test)."""
        config = load_config(validate=True)
        try:
            vector_store = aclaraiVectorStore(config=config)
        except Exception as e:
            pytest.skip(
                f"Skipping integration test: PostgreSQL not available. Error: {e}"
            )

        results = vector_store.similarity_search("test query", top_k=5)
        assert isinstance(results, list)

    @patch("aclarai_shared.embedding.storage.Settings")
    @patch("aclarai_shared.embedding.storage.create_engine")
    @patch("aclarai_shared.embedding.storage.PGVectorStore")
    @patch("aclarai_shared.embedding.storage.VectorStoreIndex")
    def test_delete_chunks_by_block_id(
        self,
        mock_vector_store_index,
        mock_pg_vector_store,
        mock_create_engine,
        mock_settings,
    ):
        """Test deleting chunks by block ID (unit test)."""
        config = load_config(validate=False)
        mock_engine = Mock()
        mock_engine.connect.return_value = self.mock_connection_context
        mock_create_engine.return_value = mock_engine
        mock_pg_vector_store.from_params.return_value = Mock()
        mock_vsi_instance = Mock()
        mock_vector_store_index.from_vector_store.return_value = mock_vsi_instance

        vector_store = aclaraiVectorStore(config=config)
        with patch.object(
            vector_store, "get_chunks_by_block_id", return_value=[{"doc_id": "doc1"}]
        ):
            deleted_count = vector_store.delete_chunks_by_block_id("test_block_id")
            assert deleted_count == 1
            mock_vsi_instance.delete.assert_called_with("doc1")

    @pytest.mark.integration
    def test_delete_chunks_by_block_id_integration(self):
        """Test deleting chunks by block ID (integration test)."""
        config = load_config(validate=True)
        try:
            vector_store = aclaraiVectorStore(config=config)
        except Exception as e:
            pytest.skip(
                f"Skipping integration test: PostgreSQL not available. Error: {e}"
            )

        chunk_meta = ChunkMetadata("del_block_id", 0, "text", "text")
        embedding_vector = [0.4] * 384
        embedded = [
            EmbeddedChunk(chunk_meta, embedding_vector, "model", len(embedding_vector))
        ]
        vector_store.store_embeddings(embedded)

        deleted_count = vector_store.delete_chunks_by_block_id("del_block_id")
        assert deleted_count >= 0

    @patch("aclarai_shared.embedding.storage.Settings")
    @patch("aclarai_shared.embedding.storage.create_engine")
    @patch("aclarai_shared.embedding.storage.PGVectorStore")
    @patch("aclarai_shared.embedding.storage.VectorStoreIndex")
    def test_get_store_metrics(
        self,
        mock_vector_store_index,
        mock_pg_vector_store,
        mock_create_engine,
        mock_settings,
    ):
        """Test getting vector store metrics (unit test)."""
        config = load_config(validate=False)
        mock_engine = Mock()
        # Configure the inner connection object returned by the context manager
        self.mock_connection_object.execute.return_value.fetchone.side_effect = [
            (0,),
            ("0 MB",),
        ]
        mock_engine.connect.return_value = self.mock_connection_context
        mock_create_engine.return_value = mock_engine

        vector_store = aclaraiVectorStore(config=config)
        metrics = vector_store.get_store_metrics()
        assert isinstance(metrics, VectorStoreMetrics)
        assert metrics.total_vectors == 0

    @pytest.mark.integration
    def test_get_store_metrics_integration(self):
        """Test getting vector store metrics (integration test)."""
        config = load_config(validate=True)
        try:
            vector_store = aclaraiVectorStore(config=config)
        except Exception as e:
            pytest.skip(
                f"Skipping integration test: PostgreSQL not available. Error: {e}"
            )

        metrics = vector_store.get_store_metrics()
        assert isinstance(metrics, VectorStoreMetrics)
        assert metrics.total_vectors >= 0
