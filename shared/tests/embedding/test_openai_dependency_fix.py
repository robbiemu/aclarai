"""
Test to verify the OpenAI embedding dependency fix.
This test ensures that aclaraiVectorStore can be created without
requiring the llama-index-embeddings-openai package.
"""

from unittest.mock import MagicMock, patch

from aclarai_shared.config import DatabaseConfig, aclaraiConfig
from aclarai_shared.embedding.chunking import ChunkMetadata
from aclarai_shared.embedding.models import EmbeddedChunk
from aclarai_shared.embedding.storage import VectorStoreMetrics, aclaraiVectorStore
from llama_index.core.embeddings import BaseEmbedding


class TestOpenAIDependencyFix:
    """Test that aclaraiVectorStore doesn't require OpenAI embeddings package."""

    def test_vector_store_creation_without_openai_package(self):
        """Test creating aclaraiVectorStore without OpenAI dependency."""
        # Create test config
        config = aclaraiConfig()
        config.database = DatabaseConfig(
            host="localhost",
            port=5432,
            user="test_user",
            password="test_pass",
            database="test_db",
        )
        config.embedding_model = "test-model"
        # Mock external dependencies to focus on the OpenAI import issue
        with (
            patch("aclarai_shared.embedding.storage.create_engine") as mock_engine,
            patch("aclarai_shared.embedding.storage.PGVectorStore") as mock_pgvector,
            patch(
                "aclarai_shared.embedding.models.HuggingFaceEmbedding"
            ) as mock_hf_embedding,
            patch(
                "aclarai_shared.embedding.models.EmbeddingGenerator._initialize_embedding_model"
            ) as mock_init_model,
        ):
            # Setup mocks
            mock_engine.return_value = MagicMock()
            mock_pgvector.from_params.return_value = MagicMock()

            # Create a proper mock embedding model that extends BaseEmbedding
            class MockEmbedding(BaseEmbedding):
                def _get_query_embedding(self, _query: str):
                    return [0.1, 0.2, 0.3]

                def _get_text_embedding(self, _text: str):
                    return [0.1, 0.2, 0.3]

                async def _aget_query_embedding(self, _query: str):
                    return [0.1, 0.2, 0.3]

                async def _aget_text_embedding(self, _text: str):
                    return [0.1, 0.2, 0.3]

            mock_embedding_instance = MockEmbedding()
            mock_init_model.return_value = mock_embedding_instance
            mock_hf_embedding.return_value = mock_embedding_instance
            # Mock the database connection methods
            mock_connection = MagicMock()
            mock_engine.return_value.connect.return_value.__enter__.return_value = (
                mock_connection
            )
            # This should NOT raise ImportError about llama-index-embeddings-openai
            vector_store = aclaraiVectorStore(config=config)
            # Verify the vector store was created successfully
            assert vector_store is not None
            assert vector_store.embedding_generator is not None
            assert vector_store.vector_index is not None

    def test_store_embeddings_without_openai_package(self):
        """Test the store_embeddings method that was mentioned in the issue."""
        # Create test config exactly as in the failing test
        config = aclaraiConfig()
        config.database = DatabaseConfig(
            host="localhost",
            port=5432,
            user="test_user",
            password="test_pass",
            database="test_db",
        )
        config.embedding_model = "test-model"
        # Create test embedded chunks as in the failing test
        chunk_metadata = ChunkMetadata(
            aclarai_block_id="blk_store_test",
            chunk_index=0,
            original_text="Test original text",
            text="Test chunk text",
        )
        embedded_chunks = [
            EmbeddedChunk(
                chunk_metadata=chunk_metadata,
                embedding=[0.1, 0.2, 0.3],
                model_name="test-model",
                embedding_dim=3,
            )
        ]
        # Mock all external dependencies
        with (
            patch("aclarai_shared.embedding.storage.create_engine") as mock_engine,
            patch("aclarai_shared.embedding.storage.PGVectorStore") as mock_pgvector,
            patch(
                "aclarai_shared.embedding.models.HuggingFaceEmbedding"
            ) as mock_hf_embedding,
            patch(
                "aclarai_shared.embedding.models.EmbeddingGenerator._initialize_embedding_model"
            ) as mock_init_model,
        ):
            # Setup mocks
            mock_engine.return_value = MagicMock()
            mock_pgvector.from_params.return_value = MagicMock()

            # Create a proper mock embedding model
            class MockEmbedding(BaseEmbedding):
                def _get_query_embedding(self, _query: str):
                    return [0.1, 0.2, 0.3]

                def _get_text_embedding(self, _text: str):
                    return [0.1, 0.2, 0.3]

                async def _aget_query_embedding(self, _query: str):
                    return [0.1, 0.2, 0.3]

                async def _aget_text_embedding(self, _text: str):
                    return [0.1, 0.2, 0.3]

            mock_embedding_instance = MockEmbedding()
            mock_init_model.return_value = mock_embedding_instance
            mock_hf_embedding.return_value = mock_embedding_instance
            # Mock the database connection methods
            mock_connection = MagicMock()
            mock_engine.return_value.connect.return_value.__enter__.return_value = (
                mock_connection
            )
            # Mock the vector index insert method
            mock_vector_index = MagicMock()
            # This should NOT raise ImportError about llama-index-embeddings-openai
            vector_store = aclaraiVectorStore(config=config)
            vector_store.vector_index = mock_vector_index
            # Test the store_embeddings method that was failing
            metrics = vector_store.store_embeddings(embedded_chunks)
            # Verify the operation completed successfully
            assert isinstance(metrics, VectorStoreMetrics)
            assert metrics.total_vectors == 1
