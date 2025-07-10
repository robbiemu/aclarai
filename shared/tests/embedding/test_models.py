"""
Tests for embedding models module.
"""

from unittest.mock import patch

from aclarai_shared import load_config
from aclarai_shared.embedding.chunking import ChunkMetadata
from aclarai_shared.embedding.models import EmbeddedChunk, EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test cases for EmbeddingGenerator."""

    @patch(
        "aclarai_shared.embedding.models.EmbeddingGenerator._initialize_embedding_model"
    )
    def test_embedding_generator_init_with_config(self, mock_init_model):
        """
        Test that EmbeddingGenerator correctly uses the `utterance` model
        from a properly loaded configuration.
        """
        # The _initialize_embedding_model is mocked to prevent network calls.
        mock_init_model.return_value = None  # Return a dummy value

        # Use the proper config loader to get a valid config object
        config = load_config(validate=False)

        # Modify the config to set specific models for the test
        # This simulates a user's custom configuration
        config.embedding.utterance = "model-for-utterance"
        config.embedding.default_model = "different-default-model"

        # Initialize the generator with the modified, but valid, config
        generator = EmbeddingGenerator(config=config)

        # Assert that the generator picked the specific 'utterance' model,
        # not the default one. This is its correct behavior.
        assert generator.model_name == "model-for-utterance"
        # Assert that the mocked initialization was called
        mock_init_model.assert_called_once()

    def test_embedding_generator_init_with_model_override(self):
        """Test EmbeddingGenerator initialization with model override."""
        # Use the proper config loader
        config = load_config(validate=False)
        config.embedding.default_model = "sentence-transformers/all-MiniLM-L6-v2"
        custom_model = "sentence-transformers/all-mpnet-base-v2"
        generator = EmbeddingGenerator(config=config, model_name=custom_model)
        assert generator.model_name == custom_model

    def test_embed_chunks(self):
        """Test embedding multiple chunks."""
        config = load_config(validate=False)
        generator = EmbeddingGenerator(config=config)
        chunks = [
            ChunkMetadata(
                aclarai_block_id="blk_1",
                chunk_index=0,
                original_text="Original text 1",
                text="Chunk text 1",
            ),
            ChunkMetadata(
                aclarai_block_id="blk_2",
                chunk_index=0,
                original_text="Original text 2",
                text="Chunk text 2",
            ),
        ]
        embedded_chunks = generator.embed_chunks(chunks)
        assert len(embedded_chunks) == 2
        for embedded_chunk in embedded_chunks:
            assert isinstance(embedded_chunk, EmbeddedChunk)
            assert embedded_chunk.model_name == generator.model_name

    def test_embed_single_chunk(self):
        """Test embedding a single chunk."""
        config = load_config(validate=False)
        generator = EmbeddingGenerator(config=config)
        chunk = ChunkMetadata(
            aclarai_block_id="blk_single",
            chunk_index=0,
            original_text="Original single text",
            text="Single chunk text",
        )
        embedded_chunk = generator.embed_single_chunk(chunk)
        assert isinstance(embedded_chunk, EmbeddedChunk)
        assert embedded_chunk.chunk_metadata == chunk

    def test_embed_text(self):
        """Test embedding raw text."""
        config = load_config(validate=False)
        generator = EmbeddingGenerator(config=config)
        embedding = generator.embed_text("Test text for embedding")
        assert isinstance(embedding, list)
        assert all(isinstance(x, (int, float)) for x in embedding)

    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        config = load_config(validate=False)
        generator = EmbeddingGenerator(config=config)
        dim = generator.get_embedding_dimension()
        assert isinstance(dim, int)
        assert dim > 0

    def test_validate_embeddings(self):
        """Test embedding validation."""
        config = load_config(validate=False)
        generator = EmbeddingGenerator(config=config)
        # Create mock embedded chunks
        chunk_metadata = ChunkMetadata(
            aclarai_block_id="blk_test",
            chunk_index=0,
            original_text="Test original",
            text="Test text",
        )
        embedded_chunks = [
            EmbeddedChunk(
                chunk_metadata=chunk_metadata,
                embedding=[0.1, 0.2, 0.3],
                model_name="test-model",
                embedding_dim=3,
            )
        ]
        validation_report = generator.validate_embeddings(embedded_chunks)
        assert isinstance(validation_report, dict)
        assert "status" in validation_report
        assert "total_chunks" in validation_report
        assert validation_report["total_chunks"] == 1

    def test_validate_embeddings_empty(self):
        """Test validation with empty list."""
        config = load_config(validate=False)
        generator = EmbeddingGenerator(config=config)
        validation_report = generator.validate_embeddings([])
        assert validation_report["status"] == "error"
        assert "No embedded chunks" in validation_report["message"]

    def test_initialize_embedding_model(self):
        """Test embedding model initialization."""
        config = load_config(validate=False)
        generator = EmbeddingGenerator(config=config)
        model = generator._initialize_embedding_model()
        # Should be a BaseEmbedding instance
        assert model is not None

    def test_get_device(self):
        """Test device detection."""
        config = load_config(validate=False)
        config.embedding.device = "cpu"
        generator = EmbeddingGenerator(config=config)
        device = generator._get_device()
        assert isinstance(device, str)
        assert device in ["cpu", "cuda", "mps"]

    def test_get_device_auto(self):
        """Test automatic device detection."""
        config = load_config(validate=False)
        config.embedding.device = "auto"
        generator = EmbeddingGenerator(config=config)
        device = generator._get_device()
        assert isinstance(device, str)
        assert device in ["cpu", "cuda", "mps"]

    def test_embed_texts_batch(self):
        """Test batch text embedding."""
        config = load_config(validate=False)
        config.embedding.batch_size = 2
        generator = EmbeddingGenerator(config=config)
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = generator._embed_texts_batch(texts)
        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)


class TestEmbeddedChunk:
    """Test cases for EmbeddedChunk dataclass."""

    def test_embedded_chunk_creation(self):
        """Test EmbeddedChunk creation."""
        chunk_metadata = ChunkMetadata(
            aclarai_block_id="blk_test",
            chunk_index=0,
            original_text="Original text",
            text="Chunk text",
        )
        embedded_chunk = EmbeddedChunk(
            chunk_metadata=chunk_metadata,
            embedding=[0.1, 0.2, 0.3, 0.4],
            model_name="test-model",
            embedding_dim=4,
        )
        assert embedded_chunk.chunk_metadata == chunk_metadata
        assert embedded_chunk.embedding == [0.1, 0.2, 0.3, 0.4]
        assert embedded_chunk.model_name == "test-model"
        assert embedded_chunk.embedding_dim == 4

    def test_embedded_chunk_equality(self):
        """Test EmbeddedChunk equality."""
        chunk_metadata = ChunkMetadata(
            aclarai_block_id="blk_test",
            chunk_index=0,
            original_text="Original text",
            text="Chunk text",
        )
        chunk1 = EmbeddedChunk(
            chunk_metadata=chunk_metadata,
            embedding=[0.1, 0.2],
            model_name="test-model",
            embedding_dim=2,
        )
        chunk2 = EmbeddedChunk(
            chunk_metadata=chunk_metadata,
            embedding=[0.1, 0.2],
            model_name="test-model",
            embedding_dim=2,
        )
        assert chunk1 == chunk2

    def test_embedded_chunk_different_embeddings(self):
        """Test EmbeddedChunk with different embeddings."""
        chunk_metadata = ChunkMetadata(
            aclarai_block_id="blk_test",
            chunk_index=0,
            original_text="Original text",
            text="Chunk text",
        )
        chunk1 = EmbeddedChunk(
            chunk_metadata=chunk_metadata,
            embedding=[0.1, 0.2],
            model_name="test-model",
            embedding_dim=2,
        )
        chunk2 = EmbeddedChunk(
            chunk_metadata=chunk_metadata,
            embedding=[0.3, 0.4],
            model_name="test-model",
            embedding_dim=2,
        )
        assert chunk1 != chunk2
