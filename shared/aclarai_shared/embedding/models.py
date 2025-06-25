"""
Embedding module for aclarai utterance chunks.
This module handles the generation of embeddings for utterance chunks using
configurable models via LlamaIndex. Supports multiple providers including
OpenAI, HuggingFace, and SentenceTransformers.
Key Features:
- Configurable embedding models from settings/aclarai.config.yaml
- Multiple provider support (OpenAI, HuggingFace, SentenceTransformers)
- Batch processing for efficiency
- Automatic device detection (CPU/GPU)
- Integration with LlamaIndex embedding abstractions
- Structured logging with service context
For detailed usage and configuration, see docs/guides/embedding_models_guide.md
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ..config import aclaraiConfig
from .chunking import ChunkMetadata

logger = logging.getLogger(__name__)


@dataclass
class EmbeddedChunk:
    """Chunk with its embedding vector and metadata."""

    chunk_metadata: ChunkMetadata
    embedding: List[float]
    model_name: str
    embedding_dim: int


class EmbeddingGenerator:
    """
    Generator for utterance chunk embeddings using configurable models.
    This class handles embedding generation following the architecture patterns
    from docs/arch/idea-sketch_how_to_use_a_BERT_model_with_llamaindex.md
    """

    def __init__(
        self, config: Optional[aclaraiConfig] = None, model_name: Optional[str] = None
    ):
        """
        Initialize the embedding generator.
        Args:
            config: aclarai configuration (loads default if None)
            model_name: Override model name (uses config default if None)
        """
        if config is None:
            from ..config import load_config

            config = load_config(validate=False)
        self.config = config
        self.model_name = model_name or config.embedding.default_model
        # Initialize embedding model
        self.embedding_model = self._initialize_embedding_model()
        logger.info(
            f"Initialized EmbeddingGenerator with model: {self.model_name}, "
            f"device: {config.embedding.device}, "
            f"batch_size: {config.embedding.batch_size}"
        )

    def embed_chunks(self, chunks: List[ChunkMetadata]) -> List[EmbeddedChunk]:
        """
        Generate embeddings for a list of chunks.
        Args:
            chunks: List of ChunkMetadata objects to embed
        Returns:
            List of EmbeddedChunk objects with embeddings
        """
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return []
        logger.info(
            f"Generating embeddings for {len(chunks)} chunks using {self.model_name}"
        )
        # Extract texts for batch embedding
        texts = [chunk.text for chunk in chunks]
        try:
            # Generate embeddings in batches
            embeddings = self._embed_texts_batch(texts)
            # Create EmbeddedChunk objects
            embedded_chunks = []
            for chunk, embedding in zip(chunks, embeddings, strict=False):
                embedded_chunk = EmbeddedChunk(
                    chunk_metadata=chunk,
                    embedding=embedding,
                    model_name=self.model_name,
                    embedding_dim=len(embedding),
                )
                embedded_chunks.append(embedded_chunk)
            logger.info(f"Successfully generated {len(embedded_chunks)} embeddings")
            return embedded_chunks
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def embed_single_chunk(self, chunk: ChunkMetadata) -> Optional[EmbeddedChunk]:
        """
        Generate embedding for a single chunk.
        Args:
            chunk: ChunkMetadata object to embed
        Returns:
            EmbeddedChunk object with embedding, or None if no embedding is generated
        """
        embedded_chunks = self.embed_chunks([chunk])
        return embedded_chunks[0] if embedded_chunks else None

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for raw text.
        Args:
            text: Text to embed
        Returns:
            Embedding vector as list of floats
        """
        try:
            embedding = self.embedding_model.get_text_embedding(text)
            logger.debug(f"Generated embedding for text (dim: {len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        Returns:
            Embedding dimension
        """
        # Test with a simple phrase to get dimension
        try:
            test_embedding = self.embed_text("test")
            return len(test_embedding)
        except Exception:
            # Fallback to configured dimension
            return self.config.embedding.embed_dim

    def _initialize_embedding_model(self) -> BaseEmbedding:
        """
        Initialize the embedding model based on configuration.
        Returns:
            Initialized embedding model
        """
        try:
            # Determine device
            device = self._get_device()
            # Initialize HuggingFace embedding model
            embedding_model = HuggingFaceEmbedding(
                model_name=self.model_name,
                device=device,
                max_length=512,  # Standard max length for most sentence transformers
            )
            logger.info(
                f"Successfully initialized embedding model: {self.model_name} on {device}"
            )
            return embedding_model
        except Exception as e:
            logger.error(f"Failed to initialize embedding model {self.model_name}: {e}")
            raise

    def _get_device(self) -> str:
        """
        Determine the appropriate device for the embedding model.
        Returns:
            Device string ("cpu", "cuda", "mps")
        """
        device_config = self.config.embedding.device
        if device_config == "auto":
            # Auto-detect best available device
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    device = "mps"  # Apple Silicon
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"
        else:
            device = device_config
        logger.info(f"Using device: {device}")
        return device

    def _embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching.
        Args:
            texts: List of texts to embed
        Returns:
            List of embedding vectors
        """
        batch_size = self.config.embedding.batch_size
        embeddings = []
        # Process in batches to manage memory
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            logger.debug(
                f"Processing embedding batch {i // batch_size + 1}: {len(batch_texts)} texts"
            )
            try:
                # Use LlamaIndex's embedding method
                batch_embeddings = []
                for text in batch_texts:
                    embedding = self.embedding_model.get_text_embedding(text)
                    batch_embeddings.append(embedding)
                embeddings.extend(batch_embeddings)
                logger.debug(f"Completed batch {i // batch_size + 1}")
            except Exception as e:
                logger.error(
                    f"Failed to process embedding batch {i // batch_size + 1}: {e}"
                )
                raise
        return embeddings

    def validate_embeddings(
        self, embedded_chunks: List[EmbeddedChunk]
    ) -> Dict[str, Any]:
        """
        Validate the quality and consistency of generated embeddings.
        Args:
            embedded_chunks: List of embedded chunks to validate
        Returns:
            Validation report dictionary
        """
        if not embedded_chunks:
            return {"status": "error", "message": "No embedded chunks to validate"}
        # Check embedding dimensions
        dimensions = [chunk.embedding_dim for chunk in embedded_chunks]
        expected_dim = self.config.embedding.embed_dim
        validation_report = {
            "status": "success",
            "total_chunks": len(embedded_chunks),
            "model_name": self.model_name,
            "expected_dimension": expected_dim,
            "actual_dimensions": {
                "min": min(dimensions),
                "max": max(dimensions),
                "unique": list(set(dimensions)),
            },
            "dimension_consistent": len(set(dimensions)) == 1,
            "dimension_matches_config": all(dim == expected_dim for dim in dimensions),
        }
        # Check for invalid embeddings (NaN, inf)
        invalid_embeddings = 0
        for chunk in embedded_chunks:
            if any(
                not (
                    isinstance(x, (int, float))
                    and x != float("inf")
                    and x != float("-inf")
                    and x == x
                )
                for x in chunk.embedding
            ):
                invalid_embeddings += 1
        validation_report["invalid_embeddings"] = invalid_embeddings
        validation_report["all_embeddings_valid"] = invalid_embeddings == 0
        if invalid_embeddings > 0 or not validation_report["dimension_consistent"]:
            validation_report["status"] = "warning"
        logger.info(
            f"Embedding validation: {validation_report['status']} - "
            f"{validation_report['total_chunks']} chunks, "
            f"{invalid_embeddings} invalid embeddings"
        )
        return validation_report
