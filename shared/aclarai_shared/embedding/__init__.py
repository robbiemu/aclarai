"""
aclarai Embedding Module
This module provides the main interface for utterance chunk embedding functionality.
It integrates chunking, embedding generation, and vector storage following the
architecture from docs/arch/on-vector_stores.md
Key Components:
- UtteranceChunker: Segments Tier 1 blocks using LlamaIndex SentenceSplitter
- EmbeddingGenerator: Creates embeddings using configurable models
- aclaraiVectorStore: Stores vectors in PostgreSQL with pgvector
- EmbeddingPipeline: Orchestrates the complete embedding workflow
Usage:
    from aclarai_shared.embedding import EmbeddingPipeline
    pipeline = EmbeddingPipeline()
    result = pipeline.process_tier1_content(markdown_content)
"""

from .chunking import ChunkMetadata, UtteranceChunker
from .models import EmbeddedChunk, EmbeddingGenerator
from .storage import VectorStoreMetrics, aclaraiVectorStore

__all__ = [
    "UtteranceChunker",
    "ChunkMetadata",
    "EmbeddingGenerator",
    "EmbeddedChunk",
    "aclaraiVectorStore",
    "VectorStoreMetrics",
    "EmbeddingPipeline",
]
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import aclaraiConfig

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of the embedding pipeline process."""

    success: bool
    total_chunks: int
    embedded_chunks: int
    stored_chunks: int
    failed_chunks: int
    metrics: VectorStoreMetrics
    errors: List[str]


class EmbeddingPipeline:
    """
    Complete pipeline for processing Tier 1 content through embedding storage.
    This class orchestrates the full embedding workflow:
    1. Parse and chunk Tier 1 Markdown content
    2. Generate embeddings for each chunk
    3. Store vectors in PostgreSQL with metadata
    4. Provide metrics and error reporting
    """

    def __init__(self, config: Optional[aclaraiConfig] = None):
        """
        Initialize the embedding pipeline.
        Args:
            config: aclarai configuration (loads default if None)
        """
        if config is None:
            from ..config import load_config

            config = load_config(validate=False)
        self.config = config
        # Initialize components
        self.chunker = UtteranceChunker(config)
        self.embedding_generator = EmbeddingGenerator(config)
        self.vector_store = aclaraiVectorStore(config)
        logger.info("Initialized EmbeddingPipeline with all components")

    def process_tier1_content(self, tier1_content: str) -> EmbeddingResult:
        """
        Process complete Tier 1 Markdown content through the embedding pipeline.
        Args:
            tier1_content: Raw Tier 1 Markdown content with aclarai:id blocks
        Returns:
            EmbeddingResult with processing metrics and status
        """
        logger.info("Starting embedding pipeline for Tier 1 content")
        errors: List[str] = []
        try:
            # Step 1: Chunk the content
            logger.debug("Step 1: Chunking Tier 1 content")
            chunks = self.chunker.chunk_tier1_blocks(tier1_content)
            if not chunks:
                logger.warning("No chunks generated from Tier 1 content")
                return EmbeddingResult(
                    success=False,
                    total_chunks=0,
                    embedded_chunks=0,
                    stored_chunks=0,
                    failed_chunks=0,
                    metrics=VectorStoreMetrics(0, 0, 0),
                    errors=["No chunks generated from input content"],
                )
            # Step 2: Generate embeddings
            logger.debug(f"Step 2: Generating embeddings for {len(chunks)} chunks")
            embedded_chunks = self.embedding_generator.embed_chunks(chunks)
            if not embedded_chunks:
                error_msg = "Failed to generate embeddings for chunks"
                logger.error(error_msg)
                errors.append(error_msg)
                return EmbeddingResult(
                    success=False,
                    total_chunks=len(chunks),
                    embedded_chunks=0,
                    stored_chunks=0,
                    failed_chunks=len(chunks),
                    metrics=VectorStoreMetrics(len(chunks), 0, len(chunks)),
                    errors=errors,
                )
            # Step 3: Validate embeddings
            logger.debug("Step 3: Validating embeddings")
            validation_report = self.embedding_generator.validate_embeddings(
                embedded_chunks
            )
            if validation_report["status"] == "warning":
                errors.append(f"Embedding validation warnings: {validation_report}")
            # Step 4: Store in vector database
            logger.debug(
                f"Step 4: Storing {len(embedded_chunks)} embeddings in vector store"
            )
            storage_metrics = self.vector_store.store_embeddings(embedded_chunks)
            # Calculate final results
            success = (
                storage_metrics.successful_inserts > 0
                and storage_metrics.failed_inserts == 0
            )
            result = EmbeddingResult(
                success=success,
                total_chunks=len(chunks),
                embedded_chunks=len(embedded_chunks),
                stored_chunks=storage_metrics.successful_inserts,
                failed_chunks=storage_metrics.failed_inserts,
                metrics=storage_metrics,
                errors=errors,
            )
            logger.info(
                f"Embedding pipeline completed: success={success}, "
                f"chunks={len(chunks)}, stored={storage_metrics.successful_inserts}"
            )
            return result
        except Exception as e:
            error_msg = f"Embedding pipeline failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            return EmbeddingResult(
                success=False,
                total_chunks=0,
                embedded_chunks=0,
                stored_chunks=0,
                failed_chunks=0,
                metrics=VectorStoreMetrics(0, 0, 0),
                errors=errors,
            )

    def process_single_block(
        self, text: str, aclarai_block_id: str, replace_existing: bool = True
    ) -> EmbeddingResult:
        """
        Process a single utterance block through the embedding pipeline.
        Args:
            text: The utterance text to process
            aclarai_block_id: The aclarai:id of the source block
            replace_existing: Whether to replace existing chunks for this block
        Returns:
            EmbeddingResult with processing metrics and status
        """
        logger.info(f"Processing single block: {aclarai_block_id}")
        errors: List[str] = []
        try:
            # Remove existing chunks if requested
            if replace_existing:
                deleted_count = self.vector_store.delete_chunks_by_block_id(
                    aclarai_block_id
                )
                if deleted_count > 0:
                    logger.debug(
                        f"Removed {deleted_count} existing chunks for block {aclarai_block_id}"
                    )
            # Step 1: Chunk the block
            chunks = self.chunker.chunk_utterance_block(text, aclarai_block_id)
            if not chunks:
                logger.warning(f"No chunks generated for block {aclarai_block_id}")
                return EmbeddingResult(
                    success=False,
                    total_chunks=0,
                    embedded_chunks=0,
                    stored_chunks=0,
                    failed_chunks=0,
                    metrics=VectorStoreMetrics(0, 0, 0),
                    errors=["No chunks generated from block"],
                )
            # Step 2: Generate embeddings
            embedded_chunks = self.embedding_generator.embed_chunks(chunks)
            # Step 3: Store embeddings
            storage_metrics = self.vector_store.store_embeddings(embedded_chunks)
            success = (
                storage_metrics.successful_inserts > 0
                and storage_metrics.failed_inserts == 0
            )
            result = EmbeddingResult(
                success=success,
                total_chunks=len(chunks),
                embedded_chunks=len(embedded_chunks),
                stored_chunks=storage_metrics.successful_inserts,
                failed_chunks=storage_metrics.failed_inserts,
                metrics=storage_metrics,
                errors=errors,
            )
            logger.info(
                f"Single block processing completed: {aclarai_block_id}, success={success}"
            )
            return result
        except Exception as e:
            error_msg = f"Single block processing failed for {aclarai_block_id}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            return EmbeddingResult(
                success=False,
                total_chunks=0,
                embedded_chunks=0,
                stored_chunks=0,
                failed_chunks=0,
                metrics=VectorStoreMetrics(0, 0, 0),
                errors=errors,
            )

    def search_similar_chunks(
        self,
        query_text: str,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in the vector store.
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
        Returns:
            List of chunk metadata with similarity scores
        """
        logger.debug(f"Searching for similar chunks: '{query_text[:50]}...'")
        try:
            results = self.vector_store.similarity_search(
                query_text=query_text,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            # Format results for return
            formatted_results = []
            for metadata, score in results:
                result = dict(metadata)
                result["similarity_score"] = score
                formatted_results.append(result)
            logger.debug(f"Found {len(formatted_results)} similar chunks")
            return formatted_results
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get the current status of the embedding pipeline components.
        Returns:
            Status dictionary with component health
        """
        status: Dict[str, Any] = {"components": {}, "overall_status": "healthy"}
        try:
            # Check vector store
            metrics = self.vector_store.get_store_metrics()
            status["components"]["vector_store"] = {
                "status": "healthy" if metrics.total_vectors >= 0 else "error",
                "total_vectors": metrics.total_vectors,
                "metrics": metrics,
            }
            # Check embedding model
            try:
                test_embedding = self.embedding_generator.embed_text("test")
                status["components"]["embedding_model"] = {
                    "status": "healthy",
                    "model_name": self.embedding_generator.model_name,
                    "embedding_dim": len(test_embedding),
                }
            except Exception as e:
                status["components"]["embedding_model"] = {
                    "status": "error",
                    "error": str(e),
                }
                status["overall_status"] = "degraded"
            # Check chunker
            status["components"]["chunker"] = {
                "status": "healthy",
                "chunk_size": self.config.embedding.chunk_size,
                "chunk_overlap": self.config.embedding.chunk_overlap,
            }
        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            status["overall_status"] = "error"
            status["error"] = str(e)
        return status
