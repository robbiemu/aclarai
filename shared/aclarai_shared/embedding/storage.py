"""
PGVector store integration for aclarai embeddings.
This module provides PostgreSQL vector storage using LlamaIndex's PGVectorStore
for utterance chunk embeddings. Includes metadata handling, indexing, and
similarity search capabilities.
Key Features:
- LlamaIndex PGVectorStore integration
- Automatic table and index creation
- Metadata preservation (aclarai:id, chunk_index, original text)
- Efficient similarity queries with IVFFlat indexing
- Batch insert and update operations
- Connection management with fallback
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import BaseNode, Document
from llama_index.core.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import (
    create_engine,
    text,
)

from ..config import aclaraiConfig
from .chunking import UtteranceChunker
from .models import EmbeddedChunk, EmbeddingGenerator

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreMetrics:
    """Metrics for vector store operations."""

    total_vectors: int
    successful_inserts: int
    failed_inserts: int
    index_size_mb: Optional[float] = None
    query_time_ms: Optional[float] = None


class aclaraiVectorStore(VectorStore):
    """
    PostgreSQL vector store for aclarai utterance embeddings using PGVectorStore.
    This class provides a high-level interface for storing and querying utterance
    embeddings following the architecture from docs/arch/idea-embedding_in_vectordb.md
    """

    def __init__(self, config: Optional[aclaraiConfig] = None):
        """
        Initialize the vector store.
        Args:
            config: aclarai configuration (loads default if None)
        """
        if config is None:
            from ..config import load_config

            config = load_config(validate=True)  # Require DB credentials
        self.config = config
        # Build connection string
        self.connection_string = config.postgres.get_connection_url(
            "postgresql+psycopg2"
        )
        # Initialize database engine
        self.engine = create_engine(
            self.connection_string,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=config.debug,
            connect_args={
                "host": config.postgres.host,
                "port": config.postgres.port,
                "user": config.postgres.user,
                "password": config.postgres.password,
            },
        )
        # Initialize PGVectorStore
        self.vector_store = self._initialize_pgvector_store()

        self.stores_text = self.vector_store.stores_text

        self._client = self.vector_store.client
        # Initialize embedding generator with configured model
        self.embedding_generator = EmbeddingGenerator(config=config)
        # Initialize chunker for compatibility with tests
        self.chunker = UtteranceChunker(config=config)
        # Set LlamaIndex global embedding model IMMEDIATELY to prevent default OpenAI dependency
        # This ensures VectorStoreIndex doesn't try to import llama-index-embeddings-openai
        # We set this as early as possible to prevent any race conditions
        Settings.embed_model = self.embedding_generator.embedding_model
        # Initialize LlamaIndex VectorStoreIndex with configured embedding model
        # Pass embed_model explicitly as additional safety to avoid any fallback to Settings.embed_model
        self.vector_index = VectorStoreIndex.from_vector_store(
            self.vector_store, embed_model=self.embedding_generator.embedding_model
        )
        logger.info(
            f"Initialized aclaraiVectorStore with collection: {config.embedding.collection_name}, "
            f"dimension: {config.embedding.embed_dim}"
        )

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    def add(
        self,
        nodes: List[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """Add nodes to index."""
        return self.vector_store.add(nodes, **kwargs)

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Delete nodes using ref_doc_id."""
        self.vector_store.delete(ref_doc_id, **kwargs)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query vector store."""
        return self.vector_store.query(query, **kwargs)

    def _validate_table_name(self, table_name: str) -> str:
        """
        Validate and sanitize table name to prevent SQL injection.
        Args:
            table_name: The table name to validate
        Returns:
            Sanitized table name
        Raises:
            ValueError: If table name is invalid
        """
        # Allow only alphanumeric characters, underscores, and hyphens
        # Must start with a letter or underscore
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_-]*$", table_name):
            raise ValueError(
                f"Invalid table name: {table_name}. Only alphanumeric characters, underscores, and hyphens are allowed."
            )
        # Limit length to prevent issues
        if len(table_name) > 63:  # PostgreSQL identifier limit
            raise ValueError(f"Table name too long: {table_name} (max 63 characters)")
        return table_name

    def store_embeddings(
        self, embedded_chunks: List[EmbeddedChunk]
    ) -> VectorStoreMetrics:
        """
        Store embedded chunks in the vector database.
        Args:
            embedded_chunks: List of EmbeddedChunk objects to store
        Returns:
            VectorStoreMetrics with operation results
        """
        if not embedded_chunks:
            logger.warning("No embedded chunks provided for storage")
            return VectorStoreMetrics(
                total_vectors=0, successful_inserts=0, failed_inserts=0
            )
        logger.info(f"Storing {len(embedded_chunks)} embedded chunks in vector store")
        # Convert embedded chunks to Documents for LlamaIndex
        documents = self._convert_to_documents(embedded_chunks)
        successful_inserts = 0
        failed_inserts = 0
        try:
            # Use LlamaIndex to insert documents
            # This handles embedding storage and metadata automatically
            for doc in documents:
                try:
                    self.vector_index.insert(doc)
                    successful_inserts += 1
                except Exception as e:
                    logger.error(f"Failed to insert document {doc.doc_id}: {e}")
                    failed_inserts += 1
            logger.info(
                f"Storage complete: {successful_inserts} successful, {failed_inserts} failed"
            )
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            failed_inserts = len(embedded_chunks)
        return VectorStoreMetrics(
            total_vectors=len(embedded_chunks),
            successful_inserts=successful_inserts,
            failed_inserts=failed_inserts,
        )

    def similarity_search(
        self,
        query_text: str,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform similarity search for utterance chunks.
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (optional)
            filter_metadata: Metadata filters (optional)
        Returns:
            List of (metadata, similarity_score) tuples
        """
        logger.debug(
            f"Performing similarity search: query='{query_text[:50]}...', top_k={top_k}"
        )
        try:
            # Use LlamaIndex query engine for similarity search
            query_engine = self.vector_index.as_query_engine(
                similarity_top_k=top_k,
                response_mode="no_text",  # We just want the nodes, not generated text
            )
            response = query_engine.query(query_text)
            # Extract results with similarity scores
            results = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    metadata = node.node.metadata
                    score = getattr(node, "score", 0.0)
                    # Apply similarity threshold and metadata filters if specified
                    if (
                        similarity_threshold is None or score >= similarity_threshold
                    ) and self._matches_filter(metadata, filter_metadata):
                        results.append((metadata, score))
            logger.debug(f"Similarity search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def get_chunk_by_id(
        self, aclarai_block_id: str, chunk_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chunk by its aclarai block ID and chunk index.
        Args:
            aclarai_block_id: The aclarai:id of the source block
            chunk_index: The chunk index within the block
        Returns:
            Chunk metadata if found, None otherwise
        """
        logger.debug(f"Retrieving chunk: {aclarai_block_id}[{chunk_index}]")
        try:
            # Use metadata filter to find specific chunk
            results = self.similarity_search(
                query_text="",  # Empty query since we're filtering by metadata
                top_k=1,
                filter_metadata={
                    "aclarai_block_id": aclarai_block_id,
                    "chunk_index": chunk_index,
                },
            )
            if results:
                return results[0][0]  # Return metadata
            else:
                logger.debug(f"Chunk not found: {aclarai_block_id}[{chunk_index}]")
                return None
        except Exception as e:
            logger.error(
                f"Failed to retrieve chunk {aclarai_block_id}[{chunk_index}]: {e}"
            )
            return None

    def get_chunks_by_block_id(self, aclarai_block_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific aclarai block ID.
        Args:
            aclarai_block_id: The aclarai:id of the source block
        Returns:
            List of chunk metadata dictionaries
        """
        logger.debug(f"Retrieving all chunks for block: {aclarai_block_id}")
        try:
            # Use metadata filter to find all chunks for this block
            results = self.similarity_search(
                query_text="",  # Empty query since we're filtering by metadata
                top_k=100,  # Reasonable upper limit for chunks per block
                filter_metadata={"aclarai_block_id": aclarai_block_id},
            )
            # Extract just the metadata and sort by chunk_index
            chunks = [result[0] for result in results]
            chunks.sort(key=lambda x: x.get("chunk_index", 0))
            logger.debug(f"Retrieved {len(chunks)} chunks for block {aclarai_block_id}")
            return chunks
        except Exception as e:
            logger.error(f"Failed to retrieve chunks for block {aclarai_block_id}: {e}")
            return []

    def delete_chunks_by_block_id(self, aclarai_block_id: str) -> int:
        """
        Delete all chunks for a specific aclarai block ID.
        Args:
            aclarai_block_id: The aclarai:id of the source block
        Returns:
            Number of chunks deleted
        """
        logger.info(f"Deleting chunks for block: {aclarai_block_id}")
        try:
            # First get all chunks to know what to delete
            chunks = self.get_chunks_by_block_id(aclarai_block_id)
            if not chunks:
                logger.debug(f"No chunks found to delete for block {aclarai_block_id}")
                return 0
            # Delete each chunk by its document ID
            deleted_count = 0
            for chunk in chunks:
                doc_id = chunk.get("doc_id")
                if doc_id:
                    try:
                        self.vector_index.delete(doc_id)
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete chunk {doc_id}: {e}")
            logger.info(f"Deleted {deleted_count} chunks for block {aclarai_block_id}")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete chunks for block {aclarai_block_id}: {e}")
            return 0

    def get_embeddings_for_concepts(
        self, concept_names: List[str]
    ) -> Dict[str, List[float]]:
        """
        Retrieve embeddings for a specific list of concept names using bulk retrieval.

        This method implements the "Bulk Embedding Retrieval" access pattern defined in
        docs/arch/on-vector_stores.md. It performs a single, efficient SQL query to
        retrieve embeddings for a large, known set of concepts, avoiding the N+1 query
        problem that would occur if similarity_search was used in a loop.

        **Use Cases:**
        - Concept clustering jobs that need embeddings for all canonical concepts
        - Analytics and batch processing operations
        - Any scenario where you need embeddings for a predetermined set of items

        **Performance Benefits:**
        - Single database query instead of N queries (one per concept)
        - Reduced network overhead and connection usage
        - Consistent performance regardless of concept count
        - Avoids the overhead of similarity calculations when exact matches are needed

        **Difference from similarity_search:**
        - similarity_search: "one-to-few" discovery pattern for finding similar items
        - get_embeddings_for_concepts: "many-to-many" bulk retrieval for known items

        Args:
            concept_names: A list of concept names to retrieve embeddings for.

        Returns:
            A dictionary mapping concept names to their embedding vectors.
            Only concepts that exist in the vector store will be included.
        """
        if not concept_names:
            return {}

        logger.debug(
            f"Retrieving embeddings for {len(concept_names)} concepts.",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "aclaraiVectorStore.get_embeddings_for_concepts",
                "concept_count": len(concept_names),
            },
        )

        # Sanitize table name to prevent SQL injection
        # Note: LlamaIndex PGVectorStore automatically prepends 'data_' to the table name
        actual_table_name = f"data_{self.config.embedding.collection_name}"
        table_name = self._validate_table_name(actual_table_name)

        # Use a parameterized query for safety and efficiency
        # The text field contains the concept name, not the metadata
        query = text(f"""
            SELECT
                text as name,
                embedding
            FROM {table_name}
            WHERE text IN :concept_names
        """)

        embeddings_map: Dict[str, List[float]] = {}
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {"concept_names": tuple(concept_names)})
                for row in result:
                    # The embedding is returned as a string representation of a list, e.g., '[0.1, 0.2, ...]'
                    # We need to parse it back into a list of floats.
                    embedding_str = row[1]
                    if isinstance(embedding_str, str):
                        # Strip brackets and split by comma
                        embedding_list = [
                            float(val) for val in embedding_str.strip("[]").split(",")
                        ]
                        embeddings_map[row[0]] = embedding_list
                    elif isinstance(
                        embedding_str, list
                    ):  # If the driver already parses it
                        embeddings_map[row[0]] = embedding_str

        except Exception as e:
            logger.error(
                f"Failed to retrieve embeddings for concepts: {e}",
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "aclaraiVectorStore.get_embeddings_for_concepts",
                },
            )
        return embeddings_map

    def get_store_metrics(self) -> VectorStoreMetrics:
        """
        Get metrics about the vector store.
        Returns:
            VectorStoreMetrics with current statistics
        """
        try:
            with self.engine.connect() as conn:
                # Validate table name to prevent SQL injection
                # Note: LlamaIndex PGVectorStore automatically prepends 'data_' to the table name
                actual_table_name = f"data_{self.config.embedding.collection_name}"
                table_name = self._validate_table_name(actual_table_name)
                # Get total count
                # Table name is validated above to prevent SQL injection
                count_query = text(f"""
                    SELECT COUNT(*) as total_vectors
                    FROM {table_name}
                """)  # nosec B608
                result = conn.execute(count_query)
                count_row = result.fetchone()
                total_vectors = count_row[0] if count_row else 0
                # Get table size
                size_query = text("""
                    SELECT pg_size_pretty(pg_total_relation_size(:table_name))
                """)
                size_result = conn.execute(size_query, {"table_name": table_name})
                size_row = size_result.fetchone()
                size_str = size_row[0] if size_row else None
                return VectorStoreMetrics(
                    total_vectors=total_vectors,
                    successful_inserts=total_vectors,  # Approximate
                    failed_inserts=0,
                    index_size_mb=self._parse_size_to_mb(size_str),
                )
        except Exception as e:
            logger.error(f"Failed to get store metrics: {e}")
            return VectorStoreMetrics(
                total_vectors=0, successful_inserts=0, failed_inserts=0
            )

    def _initialize_pgvector_store(self) -> PGVectorStore:
        """
        Initialize the PGVectorStore with proper configuration.
        Returns:
            Configured PGVectorStore instance
        """
        try:
            # Ensure pgvector extension is enabled
            self._ensure_pgvector_extension()
            # Initialize PGVectorStore
            vector_store = PGVectorStore.from_params(
                database=self.config.postgres.database,
                host=self.config.postgres.host,
                password=self.config.postgres.password,
                port=str(self.config.postgres.port),
                user=self.config.postgres.user,
                table_name=self.config.embedding.collection_name,
                embed_dim=self.config.embedding.embed_dim,
                hnsw_kwargs={
                    "hnsw_m": 16,
                    "hnsw_ef_construction": 64,
                    "hnsw_ef_search": 40,
                },
            )
            logger.info(
                f"Initialized PGVectorStore with table: {self.config.embedding.collection_name}"
            )
            return vector_store
        except Exception as e:
            logger.error(f"Failed to initialize PGVectorStore: {e}")
            raise

    def _ensure_pgvector_extension(self):
        """Ensure the pgvector extension is enabled in PostgreSQL."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                logger.debug("Ensured pgvector extension is enabled")
        except Exception as e:
            logger.error(f"Failed to enable pgvector extension: {e}")
            raise

    def _convert_to_documents(
        self, embedded_chunks: List[EmbeddedChunk]
    ) -> List[Document]:
        """
        Convert EmbeddedChunk objects to LlamaIndex Documents.
        Args:
            embedded_chunks: List of embedded chunks
        Returns:
            List of Document objects for LlamaIndex
        """
        documents = []
        for chunk in embedded_chunks:
            metadata = {
                "aclarai_block_id": chunk.chunk_metadata.aclarai_block_id,
                "chunk_index": chunk.chunk_metadata.chunk_index,
                "original_text": chunk.chunk_metadata.original_text,
                "model_name": chunk.model_name,
                "embedding_dim": chunk.embedding_dim,
            }
            # Create Document with embedding
            doc = Document(
                text=chunk.chunk_metadata.text,
                metadata=metadata,
                embedding=chunk.embedding,
            )
            documents.append(doc)
        return documents

    def _matches_filter(
        self, metadata: Dict[str, Any], filter_metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Check if metadata matches the provided filter.
        Args:
            metadata: Metadata to check
            filter_metadata: Filter criteria
        Returns:
            True if metadata matches filter, False otherwise
        """
        if filter_metadata is None:
            return True
        return all(metadata.get(key) == value for key, value in filter_metadata.items())

    def _parse_size_to_mb(self, size_str: Optional[str]) -> Optional[float]:
        """
        Parse PostgreSQL size string to MB.
        Args:
            size_str: Size string from PostgreSQL (e.g., "123 kB", "4567 MB")
        Returns:
            Size in MB or None if parsing fails
        """
        if not size_str:
            return None
        try:
            parts = size_str.strip().split()
            if len(parts) != 2:
                return None
            value_str, unit = parts
            value_float = float(value_str)
            unit = unit.upper()
            if unit == "BYTES":
                return value_float / (1024 * 1024)
            elif unit == "KB":
                return value_float / 1024
            elif unit == "MB":
                return value_float
            elif unit == "GB":
                return value_float * 1024
            else:
                return None
        except (ValueError, IndexError):
            return None
