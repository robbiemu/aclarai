"""
Vector storage for concept candidates.
This module provides specialized vector storage for concept candidates extracted
from Claims and Summary nodes, implementing the concept_candidates vector table
as specified in docs/arch/on-vector_stores.md.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import create_engine

from ..config import aclaraiConfig
from ..embedding import EmbeddingGenerator
from .models import NounPhraseCandidate

logger = logging.getLogger(__name__)


@dataclass
class ConceptCandidateDocument:
    """Document wrapper for concept candidates in vector storage."""

    doc_id: str
    text: str  # Normalized text that gets embedded
    metadata: Dict[str, Any]  # All candidate metadata
    embedding: Optional[List[float]] = None


class ConceptCandidatesVectorStore:
    """
    Specialized vector store for concept candidates.
    This store manages the concept_candidates vector table specifically for
    noun phrases extracted from Claims and Summary nodes, following the
    architecture from docs/arch/on-concepts.md.
    """

    def __init__(self, config: Optional[aclaraiConfig] = None):
        """
        Initialize the concept candidates vector store.
        Args:
            config: aclarai configuration (loads default if None)
        """
        if config is None:
            from ..config import load_config

            config = load_config(validate=True)
        self.config = config
        # Use the concept_candidates collection configuration
        self.collection_name = (
            config.noun_phrase_extraction.concept_candidates_collection
        )
        # Initialize embedding generator first
        self.embedding_generator = EmbeddingGenerator(config=config)
        # Get the embedding dimension dynamically from the model
        self.embed_dim = self.embedding_generator.get_embedding_dimension()
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
        )
        # Initialize PGVectorStore for concept_candidates
        self.vector_store = self._initialize_pgvector_store()
        # Set LlamaIndex embedding model
        Settings.embed_model = self.embedding_generator.embedding_model
        # Initialize VectorStoreIndex
        self.vector_index = VectorStoreIndex.from_vector_store(
            self.vector_store, embed_model=self.embedding_generator.embedding_model
        )
        logger.info(
            f"Initialized ConceptCandidatesVectorStore with collection: {self.collection_name}, "
            f"dimension: {self.embed_dim}",
            extra={
                "service": "aclarai",
                "filename.function_name": "concept_candidates_vector_store.ConceptCandidatesVectorStore.__init__",
                "collection_name": self.collection_name,
                "embed_dim": self.embed_dim,
            },
        )

    def store_candidates(self, candidates: List[NounPhraseCandidate]) -> int:
        """
        Store noun phrase candidates in the vector database using batch insertion.
        Args:
            candidates: List of NounPhraseCandidate objects to store
        Returns:
            Number of candidates successfully stored
        """
        if not candidates:
            logger.warning(
                "No candidates provided for storage",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_candidates_vector_store.ConceptCandidatesVectorStore.store_candidates",
                },
            )
            return 0
        logger.info(
            f"Storing {len(candidates)} concept candidates in vector store",
            extra={
                "service": "aclarai",
                "filename.function_name": "concept_candidates_vector_store.ConceptCandidatesVectorStore.store_candidates",
                "candidates_count": len(candidates),
            },
        )
        try:
            # Convert candidates to documents
            documents = self._convert_candidates_to_documents(candidates)
            # Generate embeddings for candidates that don't have them
            texts_to_embed = []
            indices_to_embed = []
            for i, candidate in enumerate(candidates):
                if candidate.embedding is None:
                    texts_to_embed.append(candidate.normalized_text)
                    indices_to_embed.append(i)
                else:
                    documents[i].embedding = candidate.embedding
            # Batch generate embeddings if needed
            if texts_to_embed:
                logger.debug(
                    f"Generating embeddings for {len(texts_to_embed)} candidates"
                )
                embeddings = self.embedding_generator._embed_texts_batch(texts_to_embed)
                for idx, embedding in zip(indices_to_embed, embeddings, strict=False):
                    candidates[idx].embedding = embedding
                    documents[idx].embedding = embedding
            # Use batch insertion via LlamaIndex
            logger.debug("Performing batch insertion of documents")
            # Insert all documents at once using LlamaIndex's batch capability
            self.vector_index.insert_nodes(documents)
            successful_count = len(candidates)
            logger.info(
                f"Successfully stored {successful_count}/{len(candidates)} concept candidates",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_candidates_vector_store.ConceptCandidatesVectorStore.store_candidates",
                    "successful_count": successful_count,
                    "total_count": len(candidates),
                },
            )
            return successful_count
        except Exception as e:
            logger.error(
                f"Failed to store concept candidates: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_candidates_vector_store.ConceptCandidatesVectorStore.store_candidates",
                    "candidates_count": len(candidates),
                    "error": str(e),
                },
            )
            return 0

    def find_similar_candidates(
        self,
        query_text: str,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find similar concept candidates using vector similarity.
        Args:
            query_text: Text to search for similar candidates
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
        Returns:
            List of (metadata, similarity_score) tuples
        """
        logger.debug(
            f"Searching for similar concept candidates: query='{query_text[:50]}...', top_k={top_k}",
            extra={
                "service": "aclarai",
                "filename.function_name": "concept_candidates_vector_store.ConceptCandidatesVectorStore.find_similar_candidates",
                "query_length": len(query_text),
                "top_k": top_k,
            },
        )
        try:
            # Use VectorStoreIndex for similarity search
            query_engine = self.vector_index.as_query_engine(
                similarity_top_k=top_k,
                response_mode="no_text",
            )
            response = query_engine.query(query_text)
            results = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    metadata = node.node.metadata
                    score = getattr(node, "score", 0.0)
                    # Apply similarity threshold if specified
                    if similarity_threshold is None or score >= similarity_threshold:
                        results.append((metadata, score))
            logger.debug(
                f"Found {len(results)} similar concept candidates",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_candidates_vector_store.ConceptCandidatesVectorStore.find_similar_candidates",
                    "results_count": len(results),
                },
            )
            return results
        except Exception as e:
            logger.error(
                f"Failed to find similar concept candidates: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_candidates_vector_store.ConceptCandidatesVectorStore.find_similar_candidates",
                    "error": str(e),
                },
            )
            return []

    def update_candidate_status(
        self,
        candidate_id: str,
        new_status: str,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update the status of a concept candidate.
        Args:
            candidate_id: ID of the candidate to update
            new_status: New status ("merged", "promoted", etc.)
            metadata_updates: Additional metadata to update
        Returns:
            True if update was successful, False otherwise
        """
        logger.debug(
            f"Updating candidate status: {candidate_id} -> {new_status}",
            extra={
                "service": "aclarai",
                "filename.function_name": "concept_candidates_vector_store.ConceptCandidatesVectorStore.update_candidate_status",
                "candidate_id": candidate_id,
                "new_status": new_status,
            },
        )
        try:
            # For PostgreSQL vector store, we need to update via SQL
            from sqlalchemy import text

            # Build the update query
            metadata_updates = metadata_updates or {}
            metadata_updates["status"] = new_status
            # Update metadata column by merging with existing metadata
            with self.engine.connect() as conn:
                # First get the current record
                select_stmt = text(f"""
                    SELECT metadata FROM {self.vector_store.table_name}
                    WHERE metadata->>'candidate_id' = :candidate_id
                """)  # nosec B608 - table_name is from config, not user input
                result = conn.execute(select_stmt, {"candidate_id": candidate_id})
                row = result.fetchone()
                if not row:
                    logger.warning(
                        f"Candidate not found for status update: {candidate_id}",
                        extra={
                            "service": "aclarai",
                            "filename.function_name": "concept_candidates_vector_store.ConceptCandidatesVectorStore.update_candidate_status",
                            "candidate_id": candidate_id,
                        },
                    )
                    return False
                # Merge metadata
                import json

                current_metadata = row[0] if row[0] else {}
                if isinstance(current_metadata, str):
                    current_metadata = json.loads(current_metadata)
                current_metadata.update(metadata_updates)
                # Update the record
                update_stmt = text(f"""
                    UPDATE {self.vector_store.table_name}
                    SET metadata = :metadata
                    WHERE metadata->>'candidate_id' = :candidate_id
                """)  # nosec B608 - table_name is from config, not user input
                conn.execute(
                    update_stmt,
                    {
                        "candidate_id": candidate_id,
                        "metadata": json.dumps(current_metadata),
                    },
                )
                conn.commit()
            logger.info(
                f"Successfully updated candidate status: {candidate_id} -> {new_status}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_candidates_vector_store.ConceptCandidatesVectorStore.update_candidate_status",
                    "candidate_id": candidate_id,
                    "new_status": new_status,
                },
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to update candidate status for {candidate_id}: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_candidates_vector_store.ConceptCandidatesVectorStore.update_candidate_status",
                    "candidate_id": candidate_id,
                    "new_status": new_status,
                    "error": str(e),
                },
            )
            return False

    def get_candidates_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Retrieve all candidates with a specific status (e.g., "pending").
        Args:
            status: Status to filter by
        Returns:
            List of candidate metadata dictionaries
        """
        logger.debug(
            f"Retrieving candidates with status: {status}",
            extra={
                "service": "aclarai",
                "filename.function_name": "concept_candidates_vector_store.ConceptCandidatesVectorStore.get_candidates_by_status",
                "status": status,
            },
        )
        try:
            # Use similarity search with empty query but metadata filter
            results = self.find_similar_candidates(
                query_text="",  # Empty query to get all
                top_k=1000,  # Large number to get all matches
            )
            # Filter by status
            filtered_results = [
                metadata
                for metadata, score in results
                if metadata.get("status") == status
            ]
            logger.debug(
                f"Found {len(filtered_results)} candidates with status '{status}'",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_candidates_vector_store.ConceptCandidatesVectorStore.get_candidates_by_status",
                    "status": status,
                    "count": len(filtered_results),
                },
            )
            return filtered_results
        except Exception as e:
            logger.error(
                f"Failed to retrieve candidates by status '{status}': {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_candidates_vector_store.ConceptCandidatesVectorStore.get_candidates_by_status",
                    "status": status,
                    "error": str(e),
                },
            )
            return []

    def _initialize_pgvector_store(self) -> PGVectorStore:
        """Initialize PGVectorStore for concept_candidates collection."""
        try:
            # Ensure pgvector extension is enabled
            self._ensure_pgvector_extension()
            # Initialize PGVectorStore with concept_candidates collection
            vector_store = PGVectorStore.from_params(
                database=self.config.postgres.database,
                host=self.config.postgres.host,
                password=self.config.postgres.password,
                port=str(self.config.postgres.port),
                user=self.config.postgres.user,
                table_name=self.collection_name,
                embed_dim=self.embed_dim,
                hnsw_kwargs={
                    "hnsw_m": 16,
                    "hnsw_ef_construction": 64,
                    "hnsw_ef_search": 40,
                },
            )
            logger.info(
                f"Initialized PGVectorStore for concept_candidates: {self.collection_name}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_candidates_vector_store.ConceptCandidatesVectorStore._initialize_pgvector_store",
                    "table_name": self.collection_name,
                },
            )
            return vector_store
        except Exception as e:
            logger.error(
                f"Failed to initialize PGVectorStore for concept_candidates: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_candidates_vector_store.ConceptCandidatesVectorStore._initialize_pgvector_store",
                    "error": str(e),
                },
            )
            raise

    def _ensure_pgvector_extension(self):
        """Ensure the pgvector extension is enabled in PostgreSQL."""
        try:
            from sqlalchemy import text

            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            logger.debug(
                "Ensured pgvector extension is enabled",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_candidates_vector_store.ConceptCandidatesVectorStore._ensure_pgvector_extension",
                },
            )
        except Exception as e:
            logger.error(
                f"Failed to enable pgvector extension: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_candidates_vector_store.ConceptCandidatesVectorStore._ensure_pgvector_extension",
                    "error": str(e),
                },
            )
            raise

    def _convert_candidates_to_documents(
        self, candidates: List[NounPhraseCandidate]
    ) -> List[Document]:
        """
        Convert NounPhraseCandidate objects to LlamaIndex Documents.
        Args:
            candidates: List of candidates to convert
        Returns:
            List of Document objects for LlamaIndex
        """
        documents = []
        for candidate in candidates:
            # Create comprehensive metadata
            metadata = {
                "original_text": candidate.text,
                "normalized_text": candidate.normalized_text,
                "source_node_id": candidate.source_node_id,
                "source_node_type": candidate.source_node_type,
                "aclarai_id": candidate.aclarai_id,
                "status": candidate.status,
                "timestamp": candidate.timestamp.isoformat()
                if candidate.timestamp
                else None,
            }
            # Create Document with normalized text as the content
            doc = Document(
                text=candidate.normalized_text,  # This is what gets embedded and searched
                metadata=metadata,
                doc_id=f"{candidate.source_node_type}_{candidate.source_node_id}_{candidate.text[:50]}",
                embedding=candidate.embedding,
            )
            documents.append(doc)
        return documents
