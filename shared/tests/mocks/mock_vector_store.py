"""
Mock Vector Store for testing and development.
This module provides an in-memory mock implementation of ConceptCandidatesVectorStore
that simulates vector similarity search using simple in-memory operations.
"""

import logging
import math
from typing import Any, Dict, List, Optional

from aclarai_shared.config import aclaraiConfig
from aclarai_shared.noun_phrase_extraction.models import NounPhraseCandidate
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)

logger = logging.getLogger(__name__)


class MockVectorStore:
    """
    In-memory mock implementation of ConceptCandidatesVectorStore.
    This mock maintains a simple list of documents and performs similarity search
    using cosine similarity calculated with basic math operations.
    """

    def __init__(self, config: Optional[aclaraiConfig] = None):
        """
        Initialize the mock vector store.
        Args:
            config: aclarai configuration (not used in mock but kept for compatibility)
        """
        self.config = config
        # In-memory storage
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
        # Mock embedding dimension (matches common models)
        self.embed_dim = 384
        logger.info(
            "mock_vector_store.MockVectorStore.__init__: Mock vector store initialized",
            extra={
                "service": "aclarai-test",
                "filename.function_name": "mock_vector_store.MockVectorStore.__init__",
                "embed_dim": self.embed_dim,
            },
        )

    def store_candidates(self, candidates: List[NounPhraseCandidate]) -> int:
        """
        Store concept candidates in the mock vector store.
        Args:
            candidates: List of NounPhraseCandidate objects
        Returns:
            Number of candidates successfully stored
        """
        stored_count = 0
        for candidate in candidates:
            candidate_id = getattr(
                candidate, "candidate_id", f"cand_{hash(candidate.text)}"
            )
            # Generate mock embedding if not provided
            if candidate.embedding is None:
                embedding = self._generate_mock_embedding(candidate.normalized_text)
            else:
                embedding = candidate.embedding
            # Create document entry
            doc = {
                "id": candidate_id,
                "text": candidate.normalized_text,
                "metadata": {
                    "id": candidate_id,
                    "source_node_id": candidate.source_node_id,
                    "source_node_type": candidate.source_node_type,
                    "aclarai_id": candidate.aclarai_id,
                    "status": getattr(candidate, "status", "pending"),
                },
            }
            self.documents.append(doc)
            self.embeddings.append(embedding)
            stored_count += 1
        logger.info(
            f"mock_vector_store.MockVectorStore.store_candidates: Stored {stored_count} candidates",
            extra={
                "service": "aclarai-test",
                "filename.function_name": "mock_vector_store.MockVectorStore.store_candidates",
                "candidates_stored": stored_count,
                "total_documents": len(self.documents),
            },
        )
        return stored_count

    def query(self, query: VectorStoreQuery, **_kwargs: Any) -> VectorStoreQueryResult:
        """Mock query method to conform to the VectorStore interface."""
        if not self.documents:
            return VectorStoreQueryResult(nodes=[], similarities=[])

        query_embedding = self._generate_mock_embedding(query.query_str)

        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        top_k = query.similarity_top_k
        top_results = similarities[:top_k]

        nodes: List[TextNode] = []
        scores: List[float] = []
        for doc_index, score in top_results:
            doc = self.documents[doc_index]
            # The real store returns nodes with metadata. We simulate that.
            node = TextNode(
                text=doc.get("text", ""),
                id_=doc.get("id", ""),
                metadata=doc.get("metadata", {}),
            )
            nodes.append(node)
            scores.append(score)

        return VectorStoreQueryResult(nodes=nodes, similarities=scores)

    def update_candidate_status(
        self,
        candidate_id: str,
        new_status: str,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update the status of a candidate.
        Args:
            candidate_id: ID of the candidate to update
            new_status: New status value
            metadata_updates: Additional metadata updates
        Returns:
            True if update was successful, False otherwise
        """
        for doc in self.documents:
            if doc["id"] == candidate_id:
                doc["metadata"]["status"] = new_status
                if metadata_updates:
                    doc["metadata"].update(metadata_updates)
                logger.debug(
                    f"mock_vector_store.MockVectorStore.update_candidate_status: "
                    f"Updated candidate {candidate_id} status to {new_status}",
                    extra={
                        "service": "aclarai-test",
                        "filename.function_name": "mock_vector_store.MockVectorStore.update_candidate_status",
                        "candidate_id": candidate_id,
                        "new_status": new_status,
                    },
                )
                return True
        logger.warning(
            f"mock_vector_store.MockVectorStore.update_candidate_status: "
            f"Candidate {candidate_id} not found",
            extra={
                "service": "aclarai-test",
                "filename.function_name": "mock_vector_store.MockVectorStore.update_candidate_status",
                "candidate_id": candidate_id,
            },
        )
        return False

    def get_candidates_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Get candidates by status.
        Args:
            status: Status to filter by
        Returns:
            List of candidate documents with matching status
        """
        results = [
            doc for doc in self.documents if doc["metadata"].get("status") == status
        ]
        logger.debug(
            f"mock_vector_store.MockVectorStore.get_candidates_by_status: "
            f"Found {len(results)} candidates with status {status}",
            extra={
                "service": "aclarai-test",
                "filename.function_name": "mock_vector_store.MockVectorStore.get_candidates_by_status",
                "status": status,
                "results_count": len(results),
            },
        )
        return results

    def clear_all_data(self):
        """
        Clear all data from the mock vector store.
        This is a test utility method not present in the real store.
        """
        self.documents.clear()
        self.embeddings.clear()
        logger.debug(
            "mock_vector_store.MockVectorStore.clear_all_data: All mock data cleared",
            extra={
                "service": "aclarai-test",
                "filename.function_name": "mock_vector_store.MockVectorStore.clear_all_data",
            },
        )

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """
        Generate a deterministic mock embedding for text.
        This creates a simple hash-based embedding that's deterministic
        but provides reasonable similarity for similar texts.
        Args:
            text: Text to generate embedding for
        Returns:
            List of floats representing the embedding
        """
        # A robust mock embedding based on word sets.
        # This ignores word order and common stop words, providing a better
        # simulation of semantic similarity for testing.
        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "in",
            "on",
            "of",
            "for",
            "with",
            "to",
            "related",
        }
        words = set(text.lower().split()) - stop_words

        # Create a "vocabulary" of all possible words to ensure consistent vector dimensions
        # For a real test suite, this vocab could be pre-built from all test data.
        # Here, we'll use a simple hashing approach for each word.
        embedding = [0.0] * self.embed_dim
        if not words:
            return embedding
        for word in words:
            # Use a hash to get a consistent index for each word
            index = hash(word) % self.embed_dim
            embedding[index] = 1.0  # Set the feature for this word to 1.0

        # Normalize the embedding vector
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]
        return embedding

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        Args:
            vec1: First vector
            vec2: Second vector
        Returns:
            Cosine similarity score between 0 and 1
        """
        if len(vec1) != len(vec2):
            return 0.0
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        # Calculate magnitudes
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(a * a for a in vec2))
        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0.0
        # Calculate cosine similarity
        similarity = dot_product / (mag1 * mag2)
        # Ensure result is between 0 and 1
        return max(0.0, min(1.0, similarity))
