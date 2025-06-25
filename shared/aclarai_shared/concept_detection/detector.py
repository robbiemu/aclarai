"""
Concept detector using hnswlib for embedding-based similarity detection.
This module implements the core logic for detecting similar concept candidates
using hnswlib and determining whether candidates should be merged or promoted.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import hnswlib
import numpy as np

from ..config import aclaraiConfig, load_config
from ..noun_phrase_extraction.concept_candidates_store import (
    ConceptCandidatesVectorStore,
)
from ..noun_phrase_extraction.models import NounPhraseCandidate
from .models import (
    ConceptAction,
    ConceptDetectionBatch,
    ConceptDetectionResult,
    SimilarityMatch,
)

logger = logging.getLogger(__name__)


class ConceptDetector:
    """
    Concept detector using hnswlib for fast similarity search.
    This class manages an hnswlib index of concept candidates and canonical concepts,
    providing functionality to detect similar items and make merge/promote decisions.
    """

    def __init__(self, config: Optional[aclaraiConfig] = None):
        """Initialize the concept detector."""
        self.config = config or load_config()
        # Get similarity threshold from config
        self.similarity_threshold = self.config.concepts.similarity_threshold
        # Initialize the concept candidates store
        self.candidates_store = ConceptCandidatesVectorStore(config)
        # HNSW index parameters
        self.embedding_dim = self.candidates_store.embed_dim
        self.index: Optional[hnswlib.Index] = None
        self.id_to_metadata: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0
        logger.info(
            f"Initialized ConceptDetector with similarity_threshold={self.similarity_threshold}",
            extra={
                "service": "aclarai",
                "filename.function_name": "concept_detection.detector.ConceptDetector.__init__",
                "similarity_threshold": self.similarity_threshold,
                "embedding_dim": self.embedding_dim,
            },
        )

    def _initialize_index(self, max_elements: int = 10000):
        """Initialize the HNSW index."""
        try:
            self.index = hnswlib.Index(space="cosine", dim=self.embedding_dim)
            self.index.init_index(
                max_elements=max_elements,
                M=16,  # Number of connections
                ef_construction=200,  # Size of candidate set
                random_seed=42,
            )
            self.index.set_ef(50)  # Search parameter
            logger.info(
                f"Initialized HNSW index with max_elements={max_elements}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_detection.detector.ConceptDetector._initialize_index",
                    "max_elements": max_elements,
                    "embedding_dim": self.embedding_dim,
                },
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize HNSW index: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_detection.detector.ConceptDetector._initialize_index",
                    "error": str(e),
                },
            )
            raise

    def build_index_from_candidates(self, force_rebuild: bool = False) -> int:
        """
        Build the HNSW index from existing concept candidates.
        Args:
            force_rebuild: Whether to force rebuilding even if index exists
        Returns:
            Number of items added to the index
        """
        if self.index is not None and not force_rebuild:
            logger.debug("HNSW index already exists, skipping rebuild")
            return len(self.id_to_metadata)
        logger.info(
            "Building HNSW index from concept candidates",
            extra={
                "service": "aclarai",
                "filename.function_name": "concept_detection.detector.ConceptDetector.build_index_from_candidates",
            },
        )
        try:
            # Get all pending candidates with embeddings
            pending_candidates = self.candidates_store.get_candidates_by_status(
                "pending"
            )
            if not pending_candidates:
                logger.warning("No pending candidates found to build index")
                self._initialize_index(max_elements=1000)  # Initialize empty index
                return 0
            # Filter candidates that have embeddings
            candidates_with_embeddings = []
            for candidate_data in pending_candidates:
                if candidate_data.get("embedding") is not None:
                    candidates_with_embeddings.append(candidate_data)
            if not candidates_with_embeddings:
                logger.warning("No candidates with embeddings found")
                self._initialize_index(max_elements=1000)
                return 0
            # Initialize index with appropriate size
            max_elements = max(len(candidates_with_embeddings) * 2, 1000)
            self._initialize_index(max_elements)
            # Add candidates to index
            embeddings = []
            ids = []
            for candidate_data in candidates_with_embeddings:
                embedding = candidate_data["embedding"]
                if isinstance(embedding, list):
                    embedding = np.array(embedding, dtype=np.float32)
                # Store metadata for this ID
                self.id_to_metadata[self.next_id] = candidate_data
                embeddings.append(embedding)
                ids.append(self.next_id)
                self.next_id += 1
            # Add all embeddings to index
            if embeddings:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                if self.index is not None:
                    self.index.add_items(embeddings_array, ids)
                else:
                    logger.error(
                        "HNSW index is not initialized before adding items.",
                        extra={
                            "service": "aclarai",
                            "filename.function_name": "concept_detection.detector.ConceptDetector.build_index_from_candidates",
                        },
                    )
                    raise RuntimeError("HNSW index is not initialized.")
            logger.info(
                f"Built HNSW index with {len(embeddings)} concept candidates",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_detection.detector.ConceptDetector.build_index_from_candidates",
                    "candidates_added": len(embeddings),
                },
            )
            return len(embeddings)
        except Exception as e:
            logger.error(
                f"Failed to build HNSW index: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_detection.detector.ConceptDetector.build_index_from_candidates",
                    "error": str(e),
                },
            )
            raise

    def find_similar_candidates(
        self, candidate: NounPhraseCandidate, top_k: int = 10
    ) -> List[SimilarityMatch]:
        """
        Find similar candidates for a given candidate using the HNSW index.
        Args:
            candidate: The candidate to find similarities for
            top_k: Number of similar items to return
        Returns:
            List of similarity matches
        """
        if self.index is None:
            logger.warning("HNSW index not initialized, building from candidates")
            self.build_index_from_candidates()
        if self.index is None or len(self.id_to_metadata) == 0:
            logger.debug("No index or no items in index")
            return []
        try:
            # Prepare embedding
            embedding = candidate.embedding
            if embedding is None:
                logger.warning(f"Candidate has no embedding: {candidate.text}")
                return []
            # Ensure embedding is a numpy array
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32).tolist()
            # Search for similar items
            labels, distances = self.index.knn_query(
                np.array(embedding, dtype=np.float32).reshape(1, -1),
                k=min(top_k, len(self.id_to_metadata)),
            )
            # Convert distances to similarities (cosine distance -> cosine similarity)
            similarities = 1.0 - distances[0]
            # Create similarity matches
            matches = []
            for label, similarity in zip(labels[0], similarities, strict=False):
                if int(label) in self.id_to_metadata:
                    metadata = self.id_to_metadata[int(label)]
                    # Skip self-match
                    if metadata.get("normalized_text") == candidate.normalized_text:
                        continue
                    match = SimilarityMatch(
                        candidate_id=f"{candidate.source_node_type}_{candidate.source_node_id}_{candidate.text[:50]}",
                        matched_candidate_id=metadata.get("id"),
                        matched_concept_id=None,  # We don't have canonical concepts in this index yet
                        similarity_score=float(similarity),
                        matched_text=metadata.get("normalized_text", ""),
                        metadata=metadata,
                    )
                    matches.append(match)
            logger.debug(
                f"Found {len(matches)} similar candidates for '{candidate.text}'",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_detection.detector.ConceptDetector.find_similar_candidates",
                    "candidate_text": candidate.text,
                    "matches_found": len(matches),
                },
            )
            return matches
        except Exception as e:
            logger.error(
                f"Failed to find similar candidates: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_detection.detector.ConceptDetector.find_similar_candidates",
                    "error": str(e),
                },
            )
            return []

    def detect_concept_action(
        self, candidate: NounPhraseCandidate
    ) -> ConceptDetectionResult:
        """
        Detect the appropriate action for a concept candidate.
        Args:
            candidate: The candidate to analyze
        Returns:
            ConceptDetectionResult with recommended action
        """
        candidate_id = f"{candidate.source_node_type}_{candidate.source_node_id}_{candidate.text[:50]}"
        logger.debug(
            f"Detecting action for concept candidate: {candidate.text}",
            extra={
                "service": "aclarai",
                "filename.function_name": "concept_detection.detector.ConceptDetector.detect_concept_action",
                "candidate_text": candidate.text,
                "candidate_id": candidate_id,
            },
        )
        try:
            # Find similar candidates
            matches = self.find_similar_candidates(candidate)
            # Check if any match exceeds the similarity threshold
            best_match = None
            if matches:
                best_match = max(matches, key=lambda m: m.similarity_score)
            # Make decision based on similarity threshold
            if best_match and best_match.similarity_score >= self.similarity_threshold:
                # Merge with existing similar candidate
                action = ConceptAction.MERGED
                reason = f"Found similar candidate '{best_match.matched_text}' with similarity {best_match.similarity_score:.3f} >= {self.similarity_threshold}"
                confidence = min(best_match.similarity_score, 1.0)
            else:
                # Promote to new canonical concept
                action = ConceptAction.PROMOTED
                if best_match:
                    reason = f"Best similarity {best_match.similarity_score:.3f} < {self.similarity_threshold}, promoting as new concept"
                    confidence = (
                        1.0 - best_match.similarity_score
                    )  # Higher confidence if very different
                else:
                    reason = "No similar candidates found, promoting as new concept"
                    confidence = 1.0
            result = ConceptDetectionResult(
                candidate_id=candidate_id,
                candidate_text=candidate.text,
                action=action,
                similarity_matches=matches,
                confidence=confidence,
                reason=reason,
            )
            logger.debug(
                f"Action determined for '{candidate.text}': {action.value}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_detection.detector.ConceptDetector.detect_concept_action",
                    "candidate_text": candidate.text,
                    "action": action.value,
                    "confidence": confidence,
                },
            )
            return result
        except Exception as e:
            logger.error(
                f"Failed to detect concept action: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_detection.detector.ConceptDetector.detect_concept_action",
                    "error": str(e),
                },
            )
            # Return a fallback result
            return ConceptDetectionResult(
                candidate_id=candidate_id,
                candidate_text=candidate.text,
                action=ConceptAction.PROMOTED,  # Default to promotion if error
                confidence=0.0,
                reason=f"Error during detection: {str(e)}",
            )

    def process_candidates_batch(
        self, candidates: List[NounPhraseCandidate]
    ) -> ConceptDetectionBatch:
        """
        Process a batch of concept candidates.
        Args:
            candidates: List of candidates to process
        Returns:
            ConceptDetectionBatch with results
        """
        start_time = time.time()
        logger.info(
            f"Processing batch of {len(candidates)} concept candidates",
            extra={
                "service": "aclarai",
                "filename.function_name": "concept_detection.detector.ConceptDetector.process_candidates_batch",
                "batch_size": len(candidates),
            },
        )
        try:
            # Ensure index is built
            if self.index is None:
                self.build_index_from_candidates()
            results = []
            merged_count = 0
            promoted_count = 0
            for candidate in candidates:
                result = self.detect_concept_action(candidate)
                results.append(result)
                if result.action == ConceptAction.MERGED:
                    merged_count += 1
                elif result.action == ConceptAction.PROMOTED:
                    promoted_count += 1
            processing_time = time.time() - start_time
            batch_result = ConceptDetectionBatch(
                results=results,
                total_processed=len(candidates),
                merged_count=merged_count,
                promoted_count=promoted_count,
                processing_time=processing_time,
            )
            logger.info(
                f"Processed {len(candidates)} candidates: {merged_count} merged, {promoted_count} promoted",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_detection.detector.ConceptDetector.process_candidates_batch",
                    "total_processed": len(candidates),
                    "merged_count": merged_count,
                    "promoted_count": promoted_count,
                    "processing_time": processing_time,
                },
            )
            return batch_result
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"Failed to process candidates batch: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_detection.detector.ConceptDetector.process_candidates_batch",
                    "error": str(e),
                    "processing_time": processing_time,
                },
            )
            return ConceptDetectionBatch(
                total_processed=len(candidates),
                processing_time=processing_time,
                error=str(e),
            )
