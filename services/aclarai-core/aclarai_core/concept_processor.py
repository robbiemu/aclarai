"""
Concept processing integration for aclarai-core.
This module integrates concept detection and processing with the main
aclarai-core pipeline, handling noun phrase extraction and concept
candidate management.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from aclarai_shared import load_config
from aclarai_shared.concept_detection import ConceptDetector
from aclarai_shared.concept_detection.models import (
    ConceptAction,
    ConceptDetectionBatch,
)
from aclarai_shared.config import aclaraiConfig
from aclarai_shared.graph import Neo4jGraphManager
from aclarai_shared.graph.models import ConceptInput
from aclarai_shared.noun_phrase_extraction import NounPhraseExtractor
from aclarai_shared.noun_phrase_extraction.concept_candidates_store import (
    ConceptCandidatesVectorStore,
)
from aclarai_shared.tier3_concept import ConceptFileWriter

logger = logging.getLogger(__name__)


class ConceptActionInfo(TypedDict):
    candidate_text: str
    action: str  # Action taken on the candidate
    confidence: float
    reason: str
    best_match: Optional[
        Dict[str, Any]
    ]  # Contains matched text and similarity score if available


class MergedConceptInfo(TypedDict):
    matched_id: Optional[str]  # ID of the concept candidate that was merged
    matched_text: str  # Text of the matched candidate
    similarity_score: float  # Similarity score of the match


class UpdateCandidateStatus(TypedDict):
    candidate_id: str  # ID of the concept candidate
    candidate_text: str  # Text of the candidate
    new_status: str  # New status after processing (e.g., "merged", "promoted")
    confidence: float  # Confidence in the decision
    reason: str  # Reason for the status change
    merged_with_info: Optional[
        MergedConceptInfo
    ]  # Info if merged with another candidate
    concept_id: Optional[str]  # ID of the created Concept node (if promoted)


class ConceptProcessor:
    """
    Processes concept candidates and performs similarity-based concept detection.
    This class integrates noun phrase extraction, concept candidate storage,
    and hnswlib-based similarity detection to determine whether candidates
    should be merged or promoted.
    """

    def __init__(
        self,
        config: Optional[aclaraiConfig] = None,
        noun_phrase_extractor: Optional[NounPhraseExtractor] = None,
        candidates_store: Optional[ConceptCandidatesVectorStore] = None,
        concept_detector: Optional[ConceptDetector] = None,
        concept_file_writer: Optional[ConceptFileWriter] = None,
        neo4j_manager: Optional[Neo4jGraphManager] = None,
    ):
        """
        Initialize the concept processor.
        Args:
            config: aclarai configuration (loads default if None)
            noun_phrase_extractor: Noun phrase extractor (creates default if None)
            candidates_store: Concept candidates vector store (creates default if None)
            concept_detector: Concept detector (creates default if None)
            concept_file_writer: Concept file writer (creates default if None)
            neo4j_manager: Neo4j graph manager (creates default if None)
        """
        self.config = config or load_config(validate=False)
        # Use injected dependencies or create real ones if not provided
        self.noun_phrase_extractor = noun_phrase_extractor or NounPhraseExtractor(
            self.config
        )
        self.candidates_store = candidates_store or ConceptCandidatesVectorStore(
            self.config
        )
        self.concept_detector = concept_detector or ConceptDetector(self.config)
        self.concept_file_writer = concept_file_writer or ConceptFileWriter(self.config)
        self.neo4j_manager = neo4j_manager or Neo4jGraphManager(self.config)
        logger.info(
            "Initialized ConceptProcessor",
            extra={
                "service": "aclarai-core",
                "filename.function_name": "concept_processor.ConceptProcessor.__init__",
            },
        )

    def process_block_for_concepts(
        self, block: Dict[str, Any], block_type: str = "claim"
    ) -> Dict[str, Any]:
        """
        Process a block to extract and analyze concept candidates.
        Args:
            block: The block dictionary with aclarai_id, semantic_text, etc.
            block_type: Type of block ("claim" or "summary")
        Returns:
            Dictionary with processing results and recommendations
        """
        aclarai_id = block.get("aclarai_id", "")
        semantic_text = block.get("semantic_text", "")
        logger.debug(
            f"Processing block for concepts: {aclarai_id}",
            extra={
                "service": "aclarai-core",
                "filename.function_name": "concept_processor.ConceptProcessor.process_block_for_concepts",
                "aclarai_id": aclarai_id,
                "block_type": block_type,
                "text_length": len(semantic_text),
            },
        )
        try:
            # Step 1: Extract noun phrases from the block
            extraction_result = self.noun_phrase_extractor._extract_from_node(
                text=semantic_text,
                source_node_id=aclarai_id,
                source_node_type=block_type,
                aclarai_id=aclarai_id,
            )
            if not extraction_result.is_successful or not extraction_result.candidates:
                logger.debug(
                    f"No noun phrases extracted from block {aclarai_id}",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "concept_processor.ConceptProcessor.process_block_for_concepts",
                        "aclarai_id": aclarai_id,
                        "extraction_error": extraction_result.error,
                    },
                )
                return {
                    "success": True,
                    "aclarai_id": aclarai_id,
                    "candidates_extracted": 0,
                    "candidates_stored": 0,
                    "concept_actions": [],
                    "message": "No noun phrases extracted",
                }
            # Step 2: Store candidates in vector store
            stored_count = self.candidates_store.store_candidates(
                extraction_result.candidates
            )
            # Step 3: Perform concept detection on the extracted candidates
            detection_batch = self.concept_detector.process_candidates_batch(
                extraction_result.candidates
            )
            # Build candidate metadata mapping for efficient lookup
            candidate_metadata_map = {}
            for candidate in extraction_result.candidates:
                candidate_id = f"{candidate.source_node_type}_{candidate.source_node_id}_{candidate.text[:50]}"
                candidate_metadata_map[candidate_id] = {
                    "source_node_id": candidate.source_node_id,
                    "source_node_type": candidate.source_node_type,
                    "aclarai_id": candidate.aclarai_id,
                    "text": candidate.text,
                }
            # Step 4: Update candidate statuses based on detection results
            updated_candidates = self._update_candidate_statuses(
                detection_batch, candidate_metadata_map
            )
            # Prepare results summary
            results = {
                "success": True,
                "aclarai_id": aclarai_id,
                "block_type": block_type,
                "candidates_extracted": len(extraction_result.candidates),
                "candidates_stored": stored_count,
                "concept_actions": [],
                "merged_count": detection_batch.merged_count,
                "promoted_count": detection_batch.promoted_count,
                "processing_time": detection_batch.processing_time,
                "updated_candidates": updated_candidates,
            }
            # Add concept action recommendations
            for result in detection_batch.results:
                action_info: ConceptActionInfo = {
                    "candidate_text": result.candidate_text,
                    "action": result.action.value,
                    "confidence": result.confidence,
                    "reason": result.reason,
                    "best_match": None,
                }
                if result.best_match:
                    action_info["best_match"] = {
                        "matched_text": result.best_match.matched_text,
                        "similarity_score": result.best_match.similarity_score,
                        "matched_candidate_id": result.best_match.matched_candidate_id,
                    }
                results["concept_actions"].append(action_info)
            logger.info(
                f"Processed {len(extraction_result.candidates)} concept candidates from block {aclarai_id}: "
                f"{detection_batch.merged_count} merged, {detection_batch.promoted_count} promoted",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "concept_processor.ConceptProcessor.process_block_for_concepts",
                    "aclarai_id": aclarai_id,
                    "candidates_extracted": len(extraction_result.candidates),
                    "merged_count": detection_batch.merged_count,
                    "promoted_count": detection_batch.promoted_count,
                },
            )
            return results
        except Exception as e:
            logger.error(
                f"Error processing block for concepts: {e}",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "concept_processor.ConceptProcessor.process_block_for_concepts",
                    "aclarai_id": aclarai_id,
                    "error": str(e),
                },
            )
            return {
                "success": False,
                "aclarai_id": aclarai_id,
                "error": str(e),
                "candidates_extracted": 0,
                "candidates_stored": 0,
                "concept_actions": [],
            }

    def _update_candidate_statuses(
        self,
        detection_batch: ConceptDetectionBatch,
        candidate_metadata_map: Dict[str, Dict[str, Any]],
    ) -> List[UpdateCandidateStatus]:
        """
        Update the status of candidates based on detection results and create Concept nodes for promoted candidates.
        Args:
            detection_batch: Results from concept detection
            candidate_metadata_map: Mapping of candidate IDs to their metadata for efficient lookup
        Returns:
            List of candidate updates that were applied
        """
        updates: List[UpdateCandidateStatus] = []
        promoted_concepts = []
        for result in detection_batch.results:
            update: UpdateCandidateStatus = {
                "candidate_id": result.candidate_id,
                "candidate_text": result.candidate_text,
                "new_status": result.action.value,
                "confidence": result.confidence,
                "reason": result.reason,
                "merged_with_info": None,
                "concept_id": None,
            }
            # Update candidate status in vector store
            metadata_updates = {
                "confidence": result.confidence,
                "reason": result.reason,
                "updated_at": datetime.now().isoformat(),
            }
            if result.action == ConceptAction.MERGED and result.best_match:
                update["merged_with_info"] = {
                    "matched_id": result.best_match.matched_candidate_id,
                    "matched_text": result.best_match.matched_text,
                    "similarity_score": result.best_match.similarity_score,
                }
                metadata_updates["merged_with_id"] = (
                    result.best_match.matched_candidate_id
                )
                metadata_updates["similarity_score"] = (
                    result.best_match.similarity_score
                )
            # Update the candidate status in the vector store
            success = self.candidates_store.update_candidate_status(
                result.candidate_id, result.action.value, metadata_updates
            )
            if success:
                updates.append(update)
                # If promoted, prepare for Concept node creation
                if result.action == ConceptAction.PROMOTED:
                    # Get candidate details for Concept creation from metadata map
                    candidate_metadata = candidate_metadata_map.get(result.candidate_id)
                    if candidate_metadata:
                        concept_input = ConceptInput(
                            text=result.candidate_text,
                            source_candidate_id=result.candidate_id,
                            source_node_id=candidate_metadata.get("source_node_id", ""),
                            source_node_type=candidate_metadata.get(
                                "source_node_type", ""
                            ),
                            aclarai_id=candidate_metadata.get("aclarai_id", ""),
                        )
                        promoted_concepts.append(concept_input)
                    else:
                        logger.warning(
                            f"Missing metadata for promoted candidate {result.candidate_id}",
                            extra={
                                "service": "aclarai-core",
                                "filename.function_name": "concept_processor.ConceptProcessor._update_candidate_statuses",
                                "candidate_id": result.candidate_id,
                            },
                        )
            else:
                logger.warning(
                    f"Failed to update candidate status for {result.candidate_id}",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "concept_processor.ConceptProcessor._update_candidate_statuses",
                        "candidate_id": result.candidate_id,
                        "action": result.action.value,
                    },
                )
        # Create Concept nodes for promoted candidates
        if promoted_concepts:
            try:
                created_concepts = self.neo4j_manager.create_concepts(promoted_concepts)
                logger.info(
                    f"Created {len(created_concepts)} Concept nodes from promoted candidates",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "concept_processor.ConceptProcessor._update_candidate_statuses",
                        "concepts_created": len(created_concepts),
                    },
                )
                # Write Tier 3 Markdown files for created concepts
                files_written = 0
                file_write_errors = []
                for concept in created_concepts:
                    try:
                        if self.concept_file_writer.write_concept_file(concept):
                            files_written += 1
                        else:
                            file_write_errors.append(
                                f"Failed to write Tier 3 file for concept {concept.concept_id}: write_concept_file returned False"
                            )
                    except Exception as e:
                        file_write_errors.append(
                            f"Error writing Tier 3 file for concept {concept.concept_id}: {e}"
                        )
                        logger.error(
                            f"Failed to write Tier 3 file for concept {concept.concept_id}: {e}",
                            extra={
                                "service": "aclarai-core",
                                "filename.function_name": "concept_processor.ConceptProcessor._update_candidate_statuses",
                                "concept_id": concept.concept_id,
                                "error": str(e),
                            },
                        )
                if file_write_errors:
                    logger.warning(
                        f"Some Tier 3 file writes failed: {len(file_write_errors)} errors",
                        extra={
                            "service": "aclarai-core",
                            "filename.function_name": "concept_processor.ConceptProcessor._update_candidate_statuses",
                            "file_write_errors": len(file_write_errors),
                            "files_written": files_written,
                            "concepts_created": len(created_concepts),
                        },
                    )
                logger.info(
                    f"Created {files_written} Tier 3 Markdown files for promoted concepts",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "concept_processor.ConceptProcessor._update_candidate_statuses",
                        "files_written": files_written,
                        "concepts_created": len(created_concepts),
                    },
                )
                # Update the results to include concept creation info
                for i, update in enumerate(updates):
                    if update["new_status"] == "promoted" and i < len(created_concepts):
                        update["concept_id"] = created_concepts[i].concept_id
            except Exception as e:
                logger.error(
                    f"Failed to create Concept nodes for promoted candidates: {e}",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "concept_processor.ConceptProcessor._update_candidate_statuses",
                        "error": str(e),
                        "promoted_count": len(promoted_concepts),
                    },
                )
        logger.info(
            f"Applied {len(updates)} candidate status updates",
            extra={
                "service": "aclarai-core",
                "filename.function_name": "concept_processor.ConceptProcessor._update_candidate_statuses",
                "updates_count": len(updates),
                "promoted_concepts": len(promoted_concepts),
            },
        )
        return updates

    def build_concept_index(self, force_rebuild: bool = False) -> int:
        """
        Build or rebuild the concept detection index.
        Args:
            force_rebuild: Whether to force rebuild even if index exists
        Returns:
            Number of items added to the index
        """
        logger.info(
            "Building concept detection index",
            extra={
                "service": "aclarai-core",
                "filename.function_name": "concept_processor.ConceptProcessor.build_concept_index",
                "force_rebuild": force_rebuild,
            },
        )
        try:
            items_added = self.concept_detector.build_index_from_candidates(
                force_rebuild
            )
            logger.info(
                f"Built concept detection index with {items_added} items",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "concept_processor.ConceptProcessor.build_concept_index",
                    "items_added": items_added,
                },
            )
            return items_added
        except Exception as e:
            logger.error(
                f"Error building concept index: {e}",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "concept_processor.ConceptProcessor.build_concept_index",
                    "error": str(e),
                },
            )
            raise

    def get_concept_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about concept candidates and detection.
        Returns:
            Dictionary with various statistics
        """
        try:
            # Get candidates by status
            pending_candidates = self.candidates_store.get_candidates_by_status(
                "pending"
            )
            merged_candidates = self.candidates_store.get_candidates_by_status("merged")
            promoted_candidates = self.candidates_store.get_candidates_by_status(
                "promoted"
            )
            stats = {
                "total_candidates": len(pending_candidates)
                + len(merged_candidates)
                + len(promoted_candidates),
                "pending_candidates": len(pending_candidates),
                "merged_candidates": len(merged_candidates),
                "promoted_candidates": len(promoted_candidates),
                "index_size": len(self.concept_detector.id_to_metadata),
                "similarity_threshold": self.concept_detector.similarity_threshold,
            }
            logger.debug(
                f"Concept statistics: {stats}",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "concept_processor.ConceptProcessor.get_concept_statistics",
                    **stats,
                },
            )
            return stats
        except Exception as e:
            logger.error(
                f"Error getting concept statistics: {e}",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "concept_processor.ConceptProcessor.get_concept_statistics",
                    "error": str(e),
                },
            )
            return {
                "error": str(e),
                "total_candidates": 0,
                "pending_candidates": 0,
                "merged_candidates": 0,
                "promoted_candidates": 0,
                "index_size": 0,
            }
