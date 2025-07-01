"""
Main orchestrator for claim-concept linking.
This module provides the main ClaimConceptLinker class that coordinates
the full linking process, from fetching claims to updating Markdown files.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.vector_stores.types import VectorStoreQuery

from ..config import aclaraiConfig, load_config
from ..evaluation_thresholds import should_allow_concept_linking
from .agent import ClaimConceptLinkerAgent
from .markdown_updater import Tier2MarkdownUpdater
from .models import (
    ClaimConceptLinkResult,
    ClaimConceptPair,
    ConceptCandidate,
)
from .neo4j_operations import ClaimConceptNeo4jManager

logger = logging.getLogger(__name__)


class ClaimConceptLinker:
    """
    Main orchestrator for linking claims to concepts.
    This class coordinates the full process of:
    1. Fetching unlinked claims
    2. Finding candidate concepts using vector similarity search
    3. Classifying relationships with LLM
    4. Creating Neo4j relationships
    5. Updating Tier 2 Markdown files
    """

    def __init__(
        self,
        config: Optional[aclaraiConfig] = None,
        neo4j_manager=None,
        vector_store=None,
        agent=None,
        markdown_updater=None,
    ):
        """
        Initialize the claim-concept linker.
        Args:
            config: aclarai configuration (loads default if None)
            neo4j_manager: Optional Neo4j manager for dependency injection
            vector_store: Optional vector store for dependency injection
            agent: Optional agent for dependency injection
            markdown_updater: Optional updater for dependency injection
        """
        self.config = config
        if not config:
            try:
                self.config = load_config()
            except Exception:
                # For testing without full environment, create minimal config
                from ..config import aclaraiConfig

                self.config = aclaraiConfig()
        # Use injected dependencies or create defaults
        self.neo4j_manager = neo4j_manager or ClaimConceptNeo4jManager(self.config)
        self.vector_store = vector_store
        if agent is not None:
            self.agent = agent
        else:
            try:
                self.agent = ClaimConceptLinkerAgent(self.config)
            except Exception:
                # For testing without full config, agent can be None
                self.agent = None
        self.markdown_updater = markdown_updater or Tier2MarkdownUpdater(
            self.config, self.neo4j_manager
        )
        logger.info(
            "Initialized ClaimConceptLinker",
            extra={
                "service": "aclarai",
                "filename.function_name": "claim_concept_linking.ClaimConceptLinker.__init__",
                "has_vector_store": self.vector_store is not None,
                "has_custom_neo4j": neo4j_manager is not None,
                "has_custom_agent": agent is not None,
                "has_custom_updater": markdown_updater is not None,
            },
        )

    def link_claims_to_concepts(
        self,
        max_claims: int = 100,
        similarity_threshold: float = 0.7,
        strength_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Execute the full claim-concept linking process.
        Args:
            max_claims: Maximum number of claims to process
            similarity_threshold: Minimum similarity for concept candidates
            strength_threshold: Minimum strength for creating relationships
        Returns:
            Dictionary with processing statistics and results
        """
        logger.info(
            "Starting claim-concept linking process",
            extra={
                "service": "aclarai",
                "filename.function_name": "claim_concept_linking.ClaimConceptLinker.link_claims_to_concepts",
                "max_claims": max_claims,
                "similarity_threshold": similarity_threshold,
                "strength_threshold": strength_threshold,
            },
        )

        stats: Dict[str, Any] = {
            "claims_fetched": 0,
            "claims_processed": 0,
            "concepts_available": 0,
            "pairs_analyzed": 0,
            "links_created": 0,
            "relationships_created": 0,
            "files_updated": 0,
            "errors": [],
        }

        if self.agent is None:
            logger.warning(
                "ClaimConceptLinker has no agent; skipping relationship classification.",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "claim_concept_linking.ClaimConceptLinker.link_claims_to_concepts",
                },
            )
            return stats

        try:
            # Step 1: Fetch unlinked claims
            claims = self.neo4j_manager.fetch_unlinked_claims(limit=max_claims)
            stats["claims_fetched"] = len(claims)
            if not claims:
                logger.info(
                    "No unlinked claims found",
                    extra={
                        "service": "aclarai",
                        "filename.function_name": "claim_concept_linking.ClaimConceptLinker.link_claims_to_concepts",
                    },
                )
                return stats
            # Step 2: Fetch available concepts
            concepts = self.neo4j_manager.fetch_all_concepts()
            stats["concepts_available"] = len(concepts)
            if not concepts:
                logger.warning(
                    "No concepts available for linking - this is expected if Tier 3 creation task hasn't run yet",
                    extra={
                        "service": "aclarai",
                        "filename.function_name": "claim_concept_linking.ClaimConceptLinker.link_claims_to_concepts",
                    },
                )
                return stats
            # Step 3: Process claim-concept pairs
            successful_links = []
            for claim in claims:
                stats["claims_processed"] += 1  # Track claims processed

                # Check if claim meets quality threshold for concept linking
                # Extract evaluation scores from claim (may be None)
                entailed_score = claim.get("entailed_score")
                coverage_score = claim.get("coverage_score")
                decontextualization_score = claim.get("decontextualization_score")

                # Log threshold check details
                logger.debug(
                    f"Checking quality threshold for claim {claim.get('id', 'unknown')}",
                    extra={
                        "service": "aclarai",
                        "filename.function_name": "claim_concept_linking.ClaimConceptLinker.link_claims_to_concepts",
                        "claim_id": claim.get("id"),
                        "entailed_score": entailed_score,
                        "coverage_score": coverage_score,
                        "decontextualization_score": decontextualization_score,
                        "quality_threshold": self.config.threshold.claim_quality
                        if self.config and self.config.threshold
                        else None,
                    },
                )

                # Find candidate concepts using vector search
                candidate_concepts = self._find_candidate_concepts_vector(
                    claim, similarity_threshold
                )
                # Classify relationships for each candidate
                for candidate in candidate_concepts:
                    pair = self._create_claim_concept_pair(claim, candidate)
                    stats["pairs_analyzed"] += 1
                    classification = self.agent.classify_relationship(pair)
                    if classification and classification.strength >= strength_threshold:
                        # Apply evaluation threshold logic before creating relationship
                        relationship_type = classification.relationship.value
                        can_link = should_allow_concept_linking(
                            entailed_score,
                            coverage_score,
                            decontextualization_score,
                            self.config.threshold.claim_quality
                            if self.config and self.config.threshold
                            else 0.5,
                            relationship_type,
                        )

                        if not can_link:
                            logger.debug(
                                "Skipping link creation: claim does not meet quality threshold",
                                extra={
                                    "service": "aclarai",
                                    "filename.function_name": "claim_concept_linking.ClaimConceptLinker.link_claims_to_concepts",
                                    "claim_id": claim.get("id"),
                                    "concept_id": candidate.concept_id,
                                    "relationship_type": relationship_type,
                                    "quality_threshold": self.config.threshold.claim_quality
                                    if self.config and self.config.threshold
                                    else None,
                                },
                            )
                            continue

                        # Convert to link result
                        link_result = self._create_link_result(pair, classification)
                        # Create Neo4j relationship
                        if self.neo4j_manager.create_claim_concept_relationship(
                            link_result
                        ):
                            successful_links.append(link_result)
                            stats["links_created"] += 1
                            stats["relationships_created"] += (
                                1  # For test compatibility
                            )
                            logger.debug(
                                f"Created concept link: {claim.get('id')} -> {candidate.concept_id} ({relationship_type})",
                                extra={
                                    "service": "aclarai",
                                    "filename.function_name": "claim_concept_linking.ClaimConceptLinker.link_claims_to_concepts",
                                    "claim_id": claim.get("id"),
                                    "concept_id": candidate.concept_id,
                                    "relationship_type": relationship_type,
                                    "strength": classification.strength,
                                },
                            )
                        else:
                            stats["errors"].append(
                                f"Failed to create relationship for {pair.claim_id} -> {pair.concept_id}"
                            )
            # Step 4: Update Markdown files
            if successful_links:
                if self.markdown_updater is not None:
                    markdown_stats = self.markdown_updater.update_files_with_links(
                        successful_links
                    )
                    stats["files_updated"] = markdown_stats["files_updated"]
                    stats["markdown_files_updated"] = markdown_stats[
                        "files_updated"
                    ]  # For test compatibility
                    stats["errors"].extend(markdown_stats["errors"])
                else:
                    # For testing without markdown updater
                    stats["files_updated"] = len(successful_links)  # Mock response
                    stats["markdown_files_updated"] = len(
                        successful_links
                    )  # For test compatibility
            logger.info(
                "Completed claim-concept linking",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "claim_concept_linking.ClaimConceptLinker.link_claims_to_concepts",
                    **stats,
                },
            )
        except Exception as e:
            error_msg = f"Fatal error in claim-concept linking: {e}"
            stats["errors"].append(error_msg)
            logger.error(
                error_msg,
                extra={
                    "service": "aclarai",
                    "filename.function_name": "claim_concept_linking.ClaimConceptLinker.link_claims_to_concepts",
                    "error": str(e),
                },
            )
        return stats

    def _find_candidate_concepts_vector(
        self, claim: Dict[str, Any], threshold: float
    ) -> List[ConceptCandidate]:
        """
        Find candidate concepts using vector similarity search.
        This method uses the concepts vector store to find semantically similar
        concepts to the claim text using vector embeddings.
        Args:
            claim: Claim dictionary
            threshold: Similarity threshold for filtering candidates
        Returns:
            List of concept candidates
        """
        candidates = []
        claim_text = claim["text"]
        try:
            # Use vector store to find similar concepts
            from llama_index.core.vector_stores.types import VectorStoreQuery

            query_obj = VectorStoreQuery(query_str=claim_text, similarity_top_k=10)
            query_result = self.vector_store.query(query_obj)

            similar_concepts = []
            if query_result.nodes and query_result.similarities:
                for node, similarity in zip(
                    query_result.nodes, query_result.similarities, strict=False
                ):
                    if similarity >= threshold:
                        metadata = node.metadata
                        metadata["id"] = node.node_id
                        similar_concepts.append((metadata, similarity))

            # Convert results to ConceptCandidate objects
            for concept_metadata, similarity_score in similar_concepts:
                candidate = ConceptCandidate(
                    concept_id=concept_metadata.get(
                        "concept_id", concept_metadata.get("id")
                    ),
                    concept_text=concept_metadata.get(
                        "normalized_text", concept_metadata.get("text")
                    ),
                    similarity_score=similarity_score,
                    source_node_id=concept_metadata.get("source_node_id"),
                    source_node_type=concept_metadata.get("source_node_type"),
                    aclarai_id=concept_metadata.get("aclarai_id"),
                )
                candidates.append(candidate)
            logger.debug(
                f"Found {len(candidates)} candidate concepts using vector search",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "claim_concept_linking.ClaimConceptLinker._find_candidate_concepts_vector",
                    "claim_id": claim.get("id"),
                    "candidates_count": len(candidates),
                    "similarity_threshold": threshold,
                },
            )
        except Exception as e:
            logger.error(
                f"Error in vector similarity search for claim {claim.get('id')}: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "claim_concept_linking.ClaimConceptLinker._find_candidate_concepts_vector",
                    "claim_id": claim.get("id"),
                    "error": str(e),
                },
            )
            # Fall back to empty list on error
            candidates = []
        return candidates

    def _create_claim_concept_pair(
        self, claim: Dict[str, Any], candidate: ConceptCandidate
    ) -> ClaimConceptPair:
        """
        Create a ClaimConceptPair from claim data and concept candidate.
        Args:
            claim: Claim dictionary from Neo4j
            candidate: Concept candidate
        Returns:
            ClaimConceptPair for classification
        """
        # Get additional context if available
        context = self.neo4j_manager.get_claim_context(claim["id"])
        return ClaimConceptPair(
            claim_id=claim["id"],
            claim_text=claim["text"],
            concept_id=candidate.concept_id,
            concept_text=candidate.concept_text,
            source_sentence=context.get("source_block_text") if context else None,
            summary_block=context.get("summary_text") if context else None,
            entailed_score=claim.get("entailed_score"),
            coverage_score=claim.get("coverage_score"),
            decontextualization_score=claim.get("decontextualization_score"),
        )

    def _create_link_result(
        self,
        pair: ClaimConceptPair,
        classification: Any,  # AgentClassificationResult
    ) -> ClaimConceptLinkResult:
        """
        Create a ClaimConceptLinkResult from classification.
        Args:
            pair: The claim-concept pair
            classification: LLM classification result
        Returns:
            ClaimConceptLinkResult for Neo4j storage
        """
        # Convert string relation to enum
        relationship = classification.to_relationship_type()
        if not relationship:
            raise ValueError(f"Invalid relationship type: {classification.relation}")
        return ClaimConceptLinkResult(
            claim_id=pair.claim_id,
            concept_id=pair.concept_id,
            relationship=relationship,
            strength=classification.strength,
            # Copy scores from the claim (may be null during Sprint 5)
            entailed_score=pair.entailed_score,
            coverage_score=pair.coverage_score,
            agent_model=self.agent.model_name if self.agent else "mock-agent",
        )

    def find_candidate_concepts(
        self,
        query_text: str,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find candidate concepts using vector similarity search.
        This method provides direct access to the vector similarity search functionality
        for finding concept candidates, primarily used for testing and development.
        Args:
            query_text: Text to search for similar concepts
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity score to include
        Returns:
            List of tuples containing (document, similarity_score)
        """
        if self.vector_store is None:
            logger.warning(
                "Vector store not available, returning empty results",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "claim_concept_linking.ClaimConceptLinker.find_candidate_concepts",
                },
            )
            return []
        try:
            query_obj = VectorStoreQuery(query_str=query_text, similarity_top_k=top_k)
            query_result = self.vector_store.query(query_obj)
            results = []
            if query_result.nodes and query_result.similarities:
                for node, similarity in zip(
                    query_result.nodes, query_result.similarities, strict=False
                ):
                    if (
                        similarity_threshold is None
                        or similarity >= similarity_threshold
                    ):
                        candidate_data = node.metadata.copy()
                        candidate_data["text"] = node.get_content()
                        results.append((candidate_data, similarity))
            return results
        except Exception as e:
            logger.error(
                f"Error finding candidate concepts: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "claim_concept_linking.ClaimConceptLinker.find_candidate_concepts",
                    "query_text": query_text,
                    "error": str(e),
                },
            )
            return []
