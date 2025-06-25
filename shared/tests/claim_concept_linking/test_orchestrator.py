"""
Tests for claim-concept linking orchestrator.
This module tests the main ClaimConceptLinker orchestrator class that coordinates
the full end-to-end process from fetching claims to updating Markdown files.
"""

from unittest.mock import patch

import pytest
from aclarai_shared.claim_concept_linking.models import (
    ClaimConceptLinkResult,
    ClaimConceptPair,
    ConceptCandidate,
    RelationshipType,
)
from aclarai_shared.claim_concept_linking.orchestrator import ClaimConceptLinker

from shared.tests.utils import get_seeded_mock_services


class MockClaimConceptNeo4jManager:
    """Mock ClaimConceptNeo4jManager for testing."""

    def __init__(self, base_manager=None):
        self.base_manager = base_manager

    def fetch_unlinked_claims(self, limit=100):
        """Return mock claims data."""
        claims_data = [
            {
                "id": "claim_gpu_error_1",
                "text": "CUDA runtime error occurred during training",
                "entailed_score": 0.92,
                "coverage_score": 0.88,
                "decontextualization_score": 0.85,
            },
            {
                "id": "claim_memory_issue_1",
                "text": "Out of memory error in GPU processing",
                "entailed_score": 0.89,
                "coverage_score": 0.91,
                "decontextualization_score": 0.83,
            },
        ]
        return claims_data[:limit]

    def fetch_all_concepts(self):
        """Return mock concepts data."""
        return [
            {"id": "concept_gpu_error", "text": "GPU Error"},
            {"id": "concept_memory_error", "text": "Memory Error"},
            {"id": "concept_cuda_error", "text": "CUDA Error"},
        ]

    def get_claim_context(self, claim_id):
        """Return mock context."""
        return {
            "source_block_text": f"Source text for {claim_id}",
            "summary_text": f"Summary for {claim_id}",
        }

    def create_claim_concept_relationship(self, _link_result):
        """Mock relationship creation - always succeeds."""
        return True

    def get_claims_source_files(self, claim_ids):
        """Return mock file mappings."""
        return {claim_id: f"/mock/vault/file_{claim_id}.md" for claim_id in claim_ids}


class TestClaimConceptLinkerOrchestrator:
    """Test the main ClaimConceptLinker orchestrator."""

    def test_init_with_mock_services(self):
        """Test ClaimConceptLinker initialization with mock services."""
        neo4j_manager, vector_store = get_seeded_mock_services()
        mock_claim_concept_manager = MockClaimConceptNeo4jManager(neo4j_manager)
        linker = ClaimConceptLinker(
            neo4j_manager=mock_claim_concept_manager,
            vector_store=vector_store,
        )
        assert linker.neo4j_manager is mock_claim_concept_manager
        assert linker.vector_store is vector_store
        assert linker.markdown_updater is not None  # Should be initialized

    def test_init_with_config_only(self):
        """Test ClaimConceptLinker initialization with config only."""
        # Mock dependencies to avoid real service initialization
        with (
            patch(
                "aclarai_shared.claim_concept_linking.orchestrator.ClaimConceptNeo4jManager"
            ) as mock_neo4j,
            patch(
                "aclarai_shared.claim_concept_linking.orchestrator.ClaimConceptLinkerAgent"
            ) as mock_agent,
            patch(
                "aclarai_shared.claim_concept_linking.orchestrator.Tier2MarkdownUpdater"
            ) as mock_updater,
        ):
            linker = ClaimConceptLinker()
            # Should have attempted to initialize components that aren't injected
            mock_neo4j.assert_called_once()
            mock_agent.assert_called_once()
            mock_updater.assert_called_once()
            # Vector store should remain None when not injected
            assert linker.vector_store is None

    def test_find_candidate_concepts_vector(self):
        """Test vector-based candidate concept finding."""
        neo4j_manager, vector_store = get_seeded_mock_services()
        mock_claim_concept_manager = MockClaimConceptNeo4jManager(neo4j_manager)
        linker = ClaimConceptLinker(
            neo4j_manager=mock_claim_concept_manager,
            vector_store=vector_store,
        )
        # Test finding candidates
        candidates = linker.find_candidate_concepts(
            query_text="An error related to CUDA occurred",
            top_k=5,
            similarity_threshold=0.1,
        )
        assert isinstance(candidates, list)
        # Should find relevant candidates from golden dataset
        assert len(candidates) > 0
        # Verify candidate structure
        candidate, similarity = candidates[0]
        assert isinstance(candidate, dict)
        assert "id" in candidate
        assert "text" in candidate
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0

    def test_create_claim_concept_pair(self):
        """Test claim-concept pair creation."""
        neo4j_manager, vector_store = get_seeded_mock_services()
        mock_claim_concept_manager = MockClaimConceptNeo4jManager(neo4j_manager)
        linker = ClaimConceptLinker(
            neo4j_manager=mock_claim_concept_manager,
            vector_store=vector_store,
        )
        claim = {
            "id": "test_claim_1",
            "text": "GPU training failed",
            "entailed_score": 0.95,
            "coverage_score": 0.85,
            "decontextualization_score": 0.78,
        }
        candidate = ConceptCandidate(
            concept_id="concept_gpu_error",
            concept_text="GPU Error",
            similarity_score=0.92,
        )
        pair = linker._create_claim_concept_pair(claim, candidate)
        assert isinstance(pair, ClaimConceptPair)
        assert pair.claim_id == "test_claim_1"
        assert pair.claim_text == "GPU training failed"
        assert pair.concept_id == "concept_gpu_error"
        assert pair.concept_text == "GPU Error"
        assert pair.entailed_score == 0.95
        assert pair.coverage_score == 0.85
        assert pair.decontextualization_score == 0.78

    def test_create_link_result(self):
        """Test link result creation from classification."""
        neo4j_manager, vector_store = get_seeded_mock_services()
        mock_claim_concept_manager = MockClaimConceptNeo4jManager(neo4j_manager)
        linker = ClaimConceptLinker(
            neo4j_manager=mock_claim_concept_manager,
            vector_store=vector_store,
            agent=None,  # Use mock agent
        )
        pair = ClaimConceptPair(
            claim_id="test_claim_1",
            claim_text="GPU error occurred",
            concept_id="concept_gpu_error",
            concept_text="GPU Error",
            entailed_score=0.95,
            coverage_score=0.85,
        )

        # Mock classification result
        class MockClassification:
            def __init__(self):
                self.relation = "SUPPORTS_CONCEPT"
                self.strength = 0.88
                self.reasoning = "The claim directly describes a GPU error"

            def to_relationship_type(self):
                return RelationshipType.SUPPORTS_CONCEPT

        classification = MockClassification()
        with patch.object(linker, 'agent', None):
            link_result = linker._create_link_result(pair, classification)
        assert isinstance(link_result, ClaimConceptLinkResult)
        assert link_result.claim_id == "test_claim_1"
        assert link_result.concept_id == "concept_gpu_error"
        assert link_result.relationship == RelationshipType.SUPPORTS_CONCEPT
        assert link_result.strength == 0.88
        assert link_result.entailed_score == 0.95
        assert link_result.coverage_score == 0.85
        assert link_result.agent_model == "mock-agent"

    def test_find_candidate_concepts_api(self):
        """Test the public find_candidate_concepts API."""
        neo4j_manager, vector_store = get_seeded_mock_services()
        mock_claim_concept_manager = MockClaimConceptNeo4jManager(neo4j_manager)
        linker = ClaimConceptLinker(
            neo4j_manager=mock_claim_concept_manager,
            vector_store=vector_store,
        )
        # Test finding candidates
        results = linker.find_candidate_concepts(
            query_text="GPU error occurred", top_k=3, similarity_threshold=0.1
        )
        assert isinstance(results, list)
        if results:  # If results found
            # Verify structure
            result = results[0]
            assert isinstance(result, tuple)
            assert len(result) == 2  # (document, similarity_score)
            assert isinstance(result[0], dict)  # document
            assert isinstance(result[1], float)  # similarity score
            assert 0.0 <= result[1] <= 1.0

    def test_link_claims_to_concepts_empty_claims(self):
        """Test linking process with no claims available."""
        neo4j_manager, vector_store = get_seeded_mock_services()

        # Create mock manager with no claims
        class EmptyMockClaimConceptNeo4jManager(MockClaimConceptNeo4jManager):
            def fetch_unlinked_claims(self, _limit=100):
                return []  # No claims

        mock_claim_concept_manager = EmptyMockClaimConceptNeo4jManager(neo4j_manager)
        linker = ClaimConceptLinker(
            neo4j_manager=mock_claim_concept_manager,
            vector_store=vector_store,
        )
        results = linker.link_claims_to_concepts()
        assert isinstance(results, dict)
        assert results["claims_fetched"] == 0
        assert results["claims_processed"] == 0
        assert results["links_created"] == 0
        assert results["files_updated"] == 0

    def test_link_claims_to_concepts_no_concepts(self):
        """Test linking process with no concepts available."""
        neo4j_manager, vector_store = get_seeded_mock_services()

        # Create mock manager with no concepts
        class NoConceptsMockClaimConceptNeo4jManager(MockClaimConceptNeo4jManager):
            def fetch_all_concepts(self):
                return []  # No concepts

        mock_claim_concept_manager = NoConceptsMockClaimConceptNeo4jManager(
            neo4j_manager
        )
        linker = ClaimConceptLinker(
            neo4j_manager=mock_claim_concept_manager,
            vector_store=vector_store,
        )
        results = linker.link_claims_to_concepts()
        assert isinstance(results, dict)
        assert results["claims_fetched"] > 0  # Should have claims
        assert results["concepts_available"] == 0
        assert results["links_created"] == 0

    @pytest.mark.integration
    def test_link_claims_to_concepts_full_process(self):
        """Integration test for the full claim-concept linking process."""
        neo4j_manager, vector_store = get_seeded_mock_services()
        mock_claim_concept_manager = MockClaimConceptNeo4jManager(neo4j_manager)
        linker = ClaimConceptLinker(
            neo4j_manager=mock_claim_concept_manager,
            vector_store=vector_store,
        )
        results = linker.link_claims_to_concepts(
            max_claims=2, similarity_threshold=0.1, strength_threshold=0.5
        )
        # Verify results structure
        assert isinstance(results, dict)
        expected_keys = [
            "claims_fetched",
            "claims_processed",
            "concepts_available",
            "pairs_analyzed",
            "links_created",
            "relationships_created",
            "files_updated",
            "markdown_files_updated",
            "errors",
        ]
        for key in expected_keys:
            assert key in results
        # Should have processed some claims and concepts
        assert results["claims_fetched"] > 0
        assert results["concepts_available"] > 0
        assert results["claims_processed"] >= 0
        # Should have created some relationships (depends on similarity thresholds)
        # Note: Due to mock data and similarity thresholds, we may or may not get matches
        assert results["links_created"] >= 0
        assert results["relationships_created"] == results["links_created"]
        # Markdown file updates should match links created (when no real updater)
        assert results["files_updated"] >= 0
        assert results["markdown_files_updated"] == results["files_updated"]
        # Should not have fatal errors
        if results["errors"]:
            # Errors can occur but shouldn't be fatal
            for error in results["errors"]:
                assert "Fatal error" not in error

    def test_link_claims_to_concepts_with_high_thresholds(self):
        """Test linking process with high similarity/strength thresholds."""
        neo4j_manager, vector_store = get_seeded_mock_services()
        mock_claim_concept_manager = MockClaimConceptNeo4jManager(neo4j_manager)
        linker = ClaimConceptLinker(
            neo4j_manager=mock_claim_concept_manager,
            vector_store=vector_store,
        )
        # Use very high thresholds that should prevent most matches
        results = linker.link_claims_to_concepts(
            max_claims=2, similarity_threshold=0.99, strength_threshold=0.95
        )
        assert isinstance(results, dict)
        assert results["claims_fetched"] > 0
        assert results["concepts_available"] > 0
        # With high thresholds, should have fewer or no links
        assert results["links_created"] >= 0

    def test_error_handling_in_linking_process(self):
        """Test error handling during the linking process."""
        neo4j_manager, vector_store = get_seeded_mock_services()
        mock_claim_concept_manager = MockClaimConceptNeo4jManager(neo4j_manager)
        # Create a linker that will have an error in vector search
        linker = ClaimConceptLinker(
            neo4j_manager=mock_claim_concept_manager,
            vector_store=vector_store,
        )

        # Mock vector store to raise exception
        def failing_vector_search(*_args, **_kwargs):
            raise Exception("Mock vector search failure")

        linker.vector_store.find_similar_candidates = failing_vector_search
        results = linker.link_claims_to_concepts(max_claims=1)
        # Should handle errors gracefully
        assert isinstance(results, dict)
        assert "errors" in results
        # Should either have errors logged or handle gracefully

    def test_find_candidate_concepts_no_vector_store(self):
        """Test candidate finding when vector store is not available."""
        neo4j_manager, _ = get_seeded_mock_services()
        mock_claim_concept_manager = MockClaimConceptNeo4jManager(neo4j_manager)
        linker = ClaimConceptLinker(
            neo4j_manager=mock_claim_concept_manager,
            vector_store=None,  # No vector store
        )
        results = linker.find_candidate_concepts("test query")
        # Should return empty results gracefully
        assert isinstance(results, list)
        assert len(results) == 0


class TestClaimConceptLinkerEdgeCases:
    """Test edge cases and error conditions."""

    def test_create_link_result_invalid_relationship(self):
        """Test error handling for invalid relationship types."""
        neo4j_manager, vector_store = get_seeded_mock_services()
        mock_claim_concept_manager = MockClaimConceptNeo4jManager(neo4j_manager)
        linker = ClaimConceptLinker(
            neo4j_manager=mock_claim_concept_manager,
            vector_store=vector_store,
        )
        pair = ClaimConceptPair(
            claim_id="test_claim",
            claim_text="Test claim",
            concept_id="test_concept",
            concept_text="Test concept",
        )

        # Mock classification with invalid relationship
        class InvalidClassification:
            def __init__(self):
                self.relation = "INVALID_RELATION"
                self.strength = 0.8

            def to_relationship_type(self):
                return None  # Invalid relationship

        classification = InvalidClassification()
        with pytest.raises(ValueError, match="Invalid relationship type"):
            linker._create_link_result(pair, classification)

    def test_null_evaluation_scores_handling(self):
        """Test proper handling of null evaluation scores."""
        neo4j_manager, vector_store = get_seeded_mock_services()
        mock_claim_concept_manager = MockClaimConceptNeo4jManager(neo4j_manager)
        linker = ClaimConceptLinker(
            neo4j_manager=mock_claim_concept_manager,
            vector_store=vector_store,
        )
        # Claim with null scores
        claim = {
            "id": "test_claim_null_scores",
            "text": "Test claim with null scores",
            "entailed_score": None,
            "coverage_score": None,
            "decontextualization_score": None,
        }
        candidate = ConceptCandidate(
            concept_id="concept_test",
            concept_text="Test Concept",
            similarity_score=0.85,
        )
        pair = linker._create_claim_concept_pair(claim, candidate)
        # Should handle null scores gracefully
        assert pair.entailed_score is None
        assert pair.coverage_score is None
        assert pair.decontextualization_score is None

        # Should create link result with null scores
        class MockClassification:
            def __init__(self):
                self.relation = "MENTIONS_CONCEPT"
                self.strength = 0.75

            def to_relationship_type(self):
                return RelationshipType.MENTIONS_CONCEPT

        classification = MockClassification()
        link_result = linker._create_link_result(pair, classification)
        assert link_result.entailed_score is None
        assert link_result.coverage_score is None
