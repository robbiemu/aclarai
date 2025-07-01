"""
Tests for mock services used in claim-concept linking development.
This module tests the MockNeo4jGraphManager, MockVectorStore, and seeding utilities
to ensure they provide a stable environment for claim-concept linking development.
"""

from aclarai_shared.claim_concept_linking.orchestrator import ClaimConceptLinker
from aclarai_shared.embedding.storage import VectorStoreQuery
from aclarai_shared.graph.models import ClaimInput, ConceptInput
from aclarai_shared.noun_phrase_extraction.models import NounPhraseCandidate


class TestMockServices:
    """Test cases for mock services."""

    def test_mock_neo4j_manager_basic_operations(self):
        """Test basic operations of MockNeo4jGraphManager."""
        from shared.tests.mocks import MockNeo4jGraphManager

        manager = MockNeo4jGraphManager()
        # Test initial state
        assert len(manager.claims) == 0
        assert len(manager.concepts) == 0
        # Test creating concepts
        concept_inputs = [
            ConceptInput(
                text="Machine Learning",
                source_candidate_id="candidate_1",
                source_node_id="source_1",
                source_node_type="Summary",
                aclarai_id="aclarai_1",
            )
        ]
        concepts = manager.create_concepts(concept_inputs)
        assert len(concepts) == 1
        assert concepts[0].text == "Machine Learning"
        assert len(manager.concepts) == 1
        # Test creating claims
        claim_inputs = [
            ClaimInput(
                text="GPU memory issues are common in deep learning",
                block_id="block_1",
            )
        ]
        claims = manager.create_claims(claim_inputs)
        assert len(claims) == 1
        assert claims[0].text == "GPU memory issues are common in deep learning"
        assert len(manager.claims) == 1
        # Test count nodes
        counts = manager.count_nodes()
        assert counts["Claim"] == 1
        assert counts["Concept"] == 1

    def test_mock_vector_store_basic_operations(self):
        """Test basic operations of MockVectorStore."""
        from shared.tests.mocks import MockVectorStore

        store = MockVectorStore()
        # Test initial state
        assert len(store.documents) == 0
        assert len(store.embeddings) == 0
        # Test storing candidates
        candidates = [
            NounPhraseCandidate(
                text="Machine Learning",
                normalized_text="machine learning",
                source_node_id="source_1",
                source_node_type="Summary",
                aclarai_id="aclarai_1",
            ),
            NounPhraseCandidate(
                text="Deep Learning",
                normalized_text="deep learning",
                source_node_id="source_2",
                source_node_type="Summary",
                aclarai_id="aclarai_2",
            ),
        ]
        stored_count = store.store_candidates(candidates)
        assert stored_count == 2
        assert len(store.documents) == 2
        assert len(store.embeddings) == 2
        # Test similarity search
        query = VectorStoreQuery(
            query_str="machine learning",
            similarity_top_k=5,
        )
        results = store.query(query)
        assert len(results.nodes) > 0
        # Should find the exact match with high similarity
        best_match_node = results.nodes[0]
        assert best_match_node.get_content() == "machine learning"
        assert results.similarities[0] > 0.8  # High similarity for exact match

    def test_seeded_mock_services(self):
        """Test the seeded mock services utility."""
        from shared.tests.utils import get_seeded_mock_services

        neo4j_manager, vector_store = get_seeded_mock_services()
        # Test that services are populated with golden data
        assert len(neo4j_manager.concepts) > 0
        assert len(neo4j_manager.claims) > 0
        assert len(vector_store.documents) > 0
        assert len(vector_store.embeddings) == len(vector_store.documents)
        # Test that we can find similar concepts
        query = VectorStoreQuery(
            query_str="machine learning",
            similarity_top_k=3,
        )
        results = vector_store.query(query)
        assert len(results.nodes) > 0
        # Test that concepts exist in Neo4j
        counts = neo4j_manager.count_nodes()
        assert counts["Concept"] > 0
        assert counts["Claim"] > 0

    def test_claim_concept_linker_with_mock_services(self):
        """Test ClaimConceptLinker can be initialized with mock services."""
        from shared.tests.utils import get_seeded_mock_services

        neo4j_manager, vector_store = get_seeded_mock_services()
        # Test that ClaimConceptLinker accepts mock services
        linker = ClaimConceptLinker(
            neo4j_manager=neo4j_manager,
            vector_store=vector_store,
        )
        assert linker.neo4j_manager is neo4j_manager
        assert linker.vector_store is vector_store
        # Test that the linker can perform basic operations
        results = linker.link_claims_to_concepts()
        assert isinstance(results, dict)
        assert "claims_processed" in results
        assert "relationships_created" in results
        assert "files_updated" in results

    def test_claim_concept_linker_find_candidate_concepts(self):
        """Test ClaimConceptLinker can find candidate concepts using mock vector store."""
        from shared.tests.utils import get_seeded_mock_services

        neo4j_manager, vector_store = get_seeded_mock_services()
        linker = ClaimConceptLinker(
            neo4j_manager=neo4j_manager,
            vector_store=vector_store,
        )
        # Test finding candidate concepts
        candidates = linker.find_candidate_concepts("GPU error occurred", top_k=3)
        assert isinstance(candidates, list)
        # Should find relevant concepts from the golden dataset
        if candidates:
            # Verify structure of returned candidates
            candidate = candidates[0]
            assert isinstance(candidate, tuple)
            assert len(candidate) == 2  # (document, similarity_score)
            assert isinstance(candidate[0], dict)  # document
            assert isinstance(candidate[1], float)  # similarity score
            assert 0.0 <= candidate[1] <= 1.0  # valid similarity range

    def test_mock_services_isolation(self):
        """Test that mock services provide isolated environments."""
        from shared.tests.mocks import MockNeo4jGraphManager, MockVectorStore

        # Create two separate instances
        manager1 = MockNeo4jGraphManager()
        manager2 = MockNeo4jGraphManager()
        store1 = MockVectorStore()
        store2 = MockVectorStore()
        # Add data to first instances
        concept_input = ConceptInput(
            text="Test Concept",
            source_candidate_id="candidate_1",
            source_node_id="source_1",
            source_node_type="Summary",
            aclarai_id="aclarai_1",
        )
        manager1.create_concepts([concept_input])
        candidate = NounPhraseCandidate(
            text="Test Candidate",
            normalized_text="test candidate",
            source_node_id="source_1",
            source_node_type="Summary",
            aclarai_id="aclarai_1",
        )
        store1.store_candidates([candidate])
        # Verify second instances are still empty
        assert len(manager1.concepts) == 1
        assert len(manager2.concepts) == 0
        assert len(store1.documents) == 1
        assert len(store2.documents) == 0

    def test_mock_services_clear_data(self):
        """Test that mock services can be cleared for test isolation."""
        from shared.tests.mocks import MockNeo4jGraphManager, MockVectorStore

        manager = MockNeo4jGraphManager()
        store = MockVectorStore()
        # Add some data
        concept_input = ConceptInput(
            text="Test Concept",
            source_candidate_id="candidate_1",
            source_node_id="source_1",
            source_node_type="Summary",
            aclarai_id="aclarai_1",
        )
        manager.create_concepts([concept_input])
        candidate = NounPhraseCandidate(
            text="Test Candidate",
            normalized_text="test candidate",
            source_node_id="source_1",
            source_node_type="Summary",
            aclarai_id="aclarai_1",
        )
        store.store_candidates([candidate])
        # Verify data exists
        assert len(manager.concepts) == 1
        assert len(store.documents) == 1
        # Clear data
        manager.clear_all_data()
        store.clear_all_data()
        # Verify data is cleared
        assert len(manager.concepts) == 0
        assert len(store.documents) == 0
