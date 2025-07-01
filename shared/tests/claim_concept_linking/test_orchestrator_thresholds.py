"""
Tests for evaluation threshold integration in claim-concept linking.
"""

import pytest
from unittest.mock import Mock, patch
from aclarai_shared.config import aclaraiConfig, ThresholdConfig
from aclarai_shared.claim_concept_linking.orchestrator import ClaimConceptLinker
from aclarai_shared.claim_concept_linking.models import (
    RelationshipType,
    AgentClassificationResult,
    ClaimConceptLinkResult,
)


class TestClaimConceptLinkingThresholds:
    """Test integration of evaluation thresholds in claim-concept linking."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock config with threshold
        self.config = Mock(spec=aclaraiConfig)
        self.config.threshold = ThresholdConfig(claim_quality=0.7)
        
        # Create mocks for dependencies
        self.neo4j_manager = Mock()
        self.vector_store = Mock()
        self.agent = Mock()
        self.markdown_updater = Mock()
        
        # Create linker instance
        self.linker = ClaimConceptLinker(
            config=self.config,
            neo4j_manager=self.neo4j_manager,
            vector_store=self.vector_store,
            agent=self.agent,
            markdown_updater=self.markdown_updater,
        )

    def test_link_claims_supports_concept_above_threshold(self):
        """Test SUPPORTS_CONCEPT allowed when claim meets quality threshold."""
        # Set up mock data
        claims = [{
            'id': 'claim1',
            'text': 'Test claim',
            'entailed_score': 0.9,
            'coverage_score': 0.8,
            'decontextualization_score': 0.85,
        }]
        concepts = [{'id': 'concept1', 'text': 'Test concept'}]
        
        self.neo4j_manager.fetch_unlinked_claims.return_value = claims
        self.neo4j_manager.fetch_all_concepts.return_value = concepts
        self.neo4j_manager.create_claim_concept_relationship.return_value = True
        
        # Mock vector search and agent classification
        with patch.object(self.linker, '_find_candidate_concepts_vector') as mock_find, \
             patch.object(self.linker, '_create_claim_concept_pair') as mock_pair, \
             patch.object(self.linker, '_create_link_result') as mock_result:
            
            mock_candidate = Mock()
            mock_candidate.concept_id = 'concept1'
            mock_find.return_value = [mock_candidate]
            
            mock_pair.return_value = Mock()
            
            # Agent returns SUPPORTS_CONCEPT with high strength
            classification = AgentClassificationResult(
                relation="SUPPORTS_CONCEPT",
                strength=0.8,
                entailed_score=0.9,
                coverage_score=0.8
            )
            classification.relationship = RelationshipType.SUPPORTS_CONCEPT
            self.agent.classify_relationship.return_value = classification
            
            mock_link_result = Mock(spec=ClaimConceptLinkResult)
            mock_result.return_value = mock_link_result
            
            # Run linking
            stats = self.linker.link_claims_to_concepts()
            
            # Should create relationship
            assert stats['links_created'] == 1
            self.neo4j_manager.create_claim_concept_relationship.assert_called_once()

    def test_link_claims_supports_concept_below_threshold(self):
        """Test SUPPORTS_CONCEPT rejected when claim below quality threshold."""
        # Set up mock data with low scores
        claims = [{
            'id': 'claim1',
            'text': 'Test claim',
            'entailed_score': 0.5,
            'coverage_score': 0.6,
            'decontextualization_score': 0.4,  # Geomean ≈ 0.49 < 0.7
        }]
        concepts = [{'id': 'concept1', 'text': 'Test concept'}]
        
        self.neo4j_manager.fetch_unlinked_claims.return_value = claims
        self.neo4j_manager.fetch_all_concepts.return_value = concepts
        
        # Mock vector search and agent classification
        with patch.object(self.linker, '_find_candidate_concepts_vector') as mock_find, \
             patch.object(self.linker, '_create_claim_concept_pair') as mock_pair:
            
            mock_candidate = Mock()
            mock_candidate.concept_id = 'concept1'
            mock_find.return_value = [mock_candidate]
            
            mock_pair.return_value = Mock()
            
            # Agent returns SUPPORTS_CONCEPT with high strength
            classification = AgentClassificationResult(
                relation="SUPPORTS_CONCEPT",
                strength=0.8,
                entailed_score=0.5,
                coverage_score=0.6
            )
            classification.relationship = RelationshipType.SUPPORTS_CONCEPT
            self.agent.classify_relationship.return_value = classification
            
            # Run linking
            stats = self.linker.link_claims_to_concepts()
            
            # Should NOT create relationship
            assert stats['links_created'] == 0
            self.neo4j_manager.create_claim_concept_relationship.assert_not_called()

    def test_link_claims_mentions_concept_below_threshold(self):
        """Test MENTIONS_CONCEPT allowed even when below quality threshold."""
        # Set up mock data with low scores but non-null
        claims = [{
            'id': 'claim1',
            'text': 'Test claim',
            'entailed_score': 0.5,
            'coverage_score': 0.6,
            'decontextualization_score': 0.4,  # Geomean ≈ 0.49 < 0.7
        }]
        concepts = [{'id': 'concept1', 'text': 'Test concept'}]
        
        self.neo4j_manager.fetch_unlinked_claims.return_value = claims
        self.neo4j_manager.fetch_all_concepts.return_value = concepts
        self.neo4j_manager.create_claim_concept_relationship.return_value = True
        
        # Mock vector search and agent classification
        with patch.object(self.linker, '_find_candidate_concepts_vector') as mock_find, \
             patch.object(self.linker, '_create_claim_concept_pair') as mock_pair, \
             patch.object(self.linker, '_create_link_result') as mock_result:
            
            mock_candidate = Mock()
            mock_candidate.concept_id = 'concept1'
            mock_find.return_value = [mock_candidate]
            
            mock_pair.return_value = Mock()
            
            # Agent returns MENTIONS_CONCEPT with high strength
            classification = AgentClassificationResult(
                relation="MENTIONS_CONCEPT",
                strength=0.8,
                entailed_score=0.5,
                coverage_score=0.6
            )
            classification.relationship = RelationshipType.MENTIONS_CONCEPT
            self.agent.classify_relationship.return_value = classification
            
            mock_link_result = Mock(spec=ClaimConceptLinkResult)
            mock_result.return_value = mock_link_result
            
            # Run linking
            stats = self.linker.link_claims_to_concepts()
            
            # Should create relationship (MENTIONS_CONCEPT allowed below threshold)
            assert stats['links_created'] == 1
            self.neo4j_manager.create_claim_concept_relationship.assert_called_once()

    def test_link_claims_null_scores_rejected_all_types(self):
        """Test all relationship types rejected when scores are null."""
        # Set up mock data with null scores
        claims = [{
            'id': 'claim1',
            'text': 'Test claim',
            'entailed_score': None,
            'coverage_score': 0.8,
            'decontextualization_score': 0.9,
        }]
        concepts = [{'id': 'concept1', 'text': 'Test concept'}]
        
        self.neo4j_manager.fetch_unlinked_claims.return_value = claims
        self.neo4j_manager.fetch_all_concepts.return_value = concepts
        
        # Test with each relationship type
        for relationship_type in ["SUPPORTS_CONCEPT", "CONTRADICTS_CONCEPT", "MENTIONS_CONCEPT"]:
            # Mock vector search and agent classification
            with patch.object(self.linker, '_find_candidate_concepts_vector') as mock_find, \
                 patch.object(self.linker, '_create_claim_concept_pair') as mock_pair:
                
                mock_candidate = Mock()
                mock_candidate.concept_id = 'concept1'
                mock_find.return_value = [mock_candidate]
                
                mock_pair.return_value = Mock()
                
                # Agent returns relationship with high strength
                classification = AgentClassificationResult(
                    relation=relationship_type,
                    strength=0.8,
                    entailed_score=None,
                    coverage_score=0.8
                )
                classification.relationship = RelationshipType(relationship_type)
                self.agent.classify_relationship.return_value = classification
                
                # Run linking
                stats = self.linker.link_claims_to_concepts()
                
                # Should NOT create relationship (null scores always rejected)
                assert stats['links_created'] == 0
                self.neo4j_manager.create_claim_concept_relationship.assert_not_called()

    def test_link_claims_contradicts_concept_above_threshold(self):
        """Test CONTRADICTS_CONCEPT allowed when claim meets quality threshold."""
        # Set up mock data with good scores
        claims = [{
            'id': 'claim1',
            'text': 'Test claim',
            'entailed_score': 0.9,
            'coverage_score': 0.8,
            'decontextualization_score': 0.85,
        }]
        concepts = [{'id': 'concept1', 'text': 'Test concept'}]
        
        self.neo4j_manager.fetch_unlinked_claims.return_value = claims
        self.neo4j_manager.fetch_all_concepts.return_value = concepts
        self.neo4j_manager.create_claim_concept_relationship.return_value = True
        
        # Mock vector search and agent classification
        with patch.object(self.linker, '_find_candidate_concepts_vector') as mock_find, \
             patch.object(self.linker, '_create_claim_concept_pair') as mock_pair, \
             patch.object(self.linker, '_create_link_result') as mock_result:
            
            mock_candidate = Mock()
            mock_candidate.concept_id = 'concept1'
            mock_find.return_value = [mock_candidate]
            
            mock_pair.return_value = Mock()
            
            # Agent returns CONTRADICTS_CONCEPT with high strength
            classification = AgentClassificationResult(
                relation="CONTRADICTS_CONCEPT",
                strength=0.8,
                entailed_score=0.9,
                coverage_score=0.8
            )
            classification.relationship = RelationshipType.CONTRADICTS_CONCEPT
            self.agent.classify_relationship.return_value = classification
            
            mock_link_result = Mock(spec=ClaimConceptLinkResult)
            mock_result.return_value = mock_link_result
            
            # Run linking
            stats = self.linker.link_claims_to_concepts()
            
            # Should create relationship
            assert stats['links_created'] == 1
            self.neo4j_manager.create_claim_concept_relationship.assert_called_once()

    def test_link_claims_contradicts_concept_below_threshold(self):
        """Test CONTRADICTS_CONCEPT rejected when claim below quality threshold."""
        # Set up mock data with low scores
        claims = [{
            'id': 'claim1',
            'text': 'Test claim',
            'entailed_score': 0.5,
            'coverage_score': 0.6,
            'decontextualization_score': 0.4,  # Geomean ≈ 0.49 < 0.7
        }]
        concepts = [{'id': 'concept1', 'text': 'Test concept'}]
        
        self.neo4j_manager.fetch_unlinked_claims.return_value = claims
        self.neo4j_manager.fetch_all_concepts.return_value = concepts
        
        # Mock vector search and agent classification
        with patch.object(self.linker, '_find_candidate_concepts_vector') as mock_find, \
             patch.object(self.linker, '_create_claim_concept_pair') as mock_pair:
            
            mock_candidate = Mock()
            mock_candidate.concept_id = 'concept1'
            mock_find.return_value = [mock_candidate]
            
            mock_pair.return_value = Mock()
            
            # Agent returns CONTRADICTS_CONCEPT with high strength
            classification = AgentClassificationResult(
                relation="CONTRADICTS_CONCEPT",
                strength=0.8,
                entailed_score=0.5,
                coverage_score=0.6
            )
            classification.relationship = RelationshipType.CONTRADICTS_CONCEPT
            self.agent.classify_relationship.return_value = classification
            
            # Run linking
            stats = self.linker.link_claims_to_concepts()
            
            # Should NOT create relationship
            assert stats['links_created'] == 0
            self.neo4j_manager.create_claim_concept_relationship.assert_not_called()

    def test_link_claims_no_agent_early_return(self):
        """Test early return when no agent is available."""
        # Create linker without agent
        linker = ClaimConceptLinker(
            config=self.config,
            neo4j_manager=self.neo4j_manager,
            vector_store=self.vector_store,
            agent=None,  # No agent
            markdown_updater=self.markdown_updater,
        )
        
        # Run linking
        stats = linker.link_claims_to_concepts()
        
        # Should return empty stats without processing
        assert stats['claims_fetched'] == 0
        assert stats['claims_processed'] == 0
        assert stats['links_created'] == 0

    def test_link_claims_low_agent_strength_below_threshold(self):
        """Test that low agent strength scores are filtered before threshold check."""
        # Set up mock data with good quality scores
        claims = [{
            'id': 'claim1',
            'text': 'Test claim',
            'entailed_score': 0.9,
            'coverage_score': 0.8,
            'decontextualization_score': 0.85,
        }]
        concepts = [{'id': 'concept1', 'text': 'Test concept'}]
        
        self.neo4j_manager.fetch_unlinked_claims.return_value = claims
        self.neo4j_manager.fetch_all_concepts.return_value = concepts
        
        # Mock vector search and agent classification
        with patch.object(self.linker, '_find_candidate_concepts_vector') as mock_find, \
             patch.object(self.linker, '_create_claim_concept_pair') as mock_pair:
            
            mock_candidate = Mock()
            mock_candidate.concept_id = 'concept1'
            mock_find.return_value = [mock_candidate]
            
            mock_pair.return_value = Mock()
            
            # Agent returns low strength (below default 0.5 threshold)
            classification = AgentClassificationResult(
                relation="SUPPORTS_CONCEPT",
                strength=0.3,  # Below default strength_threshold
                entailed_score=0.9,
                coverage_score=0.8
            )
            classification.relationship = RelationshipType.SUPPORTS_CONCEPT
            self.agent.classify_relationship.return_value = classification
            
            # Run linking
            stats = self.linker.link_claims_to_concepts()
            
            # Should NOT create relationship (agent strength too low)
            assert stats['links_created'] == 0
            self.neo4j_manager.create_claim_concept_relationship.assert_not_called()