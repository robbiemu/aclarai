"""
Mock Neo4j Graph Manager for testing and development.
This module provides an in-memory mock implementation of Neo4jGraphManager
that mimics the essential methods by operating on internal dictionaries
instead of a live database.
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from aclarai_shared.config import aclaraiConfig
from aclarai_shared.graph.models import (
    Claim,
    ClaimInput,
    Concept,
    ConceptInput,
    Sentence,
    SentenceInput,
)

logger = logging.getLogger(__name__)


class MockNeo4jGraphManager:
    """
    In-memory mock implementation of Neo4jGraphManager.
    This mock maintains dictionaries to simulate the Neo4j graph database,
    providing the same interface as the real manager for testing and development.
    """

    def __init__(self, config: Optional[aclaraiConfig] = None):
        """
        Initialize the mock Neo4j manager.
        Args:
            config: aclarai configuration (not used in mock but kept for compatibility)
        """
        self.config = config
        # In-memory storage
        self.claims: Dict[str, Dict[str, Any]] = {}
        self.sentences: Dict[str, Dict[str, Any]] = {}
        self.concepts: Dict[str, Dict[str, Any]] = {}
        self.relationships: List[Dict[str, Any]] = []
        # Query execution history for testing
        self.executed_queries: List[Dict[str, Any]] = []
        logger.info(
            "mock_neo4j_manager.MockNeo4jGraphManager.__init__: Mock Neo4j manager initialized",
            extra={
                "service": "aclarai-test",
                "filename.function_name": "mock_neo4j_manager.MockNeo4jGraphManager.__init__",
            },
        )

    @contextmanager
    def session(self):
        """Mock session context manager that yields self."""
        try:
            yield self
        finally:
            pass

    def execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Mock query execution that simulates Cypher queries.
        Args:
            query: Cypher query string
            parameters: Query parameters
        Returns:
            List of result records
        """
        if parameters is None:
            parameters = {}
        # Log the query execution
        query_record = {
            "query": query,
            "parameters": parameters,
        }
        self.executed_queries.append(query_record)
        logger.debug(
            f"mock_neo4j_manager.MockNeo4jGraphManager.execute_query: Executing query: {query}",
            extra={
                "service": "aclarai-test",
                "filename.function_name": "mock_neo4j_manager.MockNeo4jGraphManager.execute_query",
                "query": query,
                "parameters": parameters,
            },
        )
        # Simple pattern matching for common queries
        query_lower = query.lower().strip()
        if "match (c:claim)" in query_lower and "return" in query_lower:
            return list(self.claims.values())
        elif "match (k:concept)" in query_lower and "return" in query_lower:
            return list(self.concepts.values())
        elif "count" in query_lower:
            return [{"claims": len(self.claims), "concepts": len(self.concepts)}]
        else:
            # Default empty result for unhandled queries
            return []

    def setup_schema(self):
        """Mock schema setup - no-op for in-memory implementation."""
        logger.info(
            "mock_neo4j_manager.MockNeo4jGraphManager.setup_schema: Schema setup completed (mock)",
            extra={
                "service": "aclarai-test",
                "filename.function_name": "mock_neo4j_manager.MockNeo4jGraphManager.setup_schema",
            },
        )

    def create_claims(self, claim_inputs: List[ClaimInput]) -> List[Claim]:
        """
        Create Claim nodes in the mock database.
        Args:
            claim_inputs: List of ClaimInput objects
        Returns:
            List of created Claim objects
        """
        created_claims = []
        for claim_input in claim_inputs:
            claim = Claim.from_input(claim_input)
            self.claims[claim.claim_id] = claim.to_dict()
            created_claims.append(claim)
        logger.info(
            f"mock_neo4j_manager.MockNeo4jGraphManager.create_claims: Created {len(created_claims)} claims",
            extra={
                "service": "aclarai-test",
                "filename.function_name": "mock_neo4j_manager.MockNeo4jGraphManager.create_claims",
                "claims_created": len(created_claims),
            },
        )
        return created_claims

    def create_sentences(self, sentence_inputs: List[SentenceInput]) -> List[Sentence]:
        """
        Create Sentence nodes in the mock database.
        Args:
            sentence_inputs: List of SentenceInput objects
        Returns:
            List of created Sentence objects
        """
        created_sentences = []
        for sentence_input in sentence_inputs:
            sentence = Sentence.from_input(sentence_input)
            self.sentences[sentence.sentence_id] = sentence.to_dict()
            created_sentences.append(sentence)
        logger.info(
            f"mock_neo4j_manager.MockNeo4jGraphManager.create_sentences: Created {len(created_sentences)} sentences",
            extra={
                "service": "aclarai-test",
                "filename.function_name": "mock_neo4j_manager.MockNeo4jGraphManager.create_sentences",
                "sentences_created": len(created_sentences),
            },
        )
        return created_sentences

    def create_concepts(self, concept_inputs: List[ConceptInput]) -> List[Concept]:
        """
        Create Concept nodes in the mock database.
        Args:
            concept_inputs: List of ConceptInput objects
        Returns:
            List of created Concept objects
        """
        created_concepts = []
        for concept_input in concept_inputs:
            concept = Concept.from_input(concept_input)
            self.concepts[concept.concept_id] = concept.to_dict()
            created_concepts.append(concept)
        logger.info(
            f"mock_neo4j_manager.MockNeo4jGraphManager.create_concepts: Created {len(created_concepts)} concepts",
            extra={
                "service": "aclarai-test",
                "filename.function_name": "mock_neo4j_manager.MockNeo4jGraphManager.create_concepts",
                "concepts_created": len(created_concepts),
            },
        )
        return created_concepts

    def fetch_unlinked_claims(self, limit: int = 100) -> List[Dict[str, Any]]:
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

    def fetch_all_concepts(self) -> List[Dict[str, Any]]:
        """Return mock concepts data."""
        return list(self.concepts.values())

    def get_claim_context(self, claim_id: str) -> Dict[str, Any]:
        """Return mock context."""
        return {
            "source_block_text": f"Source text for {claim_id}",
            "summary_text": f"Summary for {claim_id}",
        }

    def create_claim_concept_relationship(self, _link_result: Any) -> bool:
        """Mock relationship creation - always succeeds."""
        return True

    def get_claim_by_id(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a claim by ID.
        Args:
            claim_id: The claim ID to retrieve
        Returns:
            Claim data dictionary or None if not found
        """
        return self.claims.get(claim_id)

    def get_sentence_by_id(self, sentence_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a sentence by ID.
        Args:
            sentence_id: The sentence ID to retrieve
        Returns:
            Sentence data dictionary or None if not found
        """
        return self.sentences.get(sentence_id)

    def count_nodes(self) -> Dict[str, int]:
        """
        Count nodes in the mock database.
        Returns:
            Dictionary with node counts by type
        """
        return {
            "Claim": len(self.claims),
            "Sentence": len(self.sentences),
            "Concept": len(self.concepts),
        }

    def close(self):
        """Mock close method - no-op for in-memory implementation."""
        logger.info(
            "mock_neo4j_manager.MockNeo4jGraphManager.close: Mock Neo4j manager closed",
            extra={
                "service": "aclarai-test",
                "filename.function_name": "mock_neo4j_manager.MockNeo4jGraphManager.close",
            },
        )

    def clear_all_data(self):
        """
        Clear all data from the mock database.
        This is a test utility method not present in the real manager.
        """
        self.claims.clear()
        self.sentences.clear()
        self.concepts.clear()
        self.relationships.clear()
        self.executed_queries.clear()
        logger.debug(
            "mock_neo4j_manager.MockNeo4jGraphManager.clear_all_data: All mock data cleared",
            extra={
                "service": "aclarai-test",
                "filename.function_name": "mock_neo4j_manager.MockNeo4jGraphManager.clear_all_data",
            },
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
