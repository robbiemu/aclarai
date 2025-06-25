"""
Neo4j operations for claim-concept linking.
This module handles the creation and management of claim-concept relationships
in the Neo4j database, following the graph schema from technical_overview.md.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..config import aclaraiConfig, load_config
from ..graph.neo4j_manager import Neo4jGraphManager
from .models import (
    ClaimConceptLinkResult,
)

logger = logging.getLogger(__name__)


class ClaimConceptNeo4jManager:
    """
    Manages Neo4j operations for claim-concept linking.
    This class handles fetching claims, creating relationships between claims
    and concepts, and querying for existing relationships.
    """

    def __init__(self, config: Optional[aclaraiConfig] = None):
        """
        Initialize the Neo4j manager for claim-concept operations.
        Args:
            config: aclarai configuration (loads default if None)
        """
        self.config = config or load_config()
        self.neo4j_manager = Neo4jGraphManager(config)
        logger.info(
            "Initialized ClaimConceptNeo4jManager",
            extra={
                "service": "aclarai",
                "filename.function_name": "claim_concept_linking.ClaimConceptNeo4jManager.__init__",
            },
        )

    def fetch_unlinked_claims(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch claim nodes that need to be linked to concepts.
        Prioritizes recently created or unlinked claims.
        Args:
            limit: Maximum number of claims to fetch
        Returns:
            List of claim dictionaries with id, text, and score properties
        """
        try:
            # Query for claims that don't have concept relationships yet
            # and prioritize recent ones
            query = """
            MATCH (c:Claim)
            WHERE NOT (c)-[:SUPPORTS_CONCEPT|:MENTIONS_CONCEPT|:CONTRADICTS_CONCEPT]->(:Concept)
            RETURN c.id as id, c.text as text,
                   c.entailed_score as entailed_score,
                   c.coverage_score as coverage_score,
                   c.decontextualization_score as decontextualization_score,
                   c.version as version, c.timestamp as timestamp
            ORDER BY c.timestamp DESC
            LIMIT $limit
            """
            result = self.neo4j_manager.execute_query(query, {"limit": limit})
            claims = []
            for record in result:
                claims.append(
                    {
                        "id": record["id"],
                        "text": record["text"],
                        "entailed_score": record["entailed_score"],
                        "coverage_score": record["coverage_score"],
                        "decontextualization_score": record[
                            "decontextualization_score"
                        ],
                        "version": record["version"],
                        "timestamp": record["timestamp"],
                    }
                )
            logger.debug(
                f"Fetched {len(claims)} unlinked claims",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "claim_concept_linking.ClaimConceptNeo4jManager.fetch_unlinked_claims",
                    "claims_count": len(claims),
                    "limit": limit,
                },
            )
            return claims
        except Exception as e:
            logger.error(
                f"Failed to fetch unlinked claims: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "claim_concept_linking.ClaimConceptNeo4jManager.fetch_unlinked_claims",
                    "error": str(e),
                },
            )
            return []

    def fetch_all_concepts(self) -> List[Dict[str, Any]]:
        """
        Fetch all concept nodes for linking.
        Returns:
            List of concept dictionaries with id, text, and metadata
        """
        try:
            query = """
            MATCH (k:Concept)
            RETURN k.id as id, k.text as text,
                   k.source_node_id as source_node_id,
                   k.source_node_type as source_node_type,
                   k.aclarai_id as aclarai_id,
                   k.version as version, k.timestamp as timestamp
            ORDER BY k.timestamp DESC
            """
            result = self.neo4j_manager.execute_query(query)
            concepts = []
            for record in result:
                concepts.append(
                    {
                        "id": record["id"],
                        "text": record["text"],
                        "source_node_id": record["source_node_id"],
                        "source_node_type": record["source_node_type"],
                        "aclarai_id": record["aclarai_id"],
                        "version": record["version"],
                        "timestamp": record["timestamp"],
                    }
                )
            logger.debug(
                f"Fetched {len(concepts)} concepts",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "claim_concept_linking.ClaimConceptNeo4jManager.fetch_all_concepts",
                    "concepts_count": len(concepts),
                },
            )
            return concepts
        except Exception as e:
            logger.error(
                f"Failed to fetch concepts: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "claim_concept_linking.ClaimConceptNeo4jManager.fetch_all_concepts",
                    "error": str(e),
                },
            )
            return []

    def create_claim_concept_relationship(
        self, link_result: ClaimConceptLinkResult
    ) -> bool:
        """
        Create a relationship between a claim and concept in Neo4j.
        Args:
            link_result: The linking result with relationship details
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build the relationship creation query
            relationship_type = link_result.relationship.value
            properties = link_result.to_neo4j_properties()
            query = f"""
            MATCH (c:Claim {{id: $claim_id}}), (k:Concept {{id: $concept_id}})
            MERGE (c)-[r:{relationship_type}]->(k)
            SET r += $properties
            RETURN r
            """
            params = {
                "claim_id": link_result.claim_id,
                "concept_id": link_result.concept_id,
                "properties": properties,
            }
            result = self.neo4j_manager.execute_query(query, params)
            if result:
                logger.info(
                    f"Created {relationship_type} relationship",
                    extra={
                        "service": "aclarai",
                        "filename.function_name": "claim_concept_linking.ClaimConceptNeo4jManager.create_claim_concept_relationship",
                        "claim_id": link_result.claim_id,
                        "concept_id": link_result.concept_id,
                        "relationship": relationship_type,
                        "strength": link_result.strength,
                    },
                )
                return True
            else:
                logger.warning(
                    f"No relationship created for {link_result.claim_id} -> {link_result.concept_id}",
                    extra={
                        "service": "aclarai",
                        "filename.function_name": "claim_concept_linking.ClaimConceptNeo4jManager.create_claim_concept_relationship",
                        "claim_id": link_result.claim_id,
                        "concept_id": link_result.concept_id,
                    },
                )
                return False
        except Exception as e:
            logger.error(
                f"Failed to create claim-concept relationship: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "claim_concept_linking.ClaimConceptNeo4jManager.create_claim_concept_relationship",
                    "claim_id": link_result.claim_id,
                    "concept_id": link_result.concept_id,
                    "error": str(e),
                },
            )
            return False

    def batch_create_relationships(
        self, link_results: List[ClaimConceptLinkResult]
    ) -> Tuple[int, int]:
        """
        Create multiple claim-concept relationships in a batch.
        Args:
            link_results: List of linking results to create
        Returns:
            Tuple of (successful_count, failed_count)
        """
        successful_count = 0
        failed_count = 0
        for link_result in link_results:
            if self.create_claim_concept_relationship(link_result):
                successful_count += 1
            else:
                failed_count += 1
        logger.info(
            "Batch relationship creation completed",
            extra={
                "service": "aclarai",
                "filename.function_name": "claim_concept_linking.ClaimConceptNeo4jManager.batch_create_relationships",
                "total_attempted": len(link_results),
                "successful": successful_count,
                "failed": failed_count,
            },
        )
        return successful_count, failed_count

    def get_claim_context(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """
        Get contextual information for a claim to improve classification.
        Args:
            claim_id: The ID of the claim
        Returns:
            Dictionary with context information or None if not found
        """
        try:
            # Get the source block and any summary information
            query = """
            MATCH (c:Claim {id: $claim_id})-[:REFERENCES]->(b:Block)
            OPTIONAL MATCH (b)<-[:SUMMARIZES]-(s:Summary)
            RETURN b.text as source_block_text,
                   s.text as summary_text,
                   b.aclarai_id as aclarai_id
            """
            result = self.neo4j_manager.execute_query(query, {"claim_id": claim_id})
            if result and len(result) > 0:
                record = result[0]
                return {
                    "source_block_text": record["source_block_text"],
                    "summary_text": record["summary_text"],
                    "aclarai_id": record["aclarai_id"],
                }
            logger.debug(
                f"No context found for claim {claim_id}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "claim_concept_linking.ClaimConceptNeo4jManager.get_claim_context",
                    "claim_id": claim_id,
                },
            )
            return None
        except Exception as e:
            logger.error(
                f"Failed to get claim context: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "claim_concept_linking.ClaimConceptNeo4jManager.get_claim_context",
                    "claim_id": claim_id,
                    "error": str(e),
                },
            )
            return None

    def get_claims_source_files(self, claim_ids: List[str]) -> Dict[str, str]:
        """
        Get the source file aclarai_id for multiple claims.
        Args:
            claim_ids: List of claim IDs
        Returns:
            Dictionary mapping claim_id to aclarai_id of source file
        """
        try:
            query = """
            MATCH (c:Claim)-[:REFERENCES]->(b:Block)
            WHERE c.id IN $claim_ids
            RETURN c.id as claim_id, b.aclarai_id as aclarai_id
            """
            result = self.neo4j_manager.execute_query(query, {"claim_ids": claim_ids})
            file_mapping = {}
            for record in result:
                file_mapping[record["claim_id"]] = record["aclarai_id"]
            logger.debug(
                f"Found source files for {len(file_mapping)} claims",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "claim_concept_linking.ClaimConceptNeo4jManager.get_claims_source_files",
                    "claim_count": len(claim_ids),
                    "file_count": len(set(file_mapping.values())),
                },
            )
            return file_mapping
        except Exception as e:
            logger.error(
                f"Failed to get claims source files: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "claim_concept_linking.ClaimConceptNeo4jManager.get_claims_source_files",
                    "claim_ids_count": len(claim_ids),
                    "error": str(e),
                },
            )
            return {}

    def get_concepts_for_claims(
        self, claim_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all concepts linked to the specified claims.
        Args:
            claim_ids: List of claim IDs to find concepts for
        Returns:
            Dictionary mapping claim_id to list of linked concept info
        """
        if not claim_ids:
            return {}
        try:
            query = """
            MATCH (c:Claim)-[r:SUPPORTS_CONCEPT|MENTIONS_CONCEPT|CONTRADICTS_CONCEPT]->(k:Concept)
            WHERE c.id IN $claim_ids
            RETURN c.id as claim_id, k.id as concept_id, k.text as concept_text,
                   type(r) as relationship_type, r.strength as strength
            ORDER BY c.id, r.strength DESC
            """
            result = self.neo4j_manager.execute_query(query, {"claim_ids": claim_ids})
            concepts_mapping: Dict[str, List[Dict[str, Any]]] = {}
            for record in result:
                claim_id = record["claim_id"]
                if claim_id not in concepts_mapping:
                    concepts_mapping[claim_id] = []
                concepts_mapping[claim_id].append(
                    {
                        "concept_id": record["concept_id"],
                        "concept_text": record["concept_text"],
                        "relationship_type": record["relationship_type"],
                        "strength": record["strength"],
                    }
                )
            logger.debug(
                f"Found concepts for {len(concepts_mapping)} claims",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "claim_concept_linking.ClaimConceptNeo4jManager.get_concepts_for_claims",
                    "claim_count": len(claim_ids),
                    "claims_with_concepts": len(concepts_mapping),
                },
            )
            return concepts_mapping
        except Exception as e:
            logger.error(
                f"Failed to get concepts for claims: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "claim_concept_linking.ClaimConceptNeo4jManager.get_concepts_for_claims",
                    "claim_ids_count": len(claim_ids),
                    "error": str(e),
                },
            )
            return {}
