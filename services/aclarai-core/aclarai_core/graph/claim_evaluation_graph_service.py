"""
Service for updating claim evaluation scores in Neo4j.
"""

import logging
from typing import Any, Dict, List, Optional

from aclarai_shared.config import aclaraiConfig
from neo4j import (
    Driver,
)  # Using `Driver` for type hinting, actual driver instance passed in __init__

logger = logging.getLogger(__name__)


class ClaimEvaluationGraphService:
    """
    Manages graph database updates for claim evaluation scores.
    """

    # Valid score property names to prevent Cypher injection
    VALID_SCORE_NAMES = {
        "decontextualization_score",
        "entailed_score",
        "coverage_score",
    }

    def __init__(self, neo4j_driver: Driver, config: aclaraiConfig):
        """
        Initializes the ClaimEvaluationGraphService.

        Args:
            neo4j_driver: An instance of the Neo4j Python Driver.
            config: system configurations.
        """
        self.driver = neo4j_driver
        self.config = config
        # Potential configuration for retries on DB operations if needed
        # self.db_retries = self.config.processing.get("retries", {}).get("max_attempts", 3)

    def update_decontextualization_score(
        self, claim_id: str, block_id: str, score: Optional[float]
    ) -> bool:
        """
        Updates the decontextualization score for a claim's relationship in Neo4j.

        The score is stored on the [:ORIGINATES_FROM] relationship between
        the (:Claim {id: claim_id}) and its (:Block {id: block_id}).

        Args:
            claim_id: The unique ID of the claim.
            block_id: The unique ID of the block from which the claim originates.
            score: The decontextualization score (float or None for null).

        Returns:
            True if the update was successful, False otherwise.
        """
        query = """
        MATCH (c:Claim {id: $claim_id})-[r:ORIGINATES_FROM]->(b:Block {id: $block_id})
        SET r.decontextualization_score = $score
        RETURN count(r) AS updated_count
        """
        parameters = {
            "claim_id": claim_id,
            "block_id": block_id,
            "score": score,  # Neo4j driver handles None as null
        }
        log_details = {
            "service": "aclarai-core",
            "filename_function_name": "claim_evaluation_graph_service.ClaimEvaluationGraphService.update_decontextualization_score",
            "aclarai_id_claim": claim_id,
            "aclarai_id_block": block_id,
            "decontextualization_score": score,
        }

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                record = result.single()
                updated_count = record["updated_count"] if record else 0

                if updated_count > 0:
                    logger.info(
                        f"Successfully updated decontextualization_score for claim {claim_id} on relationship to block {block_id}.",
                        extra=log_details,
                    )
                    return True
                else:
                    logger.warning(
                        f"Failed to update decontextualization_score: No relationship found for claim {claim_id} and block {block_id}.",
                        extra=log_details,
                    )
                    return False
        except Exception as e:
            logger.error(
                f"Error updating decontextualization_score for claim {claim_id} in Neo4j: {e}",
                exc_info=True,
                extra=log_details,
            )
            return False

    def batch_update_decontextualization_scores(
        self, scores_data: List[Dict[str, Any]]
    ) -> bool:
        """
        Updates decontextualization scores for multiple claims in a batch.

        Args:
            scores_data: A list of dictionaries, where each dictionary contains:
                         'claim_id': str,
                         'block_id': str,
                         'score': Optional[float]

        Returns:
            True if all updates were successful (or attempted), False if a general error occurs.
            Individual errors are logged.
        """
        query = """
        UNWIND $scores_batch AS score_entry
        MATCH (c:Claim {id: score_entry.claim_id})-[r:ORIGINATES_FROM]->(b:Block {id: score_entry.block_id})
        SET r.decontextualization_score = score_entry.score
        RETURN score_entry.claim_id AS processed_claim_id, count(r) AS updated_count
        """
        parameters = {"scores_batch": scores_data}

        processed_claims = {}
        log_details_base = {
            "service": "aclarai-core",
            "filename_function_name": "claim_evaluation_graph_service.ClaimEvaluationGraphService.batch_update_decontextualization_scores",
        }

        try:
            with self.driver.session() as session:
                results = session.run(query, parameters)
                for record in results:
                    processed_claims[record["processed_claim_id"]] = record[
                        "updated_count"
                    ]

                for score_item in scores_data:
                    claim_id = score_item["claim_id"]
                    block_id = score_item["block_id"]  # For logging
                    score_val = score_item["score"]  # For logging
                    log_item_details = {
                        **log_details_base,
                        "aclarai_id_claim": claim_id,
                        "aclarai_id_block": block_id,
                        "decontextualization_score": score_val,
                    }
                    if processed_claims.get(claim_id, 0) > 0:
                        logger.info(
                            f"Successfully batch-updated decontextualization_score for claim {claim_id}.",
                            extra=log_item_details,
                        )
                    else:
                        logger.warning(
                            f"Failed to batch-update decontextualization_score: No relationship found for claim {claim_id} and block {block_id} or claim not processed.",
                            extra=log_item_details,
                        )
                return True  # Indicates the batch operation itself completed

        except Exception as e:
            logger.error(
                f"General error during batch update of decontextualization_scores in Neo4j: {e}",
                exc_info=True,
                extra=log_details_base,
            )
            return False

    def update_relationship_score(
        self,
        claim_id: str,
        block_id: str,
        score_name: str,
        score_value: Optional[float],
    ) -> bool:
        """
        Updates any score on the [:ORIGINATES_FROM] relationship between a claim and block.

        This is a generic method that can update any score type (entailed_score,
        decontextualization_score, coverage_score, etc.) on the relationship.

        Args:
            claim_id: The unique ID of the claim.
            block_id: The unique ID of the block from which the claim originates.
            score_name: The name of the score property to update (e.g., "entailed_score").
            score_value: The score value (float or None for null).

        Returns:
            True if the update was successful, False otherwise.

        Raises:
            ValueError: If score_name is not in the list of valid score names.
        """
        # Validate score_name to prevent Cypher injection
        if score_name not in self.VALID_SCORE_NAMES:
            raise ValueError(
                f"Invalid score_name '{score_name}'. Must be one of: {self.VALID_SCORE_NAMES}"
            )

        # Build query with validated score_name
        query = f"""
        MATCH (c:Claim {{id: $claim_id}})-[r:ORIGINATES_FROM]->(b:Block {{id: $block_id}})
        SET r.{score_name} = $score_value
        RETURN count(r) AS updated_count
        """
        parameters = {
            "claim_id": claim_id,
            "block_id": block_id,
            "score_value": score_value,  # Neo4j driver handles None as null
        }
        log_details = {
            "service": "aclarai-core",
            "filename_function_name": "claim_evaluation_graph_service.ClaimEvaluationGraphService.update_relationship_score",
            "aclarai_id_claim": claim_id,
            "aclarai_id_block": block_id,
            "score_name": score_name,
            "score_value": score_value,
        }

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                record = result.single()
                updated_count = record["updated_count"] if record else 0

                if updated_count > 0:
                    logger.info(
                        f"Successfully updated {score_name} for claim {claim_id} on relationship to block {block_id}.",
                        extra=log_details,
                    )
                    return True
                else:
                    logger.warning(
                        f"Failed to update {score_name}: No relationship found for claim {claim_id} and block {block_id}.",
                        extra=log_details,
                    )
                    return False
        except Exception as e:
            logger.error(
                f"Error updating {score_name} for claim {claim_id} in Neo4j: {e}",
                exc_info=True,
                extra=log_details,
            )
            return False
