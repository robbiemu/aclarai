"""
Service for updating claim evaluation scores in Neo4j.
"""

import logging
from typing import Any, Dict, List, Optional

from aclarai_core.utils.config_manager import ConfigManager
from neo4j import (
    Driver,
)  # Using `Driver` for type hinting, actual driver instance passed in __init__

logger = logging.getLogger(__name__)


class ClaimEvaluationGraphService:
    """
    Manages graph database updates for claim evaluation scores.
    """

    def __init__(self, neo4j_driver: Driver, config_manager: ConfigManager):
        """
        Initializes the ClaimEvaluationGraphService.

        Args:
            neo4j_driver: An instance of the Neo4j Python Driver.
            config_manager: Manages access to system configurations.
        """
        self.driver = neo4j_driver
        self.config_manager = config_manager
        # Potential configuration for retries on DB operations if needed
        # self.db_retries = self.config_manager.get("database.neo4j.retries", 3)

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


# Example Usage (Conceptual)
# if __name__ == "__main__":
#     # This requires a running Neo4j instance and proper driver setup
#     # from neo4j import GraphDatabase
#     # URI = "bolt://localhost:7687"
#     # AUTH = ("neo4j", "your_password")
#     # driver = GraphDatabase.driver(URI, auth=AUTH)
#     #
#     # class MockConfigManager:
#     #     def get(self, key: str, default: Any = None) -> Any:
#     #         return default
#     #
#     # config_manager = MockConfigManager()
#     # graph_service = ClaimEvaluationGraphService(driver, config_manager)
#     #
#     # # Create dummy data in Neo4j first for this to work:
#     # # MERGE (c:Claim {id: "claim_test_123"}) MERGE (b:Block {id: "block_test_456"}) MERGE (c)-[:ORIGINATES_FROM]->(b)
#     #
#     # success_single = graph_service.update_decontextualization_score("claim_test_123", "block_test_456", 0.95)
#     # print(f"Single update successful: {success_single}")
#     #
#     # success_single_null = graph_service.update_decontextualization_score("claim_test_123", "block_test_456", None)
#     # print(f"Single update with null successful: {success_single_null}")
#     #
#     # # Dummy data for batch
#     # # MERGE (c2:Claim {id: "claim_batch_1"}) MERGE (b2:Block {id: "block_batch_1"}) MERGE (c2)-[:ORIGINATES_FROM]->(b2)
#     # # MERGE (c3:Claim {id: "claim_batch_2"}) MERGE (b3:Block {id: "block_batch_2"}) MERGE (c3)-[:ORIGINATES_FROM]->(b3)
#     # scores_to_batch = [
#     #     {"claim_id": "claim_batch_1", "block_id": "block_batch_1", "score": 0.88},
#     #     {"claim_id": "claim_batch_2", "block_id": "block_batch_2", "score": None},
#     #     {"claim_id": "claim_non_existent", "block_id": "block_non_existent", "score": 0.5}, # Test missing
#     # ]
#     # success_batch = graph_service.batch_update_decontextualization_scores(scores_to_batch)
#     # print(f"Batch update attempt completed: {success_batch}")
#     #
#     # driver.close()
#     print("ClaimEvaluationGraphService file created. Example usage commented out.")
