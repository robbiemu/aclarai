"""
Neo4j Graph Manager for aclarai knowledge graph operations.
This module provides the main interface for creating and managing
Claim and Sentence nodes in Neo4j, following the architectural patterns
from docs/arch/idea-neo4J-ineteraction.md.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, TypeVar

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable, TransientError

from ..config import aclaraiConfig
from .models import Claim, ClaimInput, Concept, ConceptInput, Sentence, SentenceInput

logger = logging.getLogger(__name__)

# Type variable for generic retry function
T = TypeVar("T")


class Neo4jGraphManager:
    """
    Manager for Neo4j graph operations following aclarai architecture.
    Handles creation of Claim and Sentence nodes with proper relationships
    and indexing as specified in the technical requirements.
    """

    def __init__(self, config: Optional[aclaraiConfig] = None):
        """
        Initialize Neo4j connection.
        Args:
            config: aclarai configuration (loads default if None)
        """
        if config is None:
            from ..config import load_config

            config = load_config(validate=False)
        self.config = config
        self._driver: Optional[Driver] = None
        # Connection details
        self.uri = config.neo4j.get_neo4j_bolt_url()
        self.auth = (config.neo4j.user, config.neo4j.password)
        logger.info(
            f"neo4j_manager.__init__: Initialized Neo4jGraphManager for {self.uri}",
            extra={
                "service": "aclarai-core",
                "filename.function_name": "neo4j_manager.__init__",
            },
        )

    @property
    def driver(self) -> Driver:
        """Get or create Neo4j driver."""
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(self.uri, auth=self.auth)
                self._driver.verify_connectivity()
                logger.info(
                    "neo4j_manager.driver: Neo4j driver connected successfully",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "neo4j_manager.driver",
                    },
                )
            except (ServiceUnavailable, AuthError) as e:
                logger.error(
                    f"neo4j_manager.driver: Failed to connect to Neo4j: {e}",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "neo4j_manager.driver",
                    },
                )
                raise
        return self._driver

    def close(self):
        """Close Neo4j driver connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info(
                "neo4j_manager.close: Neo4j driver connection closed",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "neo4j_manager.close",
                },
            )

    @contextmanager
    def session(self):
        """Context manager for Neo4j sessions."""
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()

    def _retry_with_backoff(
        self, func: Callable[..., T], *args: Any, **kwargs: Any
    ) -> T:
        """
        Execute function with retry logic and exponential backoff.
        Following guidelines from docs/arch/on-error-handling-and-resilience.md
        for handling transient Neo4j errors.
        """
        max_attempts = (
            getattr(self.config, "processing", {})
            .get("retries", {})
            .get("max_attempts", 3)
        )
        backoff_factor = (
            getattr(self.config, "processing", {})
            .get("retries", {})
            .get("backoff_factor", 2)
        )
        max_wait_time = (
            getattr(self.config, "processing", {})
            .get("retries", {})
            .get("max_wait_time", 60)
        )
        for attempt in range(max_attempts):
            try:
                logger.debug(
                    f"neo4j_manager._retry_with_backoff: Attempt {attempt + 1}/{max_attempts}",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": f"neo4j_manager.{func.__name__}",
                    },
                )
                return func(*args, **kwargs)
            except (ServiceUnavailable, TransientError) as e:
                if attempt == max_attempts - 1:
                    logger.error(
                        f"neo4j_manager._retry_with_backoff: Final attempt failed: {e}",
                        extra={
                            "service": "aclarai-core",
                            "filename.function_name": f"neo4j_manager.{func.__name__}",
                        },
                    )
                    raise
                wait_time = min(backoff_factor**attempt, max_wait_time)
                logger.warning(
                    f"neo4j_manager._retry_with_backoff: Transient error on attempt {attempt + 1}, "
                    f"retrying in {wait_time}s: {e}",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": f"neo4j_manager.{func.__name__}",
                    },
                )
                time.sleep(wait_time)
            except Exception as e:
                # Non-transient error, fail immediately
                logger.error(
                    f"neo4j_manager._retry_with_backoff: Non-transient error: {e}",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": f"neo4j_manager.{func.__name__}",
                    },
                )
                raise
        # This part is unreachable due to the raise in the loop, but mypy requires it
        # for functions with a return type.
        raise RuntimeError("Retry logic failed unexpectedly.")

    def setup_schema(self):
        """
        Set up Neo4j schema with constraints and indexes.
        Creates constraints and indexes as specified in graph_schema.cypher
        and technical requirements.
        """
        schema_queries = [
            # Constraints for unique IDs
            "CREATE CONSTRAINT claim_id_unique IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT sentence_id_unique IF NOT EXISTS FOR (s:Sentence) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT block_id_unique IF NOT EXISTS FOR (b:Block) REQUIRE b.id IS UNIQUE",
            # Indexes for performance (technical requirements specify aclarai:id and text)
            "CREATE INDEX claim_text_index IF NOT EXISTS FOR (c:Claim) ON (c.text)",
            "CREATE INDEX sentence_text_index IF NOT EXISTS FOR (s:Sentence) ON (s.text)",
            "CREATE INDEX block_text_index IF NOT EXISTS FOR (b:Block) ON (b.text)",
            "CREATE INDEX block_hash_index IF NOT EXISTS FOR (b:Block) ON (b.hash)",
            # Additional performance indexes from graph_schema.cypher
            "CREATE INDEX claim_entailed_score_index IF NOT EXISTS FOR (c:Claim) ON (c.entailed_score)",
            "CREATE INDEX claim_coverage_score_index IF NOT EXISTS FOR (c:Claim) ON (c.coverage_score)",
            "CREATE INDEX claim_decontextualization_score_index IF NOT EXISTS FOR (c:Claim) ON (c.decontextualization_score)",
        ]

        def _execute_schema():
            with self.session() as session:
                for query in schema_queries:
                    try:
                        session.run(query)
                        logger.debug(
                            f"neo4j_manager.setup_schema: Executed schema query: {query}",
                            extra={
                                "service": "aclarai-core",
                                "filename.function_name": "neo4j_manager.setup_schema",
                            },
                        )
                    except Exception as e:
                        logger.warning(
                            f"neo4j_manager.setup_schema: Schema query failed (may already exist): {query}, error: {e}",
                            extra={
                                "service": "aclarai-core",
                                "filename.function_name": "neo4j_manager.setup_schema",
                            },
                        )

        self._retry_with_backoff(_execute_schema)
        logger.info(
            "neo4j_manager.setup_schema: Neo4j schema setup completed",
            extra={
                "service": "aclarai-core",
                "filename.function_name": "neo4j_manager.setup_schema",
            },
        )

    def create_claims(self, claim_inputs: List[ClaimInput]) -> List[Claim]:
        """
        Create Claim nodes in batch with ORIGINATES_FROM relationships.
        Args:
            claim_inputs: List of ClaimInput objects
        Returns:
            List of created Claim objects
        """
        if not claim_inputs:
            logger.warning(
                "neo4j_manager.create_claims: No claims provided for creation",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "neo4j_manager.create_claims",
                },
            )
            return []
        # Convert inputs to Claim objects
        claims = [Claim.from_input(claim_input) for claim_input in claim_inputs]
        # Prepare data for batch creation
        claims_data = []
        for claim, claim_input in zip(claims, claim_inputs, strict=False):
            claim_dict = claim.to_dict()
            claim_dict["block_id"] = claim_input.block_id
            claims_data.append(claim_dict)
        logger.info(
            f"neo4j_manager.create_claims: Preparing to create {len(claims_data)} claims",
            extra={
                "service": "aclarai-core",
                "filename.function_name": "neo4j_manager.create_claims",
                "claims_count": len(claims_data),
            },
        )
        # Batch create using UNWIND (following architecture guidelines)
        cypher_query = """
        UNWIND $claims_data AS data
        MERGE (c:Claim {id: data.id})
        ON CREATE SET
            c.text = data.text,
            c.entailed_score = data.entailed_score,
            c.coverage_score = data.coverage_score,
            c.decontextualization_score = data.decontextualization_score,
            c.version = data.version,
            c.timestamp = datetime(data.timestamp)
        MERGE (b:Block {id: data.block_id})
        MERGE (c)-[:ORIGINATES_FROM]->(b)
        RETURN c.id as claim_id
        """

        def _execute_claim_creation() -> List[str]:
            with self.session() as session:
                result = session.run(cypher_query, claims_data=claims_data)
                created_ids = [record["claim_id"] for record in result]
                return created_ids

        try:
            created_ids = self._retry_with_backoff(_execute_claim_creation)
            logger.info(
                f"neo4j_manager.create_claims: Successfully created {len(created_ids)} Claim nodes",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "neo4j_manager.create_claims",
                    "created_count": len(created_ids),
                },
            )
            return claims
        except Exception as e:
            logger.error(
                f"neo4j_manager.create_claims: Failed to create Claims: {e}",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "neo4j_manager.create_claims",
                    "claims_count": len(claims_data),
                    "error": str(e),
                },
            )
            raise

    def create_sentences(self, sentence_inputs: List[SentenceInput]) -> List[Sentence]:
        """
        Create Sentence nodes in batch with ORIGINATES_FROM relationships.
        Args:
            sentence_inputs: List of SentenceInput objects
        Returns:
            List of created Sentence objects
        """
        if not sentence_inputs:
            logger.warning(
                "neo4j_manager.create_sentences: No sentences provided for creation",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "neo4j_manager.create_sentences",
                },
            )
            return []
        # Convert inputs to Sentence objects
        sentences = [
            Sentence.from_input(sentence_input) for sentence_input in sentence_inputs
        ]
        # Prepare data for batch creation
        sentences_data = []
        for sentence, sentence_input in zip(sentences, sentence_inputs, strict=False):
            sentence_dict = sentence.to_dict()
            sentence_dict["block_id"] = sentence_input.block_id
            sentences_data.append(sentence_dict)
        logger.info(
            f"neo4j_manager.create_sentences: Preparing to create {len(sentences_data)} sentences",
            extra={
                "service": "aclarai-core",
                "filename.function_name": "neo4j_manager.create_sentences",
                "sentences_count": len(sentences_data),
            },
        )
        # Batch create using UNWIND
        cypher_query = """
        UNWIND $sentences_data AS data
        MERGE (s:Sentence {id: data.id})
        ON CREATE SET
            s.text = data.text,
            s.ambiguous = data.ambiguous,
            s.verifiable = data.verifiable,
            s.failed_decomposition = data.failed_decomposition,
            s.rejection_reason = data.rejection_reason,
            s.version = data.version,
            s.timestamp = datetime(data.timestamp)
        MERGE (b:Block {id: data.block_id})
        MERGE (s)-[:ORIGINATES_FROM]->(b)
        RETURN s.id as sentence_id
        """

        def _execute_sentence_creation() -> List[str]:
            with self.session() as session:
                result = session.run(cypher_query, sentences_data=sentences_data)
                created_ids = [record["sentence_id"] for record in result]
                return created_ids

        try:
            created_ids = self._retry_with_backoff(_execute_sentence_creation)
            logger.info(
                f"neo4j_manager.create_sentences: Successfully created {len(created_ids)} Sentence nodes",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "neo4j_manager.create_sentences",
                    "created_count": len(created_ids),
                },
            )
            return sentences
        except Exception as e:
            logger.error(
                f"neo4j_manager.create_sentences: Failed to create Sentences: {e}",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "neo4j_manager.create_sentences",
                    "sentences_count": len(sentences_data),
                    "error": str(e),
                },
            )
            raise

    def get_claim_by_id(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a Claim node by ID.
        Args:
            claim_id: The claim ID to search for
        Returns:
            Claim node data as dictionary, or None if not found
        """
        cypher_query = """
        MATCH (c:Claim {id: $claim_id})
        RETURN c.id as id, c.text as text, c.entailed_score as entailed_score,
               c.coverage_score as coverage_score, c.decontextualization_score as decontextualization_score,
               c.version as version, c.timestamp as timestamp
        """

        def _execute_get_claim() -> Optional[Dict[str, Any]]:
            with self.session() as session:
                result = session.run(cypher_query, claim_id=claim_id)
                record = result.single()
                return dict(record) if record else None

        try:
            result = self._retry_with_backoff(_execute_get_claim)
            if result:
                logger.debug(
                    f"neo4j_manager.get_claim_by_id: Found claim {claim_id}",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "neo4j_manager.get_claim_by_id",
                        "aclarai_id": claim_id,
                    },
                )
            else:
                logger.debug(
                    f"neo4j_manager.get_claim_by_id: Claim {claim_id} not found",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "neo4j_manager.get_claim_by_id",
                        "aclarai_id": claim_id,
                    },
                )
            return result
        except Exception as e:
            logger.error(
                f"neo4j_manager.get_claim_by_id: Failed to get Claim {claim_id}: {e}",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "neo4j_manager.get_claim_by_id",
                    "aclarai_id": claim_id,
                    "error": str(e),
                },
            )
            raise

    def get_sentence_by_id(self, sentence_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a Sentence node by ID.
        Args:
            sentence_id: The sentence ID to search for
        Returns:
            Sentence node data as dictionary, or None if not found
        """
        cypher_query = """
        MATCH (s:Sentence {id: $sentence_id})
        RETURN s.id as id, s.text as text, s.ambiguous as ambiguous,
               s.verifiable as verifiable, s.failed_decomposition as failed_decomposition,
               s.rejection_reason as rejection_reason, s.version as version, s.timestamp as timestamp
        """

        def _execute_get_sentence() -> Optional[Dict[str, Any]]:
            with self.session() as session:
                result = session.run(cypher_query, sentence_id=sentence_id)
                record = result.single()
                return dict(record) if record else None

        try:
            result = self._retry_with_backoff(_execute_get_sentence)
            if result:
                logger.debug(
                    f"neo4j_manager.get_sentence_by_id: Found sentence {sentence_id}",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "neo4j_manager.get_sentence_by_id",
                        "aclarai_id": sentence_id,
                    },
                )
            else:
                logger.debug(
                    f"neo4j_manager.get_sentence_by_id: Sentence {sentence_id} not found",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "neo4j_manager.get_sentence_by_id",
                        "aclarai_id": sentence_id,
                    },
                )
            return result
        except Exception as e:
            logger.error(
                f"neo4j_manager.get_sentence_by_id: Failed to get Sentence {sentence_id}: {e}",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "neo4j_manager.get_sentence_by_id",
                    "aclarai_id": sentence_id,
                    "error": str(e),
                },
            )
            raise

    def count_nodes(self) -> Dict[str, int]:
        """
        Get count of nodes for monitoring/validation.
        Returns:
            Dictionary with node counts
        """
        cypher_query = """
        MATCH (c:Claim) WITH count(c) as claim_count
        MATCH (s:Sentence) WITH claim_count, count(s) as sentence_count
        MATCH (b:Block) WITH claim_count, sentence_count, count(b) as block_count
        RETURN claim_count, sentence_count, block_count
        """

        def _execute_count_nodes() -> Dict[str, int]:
            with self.session() as session:
                result = session.run(cypher_query)
                record = result.single()
                if record:
                    return {
                        "claims": record["claim_count"],
                        "sentences": record["sentence_count"],
                        "blocks": record["block_count"],
                    }
                return {"claims": 0, "sentences": 0, "blocks": 0}

        try:
            result = self._retry_with_backoff(_execute_count_nodes)
            logger.debug(
                f"neo4j_manager.count_nodes: Node counts - Claims: {result['claims']}, "
                f"Sentences: {result['sentences']}, Blocks: {result['blocks']}",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "neo4j_manager.count_nodes",
                    "claims_count": result["claims"],
                    "sentences_count": result["sentences"],
                    "blocks_count": result["blocks"],
                },
            )
            return result
        except Exception as e:
            logger.error(
                f"neo4j_manager.count_nodes: Failed to count nodes: {e}",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "neo4j_manager.count_nodes",
                    "error": str(e),
                },
            )
            raise

    def create_concepts(self, concept_inputs: List[ConceptInput]) -> List[Concept]:
        """
        Create Concept nodes in the knowledge graph with proper indexing.
        Args:
            concept_inputs: List of ConceptInput objects to create
        Returns:
            List of created Concept objects
        Raises:
            Neo4jError: If concept creation fails
        """
        if not concept_inputs:
            logger.warning(
                "create_concepts: No concept inputs provided",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "neo4j_manager.create_concepts",
                },
            )
            return []
        logger.info(
            f"create_concepts: Creating {len(concept_inputs)} Concept nodes",
            extra={
                "service": "aclarai-core",
                "filename.function_name": "neo4j_manager.create_concepts",
                "concept_count": len(concept_inputs),
            },
        )

        def _execute_concept_creation() -> List[Concept]:
            with self.session() as session:
                concepts = []
                for concept_input in concept_inputs:
                    concept = Concept.from_input(concept_input)
                    concept_data = concept.to_dict()
                    # Create Concept node with proper indexing
                    cypher = """
                    CREATE (c:Concept $concept_data)
                    RETURN c
                    """
                    result = session.run(cypher, concept_data=concept_data)
                    record = result.single()
                    if record:
                        concepts.append(concept)
                        logger.debug(
                            f"create_concepts: Created Concept node: {concept.concept_id}",
                            extra={
                                "service": "aclarai-core",
                                "filename.function_name": "neo4j_manager.create_concepts",
                                "concept_id": concept.concept_id,
                                "concept_text": concept.text,
                            },
                        )
                    else:
                        logger.error(
                            f"create_concepts: Failed to create Concept node: {concept.concept_id}",
                            extra={
                                "service": "aclarai-core",
                                "filename.function_name": "neo4j_manager.create_concepts",
                                "concept_id": concept.concept_id,
                            },
                        )
                logger.info(
                    f"create_concepts: Successfully created {len(concepts)} Concept nodes",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "neo4j_manager.create_concepts",
                        "created_count": len(concepts),
                    },
                )
                return concepts

        return self._retry_with_backoff(_execute_concept_creation)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
