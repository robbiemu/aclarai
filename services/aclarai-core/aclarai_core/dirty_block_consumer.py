"""
RabbitMQ consumer for processing dirty block notifications.
This module implements the reactive sync loop that consumes dirty block messages
from vault-watcher and updates graph nodes with proper version checking.
"""

import json
import logging
import os
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from aclarai_shared import load_config
from aclarai_shared.config import aclaraiConfig
from aclarai_shared.graph.neo4j_manager import Neo4jGraphManager
from aclarai_shared.mq import RabbitMQManager
from aclarai_shared.tools.factory import ToolFactory
from aclarai_shared.tools.vector_store_manager import aclaraiVectorStoreManager
from aclarai_shared.vault import BlockParser
from llama_index.core.llms.llm import LLM as LlamaLLM

from .agents.coverage_agent import CoverageAgent
from .agents.decontextualization_agent import DecontextualizationAgent
from .agents.entailment_agent import EntailmentAgent
from .concept_processor import ConceptProcessor
from .graph.claim_evaluation_graph_service import ClaimEvaluationGraphService
from .markdown.markdown_updater_service import MarkdownUpdaterService

logger = logging.getLogger(__name__)


class DirtyBlockConsumer:
    """
    Consumer for processing dirty block notifications from RabbitMQ.
    Implements the reactive sync loop as specified in sprint_4-Block_syncing_loop.md,
    including proper version checking and conflict detection.
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialize the dirty block consumer."""
        self.config: aclaraiConfig = config or load_config(validate=True)
        self.graph_manager = Neo4jGraphManager(self.config)
        self.block_parser = BlockParser()
        self.concept_processor = ConceptProcessor(self.config)
        self.rabbitmq_manager = RabbitMQManager(
            config=self.config, service_name="aclarai-core"
        )
        self.queue_name = self.config.vault_watcher.queue_name
        self.vault_path = Path(self.config.vault_path)

        self.llm = self._initialize_llm()  # self.llm can be Optional[LlamaLLM]

        # Instantiate the concrete VectorStoreManager
        self.vector_store_manager = aclaraiVectorStoreManager(self.config)

        tool_factory_config: Dict[str, Any]
        if is_dataclass(self.config):
            tool_factory_config = asdict(self.config)
        else:
            # Use model_dump() for Pydantic v2 compatibility
            if hasattr(self.config, "model_dump") and callable(self.config.model_dump):
                tool_factory_config = self.config.model_dump()
            elif hasattr(self.config, "dict") and callable(self.config.dict):
                # Fallback for Pydantic v1 compatibility
                tool_factory_config = self.config.dict()
            else:
                tool_factory_config = {}
        self.tool_factory = ToolFactory(tool_factory_config, self.vector_store_manager)

        if self.llm:  # Only initialize agents if LLM loaded successfully
            self.entailment_agent: Optional[EntailmentAgent] = EntailmentAgent(
                self.llm, self.tool_factory, self.config
            )
            self.coverage_agent: Optional[CoverageAgent] = CoverageAgent(
                self.llm, self.tool_factory, self.config
            )
            self.decontextualization_agent: Optional[DecontextualizationAgent] = (
                DecontextualizationAgent(self.llm, self.tool_factory, self.config)
            )
        else:
            self.entailment_agent = None  # Explicitly set to None if LLM fails
            self.coverage_agent = None
            self.decontextualization_agent = None
            logger.error(
                "Evaluation agents could not be initialized because LLM failed to load."
            )

        # Initialize generic services for score updates
        self.graph_service = ClaimEvaluationGraphService(
            self.graph_manager.driver, self.config
        )
        self.markdown_service = MarkdownUpdaterService()

        logger.info(
            "DirtyBlockConsumer: Initialized consumer.",
            extra={
                "service": "aclarai-core",
                "filename.function_name": "dirty_block_consumer.__init__",
                "rabbitmq_host": self.config.rabbitmq_host,
                "queue_name": self.queue_name,
                "llm_initialized": self.llm is not None,
                "entailment_agent_initialized": self.entailment_agent is not None,
                "coverage_agent_initialized": self.coverage_agent is not None,
                "decontextualization_agent_initialized": self.decontextualization_agent
                is not None,
            },
        )

    def _initialize_llm(self) -> Optional[LlamaLLM]:
        # Access the model configuration via the new, reliable nested structure
        llm_model_name_entailment = self.config.model.claimify.entailment
        llm_model_name_default = self.config.model.claimify.default

        llm_model_name = llm_model_name_entailment or llm_model_name_default

        log_details = {
            "service": "aclarai-core",
            "filename.function_name": "dirty_block_consumer._initialize_llm",
            "configured_entailment_llm": llm_model_name_entailment,
            "configured_default_llm": llm_model_name_default,
            "effective_llm_model_name": llm_model_name,
        }

        if not llm_model_name:
            logger.error(
                "No LLM model configured for claimify entailment or default in dirty_block_consumer.",
                extra=log_details,
            )
            return None

        logger.info(
            f"Attempting to initialize LLM for evaluation agents with model: {llm_model_name}",
            extra=log_details,
        )

        try:
            if "gpt" in llm_model_name.lower():
                from llama_index.llms.openai import OpenAI

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.error(
                        f"OPENAI_API_KEY not set. Cannot initialize OpenAI model '{llm_model_name}'. Returning non-functional mock.",
                        extra=log_details,
                    )
                    return None

                temperature_str = self.config.processing.temperature
                max_tokens_str = self.config.processing.max_tokens

                try:
                    temperature = float(temperature_str)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid temperature value '{temperature_str}', using default 0.1",
                        extra=log_details,
                    )
                    temperature = 0.1

                try:
                    max_tokens = int(max_tokens_str)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid max_tokens value '{max_tokens_str}', using default 1000",
                        extra=log_details,
                    )
                    max_tokens = 1000

                return OpenAI(
                    model=llm_model_name,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                logger.error(
                    f"Unsupported or unknown LLM model configured: {llm_model_name}. Cannot initialize. Returning non-functional mock.",
                    extra=log_details,
                )
                return None

        except ImportError as ie:
            logger.error(
                f"Failed to import LLM library for model {llm_model_name}: {ie}. Please ensure necessary packages are installed. Returning non-functional mock.",
                extra=log_details,
            )
            return None
        except Exception as e:
            logger.error(
                f"Failed to initialize LLM model {llm_model_name}: {e}. Returning non-functional mock.",
                exc_info=True,
                extra=log_details,
            )
            return None

    def connect(self):
        """Establish connection to RabbitMQ."""
        self.rabbitmq_manager.connect()
        self.rabbitmq_manager.ensure_queue(self.queue_name, durable=True)

    def disconnect(self):
        """Close the RabbitMQ connection."""
        self.rabbitmq_manager.close()

    def start_consuming(self):
        """Start consuming messages from the dirty blocks queue."""
        self._stop_consuming_event = threading.Event()

        try:
            if not self.rabbitmq_manager.is_connected():
                self.connect()
            channel = self.rabbitmq_manager.get_channel()
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(
                queue=self.queue_name,
                on_message_callback=self._on_message_received,
                auto_ack=False,
            )
            logger.info(
                "DirtyBlockConsumer: Starting to consume messages",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "dirty_block_consumer.start_consuming",
                    "queue_name": self.queue_name,
                },
            )

            while not self._stop_consuming_event.is_set():
                # The time_limit=1 allows the loop to check the event every second.
                self.rabbitmq_manager.process_data_events(time_limit=1)

        except KeyboardInterrupt:
            logger.info(
                "DirtyBlockConsumer: Stopping consumption due to KeyboardInterrupt."
            )
        except Exception as e:
            logger.error(
                f"DirtyBlockConsumer: Unhandled exception in consuming loop: {e}",
                exc_info=True,
            )
        finally:
            logger.info("DirtyBlockConsumer: Shutting down connection.")
            if self.rabbitmq_manager.is_connected():
                self.disconnect()

    def stop_consuming(self):
        """
        Signals the consumer to stop consuming messages after the current message is processed.
        This method is thread-safe.
        """
        logger.info("DirtyBlockConsumer: Received stop signal.")
        # Check if the event exists before trying to set it, making this safe
        # to call even if the consumer hasn't started yet.
        if hasattr(self, "_stop_consuming_event"):
            self._stop_consuming_event.set()

    def _on_message_received(
        self, channel: Any, method: Any, _properties: Any, body: bytes
    ):
        message_data_for_log = {"aclarai_id": "unknown", "change_type": "unknown"}
        try:
            message = json.loads(body.decode("utf-8"))
            message_data_for_log["aclarai_id"] = message.get("aclarai_id", "unknown")
            message_data_for_log["change_type"] = message.get("change_type", "unknown")
            message_data_for_log["file_path"] = message.get("file_path", "unknown")

            logger.debug(
                "DirtyBlockConsumer: Received dirty block message",
                extra=message_data_for_log,
            )
            success = self._process_dirty_block(message)
            if success:
                channel.basic_ack(delivery_tag=method.delivery_tag)
                logger.debug(
                    "DirtyBlockConsumer: Successfully processed and acknowledged message",
                    extra=message_data_for_log,
                )
            else:
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                logger.warning(
                    "DirtyBlockConsumer: Failed to process message, requeuing",
                    extra=message_data_for_log,
                )
        except json.JSONDecodeError as e:
            logger.error(
                f"DirtyBlockConsumer: Invalid JSON in message: {e}",
                extra={"body_preview": body[:200]},
            )
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.error(
                f"DirtyBlockConsumer: Error processing message: {e}",
                exc_info=True,
                extra=message_data_for_log,
            )
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def _process_dirty_block(self, message: Dict[str, Any]) -> bool:
        """
        Processes a single dirty block message.
        This version correctly groups all post-sync processing to ensure that if
        evaluation agents are unavailable, subsequent steps like concept processing
        are also skipped, preventing latent bugs.
        """
        aclarai_id_from_message = message.get("aclarai_id")
        try:
            aclarai_id = message["aclarai_id"]
            file_path_str = message["file_path"]
            change_type = message["change_type"]
            file_path = Path(file_path_str)

            if change_type == "deleted":
                logger.info(
                    f"Skipping deleted block: {aclarai_id}",
                    extra={"aclarai_id": aclarai_id},
                )
                return True

            current_block = self._read_block_from_file(file_path, aclarai_id)
            if current_block is None:
                logger.warning(
                    f"Could not read block {aclarai_id} from file {file_path}",
                    extra={"aclarai_id": aclarai_id, "file_path": str(file_path)},
                )
                return False

            sync_success = self._sync_block_with_graph(current_block, file_path)
            if not sync_success:
                logger.warning(
                    f"Failed to sync block {aclarai_id} with graph.",
                    extra={"aclarai_id": aclarai_id},
                )
                return False

            if sync_success and change_type in ("created", "modified"):
                agents_available = (
                    self.entailment_agent is not None
                    and self.coverage_agent is not None
                    and self.decontextualization_agent is not None
                )

                if not agents_available:
                    missing_agents = [
                        name
                        for agent, name in [
                            (self.entailment_agent, "EntailmentAgent"),
                            (self.coverage_agent, "CoverageAgent"),
                            (
                                self.decontextualization_agent,
                                "DecontextualizationAgent",
                            ),
                        ]
                        if agent is None
                    ]
                    logger.error(
                        f"Evaluation agents not initialized ({', '.join(missing_agents)}), skipping evaluation and concept processing for block {aclarai_id}.",
                        extra={"aclarai_id": aclarai_id},
                    )
                else:
                    # --- All processing logic now happens inside this 'else' block ---
                    assert self.entailment_agent is not None
                    assert self.coverage_agent is not None
                    assert self.decontextualization_agent is not None

                    source_text = current_block["semantic_text"]
                    source_id = current_block["aclarai_id"]
                    absolute_file_path = (
                        self.vault_path / file_path
                        if not file_path.is_absolute()
                        else file_path
                    )
                    source_filepath_str = str(absolute_file_path)
                    claims_to_evaluate = self._get_claims_for_evaluation(source_id)

                    for claim_data in claims_to_evaluate:
                        claim_id = claim_data["id"]
                        claim_text = claim_data["text"]
                        logger.info(
                            f"Evaluating claim {claim_id} from source {source_id}",
                            extra={
                                "aclarai_id_claim": claim_id,
                                "aclarai_id_source": source_id,
                            },
                        )

                        # Entailment evaluation
                        entailment_score, err_msg_ent = (
                            self.entailment_agent.evaluate_entailment(
                                claim_id=claim_id,
                                claim_text=claim_text,
                                source_id=source_id,
                                source_text=source_text,
                            )
                        )
                        if err_msg_ent != "success":
                            logger.warning(
                                f"Entailment evaluation failed for claim {claim_id}: {err_msg_ent}",
                                extra={
                                    "aclarai_id_claim": claim_id,
                                    "error": err_msg_ent,
                                },
                            )
                        logger.info(
                            f"Claim {claim_id} entailment_score: {entailment_score}",
                            extra={
                                "aclarai_id_claim": claim_id,
                                "entailed_score": entailment_score,
                            },
                        )

                        # Coverage evaluation
                        coverage_score, omitted_elements, err_msg_cov = (
                            self.coverage_agent.evaluate_coverage(
                                claim_id=claim_id,
                                claim_text=claim_text,
                                source_id=source_id,
                                source_text=source_text,
                            )
                        )
                        if err_msg_cov != "success":
                            logger.warning(
                                f"Coverage evaluation failed for claim {claim_id}: {err_msg_cov}",
                                extra={
                                    "aclarai_id_claim": claim_id,
                                    "error": err_msg_cov,
                                },
                            )
                        logger.info(
                            f"Claim {claim_id} coverage_score: {coverage_score}",
                            extra={
                                "aclarai_id_claim": claim_id,
                                "coverage_score": coverage_score,
                            },
                        )

                        # Decontextualization evaluation
                        decontextualization_score, err_msg_decon = (
                            self.decontextualization_agent.evaluate_claim_decontextualization(
                                claim_id=claim_id,
                                claim_text=claim_text,
                                source_id=source_id,
                                source_text=source_text,
                            )
                        )
                        if err_msg_decon != "success":
                            logger.warning(
                                f"Decontextualization evaluation failed for claim {claim_id}: {err_msg_decon}",
                                extra={
                                    "aclarai_id_claim": claim_id,
                                    "error": err_msg_decon,
                                },
                            )
                        logger.info(
                            f"Claim {claim_id} decontextualization_score: {decontextualization_score}",
                            extra={
                                "aclarai_id_claim": claim_id,
                                "decontextualization_score": decontextualization_score,
                            },
                        )

                        # Update Neo4j with all scores
                        self.graph_service.update_relationship_score(
                            claim_id, source_id, "entailed_score", entailment_score
                        )
                        self.graph_service.update_relationship_score(
                            claim_id, source_id, "coverage_score", coverage_score
                        )
                        self.graph_service.update_relationship_score(
                            claim_id,
                            source_id,
                            "decontextualization_score",
                            decontextualization_score,
                        )

                        # Update Markdown if score is not None
                        if entailment_score is not None:
                            self.markdown_service.add_or_update_score(
                                source_filepath_str,
                                source_id,
                                "entailed_score",
                                entailment_score,
                            )
                        if coverage_score is not None:
                            self.markdown_service.add_or_update_score(
                                source_filepath_str,
                                source_id,
                                "coverage_score",
                                coverage_score,
                            )
                        if decontextualization_score is not None:
                            self.markdown_service.add_or_update_score(
                                source_filepath_str,
                                source_id,
                                "decontextualization_score",
                                decontextualization_score,
                            )

                    # Part 2: Concept Processing
                    try:
                        concept_result = (
                            self.concept_processor.process_block_for_concepts(
                                current_block, block_type="claim"
                            )
                        )
                        logger.debug(
                            f"Concept processing for block {aclarai_id} result: {concept_result.get('merged_count', 0)} merged, {concept_result.get('promoted_count', 0)} promoted.",
                            extra={"aclarai_id": aclarai_id},
                        )
                    except Exception as e:
                        logger.warning(
                            f"Concept processing failed for block {aclarai_id}: {e}",
                            extra={"aclarai_id": aclarai_id, "error": str(e)},
                        )

            return sync_success

        except KeyError as e:
            logger.error(
                f"DirtyBlockConsumer: Missing required field in message: {e}. Message body: {message}",
                extra={"error": f"KeyError: {str(e)}"},
            )
            return False
        except Exception as e:
            logger.error(
                f"DirtyBlockConsumer: Error processing dirty block for aclarai_id '{aclarai_id_from_message}': {e}",
                exc_info=True,
                extra={"aclarai_id": aclarai_id_from_message, "error": str(e)},
            )
            return False

    def _read_block_from_file(
        self, file_path: Path, aclarai_id: str
    ) -> Optional[Dict[str, Any]]:
        try:
            if not file_path.is_absolute():
                file_path = self.vault_path / file_path
            if not file_path.exists():
                logger.warning(
                    f"File does not exist: {file_path}",
                    extra={"file_path": str(file_path), "aclarai_id": aclarai_id},
                )
                return None
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            block = self.block_parser.find_block_by_id(content, aclarai_id)
            return block
        except Exception as e:
            logger.error(
                f"Error reading block from file: {file_path}, id: {aclarai_id}: {e}",
                exc_info=True,
                extra={
                    "file_path": str(file_path),
                    "aclarai_id": aclarai_id,
                    "error": str(e),
                },
            )
            return None

    def _sync_block_with_graph(self, block: Dict[str, Any], file_path: Path) -> bool:
        try:
            aclarai_id = block["aclarai_id"]
            vault_version = block["version"]
            existing_block = self._get_block_from_graph(aclarai_id)
            if existing_block is None:
                self._create_block_in_graph(block, file_path)
                logger.info(
                    "Created new block in graph",
                    extra={"aclarai_id": aclarai_id, "file": str(file_path)},
                )
                return True

            graph_version = existing_block.get("version", 1)
            existing_hash = existing_block.get("hash", "")
            new_hash = block["content_hash"]

            if existing_hash == new_hash:
                logger.debug(
                    "Block content unchanged, skipping graph update",
                    extra={"aclarai_id": aclarai_id},
                )
                return True

            if vault_version < graph_version:
                logger.warning(
                    "Version conflict: vault is stale. Skipping update.",
                    extra={
                        "aclarai_id": aclarai_id,
                        "vault_version": vault_version,
                        "graph_version": graph_version,
                    },
                )
                return True

            self._update_block_in_graph(block, existing_block, file_path)
            logger.info(
                "Updated block in graph",
                extra={"aclarai_id": aclarai_id, "file": str(file_path)},
            )
            return True
        except Exception as e:
            logger.error(
                f"Error syncing block with graph: {block.get('aclarai_id')}: {e}",
                exc_info=True,
                extra={
                    "aclarai_id": block.get("aclarai_id"),
                    "file": str(file_path),
                    "error": str(e),
                },
            )
            return False

    def _get_block_from_graph(self, aclarai_id: str) -> Optional[Dict[str, Any]]:
        cypher_query = """
        MATCH (b:Block {id: $aclarai_id})
        RETURN b.id as id, b.text as text, b.hash as hash,
               b.version as version, b.last_updated as last_updated,
               b.needs_reprocessing as needs_reprocessing
        """

        def _execute() -> Optional[Dict[str, Any]]:
            with self.graph_manager.session() as session:
                result = session.run(cypher_query, aclarai_id=aclarai_id)
                record = result.single()
                return dict(record) if record else None

        return self.graph_manager._retry_with_backoff(_execute)

    def _create_block_in_graph(self, block: Dict[str, Any], file_path: Path):
        current_time = datetime.now(timezone.utc).isoformat()
        cypher_query = """
        MERGE (b:Block {id: $aclarai_id})
        ON CREATE SET
            b.text = $text, b.hash = $hash, b.version = $version,
            b.last_updated = datetime($last_updated),
            b.needs_reprocessing = true, b.source_file = $source_file
        """
        params = {
            "aclarai_id": block["aclarai_id"],
            "text": block["semantic_text"],
            "hash": block["content_hash"],
            "version": block["version"],
            "last_updated": current_time,
            "source_file": str(file_path),
        }

        def _execute():
            with self.graph_manager.session() as session:
                session.run(cypher_query, **params)

        self.graph_manager._retry_with_backoff(_execute)

    def _update_block_in_graph(
        self, block: Dict[str, Any], existing_block: Dict[str, Any], file_path: Path
    ):
        current_time = datetime.now(timezone.utc).isoformat()
        current_graph_version = existing_block.get("version", 1)
        new_version = current_graph_version + 1
        cypher_query = """
        MATCH (b:Block {id: $aclarai_id})
        SET b.text = $text, b.hash = $hash, b.version = $version,
            b.last_updated = datetime($last_updated),
            b.needs_reprocessing = true, b.source_file = $source_file
        """
        params = {
            "aclarai_id": block["aclarai_id"],
            "text": block["semantic_text"],
            "hash": block["content_hash"],
            "version": new_version,
            "last_updated": current_time,
            "source_file": str(file_path),
        }

        def _execute():
            with self.graph_manager.session() as session:
                session.run(cypher_query, **params)

        self.graph_manager._retry_with_backoff(_execute)

    def _get_claims_for_evaluation(self, source_block_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (claim:Claim)-[r:ORIGINATES_FROM]->(block:Block {id: $source_block_id})
        WHERE r.entailed_score IS NULL OR claim.needs_reprocessing = true
        RETURN claim.id AS id, claim.text AS text
        """.strip()
        params = {"source_block_id": source_block_id}
        try:
            results = self.graph_manager.execute_query(query, parameters=params)
            logger.info(
                f"Found {len(results)} claims from block {source_block_id} for entailment evaluation."
            )
            return results
        except Exception as e:
            logger.error(
                f"Failed to fetch claims for evaluation from block {source_block_id}: {e}",
                exc_info=True,
                extra={"source_block_id": source_block_id, "error": str(e)},
            )
            return []
