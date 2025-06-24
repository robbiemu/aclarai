"""
RabbitMQ consumer for processing dirty block notifications.
This module implements the reactive sync loop that consumes dirty block messages
from vault-watcher and updates graph nodes with proper version checking.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from aclarai_shared import load_config
from aclarai_shared.graph.neo4j_manager import Neo4jGraphManager
from aclarai_shared.mq import RabbitMQManager
from aclarai_shared.vault import BlockParser

from .concept_processor import ConceptProcessor

logger = logging.getLogger(__name__)


class DirtyBlockConsumer:
    """
    Consumer for processing dirty block notifications from RabbitMQ.
    Implements the reactive sync loop as specified in sprint_4-Block_syncing_loop.md,
    including proper version checking and conflict detection.
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialize the dirty block consumer."""
        self.config = config or load_config(validate=True)
        self.graph_manager = Neo4jGraphManager(self.config)
        self.block_parser = BlockParser()
        self.concept_processor = ConceptProcessor(self.config)
        # Initialize RabbitMQ manager
        self.rabbitmq_manager = RabbitMQManager(self.config, "aclarai-core")
        self.queue_name = "aclarai_dirty_blocks"
        # Vault paths from config
        self.vault_path = Path(self.config.vault_path)
        logger.info(
            "DirtyBlockConsumer: Initialized consumer",
            extra={
                "service": "aclarai-core",
                "filename.function_name": "dirty_block_consumer.__init__",
                "rabbitmq_host": self.config.rabbitmq_host,
                "queue_name": self.queue_name,
            },
        )

    def connect(self):
        """Establish connection to RabbitMQ."""
        self.rabbitmq_manager.connect()
        self.rabbitmq_manager.ensure_queue(self.queue_name, durable=True)

    def disconnect(self):
        """Close the RabbitMQ connection."""
        self.rabbitmq_manager.close()

    def start_consuming(self):
        """Start consuming messages from the dirty blocks queue."""
        if not self.rabbitmq_manager.is_connected():
            self.connect()
        # Get channel from manager
        channel = self.rabbitmq_manager.get_channel()
        # Set up message consumer
        channel.basic_qos(prefetch_count=1)  # Process one message at a time
        channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self._on_message_received,
            auto_ack=False,  # Manual acknowledgment for reliability
        )
        logger.info(
            "DirtyBlockConsumer: Starting to consume messages",
            extra={
                "service": "aclarai-core",
                "filename.function_name": "dirty_block_consumer.start_consuming",
                "queue_name": self.queue_name,
            },
        )
        try:
            channel.start_consuming()
        except KeyboardInterrupt:
            logger.info(
                "DirtyBlockConsumer: Stopping consumption due to interrupt",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "dirty_block_consumer.start_consuming",
                },
            )
            channel.stop_consuming()
            self.disconnect()

    def _on_message_received(
        self, channel: Any, method: Any, _properties: Any, body: bytes
    ):
        """
        Process a received dirty block message.
        Args:
            channel: RabbitMQ channel
            method: Message delivery method
            properties: Message properties
            body: Message body (JSON)
        """
        try:
            # Parse message
            message = json.loads(body.decode("utf-8"))
            logger.debug(
                "DirtyBlockConsumer: Received dirty block message",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "dirty_block_consumer._on_message_received",
                    "aclarai_id": message.get("aclarai_id"),
                    "change_type": message.get("change_type"),
                    "file_path": message.get("file_path"),
                },
            )
            # Process the dirty block
            success = self._process_dirty_block(message)
            if success:
                # Acknowledge message
                channel.basic_ack(delivery_tag=method.delivery_tag)
                logger.debug(
                    "DirtyBlockConsumer: Successfully processed and acknowledged message",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "dirty_block_consumer._on_message_received",
                        "aclarai_id": message.get("aclarai_id"),
                    },
                )
            else:
                # Reject message and requeue for retry
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                logger.warning(
                    "DirtyBlockConsumer: Failed to process message, requeuing",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "dirty_block_consumer._on_message_received",
                        "aclarai_id": message.get("aclarai_id"),
                    },
                )
        except json.JSONDecodeError as e:
            logger.error(
                f"DirtyBlockConsumer: Invalid JSON in message: {e}",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "dirty_block_consumer._on_message_received",
                    "error": str(e),
                },
            )
            # Acknowledge bad message to remove it from queue
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.error(
                f"DirtyBlockConsumer: Error processing message: {e}",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "dirty_block_consumer._on_message_received",
                    "error": str(e),
                },
            )
            # Reject and requeue for retry
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def _process_dirty_block(self, message: Dict[str, Any]) -> bool:
        """
        Process a single dirty block message.
        Args:
            message: The dirty block message from RabbitMQ
        Returns:
            True if processing was successful, False otherwise
        """
        try:
            aclarai_id = message["aclarai_id"]
            file_path = Path(message["file_path"])
            change_type = message["change_type"]
            # Skip deleted blocks for now (could be handled in future)
            if change_type == "deleted":
                logger.info(
                    "DirtyBlockConsumer: Skipping deleted block",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "dirty_block_consumer._process_dirty_block",
                        "aclarai_id": aclarai_id,
                        "change_type": change_type,
                    },
                )
                return True
            # Read and parse the current block from the file
            current_block = self._read_block_from_file(file_path, aclarai_id)
            if current_block is None:
                logger.warning(
                    "DirtyBlockConsumer: Could not read block from file",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "dirty_block_consumer._process_dirty_block",
                        "aclarai_id": aclarai_id,
                        "file_path": str(file_path),
                    },
                )
                return False
            # Sync block with graph using proper version checking
            sync_success = self._sync_block_with_graph(current_block, file_path)
            # If sync was successful and this is a new or updated block, process for concepts
            if sync_success and change_type in ("created", "modified"):
                try:
                    concept_result = self.concept_processor.process_block_for_concepts(
                        current_block,
                        block_type="claim",  # Assume claim type for now
                    )
                    logger.debug(
                        f"Concept processing completed for block {aclarai_id}: "
                        f"{concept_result.get('merged_count', 0)} merged, "
                        f"{concept_result.get('promoted_count', 0)} promoted",
                        extra={
                            "service": "aclarai-core",
                            "filename.function_name": "dirty_block_consumer._process_dirty_block",
                            "aclarai_id": aclarai_id,
                            "concept_processing_success": concept_result.get(
                                "success", False
                            ),
                        },
                    )
                except Exception as e:
                    # Don't fail the whole message if concept processing fails
                    logger.warning(
                        f"Concept processing failed for block {aclarai_id}: {e}",
                        extra={
                            "service": "aclarai-core",
                            "filename.function_name": "dirty_block_consumer._process_dirty_block",
                            "aclarai_id": aclarai_id,
                            "concept_processing_error": str(e),
                        },
                    )
            return sync_success
        except KeyError as e:
            logger.error(
                f"DirtyBlockConsumer: Missing required field in message: {e}",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "dirty_block_consumer._process_dirty_block",
                    "error": str(e),
                },
            )
            return False
        except Exception as e:
            logger.error(
                f"DirtyBlockConsumer: Error processing dirty block: {e}",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "dirty_block_consumer._process_dirty_block",
                    "aclarai_id": message.get("aclarai_id"),
                    "error": str(e),
                },
            )
            return False

    def _read_block_from_file(
        self, file_path: Path, aclarai_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Read and parse a specific block from a Markdown file.
        Args:
            file_path: Path to the Markdown file
            aclarai_id: The aclarai:id to look for
        Returns:
            Block dictionary with aclarai_id, version, semantic_text, and content_hash
        """
        try:
            # Make file_path absolute if relative to vault
            if not file_path.is_absolute():
                file_path = self.vault_path / file_path
            if not file_path.exists():
                logger.warning(
                    "DirtyBlockConsumer: File does not exist",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "dirty_block_consumer._read_block_from_file",
                        "file_path": str(file_path),
                        "aclarai_id": aclarai_id,
                    },
                )
                return None
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Extract the specific block
            block = self.block_parser.find_block_by_id(content, aclarai_id)
            return block
        except Exception as e:
            logger.error(
                f"DirtyBlockConsumer: Error reading block from file: {e}",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "dirty_block_consumer._read_block_from_file",
                    "file_path": str(file_path),
                    "aclarai_id": aclarai_id,
                    "error": str(e),
                },
            )
            return None

    def _sync_block_with_graph(self, block: Dict[str, Any], file_path: Path) -> bool:
        """
        Synchronize a block with the Neo4j graph using proper version checking.
        Implements the optimistic locking strategy from sprint_4-Block_syncing_loop.md:
        - Compare vault_ver with graph_ver
        - If vault_ver == graph_ver: Clean update, increment version
        - If vault_ver > graph_ver: Proceed with update (vault is more recent)
        - If vault_ver < graph_ver: Conflict detected, skip update and log
        Args:
            block: Block dictionary with aclarai_id, version, semantic_text, content_hash
            file_path: Path to the source file
        Returns:
            True if sync was successful, False otherwise
        """
        try:
            aclarai_id = block["aclarai_id"]
            vault_version = block["version"]
            # Get existing block from graph
            existing_block = self._get_block_from_graph(aclarai_id)
            if existing_block is None:
                # New block - create in graph
                self._create_block_in_graph(block, file_path)
                logger.info(
                    "DirtyBlockConsumer: Created new block in graph",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "dirty_block_consumer._sync_block_with_graph",
                        "aclarai_id": aclarai_id,
                        "file": str(file_path),
                        "action": "create",
                    },
                )
                return True
            # Existing block - perform version checking
            graph_version = existing_block.get("version", 1)
            existing_hash = existing_block.get("hash", "")
            new_hash = block["content_hash"]
            # Check if content actually changed
            if existing_hash == new_hash:
                logger.debug(
                    "DirtyBlockConsumer: Block content unchanged, skipping",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "dirty_block_consumer._sync_block_with_graph",
                        "aclarai_id": aclarai_id,
                        "action": "unchanged",
                    },
                )
                return True
            # Implement version checking logic
            if vault_version < graph_version:
                # Conflict detected - vault is stale
                logger.warning(
                    "DirtyBlockConsumer: Version conflict detected - vault is stale",
                    extra={
                        "service": "aclarai-core",
                        "filename.function_name": "dirty_block_consumer._sync_block_with_graph",
                        "aclarai_id": aclarai_id,
                        "vault_version": vault_version,
                        "graph_version": graph_version,
                        "action": "skip_conflict",
                        "file": str(file_path),
                    },
                )
                return True  # Skip update but don't fail the message
            # Proceed with update (vault_ver >= graph_ver)
            self._update_block_in_graph(block, existing_block, file_path)
            logger.info(
                "DirtyBlockConsumer: Updated block in graph",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "dirty_block_consumer._sync_block_with_graph",
                    "aclarai_id": aclarai_id,
                    "file": str(file_path),
                    "action": "update",
                    "vault_version": vault_version,
                    "graph_version": graph_version,
                    "old_hash": existing_hash[:8],
                    "new_hash": new_hash[:8],
                },
            )
            return True
        except Exception as e:
            logger.error(
                f"DirtyBlockConsumer: Error syncing block with graph: {e}",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "dirty_block_consumer._sync_block_with_graph",
                    "aclarai_id": block.get("aclarai_id"),
                    "file": str(file_path),
                    "error": str(e),
                },
            )
            return False

    def _get_block_from_graph(self, aclarai_id: str) -> Optional[Dict[str, Any]]:
        """Get block properties from Neo4j graph by aclarai_id."""
        cypher_query = """
        MATCH (b:Block {id: $aclarai_id})
        RETURN b.id as id, b.text as text, b.hash as hash,
               b.version as version, b.last_updated as last_updated,
               b.needs_reprocessing as needs_reprocessing
        """

        def _execute_get_block() -> Optional[Dict[str, Any]]:
            with self.graph_manager.session() as session:
                result = session.run(cypher_query, aclarai_id=aclarai_id)
                record = result.single()
                return dict(record) if record else None

        return self.graph_manager._retry_with_backoff(_execute_get_block)

    def _create_block_in_graph(self, block: Dict[str, Any], file_path: Path):
        """Create a new block in the Neo4j graph."""
        current_time = datetime.now(timezone.utc).isoformat()
        cypher_query = """
        MERGE (b:Block {id: $aclarai_id})
        ON CREATE SET
            b.text = $text,
            b.hash = $hash,
            b.version = $version,
            b.last_updated = datetime($last_updated),
            b.needs_reprocessing = true,
            b.source_file = $source_file
        """

        def _execute_create_block():
            with self.graph_manager.session() as session:
                session.run(
                    cypher_query,
                    aclarai_id=block["aclarai_id"],
                    text=block["semantic_text"],
                    hash=block["content_hash"],
                    version=block["version"],
                    last_updated=current_time,
                    source_file=str(file_path),
                )

        self.graph_manager._retry_with_backoff(_execute_create_block)

    def _update_block_in_graph(
        self, block: Dict[str, Any], existing_block: Dict[str, Any], file_path: Path
    ):
        """
        Update an existing block in the Neo4j graph.
        This implements the proper version incrementing as specified:
        - Increment the graph version (ver = ver + 1)
        - Mark with needs_reprocessing: true
        """
        current_time = datetime.now(timezone.utc).isoformat()
        # Increment the graph version, not use the vault version
        current_graph_version = existing_block.get("version", 1)
        new_version = current_graph_version + 1
        cypher_query = """
        MATCH (b:Block {id: $aclarai_id})
        SET b.text = $text,
            b.hash = $hash,
            b.version = $version,
            b.last_updated = datetime($last_updated),
            b.needs_reprocessing = true,
            b.source_file = $source_file
        """

        def _execute_update_block():
            with self.graph_manager.session() as session:
                session.run(
                    cypher_query,
                    aclarai_id=block["aclarai_id"],
                    text=block["semantic_text"],
                    hash=block["content_hash"],
                    version=new_version,
                    last_updated=current_time,
                    source_file=str(file_path),
                )

        self.graph_manager._retry_with_backoff(_execute_update_block)
