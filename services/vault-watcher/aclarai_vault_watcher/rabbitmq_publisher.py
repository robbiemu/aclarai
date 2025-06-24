"""
RabbitMQ publisher for dirty block notifications.
This module handles publishing dirty block messages to RabbitMQ queues
for downstream processing by the sync_vault_to_graph job.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pika
from aclarai_shared.config import aclaraiConfig
from aclarai_shared.mq import RabbitMQManager
from pika.exceptions import AMQPChannelError

logger = logging.getLogger(__name__)


class DirtyBlockPublisher:
    """
    Publisher for sending dirty block notifications to RabbitMQ.
    Publishes messages to the aclarai_dirty_blocks queue when blocks
    are detected as dirty (created, modified, or deleted).
    """

    def __init__(
        self,
        rabbitmq_host: str,
        rabbitmq_port: int = 5672,
        rabbitmq_user: Optional[str] = None,
        rabbitmq_password: Optional[str] = None,
        queue_name: str = "aclarai_dirty_blocks",
    ):
        """
        Initialize the RabbitMQ publisher.
        Args:
            rabbitmq_host: RabbitMQ host address
            rabbitmq_port: RabbitMQ port (default 5672)
            rabbitmq_user: Username for authentication (optional)
            rabbitmq_password: Password for authentication (optional)
            queue_name: Name of the queue to publish to
        """
        self.queue_name = queue_name
        # Create a minimal config object for the RabbitMQ manager
        # This preserves the existing interface while using the shared manager
        config = aclaraiConfig()
        config.rabbitmq_host = rabbitmq_host
        config.rabbitmq_port = rabbitmq_port
        config.rabbitmq_user = rabbitmq_user or ""
        config.rabbitmq_password = rabbitmq_password or ""
        self.rabbitmq_manager = RabbitMQManager(config, "vault-watcher")
        logger.info(
            "vault_watcher.DirtyBlockPublisher: Initialized RabbitMQ publisher",
            extra={
                "service": "vault-watcher",
                "filename.function_name": "rabbitmq_publisher.__init__",
                "rabbitmq_host": rabbitmq_host,
                "rabbitmq_port": rabbitmq_port,
                "queue_name": queue_name,
            },
        )

    def connect(self):
        """Establish connection to RabbitMQ."""
        self.rabbitmq_manager.connect()
        self.rabbitmq_manager.ensure_queue(self.queue_name, durable=True)

    def disconnect(self):
        """Close the RabbitMQ connection."""
        self.rabbitmq_manager.close()

    def publish_dirty_blocks(self, file_path: Path, dirty_blocks: Dict[str, Any]):
        """
        Publish dirty block notifications.
        Args:
            file_path: Path to the file containing dirty blocks
            dirty_blocks: Dictionary with 'added', 'modified', 'deleted' block lists
        """
        if not self.rabbitmq_manager.is_connected():
            logger.warning(
                "vault_watcher.DirtyBlockPublisher: Not connected to RabbitMQ, attempting to reconnect",
                extra={
                    "service": "vault-watcher",
                    "filename.function_name": "rabbitmq_publisher.publish_dirty_blocks",
                },
            )
            try:
                self.connect()
            except Exception as e:
                logger.error(
                    f"vault_watcher.DirtyBlockPublisher: Failed to reconnect to RabbitMQ: {e}",
                    extra={
                        "service": "vault-watcher",
                        "filename.function_name": "rabbitmq_publisher.publish_dirty_blocks",
                        "error": str(e),
                    },
                )
                return
        # Publish messages for each type of change
        for change_type in ["added", "modified", "deleted"]:
            blocks = dirty_blocks.get(change_type, [])
            for block in blocks:
                message = self._create_message(file_path, change_type, block)
                self._publish_message(message)

    def _create_message(
        self, file_path: Path, change_type: str, block: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a message for a dirty block.
        Args:
            file_path: Path to the file containing the block
            change_type: Type of change ('added', 'modified', 'deleted')
            block: Block information dictionary
        Returns:
            Message dictionary ready for JSON serialization
        """
        message = {
            "aclarai_id": block["aclarai_id"],
            "file_path": str(file_path),
            "change_type": change_type,
            "timestamp": int(time.time() * 1000),  # Milliseconds since epoch
            "version": block.get("version", block.get("new_version")),
            "block_type": block["block_type"],
        }
        # Add version-specific information for modifications
        if change_type == "modified":
            message["old_version"] = block.get("old_version")
            message["new_version"] = block.get("new_version")
            message["old_hash"] = block.get("old_hash")
            message["new_hash"] = block.get("new_hash")
        elif change_type in ["added", "deleted"]:
            message["content_hash"] = block["content_hash"]
        return message

    def _publish_message(self, message: Dict[str, Any]):
        """
        Publish a single message to RabbitMQ.
        Args:
            message: The message dictionary to publish
        """
        try:
            message_json = json.dumps(message)
            channel = self.rabbitmq_manager.get_channel()
            channel.basic_publish(
                exchange="",  # Use default exchange
                routing_key=self.queue_name,
                body=message_json,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                    content_type="application/json",
                ),
            )
            logger.debug(
                "vault_watcher.DirtyBlockPublisher: Published dirty block message",
                extra={
                    "service": "vault-watcher",
                    "filename.function_name": "rabbitmq_publisher._publish_message",
                    "aclarai_id": message["aclarai_id"],
                    "change_type": message["change_type"],
                    "file_path": message["file_path"],
                },
            )
        except AMQPChannelError as e:
            logger.error(
                f"vault_watcher.DirtyBlockPublisher: Channel error publishing message: {e}",
                extra={
                    "service": "vault-watcher",
                    "filename.function_name": "rabbitmq_publisher._publish_message",
                    "error": str(e),
                    "aclarai_id": message.get("aclarai_id"),
                    "change_type": message.get("change_type"),
                },
            )
            # Try to reconnect on channel errors
            try:
                self.connect()
            except Exception:
                logger.error(
                    "vault_watcher.DirtyBlockPublisher: Failed to reconnect after channel error",
                    extra={
                        "service": "vault-watcher",
                        "filename.function_name": "rabbitmq_publisher._publish_message",
                    },
                )
        except Exception as e:
            logger.error(
                f"vault_watcher.DirtyBlockPublisher: Unexpected error publishing message: {e}",
                extra={
                    "service": "vault-watcher",
                    "filename.function_name": "rabbitmq_publisher._publish_message",
                    "error": str(e),
                    "aclarai_id": message.get("aclarai_id"),
                    "change_type": message.get("change_type"),
                },
            )
