"""
Shared RabbitMQ connection manager for aclarai services.
This module provides centralized RabbitMQ connection management to eliminate
code duplication between the DirtyBlockConsumer and DirtyBlockPublisher.
"""

import logging
from typing import Optional

import pika
from pika.exceptions import AMQPConnectionError

from aclarai_shared.config import aclaraiConfig

logger = logging.getLogger(__name__)


class RabbitMQManager:
    """
    Centralized RabbitMQ connection manager.
    Handles connection establishment, channel management, and queue declarations
    for both consumer and publisher use cases.
    """

    def __init__(self, config: aclaraiConfig, service_name: str = "aclarai"):
        """
        Initialize the RabbitMQ manager.
        Args:
            config: aclarai configuration instance
            service_name: Name of the service for logging context
        """
        self.config = config
        self.service_name = service_name
        # RabbitMQ connection parameters from config
        self.rabbitmq_host = self.config.rabbitmq_host
        self.rabbitmq_port = getattr(self.config, "rabbitmq_port", 5672)
        self.rabbitmq_user = getattr(self.config, "rabbitmq_user", None)
        self.rabbitmq_password = getattr(self.config, "rabbitmq_password", None)
        self._connection: Optional[pika.BlockingConnection] = None
        self._channel: Optional[pika.channel.Channel] = None
        logger.info(
            f"{self.service_name}.RabbitMQManager: Initialized RabbitMQ manager",
            extra={
                "service": self.service_name,
                "filename.function_name": "rabbitmq_manager.__init__",
                "rabbitmq_host": self.rabbitmq_host,
                "rabbitmq_port": self.rabbitmq_port,
            },
        )

    def connect(self):
        """Establish connection to RabbitMQ."""
        try:
            # Set up connection parameters
            if self.rabbitmq_user and self.rabbitmq_password:
                credentials = pika.PlainCredentials(
                    self.rabbitmq_user, self.rabbitmq_password
                )
                parameters = pika.ConnectionParameters(
                    host=self.rabbitmq_host,
                    port=self.rabbitmq_port,
                    credentials=credentials,
                )
            else:
                parameters = pika.ConnectionParameters(
                    host=self.rabbitmq_host, port=self.rabbitmq_port
                )
            # Establish connection
            self._connection = pika.BlockingConnection(parameters)
            self._channel = self._connection.channel()
            logger.info(
                f"{self.service_name}.RabbitMQManager: Connected to RabbitMQ",
                extra={
                    "service": self.service_name,
                    "filename.function_name": "rabbitmq_manager.connect",
                    "rabbitmq_host": self.rabbitmq_host,
                },
            )
        except AMQPConnectionError as e:
            logger.error(
                f"{self.service_name}.RabbitMQManager: Failed to connect to RabbitMQ: {e}",
                extra={
                    "service": self.service_name,
                    "filename.function_name": "rabbitmq_manager.connect",
                    "error": str(e),
                    "rabbitmq_host": self.rabbitmq_host,
                },
            )
            raise

    def get_channel(self) -> pika.channel.Channel:
        """
        Get an active RabbitMQ channel.
        Automatically connects if not already connected or if connection is closed.
        Returns:
            Active RabbitMQ channel
        """
        if not self.is_connected():
            self.connect()
        return self._channel

    def ensure_queue(self, queue_name: str, durable: bool = True):
        """
        Ensure a queue exists by declaring it.
        Args:
            queue_name: Name of the queue to declare
            durable: Whether the queue should be durable (persistent)
        """
        channel = self.get_channel()
        channel.queue_declare(queue=queue_name, durable=durable)
        logger.debug(
            f"{self.service_name}.RabbitMQManager: Queue declared",
            extra={
                "service": self.service_name,
                "filename.function_name": "rabbitmq_manager.ensure_queue",
                "queue_name": queue_name,
                "durable": durable,
            },
        )

    def close(self):
        """Close the RabbitMQ connection."""
        if self._connection and not self._connection.is_closed:
            self._connection.close()
            self._connection = None
            self._channel = None
            logger.info(
                f"{self.service_name}.RabbitMQManager: Disconnected from RabbitMQ",
                extra={
                    "service": self.service_name,
                    "filename.function_name": "rabbitmq_manager.close",
                },
            )

    def is_connected(self) -> bool:
        """Check if the RabbitMQ connection is active."""
        return (
            self._connection is not None
            and not self._connection.is_closed
            and self._channel is not None
            and self._channel.is_open
        )
