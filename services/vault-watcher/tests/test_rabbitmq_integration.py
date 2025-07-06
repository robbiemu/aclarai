"""
End-to-end integration tests for RabbitMQ message passing.
These tests require a live RabbitMQ instance and are marked as integration tests.
They will be skipped by default in CI unless specifically requested.
"""

import contextlib
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import pika
import pytest
from aclarai_shared import load_config
from aclarai_vault_watcher.main import VaultWatcherService


@pytest.mark.integration
class TestRabbitMQIntegration:
    """End-to-end tests for RabbitMQ message passing."""

    @pytest.fixture
    def config(self):
        """Fixture that provides the aclarai config."""
        return load_config(validate=False)  # Validation not needed for tests

    @pytest.fixture
    def rabbitmq_connection(self, config):
        """Fixture for RabbitMQ connection using Docker service."""
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=config.rabbitmq_host,
                port=config.rabbitmq_port,
                credentials=pika.PlainCredentials(
                    config.rabbitmq_user, config.rabbitmq_password
                ),
            )
        )
        channel = connection.channel()
        # Ensure the test queue exists
        queue_name = "test_aclarai_dirty_blocks"
        channel.queue_declare(queue=queue_name, durable=True)
        yield connection, channel, queue_name
        # Cleanup: purge test queue and close connection
        channel.queue_purge(queue=queue_name)
        connection.close()

    @pytest.fixture
    def vault_watcher_service(self, rabbitmq_connection, config):
        """Fixture that provides a vault watcher service with proper cleanup."""
        connection, channel, queue_name = rabbitmq_connection
        with tempfile.TemporaryDirectory() as temp_dir:
            # Start with the real config and just override what we need for testing
            config.vault_path = temp_dir
            config.paths.vault = temp_dir
            config.vault_watcher.queue_name = queue_name
            service = VaultWatcherService(config)

            yield service, temp_dir, queue_name

            # Cleanup: stop service if running
            with contextlib.suppress(Exception):
                service.stop()  # Service might not be started

    def test_message_publishing_format(
        self, vault_watcher_service, rabbitmq_connection
    ):
        """Test that messages are published with correct JSON format."""
        service, temp_dir, queue_name = vault_watcher_service
        connection, channel, _ = rabbitmq_connection
        # Create a test file with aclarai block
        vault_path = Path(temp_dir)
        test_file = vault_path / "test_message.md"
        test_content = """
# Test Document
This is a test claim. <!-- aclarai:id=clm_test_format ver=1 -->
^clm_test_format
        """.strip()
        test_file.write_text(test_content)
        # Start the service to establish connections
        service.start()
        try:
            # Trigger file creation handling
            service._handle_file_created(test_file)
            # Give time for message to be published
            time.sleep(0.5)
            # Consume message from queue
            method_frame, header_frame, body = channel.basic_get(
                queue=queue_name, auto_ack=True
            )
            assert method_frame is not None, "No message was published to queue"
            # Parse message body
            message_data = json.loads(body)
            # Verify message format
            assert "aclarai_id" in message_data
            assert "file_path" in message_data
            assert "change_type" in message_data
            assert "timestamp" in message_data
            assert "version" in message_data
            assert "block_type" in message_data
            # Verify message content
            assert message_data["aclarai_id"] == "clm_test_format"
            assert message_data["change_type"] == "added"
            assert message_data["version"] == 1
            assert message_data["block_type"] == "inline"
            assert str(test_file) in message_data["file_path"]
        finally:
            service.stop()

    def test_batch_message_publishing(self, vault_watcher_service, rabbitmq_connection):
        """Test that multiple file changes result in multiple messages."""
        service, temp_dir, queue_name = vault_watcher_service
        connection, channel, _ = rabbitmq_connection
        vault_path = Path(temp_dir)
        # Create multiple test files
        test_files = []
        for i in range(3):
            test_file = vault_path / f"test_batch_{i}.md"
            test_content = f"""
# Test Document {i}
This is test claim {i}. <!-- aclarai:id=clm_batch_{i} ver=1 -->
^clm_batch_{i}
            """.strip()
            test_file.write_text(test_content)
            test_files.append(test_file)
        service.start()
        try:
            # Trigger multiple file changes
            for test_file in test_files:
                service._handle_file_created(test_file)
            # Give time for messages to be published
            time.sleep(1.0)
            # Consume all messages
            messages = []
            while True:
                method_frame, header_frame, body = channel.basic_get(
                    queue=queue_name, auto_ack=True
                )
                if method_frame is None:
                    break
                messages.append(json.loads(body))
            # Should have received multiple messages
            assert len(messages) >= 3, (
                f"Expected at least 3 messages, got {len(messages)}"
            )
            # Verify each message has a different aclarai_id
            aclarai_ids = [msg["aclarai_id"] for msg in messages]
            expected_ids = [f"clm_batch_{i}" for i in range(3)]
            for expected_id in expected_ids:
                assert expected_id in aclarai_ids, f"Missing message for {expected_id}"
        finally:
            service.stop()

    def test_connection_failure_handling(self, vault_watcher_service):
        """Test that service handles RabbitMQ connection failures gracefully."""
        service, temp_dir, queue_name = vault_watcher_service
        # Mock the publisher to simulate connection failure
        with pytest.MonkeyPatch.context() as mp:

            def mock_failing_publisher(*_args, **_kwargs):
                mock_publisher = Mock()
                mock_publisher.connect.side_effect = (
                    pika.exceptions.AMQPConnectionError("Connection failed")
                )
                mock_publisher.publish_dirty_blocks.side_effect = (
                    pika.exceptions.AMQPConnectionError("Connection failed")
                )
                return mock_publisher

            mp.setattr(
                "aclarai_vault_watcher.main.DirtyBlockPublisher",
                mock_failing_publisher,
            )
            # Service should start even if RabbitMQ connection fails
            service.start()
            try:
                # Create a test file
                vault_path = Path(temp_dir)
                test_file = vault_path / "test_failure.md"
                test_content = "# Test <!-- aclarai:id=clm_test_fail ver=1 -->"
                test_file.write_text(test_content)
                # This should not raise an exception even though RabbitMQ is failing
                service._handle_file_created(test_file)
            finally:
                service.stop()

    def test_message_consumer_skeleton(self, rabbitmq_connection):
        """Skeleton test for message consumption - prepares for future consumer testing."""
        connection, channel, queue_name = rabbitmq_connection
        # Publish a test message manually
        test_message = {
            "aclarai_id": "clm_consumer_test",
            "file_path": "/test/path.md",
            "change_type": "modified",
            "timestamp": int(time.time() * 1000),
            "version": 2,
            "block_type": "inline",
        }
        channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=json.dumps(test_message),
            properties=pika.BasicProperties(delivery_mode=2),  # Make message persistent
        )
        # Consumer skeleton - future tests can expand this
        method_frame, header_frame, body = channel.basic_get(
            queue=queue_name, auto_ack=True
        )
        assert method_frame is not None, "Message was not consumed"
        consumed_message = json.loads(body)
        assert consumed_message["aclarai_id"] == "clm_consumer_test"
        assert consumed_message["change_type"] == "modified"
        # Future: This is where aclarai-core consumer logic would be tested
        # when the sync_vault_to_graph job is ready
