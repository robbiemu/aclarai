"""
Integration tests for the aclarai-core consumer and end-to-end reactive sync loop.
Tests the complete flow from RabbitMQ message consumption to Neo4j database updates.
"""

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pika
import pytest
from aclarai_core.dirty_block_consumer import DirtyBlockConsumer
from aclarai_shared import load_config
from aclarai_shared.graph.neo4j_manager import Neo4jGraphManager


@pytest.mark.integration
class TestConsumerIntegration:
    """Integration tests for consumer requiring live Neo4j and RabbitMQ."""

    @pytest.fixture(scope="class")
    def integration_neo4j_manager(self):
        """Fixture to set up a connection to a real Neo4j database for testing."""
        if not os.getenv("NEO4J_PASSWORD"):
            pytest.skip("NEO4J_PASSWORD not set for integration tests.")

        config = load_config(validate=True)
        manager = Neo4jGraphManager(config=config)
        manager.setup_schema()
        # Clean up any existing test data
        with manager.session() as session:
            session.run(
                "MATCH (n) WHERE n.id STARTS WITH 'test_consumer_' DETACH DELETE n"
            )
        yield manager
        # Clean up after tests
        with manager.session() as session:
            session.run(
                "MATCH (n) WHERE n.id STARTS WITH 'test_consumer_' DETACH DELETE n"
            )
        manager.close()

    @pytest.fixture
    def rabbitmq_connection(self):
        """Fixture for RabbitMQ connection - requires live RabbitMQ instance."""
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host="localhost", port=5672)
            )
            channel = connection.channel()
            # Ensure the test queue exists
            queue_name = "test_aclarai_dirty_blocks_consumer"
            channel.queue_declare(queue=queue_name, durable=True)
            yield connection, channel, queue_name
            # Cleanup: purge test queue and close connection
            channel.queue_purge(queue=queue_name)
            connection.close()
        except pika.exceptions.AMQPConnectionError:
            pytest.skip("RabbitMQ not available for integration testing")

    @pytest.fixture
    def temp_vault_path(self):
        """Create a temporary vault directory with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_path = Path(temp_dir)

            # Create tier1 directory structure
            tier1_path = vault_path / "tier1"
            tier1_path.mkdir(parents=True)

            # Create a test markdown file
            test_file = tier1_path / "test_consumer_doc.md"
            test_content = """# Consumer Test Document

## Test Block
aclarai:id=test_consumer_block_001 ver=2 hash=updated_hash_456

This is updated content that should be processed by the consumer.
The version has been incremented to trigger an update.
"""
            test_file.write_text(test_content)

            yield vault_path

    @pytest.mark.integration
    def test_consumer_updates_graph_from_rabbitmq_message(
        self, integration_neo4j_manager, rabbitmq_connection, temp_vault_path
    ):
        """
        Tests the full consumer flow from RabbitMQ to Neo4j.
        1. Manually publish a valid "dirty block" message to the queue.
        2. Run the DirtyBlockConsumer in a separate thread.
        3. Wait for the message to be processed.
        4. Query Neo4j to assert that the corresponding :Block node was updated correctly.
        """
        connection, channel, queue_name = rabbitmq_connection

        # Pre-populate Neo4j with an existing block (old version)
        with integration_neo4j_manager.session() as session:
            session.run("""
                CREATE (:Block {
                    id: 'test_consumer_block_001',
                    content: 'Old content before update',
                    version: 1,
                    hash: 'old_hash_123',
                    file_path: 'tier1/test_consumer_doc.md',
                    aclarai_id: 'test_consumer_block_001'
                })
            """)

        # Create a dirty block message
        test_message = {
            "aclarai_id": "test_consumer_block_001",
            "file_path": str(temp_vault_path / "tier1" / "test_consumer_doc.md"),
            "change_type": "modified",
            "timestamp": int(time.time() * 1000),
            "version": 2,
            "block_type": "inline",
            "old_version": 1,
            "new_version": 2,
            "old_hash": "old_hash_123",
            "new_hash": "updated_hash_456",
        }

        # Publish the message to RabbitMQ
        channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=json.dumps(test_message),
            properties=pika.BasicProperties(delivery_mode=2),  # Make message persistent
        )

        # Create and configure the consumer
        config = load_config(validate=True)
        with patch.object(config, "vault_path", str(temp_vault_path)):
            with patch.object(config.vault_watcher, "queue_name", queue_name):
                consumer = DirtyBlockConsumer(config=config)

                # Track if message was processed
                processed = threading.Event()
                original_process_message = consumer.process_message

                def mock_process_message(*args, **kwargs):
                    try:
                        result = original_process_message(*args, **kwargs)
                        processed.set()
                        return result
                    except Exception as e:
                        processed.set()
                        raise e

                consumer.process_message = mock_process_message

                # Start consumer in a separate thread
                consumer_thread = threading.Thread(target=consumer.start_consuming)
                consumer_thread.daemon = True
                consumer_thread.start()

                # Wait for message to be processed (timeout after 10 seconds)
                if not processed.wait(timeout=10):
                    pytest.fail("Message was not processed within timeout")

                # Stop the consumer
                consumer.stop_consuming()
                consumer_thread.join(timeout=5)

        # Verify the block was updated in Neo4j
        with integration_neo4j_manager.session() as session:
            result = session.run("""
                MATCH (b:Block {id: 'test_consumer_block_001'}) 
                RETURN b.version as version, b.hash as hash, b.content as content
            """)
            record = result.single()
            assert record is not None
            assert record["version"] == 2  # Version should have been updated
            assert record["hash"] == "updated_hash_456"  # Hash should match new content
            assert (
                "This is updated content that should be processed by the consumer"
                in record["content"]
            )

    @pytest.mark.integration
    def test_consumer_handles_version_conflicts(
        self, integration_neo4j_manager, rabbitmq_connection, temp_vault_path
    ):
        """
        Test that the consumer properly handles version conflicts.
        When a message has an old version number, it should be ignored.
        """
        connection, channel, queue_name = rabbitmq_connection

        # Pre-populate Neo4j with a newer version of the block
        with integration_neo4j_manager.session() as session:
            session.run("""
                CREATE (:Block {
                    id: 'test_consumer_block_002',
                    content: 'Newer content already in database',
                    version: 5,
                    hash: 'newer_hash_789',
                    file_path: 'tier1/test_consumer_doc.md',
                    aclarai_id: 'test_consumer_block_002'
                })
            """)

        # Create a dirty block message with an older version
        test_message = {
            "aclarai_id": "test_consumer_block_002",
            "file_path": str(temp_vault_path / "tier1" / "test_consumer_doc.md"),
            "change_type": "modified",
            "timestamp": int(time.time() * 1000),
            "version": 3,  # Older than what's in the database
            "block_type": "inline",
            "old_version": 2,
            "new_version": 3,
            "old_hash": "old_hash_456",
            "new_hash": "outdated_hash_123",
        }

        # Publish the message
        channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=json.dumps(test_message),
            properties=pika.BasicProperties(delivery_mode=2),
        )

        # Process the message
        config = load_config(validate=True)
        with patch.object(config, "vault_path", str(temp_vault_path)):
            with patch.object(config.vault_watcher, "queue_name", queue_name):
                consumer = DirtyBlockConsumer(config=config)

                # Process one message and then stop
                processed = threading.Event()
                original_process_message = consumer.process_message

                def mock_process_message(*args, **kwargs):
                    try:
                        result = original_process_message(*args, **kwargs)
                        processed.set()
                        return result
                    except Exception as e:
                        processed.set()
                        raise e

                consumer.process_message = mock_process_message

                consumer_thread = threading.Thread(target=consumer.start_consuming)
                consumer_thread.daemon = True
                consumer_thread.start()

                if not processed.wait(timeout=10):
                    pytest.fail("Message was not processed within timeout")

                consumer.stop_consuming()
                consumer_thread.join(timeout=5)

        # Verify the block was NOT updated (version conflict)
        with integration_neo4j_manager.session() as session:
            result = session.run("""
                MATCH (b:Block {id: 'test_consumer_block_002'}) 
                RETURN b.version as version, b.hash as hash, b.content as content
            """)
            record = result.single()
            assert record is not None
            assert record["version"] == 5  # Version should remain unchanged
            assert record["hash"] == "newer_hash_789"  # Hash should remain unchanged
            assert record["content"] == "Newer content already in database"

    @pytest.mark.integration
    def test_consumer_creates_new_blocks(
        self, integration_neo4j_manager, rabbitmq_connection, temp_vault_path
    ):
        """
        Test that the consumer creates new blocks when they don't exist in Neo4j.
        """
        connection, channel, queue_name = rabbitmq_connection

        # Ensure the block doesn't exist initially
        with integration_neo4j_manager.session() as session:
            session.run("MATCH (b:Block {id: 'test_consumer_block_003'}) DELETE b")

        # Create a dirty block message for a new block
        test_message = {
            "aclarai_id": "test_consumer_block_003",
            "file_path": str(temp_vault_path / "tier1" / "test_consumer_doc.md"),
            "change_type": "added",
            "timestamp": int(time.time() * 1000),
            "version": 1,
            "block_type": "inline",
            "old_version": 0,
            "new_version": 1,
            "old_hash": "",
            "new_hash": "new_block_hash_999",
        }

        # Publish the message
        channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=json.dumps(test_message),
            properties=pika.BasicProperties(delivery_mode=2),
        )

        # Process the message
        config = load_config(validate=True)
        with patch.object(config, "vault_path", str(temp_vault_path)):
            with patch.object(config.vault_watcher, "queue_name", queue_name):
                consumer = DirtyBlockConsumer(config=config)

                processed = threading.Event()
                original_process_message = consumer.process_message

                def mock_process_message(*args, **kwargs):
                    try:
                        result = original_process_message(*args, **kwargs)
                        processed.set()
                        return result
                    except Exception as e:
                        processed.set()
                        raise e

                consumer.process_message = mock_process_message

                consumer_thread = threading.Thread(target=consumer.start_consuming)
                consumer_thread.daemon = True
                consumer_thread.start()

                if not processed.wait(timeout=10):
                    pytest.fail("Message was not processed within timeout")

                consumer.stop_consuming()
                consumer_thread.join(timeout=5)

        # Verify the new block was created
        with integration_neo4j_manager.session() as session:
            result = session.run("""
                MATCH (b:Block {id: 'test_consumer_block_003'}) 
                RETURN b.version as version, b.hash as hash, b.aclarai_id as aclarai_id
            """)
            record = result.single()
            assert record is not None
            assert record["version"] == 1
            assert record["aclarai_id"] == "test_consumer_block_003"
