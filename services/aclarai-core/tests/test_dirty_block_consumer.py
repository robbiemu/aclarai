(
    """
"""
    """
Tests for the dirty block consumer functionality.
These tests validate the reactive sync loop implementation including
version checking, conflict detection, and proper graph updates.
"""
)

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from aclarai_core.dirty_block_consumer import DirtyBlockConsumer


class TestDirtyBlockConsumer:
    """Test cases for DirtyBlockConsumer."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        config = Mock()
        config.rabbitmq_host = "localhost"
        config.rabbitmq_port = 5672
        config.rabbitmq_user = None
        config.rabbitmq_password = None
        config.vault_path = "/test/vault"
        # Provide a dictionary for the ToolFactory via the mocked .dict() method
        config.dict = Mock(
            return_value={"tools": {"agent_tool_mappings": {"entailment_agent": []}}}
        )
        return config

    @pytest.fixture
    def consumer(self, mock_config):
        """Create a DirtyBlockConsumer instance for testing."""
        with (
            patch("aclarai_core.dirty_block_consumer.Neo4jGraphManager"),
            patch("aclarai_core.dirty_block_consumer.ConceptProcessor"),
            patch("aclarai_core.agents.entailment_agent.CodeActAgent"),
            patch("aclarai_core.dirty_block_consumer.BlockParser"),
            # Patch the LLM initialization to avoid real model loading
            patch(
                "aclarai_core.dirty_block_consumer.DirtyBlockConsumer._initialize_llm",
                return_value=Mock(),
            ),
            patch(
                "aclarai_shared.tools.vector_store_manager.aclaraiVectorStoreManager"
            ),
        ):
            consumer_instance = DirtyBlockConsumer(mock_config)
            consumer_instance.entailment_agent = Mock()
            consumer_instance.entailment_agent.evaluate_entailment.return_value = (
                None,
                "success",
            )
            consumer_instance.block_parser = Mock()
            consumer_instance.block_parser.extract_aclarai_blocks.return_value = []
            consumer_instance.block_parser.calculate_content_hash.side_effect = (
                lambda text: f"hash_for_{' '.join(text.split())}"
            )
            consumer_instance.block_parser.find_block_by_id.return_value = {
                "aclarai_id": "clm_target",
                "version": 2,
                "semantic_text": "This is the target claim.",
                "content_hash": "hash_for_This is the target claim.",
            }

            return consumer_instance

    @pytest.fixture
    def sample_message(self):
        """Sample RabbitMQ message for testing."""
        return {
            "aclarai_id": "clm_test123",
            "file_path": "tier1/test.md",
            "change_type": "modified",
            "timestamp": 1234567890,
            "version": 2,
            "block_type": "inline",
            "old_version": 1,
            "new_version": 2,
            "old_hash": "abc123",
            "new_hash": "def456",
        }

    def test_extract_aclarai_blocks(self, consumer):
        """Test extraction of aclarai blocks from markdown content."""
        content = """
Some text before.
This is a claim. <!-- aclarai:id=clm_abc123 ver=2 -->
^clm_abc123
Another claim here. <!-- aclarai:id=clm_def456 ver=1 -->
^clm_def456
## File-level document
Some content for the whole file.
<!-- aclarai:id=file_summary ver=3 -->
"""
        consumer.block_parser.extract_aclarai_blocks.return_value = [
            {
                "aclarai_id": "clm_abc123",
                "version": 2,
                "semantic_text": "This is a claim.",
            },
            {
                "aclarai_id": "clm_def456",
                "version": 1,
                "semantic_text": "Another claim here.",
            },
            {
                "aclarai_id": "file_summary",
                "version": 3,
                "semantic_text": "File-level document",
            },
        ]

        blocks = consumer.block_parser.extract_aclarai_blocks(content)
        assert len(blocks) == 3
        # Check first inline block
        assert blocks[0]["aclarai_id"] == "clm_abc123"
        assert blocks[0]["version"] == 2
        assert "This is a claim." in blocks[0]["semantic_text"]
        # Check second inline block
        assert blocks[1]["aclarai_id"] == "clm_def456"
        assert blocks[1]["version"] == 1
        assert "Another claim here." in blocks[1]["semantic_text"]
        # Check file-level block
        assert blocks[2]["aclarai_id"] == "file_summary"
        assert blocks[2]["version"] == 3
        # File-level block should contain most of the content before the comment
        assert "File-level document" in blocks[2]["semantic_text"]

    def test_calculate_content_hash(self, consumer):
        """Test content hash calculation."""
        text1 = "This is some text."
        text2 = "This   is    some   text."  # Different whitespace
        text3 = "This is some other text."
        hash1 = consumer.block_parser.calculate_content_hash(text1)
        hash2 = consumer.block_parser.calculate_content_hash(text2)
        hash3 = consumer.block_parser.calculate_content_hash(text3)
        # Same content with different whitespace should have same hash
        assert hash1 == hash2
        # Different content should have different hash
        assert hash1 != hash3

    @patch("aclarai_core.dirty_block_consumer.Path.exists")
    @patch("builtins.open")
    def test_read_block_from_file(self, mock_open, mock_exists, consumer):
        """Test reading a specific block from file."""
        mock_exists.return_value = True
        file_content = """
Some content.
This is the target claim. <!-- aclarai:id=clm_target ver=2 -->
^clm_target
Another claim. <!-- aclarai:id=clm_other ver=1 -->
^clm_other
"""
        mock_open.return_value.__enter__.return_value.read.return_value = file_content
        block = consumer._read_block_from_file(Path("test.md"), "clm_target")
        assert block is not None
        assert block["aclarai_id"] == "clm_target"
        assert block["version"] == 2
        assert "This is the target claim." in block["semantic_text"]

    def test_sync_block_new_block(self, consumer):
        """Test syncing a new block (not in graph)."""
        block = {
            "aclarai_id": "clm_new123",
            "version": 1,
            "semantic_text": "This is a new claim.",
            "content_hash": "abc123",
        }
        # Mock graph manager to return None (block doesn't exist)
        consumer._get_block_from_graph = Mock(return_value=None)
        consumer._create_block_in_graph = Mock()
        result = consumer._sync_block_with_graph(block, Path("test.md"))
        assert result is True
        consumer._get_block_from_graph.assert_called_once_with("clm_new123")
        consumer._create_block_in_graph.assert_called_once()

    def test_sync_block_unchanged_content(self, consumer):
        """Test syncing a block with unchanged content."""
        block = {
            "aclarai_id": "clm_existing",
            "version": 2,
            "semantic_text": "This is an existing claim.",
            "content_hash": "same_hash",
        }
        existing_block = {
            "id": "clm_existing",
            "version": 2,
            "hash": "same_hash",  # Same hash = no change
            "text": "This is an existing claim.",
        }
        consumer._get_block_from_graph = Mock(return_value=existing_block)
        consumer._update_block_in_graph = Mock()
        result = consumer._sync_block_with_graph(block, Path("test.md"))
        assert result is True
        # Should not call update since content is unchanged
        consumer._update_block_in_graph.assert_not_called()

    def test_sync_block_version_conflict(self, consumer):
        """Test version conflict detection (vault older than graph)."""
        block = {
            "aclarai_id": "clm_conflict",
            "version": 1,  # Vault version 1
            "semantic_text": "This is a conflicted claim.",
            "content_hash": "new_hash",
        }
        existing_block = {
            "id": "clm_conflict",
            "version": 3,  # Graph version 3 > vault version 1
            "hash": "old_hash",
            "text": "Previous version",
        }
        consumer._get_block_from_graph = Mock(return_value=existing_block)
        consumer._update_block_in_graph = Mock()
        result = consumer._sync_block_with_graph(block, Path("test.md"))
        assert result is True  # Don't fail, but skip update
        # Should not call update due to version conflict
        consumer._update_block_in_graph.assert_not_called()

    def test_sync_block_successful_update(self, consumer):
        """Test successful block update (vault newer or equal)."""
        block = {
            "aclarai_id": "clm_update",
            "version": 3,  # Vault version 3
            "semantic_text": "This is an updated claim.",
            "content_hash": "new_hash",
        }
        existing_block = {
            "id": "clm_update",
            "version": 2,  # Graph version 2 < vault version 3
            "hash": "old_hash",
            "text": "Previous version",
        }
        consumer._get_block_from_graph = Mock(return_value=existing_block)
        consumer._update_block_in_graph = Mock()
        result = consumer._sync_block_with_graph(block, Path("test.md"))
        assert result is True
        # Should call update since vault is newer
        consumer._update_block_in_graph.assert_called_once()

    @patch("datetime.datetime")
    def test_update_block_increments_version(self, mock_datetime_class, consumer):
        """Test that updating a block increments the graph version."""
        mock_datetime_class.now.return_value.isoformat.return_value = (
            "2024-01-01T00:00:00"
        )
        block = {
            "aclarai_id": "clm_increment",
            "semantic_text": "Updated text",
            "content_hash": "new_hash",
        }
        existing_block = {
            "version": 5  # Current graph version
        }
        # Mock the graph manager
        mock_session = Mock()
        consumer.graph_manager.session.return_value.__enter__.return_value = (
            mock_session
        )
        consumer.graph_manager._retry_with_backoff = Mock(
            side_effect=lambda func: func()
        )
        consumer._update_block_in_graph(block, existing_block, Path("test.md"))
        # Verify the Cypher query was called with incremented version (5 + 1 = 6)
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert call_args[1]["version"] == 6  # Should be incremented
        assert call_args[1]["aclarai_id"] == "clm_increment"
        assert call_args[1]["text"] == "Updated text"
        assert call_args[1]["hash"] == "new_hash"

    def test_process_dirty_block_deleted(self, consumer):
        """Test processing deleted block messages."""
        message = {
            "aclarai_id": "clm_deleted",
            "file_path": "test.md",
            "change_type": "deleted",
        }
        # Should skip deleted blocks and return success
        result = consumer._process_dirty_block(message)
        assert result is True

    @patch.object(DirtyBlockConsumer, "_read_block_from_file")
    @patch.object(DirtyBlockConsumer, "_sync_block_with_graph")
    def test_process_dirty_block_modified(self, mock_sync, mock_read, consumer):
        """Test processing modified block messages."""
        message = {
            "aclarai_id": "clm_modified",
            "file_path": "test.md",
            "change_type": "modified",
        }
        mock_block = {
            "aclarai_id": "clm_modified",
            "version": 2,
            "semantic_text": "Modified text",
            "content_hash": "hash123",
        }
        mock_read.return_value = mock_block
        mock_sync.return_value = True
        result = consumer._process_dirty_block(message)
        assert result is True
        mock_read.assert_called_once_with(Path("test.md"), "clm_modified")
        mock_sync.assert_called_once_with(mock_block, Path("test.md"))

    def test_process_dirty_block_missing_fields(self, consumer):
        """Test processing message with missing required fields."""
        message = {
            "aclarai_id": "clm_test",
            # Missing file_path and change_type
        }
        result = consumer._process_dirty_block(message)
        assert result is False


""
