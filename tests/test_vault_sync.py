"""
Tests for vault synchronization functionality.
Tests the vault sync job implementation including markdown parsing,
hash calculation, and Neo4j synchronization.
"""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Note: These imports may fail without proper package installation
# but the test structure demonstrates the intended functionality
try:
    from aclarai_scheduler.vault_sync import VaultSyncJob

    VAULT_SYNC_AVAILABLE = True
except ImportError:
    # Define a mock VaultSyncJob for testing structure
    class VaultSyncJob:
        def __init__(self, config=None):
            pass

        def _extract_aclarai_blocks(self, content):  # noqa: ARG002
            return []

        def _calculate_content_hash(self, text):
            return hashlib.sha256(text.encode("utf-8")).hexdigest()

    VAULT_SYNC_AVAILABLE = False


def _create_mock_config():
    """Create a mock configuration for testing."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/test/vault"  # Changed from mock_config.vault_path
    mock_config.paths.tier1 = "conversations" # Using new default
    mock_config.paths.tier2 = "summaries"     # Using new default
    mock_config.paths.tier3 = "concepts"      # Using new default
    return mock_config


@patch("aclarai_scheduler.vault_sync.load_config")
@patch("aclarai_scheduler.vault_sync.Neo4jGraphManager")
def test_extract_aclarai_blocks(mock_neo4j, mock_load_config):
    """Test extraction of aclarai:id blocks from markdown content."""
    # Setup mocks to avoid database connection
    mock_load_config.return_value = _create_mock_config()
    mock_neo4j.return_value = MagicMock()
    vault_sync = VaultSyncJob()
    # Test content with inline blocks
    content = """
# Test Document
Alice: This is the first utterance. <!-- aclarai:id=blk_abc123 ver=1 -->
^blk_abc123
Bob: This is the second utterance. <!-- aclarai:id=blk_def456 ver=2 -->
^blk_def456
Some text without aclarai:id.
"""
    blocks = vault_sync.block_parser.extract_aclarai_blocks(content)
    # Should extract 2 blocks
    assert len(blocks) == 2
    # Check first block
    block1 = blocks[0]
    assert block1["aclarai_id"] == "blk_abc123"
    assert block1["version"] == 1
    assert "Alice: This is the first utterance." in block1["semantic_text"]
    # Check second block
    block2 = blocks[1]
    assert block2["aclarai_id"] == "blk_def456"
    assert block2["version"] == 2
    assert "Bob: This is the second utterance." in block2["semantic_text"]


@patch("aclarai_scheduler.vault_sync.load_config")
@patch("aclarai_scheduler.vault_sync.Neo4jGraphManager")
def test_extract_file_level_block(mock_neo4j, mock_load_config):
    """Test extraction of file-level aclarai:id blocks."""
    # Setup mocks to avoid database connection
    mock_load_config.return_value = _create_mock_config()
    mock_neo4j.return_value = MagicMock()
    vault_sync = VaultSyncJob()
    # Test content with file-level block (comment at end)
    content = """
# Top Concepts Report
- Concept A: Important topic
- Concept B: Another topic
- Concept C: Third topic
<!-- aclarai:id=file_top_concepts_20240615 ver=1 -->
"""
    blocks = vault_sync.block_parser.extract_aclarai_blocks(content)
    # Should extract 1 file-level block
    assert len(blocks) == 1
    block = blocks[0]
    assert block["aclarai_id"] == "file_top_concepts_20240615"
    assert block["version"] == 1
    # Semantic text should be the entire content excluding the comment
    assert "# Top Concepts Report" in block["semantic_text"]
    assert "aclarai:id" not in block["semantic_text"]


@patch("services.scheduler.aclarai_scheduler.vault_sync.load_config")
@patch("services.scheduler.aclarai_scheduler.vault_sync.Neo4jGraphManager")
def test_calculate_content_hash(mock_neo4j, mock_load_config):
    """Test content hash calculation."""
    # Setup mocks to avoid database connection
    mock_load_config.return_value = _create_mock_config()
    mock_neo4j.return_value = MagicMock()
    vault_sync = VaultSyncJob()
    # Test that same content produces same hash
    text1 = "Alice: This is a test utterance."
    text2 = "Alice: This is a test utterance."
    hash1 = vault_sync.block_parser.calculate_content_hash(text1)
    hash2 = vault_sync.block_parser.calculate_content_hash(text2)
    assert hash1 == hash2
    # Test that different content produces different hash
    text3 = "Bob: This is a different utterance."
    hash3 = vault_sync.block_parser.calculate_content_hash(text3)
    assert hash1 != hash3
    # Test whitespace normalization
    text4 = "Alice:   This   is   a   test   utterance."
    hash4 = vault_sync.block_parser.calculate_content_hash(text4)
    assert hash1 == hash4  # Should be same after normalization


def test_process_markdown_file_integration():
    """Integration test for processing a markdown file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test markdown file
        test_file = Path(temp_dir) / "test_conversation.md"
        test_content = """
# Test Conversation
Alice: Hello there! <!-- aclarai:id=blk_greeting ver=1 -->
^blk_greeting
Bob: Hi Alice, how are you? <!-- aclarai:id=blk_response ver=1 -->
^blk_response
Alice: I'm doing well, thanks! <!-- aclarai:id=blk_followup ver=1 -->
^blk_followup
"""
        test_file.write_text(test_content)
        # Mock the graph manager and config
        with (
            patch("services.scheduler.aclarai_scheduler.vault_sync.load_config") as mock_load_config_patch, # Renamed for clarity
            patch("services.scheduler.aclarai_scheduler.vault_sync.Neo4jGraphManager") as mock_graph,
        ):
            # Configure mocks
            mock_config_instance = _create_mock_config() # Use the helper
            mock_config_instance.paths.vault = temp_dir # Override vault path for temp_dir
            mock_config_instance.paths.tier1 = "." # Specific for this test case
            mock_load_config_patch.return_value = mock_config_instance

            mock_graph_instance = Mock()
            mock_graph.return_value = mock_graph_instance
            # Mock graph responses (no existing blocks)
            mock_graph_instance._retry_with_backoff.return_value = None
            vault_sync = VaultSyncJob()
            # Process the file
            stats = vault_sync._process_markdown_file(test_file, "tier1")
            # Should have processed 3 blocks
            assert stats["blocks_processed"] == 3
            assert stats["blocks_new"] == 3
            assert stats["blocks_updated"] == 0
            assert stats["errors"] == 0


@patch("services.scheduler.aclarai_scheduler.vault_sync.load_config")
@patch("services.scheduler.aclarai_scheduler.vault_sync.Neo4jGraphManager")
def test_vault_sync_job_stats_merging(mock_neo4j, mock_load_config):
    """Test that statistics are properly merged across files and tiers."""
    # Setup mocks to avoid database connection
    mock_load_config.return_value = _create_mock_config()
    mock_neo4j.return_value = MagicMock()
    vault_sync = VaultSyncJob()
    # Test stats merging
    target = {
        "files_processed": 1,
        "blocks_processed": 2,
        "blocks_new": 1,
        "blocks_updated": 1,
        "errors": 0,
    }
    source = {
        "files_processed": 2,
        "blocks_processed": 3,
        "blocks_new": 2,
        "blocks_updated": 0,
        "errors": 1,
    }
    vault_sync._merge_stats(target, source)
    assert target["files_processed"] == 3
    assert target["blocks_processed"] == 5
    assert target["blocks_new"] == 3
    assert target["blocks_updated"] == 1
    assert target["errors"] == 1


if __name__ == "__main__":
    # Run basic tests
    print("Running vault sync tests...")
    try:
        test_extract_aclarai_blocks()
        print("✓ test_extract_aclarai_blocks passed")
    except Exception as e:
        print(f"✗ test_extract_aclarai_blocks failed: {e}")
    try:
        test_extract_file_level_block()
        print("✓ test_extract_file_level_block passed")
    except Exception as e:
        print(f"✗ test_extract_file_level_block failed: {e}")
    try:
        test_calculate_content_hash()
        print("✓ test_calculate_content_hash passed")
    except Exception as e:
        print(f"✗ test_calculate_content_hash failed: {e}")
    try:
        test_vault_sync_job_stats_merging()
        print("✓ test_vault_sync_job_stats_merging passed")
    except Exception as e:
        print(f"✗ test_vault_sync_job_stats_merging failed: {e}")
    print("Basic tests completed.")
