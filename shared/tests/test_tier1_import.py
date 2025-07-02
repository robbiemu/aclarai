"""
Test suite for the Tier 1 import system.
"""

import tempfile
from pathlib import Path

from aclarai_shared.config import PathsConfig, aclaraiConfig
from aclarai_shared.import_system import (
    Tier1ImportSystem,
    calculate_file_hash,
    ensure_defaults,
    format_tier1_markdown,
    generate_output_filename,
    is_duplicate_import,
    record_import,
    write_file_atomically,
)
from aclarai_shared.plugin_interface import MarkdownOutput


class TestTier1ImportSystem:
    """Test cases for the Tier 1 import system."""

    def test_calculate_file_hash(self):
        """Test file hash calculation."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_path = Path(f.name)
        try:
            hash1 = calculate_file_hash(temp_path)
            hash2 = calculate_file_hash(temp_path)
            # Same file should produce same hash
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA-256 hex length
        finally:
            temp_path.unlink()

    def test_atomic_file_write(self):
        """Test atomic file writing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "test.md"
            content = "# Test Content\n\nThis is a test."
            write_file_atomically(target_path, content)
            # File should exist and have correct content
            assert target_path.exists()
            assert target_path.read_text() == content

    def test_ensure_defaults(self):
        """Test metadata defaults are applied correctly."""
        with tempfile.NamedTemporaryFile() as f:
            source_path = Path(f.name)
            # Test with minimal metadata
            md = MarkdownOutput(
                title="Test Conversation",
                markdown_text="alice: hello\nbob: hi",
                metadata={"participants": ["alice", "bob"]},
            )
            result = ensure_defaults(md, source_path)
            assert result.title == "Test Conversation"
            assert result.metadata["participants"] == ["alice", "bob"]
            assert "created_at" in result.metadata
            assert result.metadata["message_count"] == 0  # Default value

    def test_format_tier1_markdown(self):
        """Test Tier 1 Markdown formatting."""
        md = MarkdownOutput(
            title="Test Chat",
            markdown_text="alice: hello\n<!-- aclarai:id=blk_abc123 ver=1 -->\n^blk_abc123",
            metadata={
                "created_at": "2024-01-01T10:00:00",
                "participants": ["alice", "bob"],
                "message_count": 1,
                "plugin_metadata": {"source": "test"},
            },
        )
        result = format_tier1_markdown(md)
        # Should have metadata comments at top
        lines = result.split("\n")
        assert lines[0].startswith("<!-- aclarai:title=")
        assert lines[1].startswith("<!-- aclarai:created_at=")
        assert lines[2].startswith("<!-- aclarai:participants=")
        assert lines[3].startswith("<!-- aclarai:message_count=")
        assert lines[4].startswith("<!-- aclarai:plugin_metadata=")
        # Should have empty line after metadata
        assert lines[5] == ""
        # Should have original content
        assert "alice: hello" in result
        assert "aclarai:id=blk_abc123" in result

    def test_generate_output_filename(self):
        """Test output filename generation."""
        md = MarkdownOutput(
            title="Team Meeting Discussion",
            markdown_text="content",
            metadata={"created_at": "2024-01-15T10:30:00"},
        )
        source_path = Path("/tmp/chat_export.txt")
        filename = generate_output_filename(md, source_path)
        assert filename.startswith("2024-01-15_")
        assert "chat_export" in filename
        assert "Team_Meeting_Discussion" in filename
        assert filename.endswith(".md")

    def test_duplicate_detection(self):
        """Test duplicate detection system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            import_log_dir = Path(temp_dir) / "logs"
            source_path = Path("/tmp/test.txt")
            hash_value = "abc123"
            output_files = [Path("/vault/test.md")]
            # Initially should not be duplicate
            assert not is_duplicate_import(source_path, hash_value, import_log_dir)
            # Record import
            record_import(source_path, hash_value, import_log_dir, output_files)
            # Now should be detected as duplicate
            assert is_duplicate_import(source_path, hash_value, import_log_dir)

    def test_import_system_initialization(self):
        """Test import system initialization."""
        config = aclaraiConfig(
            vault_path="/test/vault",
            paths=PathsConfig(tier1="conversations", logs="logs"),
        )
        system = Tier1ImportSystem(config)
        assert system.vault_path == Path("/test/vault")
        assert system.import_log_dir == Path("/test/vault/logs")
        assert system.plugin_manager.get_plugin_count() >= 1  # At least default plugin
        
        # Test plugin info
        plugin_info = system.get_plugin_info()
        assert "plugin_count" in plugin_info
        assert plugin_info["plugin_count"] >= 1


def test_integration_simple_conversation():
    """Integration test with a simple conversation file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup
        vault_dir = Path(temp_dir) / "vault"
        config = aclaraiConfig(
            vault_path=str(vault_dir), paths=PathsConfig(tier1="tier1", logs="logs")
        )
        system = Tier1ImportSystem(config)
        # Create test input file
        input_file = Path(temp_dir) / "test_chat.txt"
        input_file.write_text("""
alice: Hello, how are you doing today?
bob: I'm doing great, thanks for asking!
alice: That's wonderful to hear.
""")
        # Import the file
        try:
            output_files = system.import_file(input_file)
            # Should create one output file
            assert len(output_files) == 1
            output_file = output_files[0]
            assert output_file.exists()
            # Check content format
            content = output_file.read_text()
            # Should have metadata comments
            assert "<!-- aclarai:title=" in content
            assert "<!-- aclarai:participants=" in content
            # Should have conversation content
            assert "alice: Hello, how are you doing today?" in content
            assert "bob: I'm doing great, thanks for asking!" in content
            # Should have block annotations
            assert "<!-- aclarai:id=blk_" in content
            assert "^blk_" in content
            # Test duplicate detection
            try:
                system.import_file(input_file)
                raise AssertionError("Should have raised DuplicateDetectionError")
            except Exception as e:
                assert "duplicate" in str(e).lower()
        except Exception as e:
            # Print debugging info if test fails
            print(f"Import failed: {e}")
            if vault_dir.exists():
                print(f"Vault contents: {list(vault_dir.rglob('*'))}")
            raise


if __name__ == "__main__":
    # Run the integration test directly
    test_integration_simple_conversation()
    print("Integration test passed!")
