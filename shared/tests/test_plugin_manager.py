"""
Tests for plugin manager and import orchestrator.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock

from aclarai_shared.plugin_interface import MarkdownOutput, Plugin
from aclarai_shared.plugin_manager import (
    ImportOrchestrator,
    ImportResult,
    ImportStatus,
    PluginManager,
    orchestrate_file_import,
)
from aclarai_shared.plugins.default_plugin import DefaultPlugin


class MockSpecificPlugin(Plugin):
    """Mock plugin for testing that accepts specific content."""

    def can_accept(self, raw_input: str) -> bool:
        return "SPECIFIC_FORMAT" in raw_input

    def convert(self, _raw_input: str, _path: Path) -> list[MarkdownOutput]:
        return [
            MarkdownOutput(
                title="Mock Specific Conversion",
                markdown_text="Mock content from specific plugin",
                metadata={"plugin_metadata": {"source_format": "mock_specific"}},
            )
        ]


class MockFailingPlugin(Plugin):
    """Mock plugin that fails during conversion."""

    def can_accept(self, raw_input: str) -> bool:
        return "FAILING_FORMAT" in raw_input

    def convert(self, raw_input: str, path: Path) -> list[MarkdownOutput]:
        raise ValueError("Simulated plugin failure")


class MockEmptyPlugin(Plugin):
    """Mock plugin that returns empty results."""

    def can_accept(self, raw_input: str) -> bool:
        return "EMPTY_FORMAT" in raw_input

    def convert(self, raw_input: str, path: Path) -> list[MarkdownOutput]:
        return []


class TestPluginManager:
    """Test cases for PluginManager class."""

    def test_plugin_manager_initialization(self):
        """Test PluginManager initializes with default plugin."""
        manager = PluginManager()

        assert manager.get_plugin_count() > 0
        plugins = manager.get_plugins()
        assert any(isinstance(p, DefaultPlugin) for p in plugins)

        metadata = manager.get_plugin_metadata()
        assert "DefaultPlugin" in metadata
        assert metadata["DefaultPlugin"]["is_fallback"] is True

    def test_register_specific_plugin(self):
        """Test registering a specific plugin before fallback."""
        manager = PluginManager()
        initial_count = manager.get_plugin_count()

        # Register specific plugin
        specific_plugin = MockSpecificPlugin()
        manager.register_plugin(specific_plugin)

        assert manager.get_plugin_count() == initial_count + 1

        plugins = manager.get_plugins()
        # Specific plugin should come before DefaultPlugin
        specific_index = next(
            i for i, p in enumerate(plugins) if isinstance(p, MockSpecificPlugin)
        )
        default_index = next(
            i for i, p in enumerate(plugins) if isinstance(p, DefaultPlugin)
        )
        assert specific_index < default_index

    def test_register_fallback_plugin(self):
        """Test registering fallback plugin goes to end."""
        manager = PluginManager()

        # Register another fallback plugin
        another_fallback = DefaultPlugin()
        manager.register_plugin(another_fallback, is_fallback=True)

        plugins = manager.get_plugins()
        # Should have two DefaultPlugin instances at the end
        assert isinstance(plugins[-1], DefaultPlugin)
        assert isinstance(plugins[-2], DefaultPlugin)

    def test_find_accepting_plugin_specific(self):
        """Test finding specific plugin that accepts input."""
        manager = PluginManager()
        specific_plugin = MockSpecificPlugin()
        manager.register_plugin(specific_plugin)

        # Test specific plugin acceptance
        accepting_plugin = manager.find_accepting_plugin(
            "This is SPECIFIC_FORMAT content"
        )
        assert isinstance(accepting_plugin, MockSpecificPlugin)

    def test_find_accepting_plugin_fallback(self):
        """Test fallback to default plugin."""
        manager = PluginManager()

        # Test fallback to default plugin
        accepting_plugin = manager.find_accepting_plugin("Random content")
        assert isinstance(accepting_plugin, DefaultPlugin)

    def test_find_accepting_plugin_none(self):
        """Test when no plugin accepts (shouldn't happen with default plugin)."""
        manager = PluginManager()
        # Remove all plugins for this test
        manager._plugins = []

        accepting_plugin = manager.find_accepting_plugin("Any content")
        assert accepting_plugin is None

    def test_find_accepting_plugin_with_exception(self):
        """Test handling plugin exceptions during can_accept()."""
        manager = PluginManager()

        # Create a plugin that raises exception in can_accept
        failing_plugin = Mock(spec=Plugin)
        failing_plugin.can_accept.side_effect = ValueError("Test exception")
        manager.register_plugin(failing_plugin)

        # Should continue to next plugin (DefaultPlugin)
        accepting_plugin = manager.find_accepting_plugin("Any content")
        assert isinstance(accepting_plugin, DefaultPlugin)

    def test_plugin_metadata(self):
        """Test plugin metadata tracking."""
        manager = PluginManager()
        specific_plugin = MockSpecificPlugin()
        manager.register_plugin(specific_plugin)

        metadata = manager.get_plugin_metadata()

        assert "MockSpecificPlugin" in metadata
        assert metadata["MockSpecificPlugin"]["is_fallback"] is False
        assert metadata["MockSpecificPlugin"]["class_name"] == "MockSpecificPlugin"

        assert "DefaultPlugin" in metadata
        assert metadata["DefaultPlugin"]["is_fallback"] is True


class TestImportOrchestrator:
    """Test cases for ImportOrchestrator class."""

    def test_orchestrator_initialization(self):
        """Test ImportOrchestrator initializes correctly."""
        orchestrator = ImportOrchestrator()
        assert orchestrator.plugin_manager is not None
        assert orchestrator.plugin_manager.get_plugin_count() > 0

    def test_orchestrator_with_custom_plugin_manager(self):
        """Test ImportOrchestrator with custom PluginManager."""
        custom_manager = PluginManager()
        orchestrator = ImportOrchestrator(plugin_manager=custom_manager)
        assert orchestrator.plugin_manager is custom_manager

    def test_import_file_not_found(self):
        """Test importing non-existent file."""
        orchestrator = ImportOrchestrator()
        non_existent_path = Path("/non/existent/file.txt")

        result = orchestrator.import_file(non_existent_path)

        assert result.status == ImportStatus.ERROR
        assert "does not exist" in result.message
        assert result.plugin_used is None
        assert result.error_details == "File not found"

    def test_import_file_successful_specific_plugin(self):
        """Test successful import with specific plugin."""
        orchestrator = ImportOrchestrator()

        # Register specific plugin
        specific_plugin = MockSpecificPlugin()
        orchestrator.plugin_manager.register_plugin(specific_plugin)

        # Create temporary file with specific format
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write("This contains SPECIFIC_FORMAT marker")
            tmp_path = Path(tmp.name)

        try:
            result = orchestrator.import_file(tmp_path)

            assert result.status == ImportStatus.SUCCESS
            assert result.plugin_used == "MockSpecificPlugin"
            assert "1 conversation(s)" in result.message
            assert result.metadata["conversation_count"] == 1

        finally:
            tmp_path.unlink()

    def test_import_file_successful_fallback_plugin(self):
        """Test successful import with fallback plugin."""
        orchestrator = ImportOrchestrator()

        # Create temporary file that triggers fallback
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write("user: Hello\nassistant: Hi there!")
            tmp_path = Path(tmp.name)

        try:
            result = orchestrator.import_file(tmp_path)

            # DefaultPlugin should handle this
            assert result.status == ImportStatus.SUCCESS
            assert result.plugin_used == "DefaultPlugin"

        finally:
            tmp_path.unlink()

    def test_import_file_plugin_conversion_failure(self):
        """Test handling plugin conversion failure."""
        orchestrator = ImportOrchestrator()

        # Register failing plugin
        failing_plugin = MockFailingPlugin()
        orchestrator.plugin_manager.register_plugin(failing_plugin)

        # Create temporary file that triggers failing plugin
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write("This contains FAILING_FORMAT marker")
            tmp_path = Path(tmp.name)

        try:
            result = orchestrator.import_file(tmp_path)

            assert result.status == ImportStatus.ERROR
            assert result.plugin_used == "MockFailingPlugin"
            assert "Plugin conversion failed" in result.message
            assert "Simulated plugin failure" in result.error_details

        finally:
            tmp_path.unlink()

    def test_import_file_no_conversations_found(self):
        """Test handling when plugin finds no conversations."""
        orchestrator = ImportOrchestrator()

        # Register plugin that returns empty results
        empty_plugin = MockEmptyPlugin()
        orchestrator.plugin_manager.register_plugin(empty_plugin)

        # Create temporary file that triggers empty plugin
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write("This contains EMPTY_FORMAT marker")
            tmp_path = Path(tmp.name)

        try:
            result = orchestrator.import_file(tmp_path)

            assert result.status == ImportStatus.IGNORED
            assert result.plugin_used == "MockEmptyPlugin"
            assert "No conversations found" in result.message
            assert result.metadata["reason"] == "no_conversations"

        finally:
            tmp_path.unlink()

    def test_import_file_unicode_decode_error(self):
        """Test handling files with encoding issues."""
        orchestrator = ImportOrchestrator()

        # Create temporary file with binary content
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as tmp:
            tmp.write(b"\xff\xfe\x00\x00")  # Invalid UTF-8
            tmp_path = Path(tmp.name)

        try:
            # Should attempt latin-1 encoding as fallback
            result = orchestrator.import_file(tmp_path)

            # The result depends on whether latin-1 decoding succeeds
            # and whether any plugin accepts the decoded content
            assert result.status in [
                ImportStatus.SUCCESS,
                ImportStatus.IGNORED,
                ImportStatus.SKIPPED,
            ]

        finally:
            tmp_path.unlink()

    def test_import_files_batch(self):
        """Test batch import of multiple files."""
        orchestrator = ImportOrchestrator()

        # Create multiple temporary files
        file_paths = []
        try:
            # File 1: Successful
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as tmp:
                tmp.write("user: Hello")
                file_paths.append(Path(tmp.name))

            # File 2: Non-existent (error)
            file_paths.append(Path("/non/existent/file.txt"))

            # File 3: Another successful
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as tmp:
                tmp.write("assistant: Hi there")
                file_paths.append(Path(tmp.name))

            results = orchestrator.import_files(file_paths)

            assert len(results) == 3
            assert results[0].status in [ImportStatus.SUCCESS, ImportStatus.IGNORED]
            assert results[1].status == ImportStatus.ERROR
            assert results[2].status in [ImportStatus.SUCCESS, ImportStatus.IGNORED]

        finally:
            # Clean up existing files
            for path in file_paths:
                if path.exists():
                    path.unlink()

    def test_get_plugin_info(self):
        """Test getting plugin information."""
        orchestrator = ImportOrchestrator()
        specific_plugin = MockSpecificPlugin()
        orchestrator.plugin_manager.register_plugin(specific_plugin)

        plugin_info = orchestrator.get_plugin_info()

        assert "plugin_count" in plugin_info
        assert "plugins" in plugin_info
        assert "plugin_order" in plugin_info

        assert plugin_info["plugin_count"] == 2  # MockSpecificPlugin + DefaultPlugin
        assert "MockSpecificPlugin" in plugin_info["plugin_order"]
        assert "DefaultPlugin" in plugin_info["plugin_order"]

        # MockSpecificPlugin should come before DefaultPlugin
        mock_index = plugin_info["plugin_order"].index("MockSpecificPlugin")
        default_index = plugin_info["plugin_order"].index("DefaultPlugin")
        assert mock_index < default_index


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_orchestrate_file_import(self):
        """Test convenience function for single file import."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write("user: Test message")
            tmp_path = Path(tmp.name)

        try:
            result = orchestrate_file_import(tmp_path)

            assert isinstance(result, ImportResult)
            assert result.file_path == tmp_path
            assert result.status in [ImportStatus.SUCCESS, ImportStatus.IGNORED]

        finally:
            tmp_path.unlink()


class TestIntegrationScenarios:
    """Integration tests for various scenarios."""

    def test_plugin_ordering_priority(self):
        """Test that plugins are executed in correct priority order."""
        orchestrator = ImportOrchestrator()

        # Register specific plugin
        specific_plugin = MockSpecificPlugin()
        orchestrator.plugin_manager.register_plugin(specific_plugin)

        # Create content that would be accepted by both specific and default plugins
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            # This contains both specific format marker AND looks like a conversation
            tmp.write("SPECIFIC_FORMAT\nuser: Hello\nassistant: Hi")
            tmp_path = Path(tmp.name)

        try:
            result = orchestrator.import_file(tmp_path)

            # Should use specific plugin, not default
            assert result.plugin_used == "MockSpecificPlugin"
            assert result.status == ImportStatus.SUCCESS

        finally:
            tmp_path.unlink()

    def test_fallback_when_specific_plugin_fails(self):
        """Test fallback behavior when specific plugin fails."""
        orchestrator = ImportOrchestrator()

        # Register failing plugin that would accept the input
        failing_plugin = MockFailingPlugin()
        orchestrator.plugin_manager.register_plugin(failing_plugin)

        # Create content that triggers the failing plugin
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write("FAILING_FORMAT content")
            tmp_path = Path(tmp.name)

        try:
            result = orchestrator.import_file(tmp_path)

            # Should fail with the failing plugin, not fallback to default
            # because the failing plugin accepts the input
            assert result.status == ImportStatus.ERROR
            assert result.plugin_used == "MockFailingPlugin"

        finally:
            tmp_path.unlink()

    def test_no_plugin_accepts_content(self):
        """Test scenario where no plugin accepts the content."""
        # Create orchestrator with empty plugin manager
        manager = PluginManager()
        manager._plugins = []  # Remove all plugins
        orchestrator = ImportOrchestrator(plugin_manager=manager)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write("Some random content")
            tmp_path = Path(tmp.name)

        try:
            result = orchestrator.import_file(tmp_path)

            assert result.status == ImportStatus.SKIPPED
            assert result.plugin_used is None
            assert "No plugin could handle" in result.message

        finally:
            tmp_path.unlink()


# Additional tests for edge cases and error handling
class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_plugin_can_accept_exception_handling(self):
        """Test handling of exceptions in plugin can_accept() method."""
        manager = PluginManager()

        # Mock plugin that raises exception in can_accept
        mock_plugin = Mock(spec=Plugin)
        mock_plugin.can_accept.side_effect = Exception("Test exception")
        type(mock_plugin).__name__ = "MockExceptionPlugin"

        manager.register_plugin(mock_plugin)

        # Should continue to next plugin despite exception
        accepting_plugin = manager.find_accepting_plugin("test content")
        assert isinstance(accepting_plugin, DefaultPlugin)

    def test_import_with_read_permission_error(self):
        """Test handling file read permission errors."""
        orchestrator = ImportOrchestrator()

        # Create a file and remove read permissions
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write("test content")
            tmp_path = Path(tmp.name)

        try:
            # Remove read permissions
            tmp_path.chmod(0o000)

            result = orchestrator.import_file(tmp_path)

            assert result.status == ImportStatus.ERROR
            assert "Cannot read file" in result.message

        finally:
            # Restore permissions and clean up
            tmp_path.chmod(0o644)
            tmp_path.unlink()

    def test_logging_behavior(self, caplog):
        """Test that proper logging occurs during orchestration."""
        with caplog.at_level(logging.INFO):
            orchestrator = ImportOrchestrator()

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as tmp:
                tmp.write("user: Test")
                tmp_path = Path(tmp.name)

            try:
                orchestrator.import_file(tmp_path)

                # Check that appropriate log messages were generated
                assert any(
                    "Starting import orchestration" in record.message
                    for record in caplog.records
                )

            finally:
                tmp_path.unlink()
