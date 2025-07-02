"""
Integration tests for plugin manager and import orchestrator.
"""

import tempfile
from pathlib import Path

from aclarai_shared.import_system import Tier1ImportSystem
from aclarai_shared.plugin_interface import MarkdownOutput, Plugin
from aclarai_shared.plugin_manager import (
    ImportOrchestrator,
    ImportStatus,
    PluginManager,
)


class ChatGPTMockPlugin(Plugin):
    """Mock plugin simulating ChatGPT JSON format detection."""

    def can_accept(self, raw_input: str) -> bool:
        return '"model":' in raw_input and '"messages":' in raw_input

    def convert(self, raw_input: str, path: Path) -> list[MarkdownOutput]:
        return [
            MarkdownOutput(
                title="ChatGPT Conversation",
                markdown_text="assistant: This is a mock ChatGPT conversation",
                metadata={
                    "participants": ["user", "assistant"],
                    "message_count": 1,
                    "plugin_metadata": {"source_format": "chatgpt_json"},
                },
            )
        ]


class SlackMockPlugin(Plugin):
    """Mock plugin simulating Slack export format detection."""

    def can_accept(self, raw_input: str) -> bool:
        return '"channel":' in raw_input and '"ts":' in raw_input

    def convert(self, raw_input: str, path: Path) -> list[MarkdownOutput]:
        return [
            MarkdownOutput(
                title="Slack Channel Export",
                markdown_text="user1: This is a mock Slack conversation",
                metadata={
                    "participants": ["user1", "user2"],
                    "message_count": 1,
                    "plugin_metadata": {"source_format": "slack_export"},
                },
            )
        ]


class TestPluginManagerIntegration:
    """Integration tests for the complete plugin manager system."""

    def test_plugin_discovery_and_ordering(self):
        """Test that plugins are discovered and ordered correctly."""
        manager = PluginManager()

        # Register specific plugins
        chatgpt_plugin = ChatGPTMockPlugin()
        slack_plugin = SlackMockPlugin()

        manager.register_plugin(chatgpt_plugin)
        manager.register_plugin(slack_plugin)

        plugins = manager.get_plugins()
        plugin_names = [type(p).__name__ for p in plugins]

        # Verify specific plugins come before DefaultPlugin
        assert "ChatGPTMockPlugin" in plugin_names
        assert "SlackMockPlugin" in plugin_names
        assert "DefaultPlugin" in plugin_names

        # DefaultPlugin should be last
        assert plugin_names[-1] == "DefaultPlugin"

        # Specific plugins should come before DefaultPlugin
        default_index = plugin_names.index("DefaultPlugin")
        chatgpt_index = plugin_names.index("ChatGPTMockPlugin")
        slack_index = plugin_names.index("SlackMockPlugin")

        assert chatgpt_index < default_index
        assert slack_index < default_index

    def test_format_specific_plugin_selection(self):
        """Test that format-specific plugins are selected correctly."""
        orchestrator = ImportOrchestrator()

        # Register specific plugins
        orchestrator.plugin_manager.register_plugin(ChatGPTMockPlugin())
        orchestrator.plugin_manager.register_plugin(SlackMockPlugin())

        # Test ChatGPT format detection
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write(
                '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
            )
            tmp_path = Path(tmp.name)

        try:
            result = orchestrator.import_file(tmp_path)

            assert result.status == ImportStatus.SUCCESS
            assert result.plugin_used == "ChatGPTMockPlugin"
            assert "ChatGPT Conversation" in result.message or result.metadata

        finally:
            tmp_path.unlink()

        # Test Slack format detection
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write('{"channel": "general", "ts": "1234567890", "text": "Hello"}')
            tmp_path = Path(tmp.name)

        try:
            result = orchestrator.import_file(tmp_path)

            assert result.status == ImportStatus.SUCCESS
            assert result.plugin_used == "SlackMockPlugin"

        finally:
            tmp_path.unlink()

    def test_fallback_plugin_usage(self):
        """Test that fallback plugin is used when no specific plugin matches."""
        orchestrator = ImportOrchestrator()

        # Create content that no specific plugin will accept
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write("user: Hello there\nassistant: Hi, how can I help?")
            tmp_path = Path(tmp.name)

        try:
            result = orchestrator.import_file(tmp_path)

            # Should use DefaultPlugin as fallback
            assert result.plugin_used == "DefaultPlugin"
            # Status depends on whether conversations are found
            assert result.status in [ImportStatus.SUCCESS, ImportStatus.IGNORED]

        finally:
            tmp_path.unlink()

    def test_batch_import_with_mixed_formats(self):
        """Test batch import with different file formats."""
        orchestrator = ImportOrchestrator()

        # Register specific plugins
        orchestrator.plugin_manager.register_plugin(ChatGPTMockPlugin())
        orchestrator.plugin_manager.register_plugin(SlackMockPlugin())

        file_paths = []

        try:
            # ChatGPT format file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp:
                tmp.write('{"model": "gpt-4", "messages": []}')
                file_paths.append(Path(tmp.name))

            # Slack format file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp:
                tmp.write('{"channel": "test", "ts": "123"}')
                file_paths.append(Path(tmp.name))

            # Generic conversation file (fallback)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as tmp:
                tmp.write("user: Test\nassistant: Response")
                file_paths.append(Path(tmp.name))

            # Non-existent file (error)
            file_paths.append(Path("/non/existent/file.txt"))

            results = orchestrator.import_files(file_paths)

            assert len(results) == 4

            # Verify different plugins were used
            plugins_used = [r.plugin_used for r in results if r.plugin_used]
            assert "ChatGPTMockPlugin" in plugins_used
            assert "SlackMockPlugin" in plugins_used
            assert "DefaultPlugin" in plugins_used

            # Verify status variety
            statuses = [r.status for r in results]
            assert ImportStatus.SUCCESS in statuses or ImportStatus.IGNORED in statuses
            assert ImportStatus.ERROR in statuses  # For non-existent file

        finally:
            # Clean up existing files
            for path in file_paths:
                if path.exists():
                    path.unlink()

    def test_tier1_import_system_integration(self):
        """Test integration with Tier1ImportSystem."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup vault structure
            vault_dir = Path(temp_dir) / "vault"
            tier1_dir = vault_dir / "tier1"
            logs_dir = vault_dir / "logs"

            tier1_dir.mkdir(parents=True)
            logs_dir.mkdir(parents=True)

            # Configure import system
            from aclarai_shared.config import PathsConfig, aclaraiConfig

            config = aclaraiConfig(
                vault_path=str(vault_dir), paths=PathsConfig(tier1="tier1", logs="logs")
            )
            system = Tier1ImportSystem(config)

            # Register additional plugin
            system.add_plugin(ChatGPTMockPlugin())

            # Test plugin info
            plugin_info = system.get_plugin_info()
            assert plugin_info["plugin_count"] >= 2  # ChatGPT + Default
            assert "ChatGPTMockPlugin" in plugin_info["plugin_order"]

            # Create test file
            test_file = vault_dir / "test_chatgpt.json"
            test_file.write_text(
                '{"model": "gpt-4", "messages": [{"role": "user", "content": "Test"}]}'
            )

            # Import file
            output_files = system.import_file(test_file)

            # Verify output
            assert len(output_files) >= 1
            assert all(f.exists() for f in output_files)
            assert all(f.parent == tier1_dir for f in output_files)

            # Verify content
            content = output_files[0].read_text()
            assert "<!-- aclarai:title=" in content
            assert "assistant: This is a mock ChatGPT conversation" in content

    def test_error_handling_and_status_tracking(self):
        """Test comprehensive error handling and status tracking."""
        orchestrator = ImportOrchestrator()

        test_cases = []

        try:
            # Case 1: Successful processing
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as tmp:
                tmp.write("user: Hello\nassistant: Hi there")
                test_cases.append(("success", Path(tmp.name)))

            # Case 2: Non-existent file
            test_cases.append(("error", Path("/non/existent/file.txt")))

            # Case 3: Empty file (likely ignored)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as tmp:
                tmp.write("")
                test_cases.append(("empty", Path(tmp.name)))

            # Process all test cases
            for case_type, file_path in test_cases:
                result = orchestrator.import_file(file_path)

                # Verify ImportResult structure
                assert hasattr(result, "file_path")
                assert hasattr(result, "status")
                assert hasattr(result, "message")
                assert hasattr(result, "plugin_used")

                # Verify status is valid
                assert result.status in [
                    ImportStatus.SUCCESS,
                    ImportStatus.IGNORED,
                    ImportStatus.ERROR,
                    ImportStatus.SKIPPED,
                ]

                # Verify error cases have proper error details
                if result.status == ImportStatus.ERROR:
                    assert result.error_details is not None
                    if case_type == "error":  # non-existent file
                        assert "does not exist" in result.message

                # Verify successful cases have plugin information
                if result.status == ImportStatus.SUCCESS:
                    assert result.plugin_used is not None
                    assert result.plugin_used in [
                        "DefaultPlugin",
                        "ChatGPTMockPlugin",
                        "SlackMockPlugin",
                    ]

        finally:
            # Clean up test files
            for _, file_path in test_cases:
                if file_path.exists():
                    file_path.unlink()

    def test_plugin_metadata_and_introspection(self):
        """Test plugin metadata tracking and introspection capabilities."""
        manager = PluginManager()

        # Register plugins
        chatgpt_plugin = ChatGPTMockPlugin()
        slack_plugin = SlackMockPlugin()

        manager.register_plugin(chatgpt_plugin)
        manager.register_plugin(slack_plugin)

        # Test metadata tracking
        metadata = manager.get_plugin_metadata()

        assert "ChatGPTMockPlugin" in metadata
        assert "SlackMockPlugin" in metadata
        assert "DefaultPlugin" in metadata

        # Verify metadata structure
        for plugin_name, meta in metadata.items():
            assert "class_name" in meta
            assert "is_fallback" in meta
            assert "module" in meta
            assert meta["class_name"] == plugin_name

        # Verify fallback status
        assert metadata["DefaultPlugin"]["is_fallback"] is True
        assert metadata["ChatGPTMockPlugin"]["is_fallback"] is False
        assert metadata["SlackMockPlugin"]["is_fallback"] is False

        # Test orchestrator plugin info
        orchestrator = ImportOrchestrator(manager)
        plugin_info = orchestrator.get_plugin_info()

        assert plugin_info["plugin_count"] == 3
        assert set(plugin_info["plugin_order"]) == {
            "ChatGPTMockPlugin",
            "SlackMockPlugin",
            "DefaultPlugin",
        }

        # Verify DefaultPlugin is last
        assert plugin_info["plugin_order"][-1] == "DefaultPlugin"
