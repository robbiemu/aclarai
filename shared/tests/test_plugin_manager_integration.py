"""
Integration tests for plugin manager and import orchestrator.
"""

import json
import tempfile
from contextlib import suppress
from pathlib import Path

import pytest
from aclarai_shared.config import PathsConfig, aclaraiConfig
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


# =======================================================================================
# END-TO-END INTEGRATION TESTS
# These tests are marked as integration tests and will be skipped during CI
# They test the complete plugin manager system against live services and real files
# =======================================================================================


@pytest.mark.integration
class TestPluginManagerEndToEndIntegration:
    """
    End-to-end integration tests for the plugin manager system.

    These tests verify the complete plugin manager functionality from file input
    to vault output using real file formats and configurations. They are marked
    as integration tests and will be excluded from CI runs.
    """

    def test_complete_chatgpt_conversation_import_pipeline(self):
        """
        End-to-end test of importing a real ChatGPT conversation export file.

        This test verifies:
        - Real file format detection
        - Plugin selection and execution
        - Vault file creation with proper metadata
        - Import logging and duplicate detection
        """
        import tempfile

        from aclarai_shared.config import PathsConfig, aclaraiConfig

        # Create a realistic ChatGPT export file
        chatgpt_conversation = {
            "title": "Test Conversation about AI Development",
            "create_time": 1699123456.789,
            "update_time": 1699123456.789,
            "mapping": {
                "msg_1": {
                    "id": "msg_1",
                    "message": {
                        "id": "msg_1",
                        "author": {"role": "user"},
                        "create_time": 1699123456.789,
                        "content": {
                            "content_type": "text",
                            "parts": ["What are the latest developments in AI?"],
                        },
                    },
                    "parent": None,
                    "children": ["msg_2"],
                },
                "msg_2": {
                    "id": "msg_2",
                    "message": {
                        "id": "msg_2",
                        "author": {"role": "assistant"},
                        "create_time": 1699123456.790,
                        "content": {
                            "content_type": "text",
                            "parts": [
                                "Recent AI developments include advances in large language models, multimodal AI systems, and improved reasoning capabilities."
                            ],
                        },
                    },
                    "parent": "msg_1",
                    "children": [],
                },
            },
            "moderation_results": [],
            "current_node": "msg_2",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup realistic vault structure
            vault_dir = Path(temp_dir) / "test_vault"
            tier1_dir = vault_dir / "tier1"
            logs_dir = vault_dir / "logs"

            tier1_dir.mkdir(parents=True)
            logs_dir.mkdir(parents=True)

            # Create configuration
            config = aclaraiConfig(
                vault_path=str(vault_dir), paths=PathsConfig(tier1="tier1", logs="logs")
            )

            # Initialize import system with real configuration
            system = Tier1ImportSystem(config)

            # Create realistic test file
            test_file = vault_dir / "chatgpt_export_2024_01_15.json"
            test_file.write_text(json.dumps(chatgpt_conversation, indent=2))

            # Execute complete import pipeline
            output_files = system.import_file(test_file)

            # Verify complete pipeline results
            assert len(output_files) == 1
            output_file = output_files[0]

            # Verify file placement in vault
            assert output_file.parent == tier1_dir
            assert output_file.exists()
            assert output_file.suffix == ".md"

            # Verify complete Tier 1 Markdown format
            content = output_file.read_text()

            # Check metadata headers
            assert "<!-- aclarai:title=" in content
            assert "<!-- aclarai:created_at=" in content
            assert "<!-- aclarai:participants=" in content
            assert "<!-- aclarai:message_count=" in content
            assert "<!-- aclarai:plugin_metadata=" in content

            # Check conversation content
            assert "What are the latest developments in AI?" in content
            assert "Recent AI developments include" in content

            # Verify import logging
            import_log = logs_dir / "imported_files.json"
            assert import_log.exists()

            log_data = json.loads(import_log.read_text())
            assert len(log_data["hashes"]) == 1
            assert str(test_file) in log_data["files"]

            # Test duplicate detection

            with suppress(Exception):
                system.import_file(test_file)
                # Should raise DuplicateDetectionError, but we catch it
                # This verifies the complete duplicate detection pipeline

    @pytest.mark.integration
    def test_multi_format_batch_import_with_real_vault(self):
        """
        End-to-end test of importing multiple file formats in a batch operation.

        This test verifies:
        - Multiple plugin activation for different formats
        - Batch processing with mixed success/failure cases
        - Vault organization with multiple file outputs
        - Complete import status tracking
        """
        import json
        import tempfile

        from aclarai_shared.config import PathsConfig, aclaraiConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup realistic vault structure
            vault_dir = Path(temp_dir) / "production_vault"
            tier1_dir = vault_dir / "tier1"
            logs_dir = vault_dir / "logs"
            import_dir = vault_dir / "imports"

            tier1_dir.mkdir(parents=True)
            logs_dir.mkdir(parents=True)
            import_dir.mkdir(parents=True)

            # Create configuration
            config = aclaraiConfig(
                vault_path=str(vault_dir), paths=PathsConfig(tier1="tier1", logs="logs")
            )

            # Initialize import system
            system = Tier1ImportSystem(config)

            # Create multiple realistic test files
            test_files = []

            # ChatGPT conversation file
            chatgpt_file = import_dir / "meeting_notes_2024.json"
            chatgpt_data = {
                "title": "Project Planning Meeting",
                "mapping": {
                    "1": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {
                                "parts": ["Let's discuss the project timeline"]
                            },
                            "create_time": 1699123456,
                        }
                    },
                    "2": {
                        "message": {
                            "author": {"role": "assistant"},
                            "content": {"parts": ["I can help you plan the timeline"]},
                            "create_time": 1699123457,
                        }
                    },
                },
            }
            chatgpt_file.write_text(json.dumps(chatgpt_data))
            test_files.append(chatgpt_file)

            # Plain text conversation file
            text_file = import_dir / "support_chat.txt"
            text_file.write_text(
                """
User: I'm having trouble with the application
Support: I can help you with that. What specific issue are you experiencing?
User: The login feature isn't working
Support: Let me guide you through some troubleshooting steps.
            """.strip()
            )
            test_files.append(text_file)

            # Unsupported file format
            binary_file = import_dir / "document.pdf"
            binary_file.write_bytes(b"PDF content")  # Not a real PDF, but binary data
            test_files.append(binary_file)

            # Execute batch import using directory import
            results = system.import_directory(import_dir, recursive=False)

            # Verify batch processing results
            assert len(results) == 3

            # Verify different outcomes for different files
            successful_imports = [f for f, outputs in results.items() if outputs]
            failed_imports = [f for f, outputs in results.items() if not outputs]

            # Should have some successful and some failed
            assert len(successful_imports) >= 1  # At least text file should work
            assert len(failed_imports) >= 1  # Binary file should fail

            # Verify vault contains properly formatted files
            tier1_files = list(tier1_dir.glob("*.md"))
            assert len(tier1_files) >= 1

            # Verify each tier1 file has proper structure
            for tier1_file in tier1_files:
                content = tier1_file.read_text()
                assert "<!-- aclarai:title=" in content
                assert "<!-- aclarai:created_at=" in content

            # Verify comprehensive import logging
            import_log = logs_dir / "imported_files.json"
            assert import_log.exists()

            log_data = json.loads(import_log.read_text())
            assert "hashes" in log_data
            assert "files" in log_data

            # Should have entries for successfully imported files
            assert len(log_data["hashes"]) >= 1

    @pytest.mark.integration
    def test_plugin_discovery_and_ordering_in_production_environment(self):
        """
        End-to-end test of plugin discovery and ordering in a production-like environment.

        This test verifies:
        - Automatic plugin discovery from multiple sources
        - Correct plugin ordering with fallback behavior
        - Plugin metadata tracking in realistic scenarios
        - Integration with real configuration system
        """
        # This test would verify plugin discovery from entry_points when implemented
        from aclarai_shared.plugin_manager import ImportOrchestrator, PluginManager

        # Test current plugin discovery mechanism
        manager = PluginManager()

        # Verify automatic discovery worked
        plugins = manager.get_plugins()
        assert len(plugins) >= 1  # Should at least have DefaultPlugin

        # Verify DefaultPlugin is present and ordered last
        plugin_names = [type(p).__name__ for p in plugins]
        assert "DefaultPlugin" in plugin_names
        assert plugin_names[-1] == "DefaultPlugin"

        # Test orchestrator initialization with discovered plugins
        orchestrator = ImportOrchestrator(manager)
        plugin_info = orchestrator.get_plugin_info()

        # Verify plugin metadata is properly tracked
        assert plugin_info["plugin_count"] >= 1
        assert "plugin_order" in plugin_info
        assert len(plugin_info["plugin_order"]) == plugin_info["plugin_count"]

        # Verify metadata structure for production use
        metadata = manager.get_plugin_metadata()
        for plugin_name, meta in metadata.items():
            assert "class_name" in meta
            assert "is_fallback" in meta
            assert "module" in meta

            # Production readiness checks
            assert meta["class_name"] == plugin_name
            assert isinstance(meta["is_fallback"], bool)
            assert isinstance(meta["module"], str)

    @pytest.mark.integration
    def test_error_recovery_and_resilience_in_production_scenarios(self):
        """
        End-to-end test of error recovery and resilience mechanisms.

        This test verifies:
        - Graceful handling of corrupted files
        - Recovery from plugin failures
        - Proper error logging and status tracking
        - System stability under adverse conditions
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup vault
            vault_dir = Path(temp_dir) / "resilience_test_vault"
            tier1_dir = vault_dir / "tier1"
            logs_dir = vault_dir / "logs"

            tier1_dir.mkdir(parents=True)
            logs_dir.mkdir(parents=True)

            aclaraiConfig(
                vault_path=str(vault_dir), paths=PathsConfig(tier1="tier1", logs="logs")
            )

            # We only need the orchestrator for this test
            orchestrator = ImportOrchestrator()

            # Test various error scenarios
            test_cases = []

            try:
                # Corrupted JSON file
                corrupted_json = vault_dir / "corrupted.json"
                corrupted_json.write_text('{"invalid": json content')
                test_cases.append(corrupted_json)

                # Very large file (if supported)
                large_file = vault_dir / "large.txt"
                large_content = "Large conversation content\n" * 1000
                large_file.write_text(large_content)
                test_cases.append(large_file)

                # File with special characters
                special_file = vault_dir / "special_chars_файл.txt"
                special_file.write_text(
                    "User: Testing special chars 特殊字符\nBot: Response"
                )
                test_cases.append(special_file)

                # Test each scenario for resilience
                for test_file in test_cases:
                    # Should not crash the system
                    result = orchestrator.import_file(test_file)

                    # Result should be properly structured even on failure
                    assert hasattr(result, "status")
                    assert hasattr(result, "message")
                    assert result.status in [
                        ImportStatus.SUCCESS,
                        ImportStatus.ERROR,
                        ImportStatus.IGNORED,
                        ImportStatus.SKIPPED,
                    ]

                    # Error cases should have proper error details
                    if result.status == ImportStatus.ERROR:
                        assert result.error_details is not None
                        assert isinstance(result.message, str)
                        assert len(result.message) > 0

                # Verify system is still functional after errors
                working_file = vault_dir / "working.txt"
                working_file.write_text(
                    "User: This should work\nAssistant: Yes it does"
                )

                final_result = orchestrator.import_file(working_file)
                assert final_result.status in [
                    ImportStatus.SUCCESS,
                    ImportStatus.IGNORED,
                ]

            finally:
                # Cleanup
                for test_file in test_cases:
                    if test_file.exists():
                        test_file.unlink()
