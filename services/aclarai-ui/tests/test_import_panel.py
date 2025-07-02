"""
Unit tests for the Gradio Import Panel integration with real plugin orchestrator.
Tests the core functionality without launching the UI.
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Add the UI module to path
sys.path.insert(0, "services/aclarai-ui")

from aclarai_shared.plugin_manager import ImportStatus as PluginImportStatus
from aclarai_ui.main import (
    ImportStatusTracker,
    clear_import_queue,
    create_import_interface,
    real_plugin_orchestrator,
)


class TestImportStatusTracker:
    """Test the ImportStatusTracker class."""

    def test_initialization(self):
        """Test that ImportStatusTracker initializes correctly."""
        tracker = ImportStatusTracker()
        assert tracker.import_queue == []
        assert tracker.orchestrator is not None
        assert tracker.orchestrator.plugin_manager.get_plugin_count() > 0

    def test_map_plugin_status_to_ui(self):
        """Test status mapping from plugin to UI."""
        # Test success with regular plugin
        assert (
            ImportStatusTracker._map_plugin_status_to_ui(
                PluginImportStatus.SUCCESS, "SomePlugin"
            )
            == "✅ Imported"
        )

        # Test success with DefaultPlugin (should be fallback)
        assert (
            ImportStatusTracker._map_plugin_status_to_ui(
                PluginImportStatus.SUCCESS, "DefaultPlugin"
            )
            == "⚠️ Fallback"
        )

        # Test error status
        assert (
            ImportStatusTracker._map_plugin_status_to_ui(
                PluginImportStatus.ERROR, "SomePlugin"
            )
            == "❌ Failed"
        )

        # Test ignored status
        assert (
            ImportStatusTracker._map_plugin_status_to_ui(
                PluginImportStatus.IGNORED, "SomePlugin"
            )
            == "⏸️ Skipped"
        )

    def test_add_file_and_update_status(self):
        """Test adding files and updating their status."""
        tracker = ImportStatusTracker()

        # Add a file
        tracker.add_file("test.txt", "/path/to/test.txt")
        assert len(tracker.import_queue) == 1
        assert tracker.import_queue[0]["filename"] == "test.txt"
        assert tracker.import_queue[0]["status"] == "Processing..."

        # Update the status
        tracker.update_file_status("test.txt", "✅ Imported", "TestPlugin")
        assert tracker.import_queue[0]["status"] == "✅ Imported"
        assert tracker.import_queue[0]["detector"] == "TestPlugin"

    def test_format_queue_display(self):
        """Test the queue display formatting."""
        tracker = ImportStatusTracker()

        # Empty queue
        display = tracker.format_queue_display()
        assert "No files in import queue" in display

        # Add some files
        tracker.add_file("test1.txt", "/path/test1.txt")
        tracker.add_file("test2.json", "/path/test2.json")
        tracker.update_file_status("test1.txt", "✅ Imported", "Plugin1")
        tracker.update_file_status("test2.json", "⚠️ Fallback", "DefaultPlugin")

        display = tracker.format_queue_display()
        assert "test1.txt" in display
        assert "test2.json" in display
        assert "✅" in display
        assert "⚠️" in display

    def test_get_summary(self):
        """Test the summary generation."""
        tracker = ImportStatusTracker()

        # Empty summary
        summary = tracker.get_summary()
        assert summary == ""

        # Add files with different statuses
        tracker.add_file("file1.txt", "/path/file1.txt")
        tracker.add_file("file2.txt", "/path/file2.txt")
        tracker.add_file("file3.txt", "/path/file3.txt")
        tracker.update_file_status("file1.txt", "✅ Imported", "Plugin1")
        tracker.update_file_status("file2.txt", "⚠️ Fallback", "DefaultPlugin")
        tracker.update_file_status("file3.txt", "❌ Failed", "None")

        summary = tracker.get_summary()
        assert "Total Files Processed:** 3" in summary
        assert "Successfully Imported:** 1" in summary
        assert "Used Fallback Plugin:** 1" in summary
        assert "Failed to Import:** 1" in summary


class TestRealPluginOrchestrator:
    """Test the real plugin orchestrator function."""

    def test_process_text_file(self):
        """Test processing a text file with conversation content."""
        # Create a test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("User: Hello!\nAssistant: Hi there! How can I help?")
            test_file = f.name

        try:
            tracker = ImportStatusTracker()
            queue_display, summary_display, updated_tracker = real_plugin_orchestrator(
                test_file, tracker
            )

            # Check that file was processed
            assert len(updated_tracker.import_queue) == 1
            assert "tmp" in updated_tracker.import_queue[0]["filename"]
            assert updated_tracker.import_queue[0]["status"] in [
                "✅ Imported",
                "⚠️ Fallback",
            ]

            # Check displays are generated
            assert queue_display is not None
            assert len(queue_display) > 0
            assert summary_display is not None
            assert len(summary_display) > 0

        finally:
            Path(test_file).unlink()

    def test_duplicate_file_handling(self):
        """Test that duplicate files are handled correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            test_file = f.name

        try:
            tracker = ImportStatusTracker()

            # Process same file twice
            _, _, tracker1 = real_plugin_orchestrator(test_file, tracker)
            _, _, tracker2 = real_plugin_orchestrator(test_file, tracker1)

            # The queue should still only have one entry for this file
            assert len(tracker2.import_queue) == 1

            # That entry's status should now be "Skipped"
            duplicate_entry = tracker2.import_queue[0]

            assert duplicate_entry["status"] == "⏸️ Skipped"
            assert duplicate_entry["detector"] == "Duplicate"

        finally:
            Path(test_file).unlink()

    def test_no_file_provided(self):
        """Test handling when no file is provided."""
        tracker = ImportStatusTracker()
        queue_display, summary_display, updated_tracker = real_plugin_orchestrator(
            None, tracker
        )

        assert "No file selected" in queue_display
        assert summary_display == ""
        assert updated_tracker == tracker


class TestClearImportQueue:
    """Test the clear import queue function."""

    def test_clear_empty_queue(self):
        """Test clearing an empty queue."""
        tracker = ImportStatusTracker()
        queue_display, summary_display, new_tracker = clear_import_queue(tracker)

        assert "Import queue cleared" in queue_display
        assert summary_display == ""
        assert len(new_tracker.import_queue) == 0

    def test_clear_populated_queue(self):
        """Test clearing a queue with files."""
        tracker = ImportStatusTracker()
        tracker.add_file("test.txt", "/path/test.txt")

        queue_display, summary_display, new_tracker = clear_import_queue(tracker)

        assert "Import queue cleared" in queue_display
        assert len(new_tracker.import_queue) == 0


class TestCreateImportInterface:
    """Test the Gradio interface creation."""

    def test_interface_creation(self):
        """Test that the interface can be created without errors."""
        interface = create_import_interface()
        assert interface is not None
        assert hasattr(interface, "launch")  # It's a Gradio Blocks object


if __name__ == "__main__":
    # Run tests if pytest is available, otherwise run a simple test
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, running basic tests...")

        # Basic test without pytest
        print("Testing ImportStatusTracker initialization...")
        tracker = ImportStatusTracker()
        print(
            f"✅ Tracker created with {tracker.orchestrator.plugin_manager.get_plugin_count()} plugins"
        )

        print("Testing status mapping...")
        status = ImportStatusTracker._map_plugin_status_to_ui(
            PluginImportStatus.SUCCESS, "DefaultPlugin"
        )
        assert status == "⚠️ Fallback"
        print("✅ Status mapping works correctly")

        print("Testing interface creation...")
        interface = create_import_interface()
        print("✅ Interface created successfully")

        print("\n🎉 All basic tests passed!")
