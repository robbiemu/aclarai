#!/usr/bin/env python3
"""
Quick integration test to verify the real plugin orchestrator integration.
This script tests the new import panel functionality without launching the full UI.
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Add the UI module to path
sys.path.insert(0, "services/aclarai-ui")

from aclarai_ui.main import (
    ImportStatusTracker,
    clear_import_queue,
    real_plugin_orchestrator,
)


@pytest.mark.integration
def test_real_plugin_integration():
    """Test the real plugin orchestrator integration."""
    print("🧪 Testing Real Plugin Orchestrator Integration")
    print("=" * 50)

    # Create test files
    test_files = []

    # Test file 1: Simple conversation
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(
            "User: Hello there!\nAssistant: Hi! How can I help you today?\nUser: What is 2+2?\nAssistant: 2+2 equals 4."
        )
        test_files.append(f.name)

    # Test file 2: JSON format (should still use fallback since no specific JSON plugin)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('{"conversation": "This is a test conversation in JSON format"}')
        test_files.append(f.name)

    # Test file 3: Empty file (should fail)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("")
        test_files.append(f.name)

    try:
        tracker = ImportStatusTracker()
        print(
            f"✅ Created ImportStatusTracker with {tracker.orchestrator.plugin_manager.get_plugin_count()} plugins"
        )

        results = []
        for i, test_file in enumerate(test_files, 1):
            filename = Path(test_file).name
            print(f"\n📁 Testing file {i}: {filename}")

            # Simulate the UI workflow
            queue_display, summary_display, updated_tracker = real_plugin_orchestrator(
                test_file, tracker
            )
            tracker = updated_tracker

            print(f"   Queue Display: {queue_display[:100]}...")
            print(f"   Summary: {summary_display[:100]}...")

            results.append(
                {"file": filename, "queue": queue_display, "summary": summary_display}
            )

        print("\n📊 Final Summary:")
        final_summary = tracker.get_summary()
        print(final_summary)

        print("\n🔄 Testing Clear Queue:")
        queue_display, summary_display, new_tracker = clear_import_queue(tracker)
        print(f"   Clear result: {queue_display}")

        print("\n✅ Integration test completed successfully!")
        
        # Add assertions to verify the test actually passed
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        assert tracker.orchestrator.plugin_manager.get_plugin_count() >= 1, "Should have at least one plugin"
        assert "**Total Files Processed:** 3" in final_summary, "Should have processed 3 files"
        assert "Import queue cleared" in queue_display, "Queue should be cleared"

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Integration test failed: {e}")

    finally:
        # Cleanup
        for test_file in test_files:
            Path(test_file).unlink()
        print(f"\n🧹 Cleaned up {len(test_files)} test files")


if __name__ == "__main__":
    try:
        test_real_plugin_integration()
        print("Test completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
