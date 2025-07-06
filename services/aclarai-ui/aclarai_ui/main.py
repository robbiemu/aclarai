"""Main Gradio application for aclarai frontend."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, cast

import gradio as gr
from aclarai_shared.plugin_manager import ImportOrchestrator, ImportResult
from aclarai_shared.plugin_manager import ImportStatus as PluginImportStatus

from .config import config
from .config_panel import create_configuration_panel
from .review_panel import create_review_panel

# Configure structured logging as per docs/arch/on-error-handling-and-resilience.md
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("aclarai-ui")


class ImportStatusTracker:
    """Class to track import status and queue for UI display."""

    def __init__(self):
        self.import_queue = []
        self.summary_stats = {"imported": 0, "failed": 0, "fallback": 0, "skipped": 0}
        # Initialize the real plugin orchestrator
        self.orchestrator = ImportOrchestrator()
        logger.info(
            "ImportStatusTracker initialized with real plugin orchestrator",
            extra={
                "service": "aclarai-ui",
                "component": "ImportStatusTracker",
                "action": "initialize",
                "plugin_count": self.orchestrator.plugin_manager.get_plugin_count(),
            },
        )

    @staticmethod
    def _map_plugin_status_to_ui(
        plugin_status: PluginImportStatus, plugin_name: Optional[str] = None
    ) -> str:
        """Map plugin ImportStatus enum to UI-friendly status strings."""
        status_mapping = {
            PluginImportStatus.SUCCESS: "✅ Imported",
            PluginImportStatus.IGNORED: "⏸️ Skipped",
            PluginImportStatus.ERROR: "❌ Failed",
            PluginImportStatus.SKIPPED: "❌ Failed",  # No plugin could handle it
        }

        status = status_mapping.get(plugin_status, "❌ Failed")

        # Special case: if successful with DefaultPlugin, mark as fallback
        if (
            plugin_status == PluginImportStatus.SUCCESS
            and plugin_name == "DefaultPlugin"
        ):
            status = "⚠️ Fallback"

        return status

    def add_file(self, filename: str, file_path: str):
        """Add a file to the import queue."""
        try:
            self.import_queue.append(
                {
                    "filename": filename,
                    "path": file_path,
                    "status": "Processing...",
                    "detector": "Detecting...",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }
            )
            logger.info(
                "File added to import queue",
                extra={
                    "service": "aclarai-ui",
                    "component": "ImportStatusTracker",
                    "action": "add_file",
                    "file_name": filename,
                    "file_path": file_path,
                },
            )
        except Exception as e:
            logger.error(
                "Failed to add file to import queue",
                extra={
                    "service": "aclarai-ui",
                    "component": "ImportStatusTracker",
                    "action": "add_file",
                    "file_name": filename,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise

    def update_file_status(self, filename: str, status: str, detector: str):
        """Update the status of a file in the queue."""
        try:
            for item in self.import_queue:
                if item["filename"] == filename:
                    item["status"] = status
                    item["detector"] = detector
                    logger.info(
                        "File status updated",
                        extra={
                            "service": "aclarai-ui",
                            "component": "ImportStatusTracker",
                            "action": "update_status",
                            "file_name": filename,
                            "status": status,
                            "detector": detector,
                        },
                    )
                    break
            else:
                logger.warning(
                    "File not found in queue for status update",
                    extra={
                        "service": "aclarai-ui",
                        "component": "ImportStatusTracker",
                        "action": "update_status",
                        "file_name": filename,
                        "attempted_status": status,
                    },
                )
        except Exception as e:
            logger.error(
                "Failed to update file status",
                extra={
                    "service": "aclarai-ui",
                    "component": "ImportStatusTracker",
                    "action": "update_status",
                    "file_name": filename,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Graceful degradation: continue operation even if status update fails

    def process_file_with_orchestrator(self, file_path: str) -> ImportResult:
        """Process a file using the real plugin orchestrator."""
        try:
            logger.info(
                "Processing file with orchestrator",
                extra={
                    "service": "aclarai-ui",
                    "component": "ImportStatusTracker",
                    "action": "process_file",
                    "file_path": file_path,
                },
            )

            # Use the real orchestrator to process the file
            result = self.orchestrator.import_file(Path(file_path))

            logger.info(
                "File processing completed",
                extra={
                    "service": "aclarai-ui",
                    "component": "ImportStatusTracker",
                    "action": "process_file",
                    "file_path": file_path,
                    "status": result.status.value,
                    "plugin_used": result.plugin_used,
                    "result_message": result.message,
                },
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to process file with orchestrator",
                extra={
                    "service": "aclarai-ui",
                    "component": "ImportStatusTracker",
                    "action": "process_file",
                    "file_path": file_path,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Return a failed result for graceful degradation
            return ImportResult(
                file_path=Path(file_path),
                status=PluginImportStatus.ERROR,
                message=f"Processing failed: {str(e)}",
                error_details=str(e),
            )

    def format_queue_display(self) -> str:
        """Format the import queue for display."""
        try:
            if not self.import_queue:
                return "No files in import queue."
            header = "| Filename | Status | Detector | Time |\n|----------|--------|----------|------|\n"
            rows = []
            for item in self.import_queue:
                status_icon = {
                    "✅ Imported": "✅",
                    "❌ Failed": "❌",
                    "⚠️ Fallback": "⚠️",
                    "⏸️ Skipped": "⏸️",
                    "Processing...": "🔄",
                }.get(item["status"], "🔄")
                row = f"| {item['filename']} | {status_icon} {item['status']} | {item['detector']} | {item['timestamp']} |"
                rows.append(row)
            return header + "\n".join(rows)
        except Exception as e:
            logger.error(
                "Failed to format queue display",
                extra={
                    "service": "aclarai-ui",
                    "component": "ImportStatusTracker",
                    "action": "format_queue_display",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Graceful degradation: return error message instead of crashing
            return "Error displaying import queue. Please check logs for details."

    def get_summary(self) -> str:
        """Get post-import summary with configurable paths."""
        try:
            if not self.import_queue:
                return ""
            total = len(self.import_queue)
            imported = sum(
                1 for item in self.import_queue if item["status"] == "✅ Imported"
            )
            failed = sum(
                1 for item in self.import_queue if item["status"] == "❌ Failed"
            )
            fallback = sum(
                1 for item in self.import_queue if item["status"] == "⚠️ Fallback"
            )
            skipped = sum(
                1 for item in self.import_queue if item["status"] == "⏸️ Skipped"
            )
            logger.info(
                "Summary generated",
                extra={
                    "service": "aclarai-ui",
                    "component": "ImportStatusTracker",
                    "action": "get_summary",
                    "total_files": total,
                    "imported": imported,
                    "failed": failed,
                    "fallback": fallback,
                    "skipped": skipped,
                },
            )
            # Get configured paths for next steps links
            next_steps_links = config.get_next_steps_links()
            summary = f"""## 📊 Import Summary
**Total Files Processed:** {total}
✅ **Successfully Imported:** {imported} files
⚠️ **Used Fallback Plugin:** {fallback} files
❌ **Failed to Import:** {failed} files
⏸️ **Skipped (Duplicates):** {skipped} files
### Next Steps:
- [View Imported Files]({next_steps_links["vault"]}) (files written to vault)
- [Download Import Log]({next_steps_links["logs"]}) (detailed processing logs)
"""
            return summary
        except Exception as e:
            logger.error(
                "Failed to generate summary",
                extra={
                    "service": "aclarai-ui",
                    "component": "ImportStatusTracker",
                    "action": "get_summary",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Graceful degradation: return basic error message
            return "## 📊 Import Summary\n\nError generating summary. Please check logs for details."


def real_plugin_orchestrator(
    file_path: Optional[str], import_status: ImportStatusTracker
) -> Tuple[str, str, ImportStatusTracker]:
    """Process a file using the real plugin orchestrator with proper error handling.
    Args:
        file_path: Path to uploaded file
        import_status: Current import status state
    Returns:
        Tuple of (queue_display, summary_display, updated_import_status)
    """
    try:
        if not file_path:
            logger.warning(
                "No file provided for import",
                extra={
                    "service": "aclarai-ui",
                    "component": "plugin_orchestrator",
                    "action": "real_import",
                },
            )
            return "No file selected for import.", "", import_status

        filename = os.path.basename(file_path)
        logger.info(
            "Starting real plugin orchestrator processing",
            extra={
                "service": "aclarai-ui",
                "component": "plugin_orchestrator",
                "action": "real_import",
                "file_name": filename,
                "file_path": file_path,
            },
        )

        # Check for duplicates
        existing_files = [item["filename"] for item in import_status.import_queue]
        if filename in existing_files:
            logger.info(
                "Duplicate file detected",
                extra={
                    "service": "aclarai-ui",
                    "component": "plugin_orchestrator",
                    "action": "duplicate_check",
                    "file_name": filename,
                },
            )
            import_status.update_file_status(filename, "⏸️ Skipped", "Duplicate")
            return (
                import_status.format_queue_display(),
                import_status.get_summary(),
                import_status,
            )

        # Add file to queue with processing status
        import_status.add_file(filename, file_path)

        # Process file with the real orchestrator
        result = import_status.process_file_with_orchestrator(file_path)

        # Map the result to UI status
        ui_status = ImportStatusTracker._map_plugin_status_to_ui(
            result.status, result.plugin_used
        )
        detector_name = result.plugin_used or "None"

        # Handle special cases in the plugin result
        if result.status == PluginImportStatus.IGNORED:
            # File processed but no conversations found
            detector_name = result.plugin_used or "No plugin"
            ui_status = "⏸️ Skipped"
        elif result.status == PluginImportStatus.SKIPPED:
            # No plugin could handle the file
            detector_name = "None"
            ui_status = "❌ Failed"

        # Update status in the queue
        import_status.update_file_status(filename, ui_status, detector_name)

        logger.info(
            "Real plugin orchestrator processing completed",
            extra={
                "service": "aclarai-ui",
                "component": "plugin_orchestrator",
                "action": "real_import",
                "file_name": filename,
                "final_status": ui_status,
                "detector": detector_name,
                "plugin_status": result.status.value,
                "result_message": result.message,
            },
        )

        return (
            import_status.format_queue_display(),
            import_status.get_summary(),
            import_status,
        )

    except Exception as e:
        logger.error(
            "Real plugin orchestrator processing failed",
            extra={
                "service": "aclarai-ui",
                "component": "plugin_orchestrator",
                "action": "real_import",
                "file_name": filename if "filename" in locals() else "unknown",
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        # Graceful degradation: return error message instead of crashing
        error_msg = f"Error processing file: {str(e)}"
        return (
            error_msg,
            "Processing failed.",
            import_status,
        )


def clear_import_queue(
    import_status: ImportStatusTracker,
) -> Tuple[str, str, ImportStatusTracker]:
    """Clear the import queue and reset statistics.
    Args:
        import_status: Current import status state
    Returns:
        Tuple of (queue_display, summary_display, new_import_status)
    """
    try:
        logger.info(
            "Clearing import queue",
            extra={
                "service": "aclarai-ui",
                "component": "queue_manager",
                "action": "clear_queue",
                "queue_size": len(import_status.import_queue),
            },
        )
        new_import_status = ImportStatusTracker()
        return "Import queue cleared.", "", new_import_status
    except Exception as e:
        logger.error(
            "Failed to clear import queue",
            extra={
                "service": "aclarai-ui",
                "component": "queue_manager",
                "action": "clear_queue",
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        # Graceful degradation: return error message
        return "Error clearing queue. Please refresh the page.", "", import_status


def create_import_interface() -> gr.Blocks:
    """Create the complete import interface following the documented design."""
    try:
        logger.info(
            "Creating import interface",
            extra={
                "service": "aclarai-ui",
                "component": "interface_creator",
                "action": "create_interface",
            },
        )
        with gr.Blocks(
            title="aclarai - Import Panel", theme=gr.themes.Soft()
        ) as interface:
            # Session-based state management for import status
            import_status_state = gr.State(ImportStatusTracker())
            gr.Markdown("# 📥 aclarai Import Panel")
            gr.Markdown(
                """Upload conversation files from various sources (ChatGPT exports, Slack logs, generic text files)
                to process and import into the aclarai system. Files are automatically detected and processed
                using the appropriate format plugin."""
            )
            # File Picker Section
            with gr.Group():
                gr.Markdown("## 📁 File Selection")
                file_input = gr.File(
                    label="Drag files here or click to browse",
                    file_types=[".json", ".txt", ".csv", ".md", ".zip"],
                    type="filepath",
                    height=100,
                )
                with gr.Row():
                    import_btn = gr.Button(
                        "🚀 Process File", variant="primary", size="lg"
                    )
                    clear_btn = gr.Button("🗑️ Clear Queue", variant="secondary")
            # Live Import Queue Section
            with gr.Group():
                gr.Markdown("## 📋 Live Import Queue")
                queue_display = gr.Markdown(
                    value="No files in import queue.", label="Import Status"
                )
            # Post-import Summary Section
            with gr.Group():
                gr.Markdown("## 📊 Import Summary")
                summary_display = gr.Markdown(
                    value="Process files to see import summary.", label="Summary"
                )

            # Event handlers with error handling and state management
            def safe_real_plugin_orchestrator(
                file_input_value, import_status
            ) -> Tuple[str, str, ImportStatusTracker]:
                try:
                    queue_display, summary_display, updated_status = (
                        real_plugin_orchestrator(file_input_value, import_status)
                    )
                    return queue_display, summary_display, updated_status
                except Exception as e:
                    logger.error(
                        "Error in real plugin orchestrator",
                        extra={
                            "service": "aclarai-ui",
                            "component": "interface_handler",
                            "action": "real_orchestrator",
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    return (
                        "Error processing file. Please try again.",
                        "Processing failed.",
                        import_status,
                    )

            def safe_clear_import_queue(
                import_status,
            ) -> Tuple[str, str, ImportStatusTracker]:
                try:
                    queue_display, summary_display, new_status = clear_import_queue(
                        import_status
                    )
                    return queue_display, summary_display, new_status
                except Exception as e:
                    logger.error(
                        "Error clearing import queue",
                        extra={
                            "service": "aclarai-ui",
                            "component": "interface_handler",
                            "action": "clear_queue",
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    return (
                        "Error clearing queue. Please refresh the page.",
                        "",
                        import_status,
                    )

            import_btn.click(
                fn=safe_real_plugin_orchestrator,
                inputs=[file_input, import_status_state],
                outputs=[queue_display, summary_display, import_status_state],
            )
            # Auto-process on file upload
            file_input.change(
                fn=safe_real_plugin_orchestrator,
                inputs=[file_input, import_status_state],
                outputs=[queue_display, summary_display, import_status_state],
            )
            # Clear queue handler
            clear_btn.click(
                fn=safe_clear_import_queue,
                inputs=[import_status_state],
                outputs=[queue_display, summary_display, import_status_state],
            )
        logger.info(
            "Import interface created successfully",
            extra={
                "service": "aclarai-ui",
                "component": "interface_creator",
                "action": "create_interface",
            },
        )
        return cast(gr.Blocks, interface)
    except Exception as e:
        logger.error(
            "Failed to create import interface",
            extra={
                "service": "aclarai-ui",
                "component": "interface_creator",
                "action": "create_interface",
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise


def create_complete_interface() -> gr.Blocks:
    """Create the complete aclarai interface with multiple panels."""
    try:
        logger.info(
            "Creating complete aclarai interface",
            extra={
                "service": "aclarai-ui",
                "component": "interface_creator",
                "action": "create_complete_interface",
            },
        )
        with gr.Blocks(title="aclarai", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🧠 aclarai - AI-Powered Knowledge Management")
            gr.Markdown(
                """Welcome to aclarai, an intelligent system for processing and organizing conversational data
                into structured knowledge graphs. Use the tabs below to access different functionality."""
            )
            with gr.Tabs():
                # Import panel content (This section remains unchanged)
                with gr.Tab("📥 Import", id="import_tab"), gr.Group():
                    # Session-based state management for import status
                    import_status_state = gr.State(ImportStatusTracker())
                    gr.Markdown(
                        """Upload conversation files from various sources (ChatGPT exports, Slack logs, generic text files)
                        to process and import into the aclarai system. Files are automatically detected and processed
                        using the appropriate format plugin."""
                    )
                    # File Picker Section
                    with gr.Group():
                        gr.Markdown("## 📁 File Selection")
                        file_input = gr.File(
                            label="Drag files here or click to browse",
                            file_types=[".json", ".txt", ".csv", ".md", ".zip"],
                            type="filepath",
                            height=100,
                        )
                        with gr.Row():
                            import_btn = gr.Button(
                                "🚀 Process File", variant="primary", size="lg"
                            )
                            clear_btn = gr.Button("🗑️ Clear Queue", variant="secondary")
                    # Live Import Queue Section
                    with gr.Group():
                        gr.Markdown("## 📋 Live Import Queue")
                        queue_display = gr.Markdown(
                            value="No files in import queue.", label="Import Status"
                        )
                    # Post-import Summary Section
                    with gr.Group():
                        gr.Markdown("## 📊 Import Summary")
                        summary_display = gr.Markdown(
                            value="Process files to see import summary.",
                            label="Summary",
                        )

                    # Event handlers for the import tab
                    def safe_real_plugin_orchestrator(
                        file_input_value, import_status
                    ) -> Tuple[str, str, ImportStatusTracker]:
                        try:
                            queue_display, summary_display, updated_status = (
                                real_plugin_orchestrator(
                                    file_input_value, import_status
                                )
                            )
                            return queue_display, summary_display, updated_status
                        except Exception as e:
                            logger.error(
                                "Error in real plugin orchestrator",
                                extra={
                                    "service": "aclarai-ui",
                                    "component": "interface_handler",
                                    "action": "real_orchestrator",
                                    "error": str(e),
                                    "error_type": type(e).__name__,
                                },
                            )
                            return (
                                "Error processing file. Please try again.",
                                "Processing failed.",
                                import_status,
                            )

                    def safe_clear_import_queue(
                        import_status,
                    ) -> Tuple[str, str, ImportStatusTracker]:
                        try:
                            queue_display, summary_display, new_status = (
                                clear_import_queue(import_status)
                            )
                            return queue_display, summary_display, new_status
                        except Exception as e:
                            logger.error(
                                "Error clearing import queue",
                                extra={
                                    "service": "aclarai-ui",
                                    "component": "interface_handler",
                                    "action": "clear_queue",
                                    "error": str(e),
                                    "error_type": type(e).__name__,
                                },
                            )
                            return (
                                "Error clearing queue. Please refresh the page.",
                                "",
                                import_status,
                            )

                    import_btn.click(
                        fn=safe_real_plugin_orchestrator,
                        inputs=[file_input, import_status_state],
                        outputs=[
                            queue_display,
                            summary_display,
                            import_status_state,
                        ],
                    )
                    file_input.change(
                        fn=safe_real_plugin_orchestrator,
                        inputs=[file_input, import_status_state],
                        outputs=[
                            queue_display,
                            summary_display,
                            import_status_state,
                        ],
                    )
                    clear_btn.click(
                        fn=safe_clear_import_queue,
                        inputs=[import_status_state],
                        outputs=[
                            queue_display,
                            summary_display,
                            import_status_state,
                        ],
                    )

                # Review Panel Tab (Unchanged)
                with gr.Tab("📊 Review", id="review_tab"):
                    create_review_panel()

                # Configuration Panel Tab (Drastically Simplified)
                with gr.Tab("⚙️ Configuration", id="config_tab"):
                    # All the logic is now encapsulated in this single function call.
                    # No more duplicated code.
                    create_configuration_panel()

        logger.info(
            "Complete interface created successfully",
            extra={
                "service": "aclarai-ui",
                "component": "interface_creator",
                "action": "create_complete_interface",
            },
        )
        return cast(gr.Blocks, interface)
    except Exception as e:
        logger.error(
            "Failed to create complete interface",
            extra={
                "service": "aclarai-ui",
                "component": "interface_creator",
                "action": "create_complete_interface",
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise


def main():
    """Launch the Gradio application with proper error handling."""
    try:
        logger.info(
            "Starting aclarai UI service",
            extra={"service": "aclarai-ui", "component": "main", "action": "startup"},
        )
        # Launch the complete interface with all tabs
        interface = create_complete_interface()
        logger.info(
            "Launching Gradio interface",
            extra={
                "service": "aclarai-ui",
                "component": "main",
                "action": "launch",
                "host": config.server_host,
                "port": config.server_port,
            },
        )
        interface.launch(
            server_name=config.server_host,
            server_port=config.server_port,
            share=False,
            debug=config.debug_mode,
        )
    except Exception as e:
        logger.error(
            "Failed to start aclarai UI service",
            extra={
                "service": "aclarai-ui",
                "component": "main",
                "action": "startup",
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise


if __name__ == "__main__":
    main()
