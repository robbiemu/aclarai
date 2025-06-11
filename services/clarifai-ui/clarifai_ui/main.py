"""Main Gradio application for ClarifAI frontend."""

import gradio as gr
import logging
import os
import time
from typing import Optional, Tuple
from datetime import datetime

from .config import config


# Configure structured logging as per docs/arch/on-error-handling-and-resilience.md
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("clarifai-ui")


class ImportStatus:
    """Class to track import status and queue."""

    def __init__(self):
        self.import_queue = []
        self.summary_stats = {"imported": 0, "failed": 0, "fallback": 0, "skipped": 0}
        logger.info(
            "ImportStatus initialized",
            extra={
                "service": "clarifai-ui",
                "component": "ImportStatus",
                "action": "initialize",
            },
        )

    def add_file(self, filename: str, file_path: str) -> None:
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
                    "service": "clarifai-ui",
                    "component": "ImportStatus",
                    "action": "add_file",
                    "file_name": filename,
                    "file_path": file_path,
                },
            )
        except Exception as e:
            logger.error(
                "Failed to add file to import queue",
                extra={
                    "service": "clarifai-ui",
                    "component": "ImportStatus",
                    "action": "add_file",
                    "file_name": filename,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise

    def update_file_status(self, filename: str, status: str, detector: str) -> None:
        """Update the status of a file in the queue."""
        try:
            for item in self.import_queue:
                if item["filename"] == filename:
                    item["status"] = status
                    item["detector"] = detector
                    logger.info(
                        "File status updated",
                        extra={
                            "service": "clarifai-ui",
                            "component": "ImportStatus",
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
                        "service": "clarifai-ui",
                        "component": "ImportStatus",
                        "action": "update_status",
                        "file_name": filename,
                        "attempted_status": status,
                    },
                )
        except Exception as e:
            logger.error(
                "Failed to update file status",
                extra={
                    "service": "clarifai-ui",
                    "component": "ImportStatus",
                    "action": "update_status",
                    "file_name": filename,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Graceful degradation: continue operation even if status update fails

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
                    "service": "clarifai-ui",
                    "component": "ImportStatus",
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
                    "service": "clarifai-ui",
                    "component": "ImportStatus",
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
                    "service": "clarifai-ui",
                    "component": "ImportStatus",
                    "action": "get_summary",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Graceful degradation: return basic error message
            return "## 📊 Import Summary\n\nError generating summary. Please check logs for details."


def detect_file_format(file_path: str, filename: str) -> Tuple[str, str]:
    """Simulate format detection logic with proper error handling.

    Returns:
        Tuple of (detector_name, status)
    """
    try:
        logger.info(
            "Starting format detection",
            extra={
                "service": "clarifai-ui",
                "component": "format_detector",
                "action": "detect_format",
                "file_name": filename,
                "file_path": file_path,
            },
        )

        # Simulate format detection based on file extension and content
        time.sleep(0.5)  # Simulate processing time

        if filename.lower().endswith(".json"):
            # Check if it's a Slack export based on filename
            if "slack" in filename.lower():
                detector, status = "slack_json", "✅ Imported"
            else:
                # Default to ChatGPT JSON format
                detector, status = "chatgpt_json", "✅ Imported"
        elif filename.lower().endswith(".csv"):
            # Check if it's a generic export based on filename
            if "generic" in filename.lower() or "tabular" in filename.lower():
                detector, status = "generic_csv", "✅ Imported"
            else:
                # Default to Slack CSV format
                detector, status = "slack_csv", "✅ Imported"
        elif filename.lower().endswith(".txt"):
            # Simulate generic text that needs fallback
            detector, status = "fallback_llm", "⚠️ Fallback"
        elif filename.lower().endswith(".md"):
            # Simulate markdown format
            detector, status = "markdown", "✅ Imported"
        else:
            # Simulate unsupported format
            detector, status = "None", "❌ Failed"

        logger.info(
            "Format detection completed",
            extra={
                "service": "clarifai-ui",
                "component": "format_detector",
                "action": "detect_format",
                "file_name": filename,
                "detector": detector,
                "status": status,
            },
        )

        return detector, status

    except Exception as e:
        logger.error(
            "Format detection failed",
            extra={
                "service": "clarifai-ui",
                "component": "format_detector",
                "action": "detect_format",
                "file_name": filename,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        # Graceful degradation: return failed status
        return "error", "❌ Failed"


def simulate_plugin_orchestrator(
    file_path: Optional[str], import_status: ImportStatus
) -> Tuple[str, str, ImportStatus]:
    """Simulate the plugin orchestrator processing a file with proper error handling.

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
                    "service": "clarifai-ui",
                    "component": "plugin_orchestrator",
                    "action": "simulate_import",
                },
            )
            return "No file selected for import.", "", import_status

        filename = os.path.basename(file_path)

        logger.info(
            "Starting plugin orchestrator simulation",
            extra={
                "service": "clarifai-ui",
                "component": "plugin_orchestrator",
                "action": "simulate_import",
                "file_name": filename,
                "file_path": file_path,
            },
        )

        # Check for duplicates (simple simulation)
        existing_files = [item["filename"] for item in import_status.import_queue]
        if filename in existing_files:
            logger.info(
                "Duplicate file detected",
                extra={
                    "service": "clarifai-ui",
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

        # Add file to queue
        import_status.add_file(filename, file_path)

        # Start with processing status
        queue_display = import_status.format_queue_display()

        # Simulate format detection and processing
        detector, status = detect_file_format(file_path, filename)

        # Update status after "processing"
        time.sleep(1)  # Simulate processing time
        import_status.update_file_status(filename, status, detector)

        logger.info(
            "Plugin orchestrator simulation completed",
            extra={
                "service": "clarifai-ui",
                "component": "plugin_orchestrator",
                "action": "simulate_import",
                "file_name": filename,
                "final_status": status,
                "detector": detector,
            },
        )

        return (
            import_status.format_queue_display(),
            import_status.get_summary(),
            import_status,
        )

    except Exception as e:
        logger.error(
            "Plugin orchestrator simulation failed",
            extra={
                "service": "clarifai-ui",
                "component": "plugin_orchestrator",
                "action": "simulate_import",
                "file_name": filename if "filename" in locals() else "unknown",
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        # Graceful degradation: return error message instead of crashing
        error_msg = f"Error processing file: {str(e)}"
        return (
            error_msg,
            "Processing failed. Please check logs for details.",
            import_status,
        )


def clear_import_queue(import_status: ImportStatus) -> Tuple[str, str, ImportStatus]:
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
                "service": "clarifai-ui",
                "component": "queue_manager",
                "action": "clear_queue",
                "queue_size": len(import_status.import_queue),
            },
        )
        new_import_status = ImportStatus()
        return "Import queue cleared.", "", new_import_status
    except Exception as e:
        logger.error(
            "Failed to clear import queue",
            extra={
                "service": "clarifai-ui",
                "component": "queue_manager",
                "action": "clear_queue",
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        # Graceful degradation: return error message
        return "Error clearing queue. Please refresh the page.", "", import_status


def create_import_interface():
    """Create the complete import interface following the documented design."""
    try:
        logger.info(
            "Creating import interface",
            extra={
                "service": "clarifai-ui",
                "component": "interface_creator",
                "action": "create_interface",
            },
        )

        with gr.Blocks(
            title="ClarifAI - Import Panel", theme=gr.themes.Soft()
        ) as interface:
            # Session-based state management for import status
            import_status_state = gr.State(ImportStatus())

            gr.Markdown("# 📥 ClarifAI Import Panel")
            gr.Markdown(
                """Upload conversation files from various sources (ChatGPT exports, Slack logs, generic text files) 
                to process and import into the ClarifAI system. Files are automatically detected and processed 
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
            def safe_simulate_plugin_orchestrator(file_input_value, import_status):
                try:
                    queue_display, summary_display, updated_status = (
                        simulate_plugin_orchestrator(file_input_value, import_status)
                    )
                    return queue_display, summary_display, updated_status
                except Exception as e:
                    logger.error(
                        "Error in plugin orchestrator simulation",
                        extra={
                            "service": "clarifai-ui",
                            "component": "interface_handler",
                            "action": "simulate_orchestrator",
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    return (
                        "Error processing file. Please try again.",
                        "Processing failed.",
                        import_status,
                    )

            def safe_clear_import_queue(import_status):
                try:
                    queue_display, summary_display, new_status = clear_import_queue(
                        import_status
                    )
                    return queue_display, summary_display, new_status
                except Exception as e:
                    logger.error(
                        "Error clearing import queue",
                        extra={
                            "service": "clarifai-ui",
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
                fn=safe_simulate_plugin_orchestrator,
                inputs=[file_input, import_status_state],
                outputs=[queue_display, summary_display, import_status_state],
            )

            # Auto-process on file upload
            file_input.change(
                fn=safe_simulate_plugin_orchestrator,
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
                "service": "clarifai-ui",
                "component": "interface_creator",
                "action": "create_interface",
            },
        )

        return interface

    except Exception as e:
        logger.error(
            "Failed to create import interface",
            extra={
                "service": "clarifai-ui",
                "component": "interface_creator",
                "action": "create_interface",
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise


def main():
    """Launch the Gradio application with proper error handling."""
    try:
        logger.info(
            "Starting ClarifAI UI service",
            extra={"service": "clarifai-ui", "component": "main", "action": "startup"},
        )

        interface = create_import_interface()

        logger.info(
            "Launching Gradio interface",
            extra={
                "service": "clarifai-ui",
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
            "Failed to start ClarifAI UI service",
            extra={
                "service": "clarifai-ui",
                "component": "main",
                "action": "startup",
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise


if __name__ == "__main__":
    main()
