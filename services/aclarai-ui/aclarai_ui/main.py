"""Main Gradio application for aclarai frontend."""

import logging
import os
import time
from datetime import datetime
from typing import Optional, Tuple, cast

import gradio as gr

from .config import config

# Configure structured logging as per docs/arch/on-error-handling-and-resilience.md
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("aclarai-ui")


class ImportStatus:
    """Class to track import status and queue."""

    def __init__(self):
        self.import_queue = []
        self.summary_stats = {"imported": 0, "failed": 0, "fallback": 0, "skipped": 0}
        logger.info(
            "ImportStatus initialized",
            extra={
                "service": "aclarai-ui",
                "component": "ImportStatus",
                "action": "initialize",
            },
        )

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
                    "service": "aclarai-ui",
                    "component": "ImportStatus",
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
                        "service": "aclarai-ui",
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
                    "service": "aclarai-ui",
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
                    "service": "aclarai-ui",
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
                    "service": "aclarai-ui",
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
                    "service": "aclarai-ui",
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
                "service": "aclarai-ui",
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
                "service": "aclarai-ui",
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
                "service": "aclarai-ui",
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
                    "service": "aclarai-ui",
                    "component": "plugin_orchestrator",
                    "action": "simulate_import",
                },
            )
            return "No file selected for import.", "", import_status
        filename = os.path.basename(file_path)
        logger.info(
            "Starting plugin orchestrator simulation",
            extra={
                "service": "aclarai-ui",
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
                "service": "aclarai-ui",
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
                "service": "aclarai-ui",
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
                "service": "aclarai-ui",
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
            import_status_state = gr.State(ImportStatus())
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
            def safe_simulate_plugin_orchestrator(
                file_input_value, import_status
            ) -> Tuple[str, str, ImportStatus]:
                try:
                    queue_display, summary_display, updated_status = (
                        simulate_plugin_orchestrator(file_input_value, import_status)
                    )
                    return queue_display, summary_display, updated_status
                except Exception as e:
                    logger.error(
                        "Error in plugin orchestrator simulation",
                        extra={
                            "service": "aclarai-ui",
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

            def safe_clear_import_queue(
                import_status,
            ) -> Tuple[str, str, ImportStatus]:
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
                # Import panel content (existing functionality)
                with gr.Tab("📥 Import", id="import_tab"), gr.Group():
                    # Session-based state management for import status
                    import_status_state = gr.State(ImportStatus())
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

                    # Event handlers with error handling and state management
                    def safe_simulate_plugin_orchestrator(
                        file_input_value, import_status
                    ) -> Tuple[str, str, ImportStatus]:
                        try:
                            queue_display, summary_display, updated_status = (
                                simulate_plugin_orchestrator(
                                    file_input_value, import_status
                                )
                            )
                            return queue_display, summary_display, updated_status
                        except Exception as e:
                            logger.error(
                                "Error in plugin orchestrator simulation",
                                extra={
                                    "service": "aclarai-ui",
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

                    def safe_clear_import_queue(
                        import_status,
                    ) -> Tuple[str, str, ImportStatus]:
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
                        fn=safe_simulate_plugin_orchestrator,
                        inputs=[file_input, import_status_state],
                        outputs=[
                            queue_display,
                            summary_display,
                            import_status_state,
                        ],
                    )
                    # Auto-process on file upload
                    file_input.change(
                        fn=safe_simulate_plugin_orchestrator,
                        inputs=[file_input, import_status_state],
                        outputs=[
                            queue_display,
                            summary_display,
                            import_status_state,
                        ],
                    )
                    # Clear queue handler
                    clear_btn.click(
                        fn=safe_clear_import_queue,
                        inputs=[import_status_state],
                        outputs=[
                            queue_display,
                            summary_display,
                            import_status_state,
                        ],
                    )
                with gr.Tab("⚙️ Configuration", id="config_tab"):
                    # Configuration panel - directly embed the components
                    from .config_panel import (
                        ConfigurationManager,
                        validate_model_name,
                        validate_threshold,
                        validate_window_param,
                    )

                    # Initialize configuration manager
                    config_manager = ConfigurationManager()
                    gr.Markdown("## ⚙️ Configuration Panel")
                    gr.Markdown("""
                    Configure aclarai's behavior by setting model selections, embedding models,
                    processing thresholds, and context window parameters. Changes are automatically
                    saved to your configuration file.
                    """)
                    # Load current configuration
                    current_config = config_manager.load_config()
                    # Model & Embedding Settings Section
                    with gr.Group():
                        gr.Markdown("### 🤖 Model & Embedding Settings")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**Claimify Models**")
                                claimify_default_input = gr.Textbox(
                                    label="Default Model",
                                    value=current_config.get("model", {})
                                    .get("claimify", {})
                                    .get("default", ""),
                                    placeholder="e.g., gpt-3.5-turbo",
                                    info="Base model for all Claimify stages",
                                )
                                claimify_selection_input = gr.Textbox(
                                    label="Selection Model (Optional)",
                                    value=current_config.get("model", {})
                                    .get("claimify", {})
                                    .get("selection", "")
                                    or "",
                                    placeholder="e.g., claude-3-opus",
                                    info="Override for claim selection",
                                )
                                claimify_disambiguation_input = gr.Textbox(
                                    label="Disambiguation Model (Optional)",
                                    value=current_config.get("model", {})
                                    .get("claimify", {})
                                    .get("disambiguation", "")
                                    or "",
                                    placeholder="e.g., mistral-7b",
                                    info="Override for ambiguity resolution",
                                )
                                claimify_decomposition_input = gr.Textbox(
                                    label="Decomposition Model (Optional)",
                                    value=current_config.get("model", {})
                                    .get("claimify", {})
                                    .get("decomposition", "")
                                    or "",
                                    placeholder="e.g., gpt-4",
                                    info="Override for claim decomposition",
                                )
                            with gr.Column():
                                gr.Markdown("**Agent Models**")
                                concept_linker_input = gr.Textbox(
                                    label="Concept Linker",
                                    value=current_config.get("model", {}).get(
                                        "concept_linker", ""
                                    ),
                                    placeholder="e.g., gpt-3.5-turbo",
                                    info="Links claims to concepts",
                                )
                                concept_summary_input = gr.Textbox(
                                    label="Concept Summary",
                                    value=current_config.get("model", {}).get(
                                        "concept_summary", ""
                                    ),
                                    placeholder="e.g., gpt-4",
                                    info="Generates [[Concept]] pages",
                                )
                                subject_summary_input = gr.Textbox(
                                    label="Subject Summary",
                                    value=current_config.get("model", {}).get(
                                        "subject_summary", ""
                                    ),
                                    placeholder="e.g., claude-3-sonnet",
                                    info="Generates [[Subject:XYZ]] pages",
                                )
                                trending_concepts_agent_input = gr.Textbox(
                                    label="Trending Concepts Agent",
                                    value=current_config.get("model", {}).get(
                                        "trending_concepts_agent", ""
                                    ),
                                    placeholder="e.g., gpt-3.5-turbo",
                                    info="Writes trending topic summaries",
                                )
                                fallback_plugin_input = gr.Textbox(
                                    label="Fallback Plugin",
                                    value=current_config.get("model", {}).get(
                                        "fallback_plugin", ""
                                    ),
                                    placeholder="e.g., gpt-3.5-turbo",
                                    info="Used when format detection fails",
                                )
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**Embedding Models**")
                                utterance_embedding_input = gr.Textbox(
                                    label="Utterance Embeddings",
                                    value=current_config.get("embedding", {}).get(
                                        "utterance", ""
                                    ),
                                    placeholder="e.g., text-embedding-3-small",
                                    info="For Tier 1 conversation blocks",
                                )
                                concept_embedding_input = gr.Textbox(
                                    label="Concept Embeddings",
                                    value=current_config.get("embedding", {}).get(
                                        "concept", ""
                                    ),
                                    placeholder="e.g., sentence-transformers/all-MiniLM-L6-v2",
                                    info="For Tier 3 concept files",
                                )
                            with gr.Column():
                                summary_embedding_input = gr.Textbox(
                                    label="Summary Embeddings",
                                    value=current_config.get("embedding", {}).get(
                                        "summary", ""
                                    ),
                                    placeholder="e.g., text-embedding-3-small",
                                    info="For Tier 2 summaries",
                                )
                                fallback_embedding_input = gr.Textbox(
                                    label="Fallback Embeddings",
                                    value=current_config.get("embedding", {}).get(
                                        "fallback", ""
                                    ),
                                    placeholder="e.g., text-embedding-3-small",
                                    info="Used when other configs fail",
                                )
                    # Thresholds & Parameters Section
                    with gr.Group():
                        gr.Markdown("### 📏 Thresholds & Parameters")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**Similarity Thresholds**")
                                concept_merge_input = gr.Number(
                                    label="Concept Merge Threshold",
                                    value=current_config.get("threshold", {}).get(
                                        "concept_merge", 0.90
                                    ),
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    info="Cosine similarity required to merge concept candidates (0.0-1.0)",
                                )
                                claim_link_strength_input = gr.Number(
                                    label="Claim Link Strength",
                                    value=current_config.get("threshold", {}).get(
                                        "claim_link_strength", 0.60
                                    ),
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    info="Minimum strength to create claim→concept edges (0.0-1.0)",
                                )
                            with gr.Column():
                                gr.Markdown("**Context Window Parameters**")
                                window_p_input = gr.Number(
                                    label="Previous Sentences (p)",
                                    value=current_config.get("window", {})
                                    .get("claimify", {})
                                    .get("p", 3),
                                    minimum=0,
                                    maximum=10,
                                    step=1,
                                    info="How many sentences before target sentence to include (0-10)",
                                )
                                window_f_input = gr.Number(
                                    label="Following Sentences (f)",
                                    value=current_config.get("window", {})
                                    .get("claimify", {})
                                    .get("f", 1),
                                    minimum=0,
                                    maximum=10,
                                    step=1,
                                    info="How many sentences after target sentence to include (0-10)",
                                )
                    # Control Buttons
                    with gr.Row():
                        save_btn = gr.Button(
                            "💾 Save Changes", variant="primary", size="lg"
                        )
                        reload_btn = gr.Button(
                            "🔄 Reload from File", variant="secondary"
                        )
                    # Status Display
                    save_status = gr.Markdown("Ready to configure settings.")

                    def save_configuration(*args) -> str:
                        """Save the configuration with validation."""
                        try:
                            # Unpack arguments in the same order as inputs
                            (
                                claimify_default,
                                claimify_selection,
                                claimify_disambiguation,
                                claimify_decomposition,
                                concept_linker,
                                concept_summary,
                                subject_summary,
                                trending_concepts_agent,
                                fallback_plugin,
                                utterance_embedding,
                                concept_embedding,
                                summary_embedding,
                                fallback_embedding,
                                concept_merge,
                                claim_link_strength,
                                window_p,
                                window_f,
                            ) = args
                            # Validate inputs
                            validation_errors = []
                            # Validate required model names
                            for name, value in [
                                ("Claimify Default", claimify_default),
                                ("Concept Linker", concept_linker),
                                ("Concept Summary", concept_summary),
                                ("Subject Summary", subject_summary),
                                ("Trending Concepts Agent", trending_concepts_agent),
                                ("Fallback Plugin", fallback_plugin),
                                ("Utterance Embedding", utterance_embedding),
                                ("Concept Embedding", concept_embedding),
                                ("Summary Embedding", summary_embedding),
                                ("Fallback Embedding", fallback_embedding),
                            ]:
                                if value and value.strip():
                                    is_valid, error = validate_model_name(value)
                                    if not is_valid:
                                        validation_errors.append(f"{name}: {error}")
                            # Validate optional model names
                            for name, value in [
                                ("Claimify Selection", claimify_selection),
                                ("Claimify Disambiguation", claimify_disambiguation),
                                ("Claimify Decomposition", claimify_decomposition),
                            ]:
                                if value and value.strip():
                                    is_valid, error = validate_model_name(value)
                                    if not is_valid:
                                        validation_errors.append(f"{name}: {error}")
                            # Validate thresholds
                            for name, value in [
                                ("Concept Merge Threshold", concept_merge),
                                ("Claim Link Strength", claim_link_strength),
                            ]:
                                is_valid, error = validate_threshold(value)
                                if not is_valid:
                                    validation_errors.append(f"{name}: {error}")
                            # Validate window parameters
                            for name, value in [
                                ("Previous Sentences (p)", window_p),
                                ("Following Sentences (f)", window_f),
                            ]:
                                is_valid, error = validate_window_param(value)
                                if not is_valid:
                                    validation_errors.append(f"{name}: {error}")
                            if validation_errors:
                                error_msg = "❌ **Validation Errors:**\n" + "\n".join(
                                    f"- {error}" for error in validation_errors
                                )
                                return error_msg
                            # Build configuration dictionary
                            new_config = {
                                "model": {
                                    "claimify": {
                                        "default": claimify_default,
                                    },
                                    "concept_linker": concept_linker,
                                    "concept_summary": concept_summary,
                                    "subject_summary": subject_summary,
                                    "trending_concepts_agent": trending_concepts_agent,
                                    "fallback_plugin": fallback_plugin,
                                },
                                "embedding": {
                                    "utterance": utterance_embedding,
                                    "concept": concept_embedding,
                                    "summary": summary_embedding,
                                    "fallback": fallback_embedding,
                                },
                                "threshold": {
                                    "concept_merge": concept_merge,
                                    "claim_link_strength": claim_link_strength,
                                },
                                "window": {
                                    "claimify": {
                                        "p": int(window_p),
                                        "f": int(window_f),
                                    }
                                },
                            }
                            # Add optional claimify models if specified
                            if claimify_selection and claimify_selection.strip():
                                new_config["model"]["claimify"]["selection"] = (
                                    claimify_selection
                                )
                            if (
                                claimify_disambiguation
                                and claimify_disambiguation.strip()
                            ):
                                new_config["model"]["claimify"]["disambiguation"] = (
                                    claimify_disambiguation
                                )
                            if (
                                claimify_decomposition
                                and claimify_decomposition.strip()
                            ):
                                new_config["model"]["claimify"]["decomposition"] = (
                                    claimify_decomposition
                                )
                            # Save configuration
                            if config_manager.save_config(new_config):
                                return "✅ **Configuration saved successfully!**\n\nChanges have been written to `settings/aclarai.config.yaml`."
                            else:
                                return "❌ **Failed to save configuration.** Please check file permissions and try again."
                        except Exception as e:
                            logger.error(
                                "Configuration save failed",
                                extra={
                                    "service": "aclarai-ui",
                                    "component": "config_panel",
                                    "action": "save_config",
                                    "error": str(e),
                                    "error_type": type(e).__name__,
                                },
                            )
                            return f"❌ **Error saving configuration:** {str(e)}"

                    def reload_configuration() -> Tuple[
                        str,
                        str,
                        str,
                        str,
                        str,
                        str,
                        str,
                        str,
                        str,
                        str,
                        str,
                        str,
                        str,
                        float,
                        float,
                        int,
                        int,
                        str,
                    ]:
                        """Reload configuration from file."""
                        try:
                            current_config = config_manager.load_config()
                            return (
                                current_config.get("model", {})
                                .get("claimify", {})
                                .get("default", ""),
                                current_config.get("model", {})
                                .get("claimify", {})
                                .get("selection", "")
                                or "",
                                current_config.get("model", {})
                                .get("claimify", {})
                                .get("disambiguation", "")
                                or "",
                                current_config.get("model", {})
                                .get("claimify", {})
                                .get("decomposition", "")
                                or "",
                                current_config.get("model", {}).get(
                                    "concept_linker", ""
                                ),
                                current_config.get("model", {}).get(
                                    "concept_summary", ""
                                ),
                                current_config.get("model", {}).get(
                                    "subject_summary", ""
                                ),
                                current_config.get("model", {}).get(
                                    "trending_concepts_agent", ""
                                ),
                                current_config.get("model", {}).get(
                                    "fallback_plugin", ""
                                ),
                                current_config.get("embedding", {}).get(
                                    "utterance", ""
                                ),
                                current_config.get("embedding", {}).get("concept", ""),
                                current_config.get("embedding", {}).get("summary", ""),
                                current_config.get("embedding", {}).get("fallback", ""),
                                current_config.get("threshold", {}).get(
                                    "concept_merge", 0.90
                                ),
                                current_config.get("threshold", {}).get(
                                    "claim_link_strength", 0.60
                                ),
                                current_config.get("window", {})
                                .get("claimify", {})
                                .get("p", 3),
                                current_config.get("window", {})
                                .get("claimify", {})
                                .get("f", 1),
                                "🔄 **Configuration reloaded from file.**",
                            )
                        except Exception as e:
                            logger.error(
                                "Configuration reload failed",
                                extra={
                                    "service": "aclarai-ui",
                                    "component": "config_panel",
                                    "action": "reload_config",
                                    "error": str(e),
                                    "error_type": type(e).__name__,
                                },
                            )
                            # Return current values unchanged, just update status
                            return (
                                *[
                                    comp.value
                                    for comp in [
                                        claimify_default_input,
                                        claimify_selection_input,
                                        claimify_disambiguation_input,
                                        claimify_decomposition_input,
                                        concept_linker_input,
                                        concept_summary_input,
                                        subject_summary_input,
                                        trending_concepts_agent_input,
                                        fallback_plugin_input,
                                        utterance_embedding_input,
                                        concept_embedding_input,
                                        summary_embedding_input,
                                        fallback_embedding_input,
                                        concept_merge_input,
                                        claim_link_strength_input,
                                        window_p_input,
                                        window_f_input,
                                    ]
                                ],
                                f"❌ **Error reloading configuration:** {str(e)}",
                            )

                    # Wire up event handlers
                    save_btn.click(
                        fn=save_configuration,
                        inputs=[
                            claimify_default_input,
                            claimify_selection_input,
                            claimify_disambiguation_input,
                            claimify_decomposition_input,
                            concept_linker_input,
                            concept_summary_input,
                            subject_summary_input,
                            trending_concepts_agent_input,
                            fallback_plugin_input,
                            utterance_embedding_input,
                            concept_embedding_input,
                            summary_embedding_input,
                            fallback_embedding_input,
                            concept_merge_input,
                            claim_link_strength_input,
                            window_p_input,
                            window_f_input,
                        ],
                        outputs=[save_status],
                    )
                    reload_btn.click(
                        fn=reload_configuration,
                        outputs=[
                            claimify_default_input,
                            claimify_selection_input,
                            claimify_disambiguation_input,
                            claimify_decomposition_input,
                            concept_linker_input,
                            concept_summary_input,
                            subject_summary_input,
                            trending_concepts_agent_input,
                            fallback_plugin_input,
                            utterance_embedding_input,
                            concept_embedding_input,
                            summary_embedding_input,
                            fallback_embedding_input,
                            concept_merge_input,
                            claim_link_strength_input,
                            window_p_input,
                            window_f_input,
                            save_status,
                        ],
                    )
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
        raise  # Always raise, never return None or Any


def main():
    """Launch the Gradio application with proper error handling."""
    try:
        logger.info(
            "Starting aclarai UI service",
            extra={"service": "aclarai-ui", "component": "main", "action": "startup"},
        )
        # For now, we'll launch the import interface as the main interface
        # The configuration panel can be accessed separately
        interface = create_import_interface()
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
