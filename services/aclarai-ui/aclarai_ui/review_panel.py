"""
Review Panel for aclarai, providing automation status and control.

This panel follows the design specified in docs/arch/design_review_panel.md,
implementing the Automation Status + Controls section with pause functionality.
"""

import logging
from typing import Optional, Tuple, cast

import gradio as gr
from aclarai_shared.automation.pause_controller import is_paused, pause, resume

logger = logging.getLogger(__name__)


def create_review_panel() -> Optional[gr.Blocks]:
    """Create the Review Panel with automation controls."""
    try:
        with gr.Blocks() as panel:
            gr.Markdown("# 📊 Review Panel")

            # Automation Status + Controls Section
            with gr.Group():
                gr.Markdown("### ⚙️ Automation Status + Controls")

                # Status Display
                with gr.Row():
                    automation_status = gr.Markdown(
                        value="Loading status...", elem_id="automation-status"
                    )

                # Pause Button
                pause_btn = gr.Button(
                    value="[ ⏸️ Pause Automation ]",
                    variant="secondary",
                    elem_id="pause-button",
                )

                # Job Status Display
                gr.DataFrame(
                    headers=["Job", "Last Run", "Next Run", "Status"],
                    value=[],
                    elem_id="job-status-table",
                )

            def update_status_display() -> str:
                """Update the automation status display.

                Returns:
                    str: Markdown formatted status message
                """
                try:
                    current_state = is_paused()
                    if current_state:
                        return "### ❌ Automation Status: Paused\nAll automated jobs are currently paused."
                    return "### ✅ Automation Status: Running\nAutomated jobs are active and running on schedule."
                except Exception as e:
                    logger.error(
                        "Failed to update automation status",
                        extra={
                            "service": "aclarai-ui",
                            "component": "review_panel",
                            "action": "update_status",
                            "error": str(e),
                        },
                    )
                    return "### ⚠️ Error: Could not determine automation status"

            def toggle_pause_state() -> Tuple[str, str]:
                """Toggle the pause state and update UI elements.

                Returns:
                    Tuple[str, str]: New button text and status display
                """
                try:
                    current_state = is_paused()
                    if current_state:
                        resume()
                        return "[ ⏸️ Pause Automation ]", update_status_display()
                    else:
                        pause()
                        return "[ ▶️ Resume Automation ]", update_status_display()
                except Exception as e:
                    logger.error(
                        "Failed to toggle pause state",
                        extra={
                            "service": "aclarai-ui",
                            "component": "review_panel",
                            "action": "toggle_pause",
                            "error": str(e),
                        },
                    )
                    return (
                        "[ ⚠️ Error ]",
                        "### ⚠️ Error: Failed to change automation state",
                    )

            # Initial status update
            automation_status.value = update_status_display()

            # Wire up the pause button
            pause_btn.click(
                fn=toggle_pause_state, outputs=[pause_btn, automation_status]
            )

        # To simulate periodic updates, an actual scheduler or async setup would be needed.

        return cast(Optional[gr.Blocks], panel)
    except Exception as e:
        logger.error(
            "Failed to create review panel",
            extra={
                "service": "aclarai-ui",
                "component": "review_panel",
                "action": "create_panel",
                "error": str(e),
            },
        )
        raise  # Re-raise to ensure the error is properly handled
