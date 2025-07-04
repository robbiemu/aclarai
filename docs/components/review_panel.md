# Review & Automation Status Panel

This document describes the Aclarai Review & Automation Status Panel, a component designed for monitoring system activities and managing automation states.

## Overview

The Review & Automation Status Panel provides a user interface to observe the status of various jobs within the Aclarai system and to control the overall automation process. Users can see whether automation is active or paused and can toggle this state. The panel is also designed to display details about file processing and claim statuses, though current implementation primarily focuses on automation controls.

## Architecture

### Core Components (Implemented)

1.  **Automation Status Display**: Shows whether the system-wide automation is currently "Running" or "Paused".
2.  **Pause/Resume Control**: A button that allows users to toggle the automation state.
    *   When automation is running, the button shows "[ ⏸️ Pause Automation ]".
    *   When automation is paused, the button shows "[ ▶️ Resume Automation ]".
3.  **Job Status Table**: A placeholder table intended to display the status of different jobs (e.g., Vault Sync, Concept Embedding Refresh). Currently, this table is not populated with live data in the UI.

### Core Components (Designed, from `design_review_panel.md`)

The full design envisions a more comprehensive panel including:

1.  **File / Block Index**: To show recent files, extracted blocks, and their processing states.
2.  **Claim Detail View**: To inspect Markdown blocks, metadata (scores, IDs), and link statuses, with actions like reprocessing or unlinking.
3.  **Claim Explorer**: A searchable, cross-file view of all extracted claims.

### Integration Points

*   **Automation State**: Integrates with `aclarai_shared.automation.pause_controller` to check (`is_paused()`) and modify (`pause()`, `resume()`) the system's automation flag (persisted via `.aclarai_pause` file in the vault root).
*   **Configuration**: Designed to pull state from `settings/aclarai.config.yaml` for job-specific overrides, though this is not fully implemented in the current UI panel.

## Implementation Details (Current)

The panel is built using Gradio.

### Automation Control Logic

*   `update_status_display()`: Fetches the current pause state and returns a Markdown string indicating "✅ Automation Status: Running" or "❌ Automation Status: Paused".
*   `toggle_pause_state()`: Calls `pause()` or `resume()` from the `pause_controller` and updates the button text and status display accordingly.

```python
# Example snippet from services/aclarai-ui/aclarai_ui/review_panel.py

def update_status_display() -> str:
    current_state = is_paused()
    if current_state:
        return "### ❌ Automation Status: Paused\nAll automated jobs are currently paused."
    return "### ✅ Automation Status: Running\nAutomated jobs are active and running on schedule."

def toggle_pause_state() -> Tuple[str, str]:
    current_state = is_paused()
    if current_state:
        resume()
        return "[ ⏸️ Pause Automation ]", update_status_display()
    else:
        pause()
        return "[ ▶️ Resume Automation ]", update_status_display()
```

## User Interface (Current)

The current UI consists of:

*   A title: "📊 Review Panel".
*   A section: "⚙️ Automation Status + Controls".
    *   A Markdown display for the automation status.
    *   A button to Pause/Resume automation.
    *   An empty `gr.DataFrame` intended for job statuses.

## Error Handling

*   The panel includes basic error handling for Gradio UI creation.
*   Functions `update_status_display()` and `toggle_pause_state()` log errors if exceptions occur during their execution and display error messages in the UI.

## Testing

*   Manual UI testing can be performed by running the Aclarai UI.
*   The underlying pause/resume logic in `aclarai_shared.automation.pause_controller` has its own tests.

## Configuration

*   The primary configuration affecting this panel is the presence or absence of the `.aclarai_pause` file in the vault root, which dictates the automation state.
*   The design document mentions `settings/aclarai.config.yaml` for job-specific pause overrides, but this is not yet reflected in the panel's direct behavior.

## Logging

The panel uses the standard Python logging module. Log entries include service, component, and action details:

```python
logger.error(
    "Failed to toggle pause state",
    extra={
        "service": "aclarai-ui",
        "component": "review_panel",
        "action": "toggle_pause",
        "error": str(e),
    },
)
```

## Future Enhancements

Based on `docs/arch/design_review_panel.md`:

1.  **Implement File / Block Index**: Display files, blocks, and their processing states.
2.  **Implement Claim Detail View**: Allow inspection of claim metadata and actions.
3.  **Implement Claim Explorer**: Provide a searchable view of all claims.
4.  **Populate Job Status Table**: Connect the job status table to live data from the scheduler or job registry.
5.  **Integrate Job-Specific Controls**: Allow pausing/resuming individual jobs as per design.

## Dependencies

*   `gradio`: For building the web interface.
*   `aclarai_shared.automation.pause_controller`: For managing the system's automation pause state.

## Related Documentation

*   [Review & Automation Status Panel Design](../arch/design_review_panel.md) - Detailed design specification.
*   [Automation Pause Controller](../../shared/aclarai_shared/automation/pause_controller.py) - Code for pause/resume logic.
*   [UI System](./ui_system.md) - Overview of the UI system.
