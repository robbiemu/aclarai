# Configuration Panel

This document describes the Aclarai Configuration Panel, a UI component for managing system settings and agent parameters.

## Overview

The Configuration Panel provides a user-friendly Gradio interface to view and modify various settings for the Aclarai system. These settings include model selections for different AI agents, processing thresholds, context window sizes for Claimify, scheduler job controls, and parameters for concept highlight generation. Changes made in this panel are persisted to the `settings/aclarai.config.yaml` file.

## Architecture

### Core Components

1.  **ConfigurationManager Class**: Handles loading configuration from `settings/aclarai.config.yaml` (with defaults from `shared/aclarai_shared/aclarai.config.default.yaml`) and saving changes back to the user's configuration file. It performs a deep merge of user settings over defaults.
2.  **ConfigData Dataclass**: A structured container holding all configuration values exposed in the UI.
3.  **Validation Functions**:
    *   `validate_model_name()`: Checks for valid model name formats (e.g., `gpt-`, `claude-`, `openrouter:`, `ollama:`, HuggingFace paths).
    *   `validate_threshold()`: Ensures numerical thresholds are within a specified range (typically 0.0 to 1.0).
    *   `validate_window_param()`: Validates integer window parameters (e.g., for Claimify context).
    *   `validate_cron_expression()`: Basic validation for cron string format (5 fields).
    *   `validate_concept_highlights_config()`: Validates parameters for Top Concepts and Trending Topics jobs, including metric choices, count/percent mutual exclusivity, and target filenames.
4.  **Gradio UI Elements**: Textboxes, number inputs, dropdowns, and checkboxes for each configurable setting, organized into sections.
5.  **Save/Reload Logic**:
    *   `load_current_config()`: Populates the `ConfigData` object from the `ConfigurationManager`.
    *   `save_configuration()`: Gathers all values from UI inputs, validates them, updates the loaded configuration dictionary, and uses `ConfigurationManager` to write to YAML.
    *   `reload_configuration()`: Reloads settings from the YAML file and updates all UI components.

### Integration Points

*   **YAML Configuration Files**:
    *   Reads from `settings/aclarai.config.yaml` (user's custom settings).
    *   Reads from `shared/aclarai_shared/aclarai.config.default.yaml` (system defaults).
    *   Writes changes to `settings/aclarai.config.yaml`.
*   **Aclarai System**: The settings modified here directly affect the behavior of various Aclarai agents, processing pipelines, and scheduled jobs that read this configuration.

## Implementation Details

### Configuration Loading and Saving

The `ConfigurationManager` class is central to persisting settings. It uses a temporary file and atomic rename pattern for safer writes.

```python
# Example snippet from services/aclarai-ui/aclarai_ui/config_panel.py
class ConfigurationManager:
    def load_config(self) -> Dict[str, Any]:
        # ... loads default and user YAML, then deep merges ...
        pass

    def save_config(self, config: Dict[str, Any]) -> bool:
        # ... writes to a temp file, then renames to actual config path ...
        pass
```

### UI Structure

The panel is organized into several collapsible sections using `gr.Group()`:

*   **🤖 Model & Embedding Settings**:
    *   🔮 Claimify Models (Default, Selection, Disambiguation, Decomposition)
    *   🧠 Agent Models (Concept Linker, Concept Summary, Subject Summary, Trending Concepts Agent, Fallback Plugin)
    *   🧬 Embedding Models (Utterance, Concept, Summary, Fallback)
*   **📏 Thresholds & Parameters**:
    *   Concept Merge Threshold
    *   Claim Link Strength
    *   🪟 Context Window Parameters (Previous Sentences `p`, Following Sentences `f` for Claimify)
*   **⏰ Scheduler Job Controls**:
    *   🔄 Concept Embedding Refresh (Enabled, Manual Only, Cron Schedule)
    *   📁 Vault Sync (Enabled, Manual Only, Cron Schedule)
*   **🧠 Highlight & Summary**:
    *   🤖 Writing Agent (Model for Trending Concepts Agent - synchronized with the one in Model Settings)
    *   🏆 Top Concepts (Ranking Metric, Count, Percent, Target File with preview)
    *   📈 Trending Topics (Window Days, Count, Percent, Min Mentions, Target File with preview)
*   **💾 Save Configuration**:
    *   "Save Changes" button
    *   "Reload from File" button
    *   Status message display area.

### Validation Feedback

Validation errors encountered during the save operation are displayed to the user in the status message area.

## User Interface

The UI provides distinct input fields for each configuration parameter, grouped logically. Tooltips (`info` parameter in Gradio components) offer explanations for each setting. Filename previews for highlight jobs update dynamically as the user types.

## Usage Examples

*   **Changing a Model**: User types "gpt-4-turbo" into the "Concept Summary" textbox and clicks "Save Changes".
*   **Adjusting a Threshold**: User slides the "Concept Merge Threshold" to `0.85` and saves.
*   **Disabling a Job**: User unchecks "Enabled" for "Vault Sync" and saves.
*   **Modifying Highlight Job**: User changes "Target File" for "Top Concepts" to "My Vault Top Concepts.md" and saves.

## Error Handling

*   **Loading**: If `settings/aclarai.config.yaml` is missing or corrupt, it attempts to use defaults. Errors are logged.
*   **Saving**: If an error occurs during saving (e.g., permission issues), it's logged, and a failure message is shown to the user.
*   **Validation**: Input values are validated before saving. If validation fails, an error message detailing the issues is shown, and the configuration is not saved.

## Testing

*   The panel can be launched standalone for manual UI testing (`if __name__ == "__main__": ...`).
*   Unit tests for configuration loading/saving and validation logic could be implemented (current structure shows test files like `test_config_panel.py` exist).

## Configuration

This panel *is* the interface for managing `settings/aclarai.config.yaml`. It also depends on `shared/aclarai_shared/aclarai.config.default.yaml` for base settings.

## Logging

The panel uses the standard Python logging module, configured for `aclarai-ui.config_panel`. Logs include contextual information like service, component, and action.

```python
logger.error(
    "Failed to save configuration",
    extra={
        "service": "aclarai-ui",
        "component": "config_panel",
        "action": "save_configuration",
        "error": str(e),
        "error_type": type(e).__name__,
    },
)
```

## Performance Considerations

*   Loading and saving YAML files are I/O operations; performance is generally acceptable for typical configuration file sizes.
*   Deep merging of configurations is done in memory.

## Future Enhancements

*   **Agent Toggles**: The design document `design_config_panel.md` mentions a section for "Agent Toggles" (e.g., `agents.claimify: true`). This is not currently present in the UI panel's code.
*   **Vault Paths and Type Detection**: The design also specifies a section for "Vault Paths and Document Type Detection". This is also not in the current UI code.
*   **More Granular Validation**: Enhance validation for specific model providers or more complex cron patterns if needed.
*   **Help Links**: Direct links to relevant documentation pages for complex settings.
*   **Backup/Restore**: Functionality to backup current settings or restore a previous version of `aclarai.config.yaml`.

## Dependencies

*   `gradio`: For building the web interface.
*   `PyYAML`: For reading and writing YAML configuration files.
*   Standard Python libraries (`pathlib`, `logging`, `copy`, `re`).

## Related Documentation

*   [Configuration Panel Design](../arch/design_config_panel.md) - Detailed design specification.
*   [Configuration Guide](../guides/configuration_guide.md) - General guide on Aclarai configuration.
*   `settings/aclarai.config.yaml` - The user's configuration file.
*   `shared/aclarai_shared/aclarai.config.default.yaml` - Default system configuration.
*   [UI System](./ui_system.md) - Overview of the UI system.
