# aclarai UI

This service provides the user interface for the aclarai project using the Gradio framework. It is the primary entry point for users to import data, manage system configurations, and monitor processing status.

## Utility Scripts

The `aclarai_ui/util/` directory contains standalone scripts for development and testing of specific UI components. These scripts can be run directly from the project root using `uv run <script_name>`.

### Configuration Panel Launcher

-   **Script:** `aclarai_ui/util/config_launcher.py`
-   **Purpose:** Launches the Configuration Panel as a standalone Gradio application on `http://localhost:7861`. This is useful for testing the config UI without running the full aclarai stack.
-   **Usage:**
    ```bash
    uv run launch-config
    ```

### Import Panel Launcher

-   **Script:** `aclarai_ui/util/import_panel_launcher.py` (formerly `launch_ui_test.py`)
-   **Purpose:** Launches the main Import Panel as a standalone Gradio application on `http://localhost:7860`. This allows for focused testing of the file import workflow.
-   **Usage:**
    ```bash
    uv run launch-import-panel
    ```
