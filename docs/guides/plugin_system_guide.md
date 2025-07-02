# Plugin System Guide

This guide provides a comprehensive overview of the aclarai plugin system, which is designed for converting various file formats into standardized Tier 1 Markdown documents. The system is built around a central `ImportOrchestrator` and a `PluginManager` that automatically discovers and manages plugins.

## Core Concepts

-   **Plugin**: A Python class that implements the `Plugin` interface. Each plugin is responsible for recognizing and converting a specific file format.
-   **`ImportOrchestrator`**: The main entry point for all conversion tasks. It takes a file, finds the right plugin using the `PluginManager`, and returns a structured `ImportResult`.
-   **`PluginManager`**: Automatically discovers, loads, and orders all available plugins using Python's `entry_points` mechanism. This means you never have to manually create a plugin registry.
-   **`ImportResult`**: A dataclass that encapsulates the result of an import operation, including a status (`ImportStatus`), a message, and the list of generated `MarkdownOutput` objects.

## Usage

The primary way to interact with the plugin system is through the `ImportOrchestrator`.

```python
from pathlib import Path
from aclarai_shared.import_system import ImportOrchestrator
from aclarai_shared.plugin_manager import PluginManager

# 1. Initialize the PluginManager and ImportOrchestrator
plugin_manager = PluginManager()
orchestrator = ImportOrchestrator(plugin_manager)

# 2. Define the input file
input_file = Path("path/to/your/conversation.txt")

# 3. Process the file
result = orchestrator.import_file(input_file)

# 4. Check the result
print(f"Import status: {result.status.name}")
if result.status == ImportStatus.SUCCESS:
    for output in result.outputs:
        print(f"  - Title: {output.title}")
        print(f"    Markdown length: {len(output.markdown_text)}")
elif result.status == ImportStatus.ERROR:
    print(f"  Error: {result.message}")
```

## Plugin Development

Creating a new plugin is straightforward. You need to create a class that inherits from `Plugin` and implement two methods: `can_accept()` and `convert()`.

### 1. Implement the `Plugin` Interface

```python
# src/my_plugin/plugin.py
from pathlib import Path
from typing import List
from aclarai_shared.plugin_interface import Plugin, MarkdownOutput

class MyCustomPlugin(Plugin):
    @property
    def priority(self) -> int:
        return 10  # Higher number means higher priority

    def can_accept(self, raw_input: str) -> bool:
        # Check if this plugin can handle the file format.
        # Be specific to avoid conflicts with other plugins.
        return raw_input.strip().startswith("MY-CUSTOM-FORMAT-V1")

    def convert(self, raw_input: str, file_path: Path) -> List[MarkdownOutput]:
        # Your conversion logic here.
        # Parse the raw_input and return a list of MarkdownOutput objects.
        return [
            MarkdownOutput(
                title="My Custom Conversation",
                markdown_text="speaker1: Hello!\n<!-- aclarai:id=blk_123 ver=1 -->\n^blk_123",
                metadata={}
            )
        ]
```

### 2. Register the Plugin via `entry_points`

To make your plugin discoverable by the `PluginManager`, you need to register it in your project's `pyproject.toml` file under the `aclarai.plugins` entry point group.

```toml
# pyproject.toml
[project.entry-points."aclarai.plugins"]
my_custom_plugin = "my_plugin.plugin:MyCustomPlugin"
```

-   `my_custom_plugin`: A unique name for your plugin's entry point.
-   `my_plugin.plugin:MyCustomPlugin`: The import path to your plugin class.

Once your package is installed (e.g., with `pip install .`), the `PluginManager` will automatically find and load it.

## The `ImportResult` and `ImportStatus`

The `ImportOrchestrator` always returns an `ImportResult` object, which provides clear feedback on the outcome of the operation. The `status` field is an `ImportStatus` enum with the following members:

-   `SUCCESS`: The file was successfully converted.
-   `SKIPPED`: The file was skipped (e.g., it was empty).
-   `ERROR`: An error occurred during conversion. The `message` field will contain details.
-   `NO_PLUGIN_FOUND`: No suitable plugin could be found to handle the file.

This structured result makes error handling robust and predictable.

## The Default Plugin

The system includes a `DefaultPlugin` that serves as a fallback. It has a low priority and its `can_accept()` method always returns `True`. This ensures that if no other plugin can handle a file, the `DefaultPlugin` will attempt to process it using general-purpose pattern matching and LLM-based extraction.

Because of automatic discovery, you do **not** need to do anything to enable it. It is always available as a fallback.
