# Tier 1 Markdown Import System

The Tier 1 import system provides a complete, orchestrated pipeline for importing conversation files into the aclarai vault as standardized Tier 1 Markdown documents. It is built on a robust, plugin-based architecture that ensures extensibility and maintainability.

## Core Components

The import system consists of three main components that work together to process and convert files:

1.  **`Tier1ImportSystem`**: The high-level entry point for all import operations. It handles file discovery, duplicate detection, and orchestrates the entire import process.
2.  **`ImportOrchestrator`**: The central coordinator that manages the import lifecycle for a single file. It uses the `PluginManager` to find a suitable plugin and returns a structured `ImportResult`.
3.  **`PluginManager`**: Responsible for discovering, loading, and ordering all available import plugins. It uses Python's `entry_points` mechanism for automatic plugin registration.

## Architecture and Data Flow

The import process follows a clear, sequential flow, ensuring that each file is handled consistently and efficiently.

```mermaid
graph TD
    A[Tier1ImportSystem] --> B{ImportOrchestrator};
    B --> C{PluginManager};
    C --> D[Discover & Order Plugins];
    D --> E[Selects Best Plugin];
    E --> F[Plugin.convert()];
    F --> G[MarkdownOutput];
    G --> B;
    B --> H[ImportResult];
    H --> A;
```

1.  **Initiation**: The `Tier1ImportSystem` receives a file path and initiates the import process.
2.  **Orchestration**: It passes the file to the `ImportOrchestrator`, which takes over the conversion logic.
3.  **Plugin Discovery**: The `ImportOrchestrator` queries the `PluginManager` to find a suitable plugin for the file format. The `PluginManager` discovers all installed plugins via `entry_points` and orders them based on priority.
4.  **Conversion**: The highest-priority plugin that can handle the file format is selected and its `convert()` method is called.
5.  **Structured Output**: The plugin returns one or more `MarkdownOutput` objects containing the converted text and metadata.
6.  **Result**: The `ImportOrchestrator` wraps the result in an `ImportResult` object, which includes a status (`ImportStatus.SUCCESS`, `ImportStatus.ERROR`, etc.) and other relevant details.
7.  **File Writing**: The `Tier1ImportSystem` receives the `ImportResult` and, if successful, performs an atomic write to save the new Tier 1 Markdown file to the vault.

## Features

-   **Automatic Plugin Discovery**: Plugins are discovered automatically via `entry_points`, eliminating the need for manual registration. See the [Plugin System Guide (`docs/guides/plugin_system_guide.md`)](<../guides/plugin_system_guide.md>) for more details.
-   **Structured Import Results**: The `ImportResult` dataclass provides detailed feedback on each import operation, including status, messages, and output files.
-   **Hash-Based Duplicate Detection**: SHA-256 hashing prevents re-importing the same content.
-   **Atomic File Writing**: A safe `.tmp` → `fsync` → `rename` pattern prevents file corruption.
-   **Configuration-Driven**: Uses vault paths from the central `aclaraiConfig` system.

## Usage

For complete usage examples and step-by-step tutorials, see:
-   **Tutorial**: `docs/tutorials/tier1_import_tutorial.md`
-   **CLI Reference**: `shared/aclarai_shared/scripts/import_cli.py --help`

## Error Handling and Statuses

The system uses the `ImportStatus` enum to communicate the outcome of an import operation:

-   `SUCCESS`: The file was successfully converted and saved.
-   `SKIPPED`: The file was skipped (e.g., it was empty or a directory).
-   `IGNORED`: The file was ignored because it was a duplicate.
-   `ERROR`: An error occurred during processing. The `message` field in the `ImportResult` will contain details.

This structured approach to error handling allows consumers of the import system (like the UI) to provide clear and actionable feedback to the user.
