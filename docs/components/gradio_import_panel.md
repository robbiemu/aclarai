# Gradio Import Panel Integration

This document describes the implementation of the Gradio Import Panel that integrates with the aclarai plugin orchestrator system for automatic file format detection and processing.

## Overview

The Gradio Import Panel provides a user-friendly web interface for uploading and processing conversation files. It automatically detects file formats using the plugin system's `can_accept()` methods and processes files through the `ImportOrchestrator`.

## Architecture

### Core Components

1. **ImportStatusTracker** - Manages import queue and integrates with the plugin orchestrator
2. **real_plugin_orchestrator()** - Processes files using the actual plugin system
3. **Gradio Interface** - Web UI with file picker, live queue, and summary display
4. **Status Mapping** - Converts plugin status to user-friendly UI indicators

### Integration Points

- **Plugin System** - Uses `ImportOrchestrator` and `PluginManager` from `aclarai_shared`
- **File Processing** - Automatically routes files to appropriate plugins via `can_accept()`
- **Fallback Handling** - Uses `DefaultPlugin` when no specific plugin accepts a file
- **Real-time Updates** - Live queue display with processing status and results

## Implementation Details

### ImportStatusTracker Class

Replaces the previous simulated `ImportStatus` class with real plugin integration:

```python
class ImportStatusTracker:
    def __init__(self):
        self.import_queue = []
        self.orchestrator = ImportOrchestrator()  # Real orchestrator
    
    def process_file_with_orchestrator(self, file_path: str) -> ImportResult:
        # Uses actual plugin system to process files
        return self.orchestrator.import_file(Path(file_path))
```

### Status Mapping

Maps plugin system status to UI-friendly indicators:

- `SUCCESS` → "✅ Imported" (regular plugins)
- `SUCCESS` + `DefaultPlugin` → "⚠️ Fallback" (fallback plugin used)
- `ERROR`/`SKIPPED` → "❌ Failed" (processing failures)
- `IGNORED` → "⏸️ Skipped" (no conversations found)

### Plugin Detection

The system automatically:
1. Reads uploaded file content
2. Iterates through available plugins calling `can_accept()`
3. Uses the first plugin that returns `True`
4. Falls back to `DefaultPlugin` if no specific plugin accepts the file
5. Displays the plugin name in the detector column

## User Interface

### File Picker Section
- Drag-and-drop file upload
- Supports `.json`, `.txt`, `.csv`, `.md`, `.zip` files
- Manual "Process File" button
- "Clear Queue" functionality

### Live Import Queue
Real-time table display showing:
- **Filename** - Original file name
- **Status** - Processing result with icons (✅⚠️❌⏸️)
- **Detector** - Plugin name that processed the file
- **Time** - Processing timestamp

### Post-import Summary
Statistics display after processing:
- Total files processed
- Successfully imported count
- Fallback plugin usage count
- Failed imports count
- Skipped files count
- Links to view results and logs

## Usage Examples

### Text File with Conversations
```
Input: conversation.txt
Content: "User: Hello\nAssistant: Hi there!"
Result: ⚠️ Fallback (DefaultPlugin)
```

### JSON File (No Specific Plugin)
```
Input: data.json  
Content: {"conversation": "test"}
Result: ⚠️ Fallback (DefaultPlugin) or ⏸️ Skipped
```

### Empty or Invalid File
```
Input: empty.txt
Content: ""
Result: ⏸️ Skipped (no conversations found)
```

## Error Handling

The implementation includes robust error handling:

- **File Processing Errors** - Graceful degradation with error messages
- **Plugin Failures** - Fallback to error status with details
- **Logging Conflicts** - Fixed to avoid LogRecord conflicts
- **Network Issues** - Timeout handling for long-running operations
- **UI State Management** - Consistent state across Gradio interactions

## Testing

### Integration Tests
- `test_integration.py` - End-to-end processing with real files
- `test_import_panel.py` - Unit tests for all components
- `launch_ui_test.py` - Manual UI testing script

### Test Scenarios
1. **File Type Detection** - Various file formats and extensions
2. **Plugin Selection** - Automatic plugin discovery and selection
3. **Fallback Behavior** - DefaultPlugin usage when no specific plugin matches
4. **Queue Management** - Adding, updating, and clearing file queue
5. **Duplicate Handling** - Skipping duplicate files
6. **Edge Cases** - Empty files, corrupted content, missing files

## Configuration

The panel uses configuration from `aclarai.config.yaml`:

```yaml
ui:
  server_host: "127.0.0.1"
  server_port: 7860
  next_steps:
    vault: "./vault/tier1/"
    logs: "./.aclarai/import_logs/"
```

## Logging

Structured logging following the project's logging standards:

```python
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
```

## Performance Considerations

- **Plugin Discovery** - Plugins are discovered once at initialization
- **File Processing** - Asynchronous processing with real-time status updates
- **Memory Usage** - Queue is maintained in memory for session duration
- **Scalability** - Designed for individual file processing, not batch operations

## Future Enhancements

1. **Additional Plugins** - Support for more file formats via plugin system
2. **Batch Processing** - Multiple file upload and processing
3. **Progress Indicators** - Real-time progress bars for large files
4. **File Preview** - Preview imported content before processing
5. **Export Functions** - Download processed results in various formats

## Dependencies

- `gradio` - Web interface framework
- `aclarai_shared.plugin_manager` - Plugin orchestration system
- `aclarai_shared.config` - Configuration management
- Standard library modules for file handling and logging

## Related Documentation

- [Plugin System Guide](../guides/plugin_system_guide.md) - Plugin development
- [Plugin Manager System](./plugin_manager_system.md) - Core orchestration
- [Import Panel Design](../arch/design_import_panel.md) - UI design specification
- [Error Handling](../arch/on-error-handling-and-resilience.md) - Error handling patterns