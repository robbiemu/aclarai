# Plugin Manager and Import Orchestrator

This document describes the centralized plugin manager and import orchestrator implemented for aclarai's pluggable format conversion system.

## Overview

The plugin manager and import orchestrator provide a centralized approach to discovering, managing, and executing format conversion plugins. They implement the requirements from `docs/project/epic_1/sprint_8-Create_plugin_manager.md` while maintaining compatibility with the existing plugin system.

## Architecture

### Core Components

1. **PluginManager** - Centralized plugin discovery and management
2. **ImportOrchestrator** - High-level import coordination with status tracking
3. **ImportStatus** - Structured status enum for UI consumption
4. **ImportResult** - Detailed result object for each import operation

### Integration Points

- **Tier1ImportSystem** - Updated to use the new components while maintaining the same interface
- **Plugin Interface** - Full compatibility with existing `Plugin` abstract class
- **Default Plugin** - Proper integration as fallback plugin
- **UI Systems** - Structured status tracking for import panel consumption

## Plugin Discovery and Management

### PluginManager

The `PluginManager` class handles:

- **Plugin Discovery** - Automatic discovery of available plugins
- **Plugin Registry** - Centralized plugin registry management  
- **Plugin Ordering** - Ensures correct execution order (specific plugins first, fallback last)
- **Plugin Metadata** - Tracking plugin information for debugging and UI

```python
from aclarai_shared.plugin_manager import PluginManager

# Create plugin manager (auto-discovers plugins)
manager = PluginManager()

# Register additional plugins
manager.register_plugin(MyCustomPlugin())

# Get plugin information
plugin_count = manager.get_plugin_count()
plugins = manager.get_plugins()
metadata = manager.get_plugin_metadata()
```

### Plugin Ordering

Plugins are automatically ordered to ensure correct priority:

1. **Format-specific plugins** - Higher priority, execute first
2. **Default/fallback plugin** - Lowest priority, executes last

This ensures that specific format handlers take precedence over the general fallback plugin.

## Import Orchestration

### ImportOrchestrator

The `ImportOrchestrator` class provides high-level import coordination:

- **Plugin Selection** - Finds the first plugin that accepts the input
- **Status Tracking** - Detailed import status for UI consumption
- **Error Handling** - Graceful handling of plugin failures
- **Result Aggregation** - Structured results for batch operations

```python
from aclarai_shared.plugin_manager import ImportOrchestrator

# Create orchestrator
orchestrator = ImportOrchestrator()

# Import single file
result = orchestrator.import_file(Path("conversation.txt"))

# Import multiple files
results = orchestrator.import_files([Path("file1.txt"), Path("file2.json")])

# Get plugin information
plugin_info = orchestrator.get_plugin_info()
```

### Import Status Tracking

The system provides structured status tracking for UI consumption:

```python
from aclarai_shared.plugin_manager import ImportStatus

# Possible import statuses
ImportStatus.SUCCESS   # File successfully processed
ImportStatus.IGNORED   # File skipped (e.g., no conversations found)
ImportStatus.ERROR     # File processing failed
ImportStatus.SKIPPED   # No plugin could handle the file
```

### ImportResult Structure

Each import operation returns detailed results:

```python
@dataclass
class ImportResult:
    file_path: Path              # Path to the imported file
    status: ImportStatus         # Import status
    message: str                 # Human-readable message
    plugin_used: Optional[str]   # Name of plugin that processed the file
    output_files: Optional[List[Path]]  # Created output files
    error_details: Optional[str] # Detailed error information
    metadata: Optional[Dict[str, Any]]  # Additional metadata
```

## Integration with Existing Systems

### Tier1ImportSystem Integration

The existing `Tier1ImportSystem` has been updated to use the new components while maintaining the same interface:

```python
from aclarai_shared.import_system import Tier1ImportSystem

# Works exactly the same as before
system = Tier1ImportSystem()
output_files = system.import_file(Path("conversation.txt"))

# New: Get plugin information
plugin_info = system.get_plugin_info()
```

### Backward Compatibility

All existing functionality is preserved:

- Same method signatures
- Same return types  
- Same exception handling
- Same configuration system

## Plugin Development

### Adding New Plugins

To add a new format-specific plugin:

1. **Implement Plugin Interface**:
```python
from aclarai_shared.plugin_interface import Plugin, MarkdownOutput

class MyFormatPlugin(Plugin):
    def can_accept(self, raw_input: str) -> bool:
        return "MY_FORMAT_MARKER" in raw_input
    
    def convert(self, raw_input: str, path: Path) -> List[MarkdownOutput]:
        # Process the input and return MarkdownOutput objects
        pass
```

2. **Register with Plugin Manager**:
```python
# Manual registration
manager = PluginManager()
manager.register_plugin(MyFormatPlugin())

# Or integrate with discovery system (future enhancement)
```

### Plugin Discovery (Future Enhancement)

The plugin manager is designed to support automatic plugin discovery using entry points:

```python
# In setup.py or pyproject.toml
entry_points={
    'aclarai.plugins': [
        'my_format = my_package.plugins:MyFormatPlugin',
    ],
}
```

## Error Handling

The system implements robust error handling:

### Plugin-Level Errors

- **can_accept() failures** - Continue to next plugin
- **convert() failures** - Return error status with details
- **Plugin exceptions** - Graceful degradation

### File-Level Errors

- **File not found** - Clear error status
- **Permission errors** - Detailed error information
- **Encoding issues** - Automatic fallback to latin-1

### System-Level Errors

- **No accepting plugin** - SKIPPED status (shouldn't happen with fallback plugin)
- **Unexpected errors** - ERROR status with full details

## Logging

The system uses structured logging throughout:

```python
import logging
logger = logging.getLogger(__name__)

# Plugin selection
logger.info(f"Using plugin {plugin_name} for {file_path}")

# Status updates  
logger.info(f"Successfully processed {len(outputs)} conversation(s)")

# Error conditions
logger.warning(f"Plugin {plugin_name} failed to convert {file_path}: {error}")
```

Log levels:
- **INFO** - Normal operation, status updates
- **DEBUG** - Detailed plugin selection, file processing steps
- **WARNING** - Plugin failures, non-fatal errors
- **ERROR** - System failures, fatal errors

## Performance Considerations

### Plugin Ordering Optimization

- Specific plugins execute first (faster failure for unsupported formats)
- Fallback plugin executes last (most expensive, general processing)

### Caching and Reuse

- Plugin manager instances can be reused across multiple imports
- Plugin registry is built once and reused
- Metadata tracking has minimal overhead

### Memory Usage

- Plugin instances are lightweight
- Import results contain minimal metadata
- Large file content is not retained in memory

## Testing

The plugin manager and orchestrator include comprehensive test coverage:

### Unit Tests

- Plugin manager functionality
- Import orchestrator behavior
- Status tracking accuracy
- Error handling scenarios

### Integration Tests

- Plugin ordering verification
- Fallback behavior testing
- Tier1ImportSystem integration
- File processing scenarios

### Test Structure

```bash
shared/tests/test_plugin_manager.py    # Comprehensive plugin manager tests
shared/tests/test_tier1_import.py      # Integration with import system
shared/tests/test_plugins.py           # Existing plugin tests
```

## Future Enhancements

### Planned Improvements

1. **Automatic Plugin Discovery** - Entry point-based plugin discovery
2. **Plugin Validation** - Runtime plugin validation and health checks
3. **Plugin Metrics** - Performance tracking and usage statistics
4. **Plugin Configuration** - Per-plugin configuration support
5. **Plugin Versioning** - Plugin version compatibility checking

### Extensibility Points

The system is designed for easy extension:

- **New Import Status Types** - Add new status enum values as needed
- **Enhanced Metadata** - Extend ImportResult with additional fields
- **Custom Discovery** - Implement custom plugin discovery mechanisms
- **Plugin Middleware** - Add plugin interceptors for monitoring/modification

## Troubleshooting

### Common Issues

**No plugin accepts file:**
- Check file format and plugin compatibility
- Verify fallback plugin is registered
- Review plugin can_accept() logic

**Plugin conversion failures:**
- Check plugin-specific error logs
- Verify input file format
- Test plugin in isolation

**Status tracking issues:**
- Verify ImportResult handling in calling code
- Check error status propagation
- Review status mapping logic

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# All plugin operations will now log detailed information
orchestrator = ImportOrchestrator()
result = orchestrator.import_file(file_path)
```

### Plugin Information

Get runtime plugin information for debugging:

```python
orchestrator = ImportOrchestrator()
plugin_info = orchestrator.get_plugin_info()

print(f"Loaded {plugin_info['plugin_count']} plugins")
print(f"Plugin order: {plugin_info['plugin_order']}")
print(f"Plugin metadata: {plugin_info['plugins']}")
```