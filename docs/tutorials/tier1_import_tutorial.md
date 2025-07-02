# Tier 1 Import System Tutorial

This tutorial shows how to use the aclarai Tier 1 import system to convert conversation files into standardized Tier 1 Markdown documents. The system is now orchestrated by the `ImportOrchestrator` and provides detailed feedback through the `ImportResult` object.

## Basic Usage

### Setting Up the Import System

First, you need to initialize the core components: the `PluginManager` and the `ImportOrchestrator`.

```python
from pathlib import Path
from aclarai_shared.config import aclaraiConfig
from aclarai_shared.import_system import Tier1ImportSystem, ImportStatus
from aclarai_shared.plugin_manager import PluginManager

# 1. Load the application configuration
config = aclaraiConfig()

# 2. Initialize the PluginManager to discover all installed plugins
plugin_manager = PluginManager()

# 3. Initialize the Tier1ImportSystem with the config and plugin manager
system = Tier1ImportSystem(config=config, plugin_manager=plugin_manager)
```

### Importing a Single File

When you import a file, the system returns an `ImportResult` object that tells you exactly what happened.

```python
# Create a dummy file for the example
dummy_file = Path("my_conversation.txt")
dummy_file.write_text("alice: Hello!\nbob: Hi there.")

# Import the conversation file
result = system.import_file(dummy_file)

# Check the result status
print(f"Import status: {result.status.name}")

if result.status == ImportStatus.SUCCESS:
    print(f"Successfully created Tier 1 file: {result.output_file}")
elif result.status in (ImportStatus.ERROR, ImportStatus.IGNORED, ImportStatus.SKIPPED):
    print(f"Import did not complete: {result.message}")

# Clean up the dummy file
dummy_file.unlink()
```

### Importing a Directory

The `import_directory` method returns a list of `ImportResult` objects, one for each file processed.

```python
# Create a dummy directory and files
import_dir = Path("my_conversations")
import_dir.mkdir()
(import_dir / "chat1.txt").write_text("dave: First chat.")
(import_dir / "chat2.txt").write_text("sara: Second chat.")

# Import all files from the directory
results = system.import_directory(import_dir, recursive=True)

# Process the results
for result in results:
    print(f"- File: {result.input_file.name}, Status: {result.status.name}, Message: {result.message}")

# Clean up the dummy directory
import shutil
shutil.rmtree(import_dir)
```

## Handling Different Statuses

The `ImportResult` object is key to understanding the outcome of an import. Here’s how to handle the different statuses:

-   `ImportStatus.SUCCESS`: The file was converted and saved successfully. The `output_file` attribute points to the new Tier 1 Markdown file.
-   `ImportStatus.IGNORED`: The file was a duplicate of already imported content. The `message` attribute will explain why it was ignored.
-   `ImportStatus.SKIPPED`: The file was skipped because it was empty or a directory. The `message` will provide details.
-   `ImportStatus.ERROR`: An error occurred. The `message` will contain the error details for debugging.
-   `ImportStatus.NO_PLUGIN_FOUND`: No suitable plugin was found to handle the file format.

Here is a more robust example of handling results:

```python
results = system.import_directory("path/to/your/conversations")

successful_imports = []
failed_imports = []

for res in results:
    if res.status == ImportStatus.SUCCESS:
        successful_imports.append(res.output_file)
    else:
        failed_imports.append((res.input_file, res.status, res.message))

print(f"Successfully imported {len(successful_imports)} files.")
if failed_imports:
    print(f"\nFailed or skipped {len(failed_imports)} files:")
    for file, status, msg in failed_imports:
        print(f"  - {file.name} ({status.name}): {msg}")
```

## Duplicate Detection

The system automatically handles duplicate detection. If you attempt to import a file with the same content as a previously imported one, the `import_file` method will return an `ImportResult` with a status of `ImportStatus.IGNORED`.

```python
# First import succeeds
result1 = system.import_file(Path("chat.txt"))
print(f"First import: {result1.status.name}")

# Second import of the same file is ignored
result2 = system.import_file(Path("chat.txt"))
print(f"Second import: {result2.status.name}") # -> IGNORED
print(f"Message: {result2.message}")
```

This prevents creating redundant files in your vault and ensures that each unique conversation is stored only once.
