# Tier 1 Import System Tutorial

This tutorial shows how to use the aclarai Tier 1 import system to convert conversation files into standardized Tier 1 Markdown documents.

## Basic Usage

### Setting Up the Import System

```python
from aclarai_shared import aclaraiConfig, Tier1ImportSystem, PathsConfig

# Configure with your vault path
config = aclaraiConfig(
    vault_path="/path/to/your/vault",
    paths=PathsConfig(
        tier1="conversations",  
        logs="import_logs"
    )
)

# Initialize the import system
system = Tier1ImportSystem(config)
```

### Importing a Single File

```python
# Import a conversation file
output_files = system.import_file("chat_export.txt")

if output_files:
    print(f"Created {len(output_files)} Tier 1 file(s):")
    for file in output_files:
        print(f"  {file}")
else:
    print("No conversations found in the file")
```

### Importing a Directory

```python
# Import all files from a directory
results = system.import_directory("conversations/", recursive=True)

# Show results
total_files = len(results)
successful = sum(1 for files in results.values() if files)
print(f"Processed {total_files} files, {successful} successful imports")
```

## Working with Different Input Formats

The system automatically detects and converts various conversation formats:

### Simple Speaker Format
```
alice: Hello, how are you?
bob: I'm doing well, thanks!
alice: Great to hear!
```

### ENTRY Format (Custom Logs)
```
ENTRY [10:00:00] alice >> Let's start the meeting.
ENTRY [10:00:30] bob >> I've prepared the agenda.
ENTRY [10:01:00] alice >> Perfect, let's begin.
```

### With Metadata
```
SESSION_ID: team_weekly_20250609
TOPIC: Weekly Team Sync
PARTICIPANTS: alice, bob, charlie

alice: Let's start with project updates
bob: The backend API is 90% complete
charlie: Frontend is ready for testing
```

## Understanding the Output

The system generates Tier 1 Markdown files with proper annotations:

```markdown
<!-- aclarai:title=Weekly Team Sync -->
<!-- aclarai:created_at=2025-06-09T23:29:03.406829 -->
<!-- aclarai:participants=["alice", "bob", "charlie"] -->
<!-- aclarai:message_count=3 -->
<!-- aclarai:plugin_metadata={"source_format": "fallback_llm", "session_id": "team_weekly_20250609"} -->

alice: Let's start with project updates
<!-- aclarai:id=blk_fkj7pn ver=1 -->
^blk_fkj7pn

bob: The backend API is 90% complete
<!-- aclarai:id=blk_xl8j4v ver=1 -->
^blk_xl8j4v

charlie: Frontend is ready for testing
<!-- aclarai:id=blk_mn3k2p ver=1 -->
^blk_mn3k2p
```

## Duplicate Detection

The system automatically prevents re-importing identical content:

```python
# First import succeeds
output_files = system.import_file("chat.txt")
print(f"Imported: {output_files}")

# Second import of same file is skipped
try:
    system.import_file("chat.txt")
except DuplicateDetectionError:
    print("File already imported (duplicate detected)")
```

## Error Handling

```python
from aclarai_shared.import_system import DuplicateDetectionError, ImportSystemError

try:
    output_files = system.import_file("conversation.txt")
except DuplicateDetectionError:
    print("File already imported")
except ImportSystemError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Complete Example

Here's a complete example that creates sample files and imports them:

```python
import tempfile
from pathlib import Path
from aclarai_shared import aclaraiConfig, Tier1ImportSystem, PathsConfig

# Create a temporary demo environment
with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = Path(temp_dir)
    
    # Create sample conversation file
    sample_file = temp_path / "team_chat.txt"
    sample_file.write_text("""alice: Hey, how's the project going?
bob: Making good progress! Almost done with the API.
alice: Awesome! Any blockers?
bob: None at the moment, should be ready by Friday.
alice: Perfect, let's review it together.""")
    
    # Setup vault and import system
    vault_dir = temp_path / "vault"
    config = aclaraiConfig(
        vault_path=str(vault_dir),
        paths=PathsConfig(tier1="conversations")
    )
    system = Tier1ImportSystem(config)
    
    # Import the file
    output_files = system.import_file(sample_file)
    print(f"Created: {output_files}")
    
    # Check the result
    if output_files:
        content = Path(output_files[0]).read_text()
        print("Generated content:")
        print(content[:200] + "...")
```

This tutorial covers the essential functionality of the Tier 1 import system. The system handles the complexity of format detection, duplicate prevention, and proper Tier 1 document generation automatically.