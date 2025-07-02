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

## Supported Input Formats

The system automatically detects and converts various conversation formats:

-   **Simple speaker format**: `alice: Hello\nbob: Hi there!`
-   **ENTRY format**: `ENTRY [10:00] alice >> message`
-   **With metadata**: Session IDs, topics, participants extraction

## Understanding the Output

The system generates Tier 1 Markdown files with proper annotations.

### Initial Import

Immediately after import, the file will contain the conversation text, block identifiers, and file-level metadata.

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

### After Claim Evaluation

Once the claims within the file have been processed by the evaluation agents, additional metadata comments for scores will be added directly into the file. Notice that the `ver` number of the block has been incremented.

```markdown
<!-- aclarai:title=Weekly Team Sync -->
<!-- aclarai:created_at=2025-06-09T23:29:03.406829 -->
<!-- aclarai:participants=["alice", "bob", "charlie"] -->
<!-- aclarai:message_count=3 -->
<!-- aclarai:plugin_metadata={"source_format": "fallback_llm", "session_id": "team_weekly_20250609"} -->

<!-- aclarai:entailed_score=0.91 -->
<!-- aclarai:coverage_score=0.77 -->
<!-- aclarai:decontextualization_score=0.88 -->
alice: Let's start with project updates
<!-- aclarai:id=blk_fkj7pn ver=4 -->
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