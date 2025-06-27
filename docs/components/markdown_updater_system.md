# Markdown Updater System

## 🎯 Purpose

The `MarkdownUpdaterService` is a dedicated component responsible for safely and atomically modifying Markdown files within the aclarai vault. Its primary goal is to provide a single, reliable interface for any backend service (such as evaluation agents) that needs to add or update `aclarai`-specific metadata comments within existing documents, without corrupting the file or losing data.

This service is the cornerstone of how aclarai enriches documents with new information (like evaluation scores) while respecting the principles of data integrity and versioning.

## ✨ Key Features

*   **Atomic Writes:** All file modifications use a `write-temp -> fsync -> rename` pattern. This is critical for preventing data corruption and ensures that file watchers (like Obsidian or the `vault-watcher` service) see either the old version of the file or the new, fully-written version—never a partial or incomplete file. This implements the strategy from `docs/arch/on-filehandle_conflicts.md`.
*   **Block-Level Targeting:** Updates are targeted to specific content blocks within a file using their unique `aclarai:id`. This allows for precise modifications without affecting the rest of the document.
*   **Automatic Versioning:** Whenever a block is modified, the service automatically increments the `ver=N` number in its `aclarai:id` comment. This signals to the vault synchronization system that the block has changed and may need reprocessing.
*   **Metadata Injection:** The service is designed to insert or update `<!-- aclarai:... -->` metadata comments in a standardized way, ensuring consistency across the vault.
*   **Resilient Error Handling:** The service includes robust error handling for common issues like file-not-found, block-not-found, and file permission errors.

## 🔄 How it Works: The Update Workflow

The service follows a clear, sequential process to update a file:

```mermaid
graph TD
    A[Service Call e.g., add_or_update_score] --> B{Read Markdown File};
    B --> C{Find Block by `aclarai:id`};
    C --> |Block Not Found| D[Log Warning & Return False];
    C --> |Block Found| E[Modify Content in Memory];
    E --> F[1. Increment `ver=N` in `aclarai:id` comment];
    F --> G[2. Add/Update Score Comment e.g., `<!-- aclarai:score=... -->`];
    G --> H{Write Atomically};
    H --> I[Write to `.tmp` file];
    I --> J[fsync() to disk];
    J --> K[Rename `.tmp` to original filename];
    K --> L[Return True];
    H --> |Write Fails| M[Log Error & Return False];
```

## 🐍 Usage Example

Any service needing to update a Markdown file can use the `MarkdownUpdaterService`. Here is how the `DecontextualizationAgent` would use it to persist a score:

```python
from aclarai_core.markdown import MarkdownUpdaterService

# Assume 'score' and other variables are defined
# score = 0.88
# claim_id = "blk_abc123"
# file_path_of_claim = "/path/to/vault/tier1/conversation.md"

# Instantiate the service
updater = MarkdownUpdaterService()

# Call the specific update method
success = updater.add_or_update_decontextualization_score(
    filepath_str=file_path_of_claim,
    block_id=claim_id,
    score=score
)

if success:
    print(f"Successfully updated block {claim_id} in {file_path_of_claim}.")
else:
    print(f"Failed to update block {claim_id}.")

```

## 🧩 Integration Points

*   **Upstream (Callers):** This service is primarily called by the **Evaluation Agents** (`Entailment`, `Coverage`, `Decontextualization`) after they have computed a score for a claim.
*   **Downstream (Consumers of Output):** The output of this service (the modified Markdown file) is consumed by:
    *   The **`vault-watcher`** service, which detects the file change.
    *   The **`block_syncing_loop`** and **`vault_sync` job**, which will detect the incremented `ver=` number and updated file hash, triggering re-processing of the block.

## ❌ Error Handling

The service is designed to fail safely and informatively:

*   **File Not Found:** If the target Markdown file does not exist, the operation fails and returns `False` with an error log.
*   **Block ID Not Found:** If the `aclarai:id` does not exist within the file, the operation fails and returns `False` with a warning log. This prevents accidental modification of the wrong file.
*   **Write Errors:** If an `OSError` or `PermissionError` occurs during the atomic write process, the operation fails, the temporary file is cleaned up, and the error is logged.

## 📚 Related Documentation

*   **File Safety:** [Obsidian File Handle Conflicts (`on-filehandle_conflicts.md`)](../arch/on-filehandle_conflicts.md)
*   **Versioning and Syncing:** [aclarai Graph–Vault Sync Design (`on-graph_vault_synchronization.md`)](../arch/on-graph_vault_synchronization.md)
*   **Consumers:** [Evaluation Agents (`on-evaluation_agents.md`)](../arch/on-evaluation_agents.md)
