# aclarai Scheduler

This service is responsible for running periodic background tasks within the aclarai ecosystem. These tasks include, but are not limited to, concept hygiene (deduplication, refinement), refreshing embeddings, reprocessing content that has been marked as dirty, and performing regular synchronization between the vault and the knowledge graph.

## Features

### Vault Synchronization Job

The core functionality implemented in this service is the **vault sync job** that maintains consistency between Markdown files in the vault and the Neo4j knowledge graph.

**What it does:**
- Scans all Tier 1 Markdown files (and optionally other tiers) for blocks with `aclarai:id` markers
- Calculates content hashes for semantic text (visible content excluding metadata comments)
- Compares hashes with stored values in Neo4j to detect changes
- Updates Neo4j nodes with new content, increments version numbers, and marks changed blocks for reprocessing
- Provides detailed logging with statistics and error reporting

**Configuration:**
The vault sync job is configured via `settings/aclarai.config.yaml`:

```yaml
scheduler:
  jobs:
    vault_sync:
      enabled: true
      cron: "*/30 * * * *"  # Every 30 minutes
      description: "Sync vault files with knowledge graph"
```

### Scheduling

The service uses **APScheduler** (Advanced Python Scheduler) to run jobs on cron schedules. Jobs can be paused system-wide via:

1. The Review Panel's pause button in the UI
2. Creation of a `.aclarai_pause` file in the vault root

When the system is paused:
- New jobs will not be executed
- The pause state is checked before running any job
- Jobs are skipped (not queued) when paused
- Normal execution resumes once the pause is lifted

## Architecture

The scheduler follows the aclarai architectural patterns:

- **Structured logging** following `docs/arch/idea-logging.md`
- **Error handling and retries** following `docs/arch/on-error-handling-and-resilience.md`
- **Neo4j interaction** following `docs/arch/on-neo4j_interaction.md`
- **Block synchronization** following `docs/arch/on-graph_vault_synchronization.md`

## Components

### VaultSyncJob (`vault_sync.py`)

Implements the vault-to-graph synchronization logic:

- **Block extraction**: Parses Markdown files to extract `aclarai:id` blocks
- **Hash calculation**: Computes SHA-256 hashes of semantic text for change detection
- **Graph synchronization**: Creates/updates Block nodes in Neo4j with version tracking
- **Statistics tracking**: Provides detailed metrics on sync operations

### SchedulerService (`main.py`)

Main service class that:

- Sets up APScheduler with thread pool executors
- Registers jobs from configuration
- Handles graceful shutdown and signal management
- Provides centralized logging and error handling

## Environment Variables

- `AUTOMATION_PAUSE`: (Deprecated) Use `.aclarai_pause` file or UI for pausing jobs
- `CONCEPT_EMBEDDING_REFRESH_ENABLED`: Enable/disable concept refresh job (future)
- `CONCEPT_EMBEDDING_REFRESH_CRON`: Override cron schedule for concept refresh

## Usage

### Running the Service

```bash
cd services/scheduler
python -m aclarai_scheduler.main
```

### Docker

The service includes a Dockerfile for containerized deployment:

```bash
docker build -t aclarai-scheduler .
docker run aclarai-scheduler
```

## Logging

All operations are logged with structured format including:

- Service identification (`aclarai-scheduler`)
- Function names for traceability
- Job IDs for tracking individual executions
- Statistics and error details
- Context IDs (`aclarai_id`, `job_id`) for correlation

Example log entry:
```json
{
  "level": "INFO",
  "service": "aclarai-scheduler",
  "filename.function_name": "vault_sync.run_sync",
  "job_id": "vault_sync_1734264000",
  "blocks_processed": 42,
  "blocks_updated": 3,
  "duration": 2.1
}
```

## Testing

Basic tests are provided in `tests/test_vault_sync.py` covering:

- Markdown block extraction
- Content hash calculation
- File-level vs inline block handling
- Statistics merging
- Integration scenarios

## Future Enhancements

- Concept embedding refresh job
- Hygiene and cleanup jobs
- Advanced scheduling options
- Performance optimizations for large vaults
