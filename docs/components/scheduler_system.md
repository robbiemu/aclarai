# Scheduler System

The scheduler system provides periodic job execution infrastructure for the aclarai ecosystem using APScheduler (Advanced Python Scheduler).

## Overview

The scheduler service runs background tasks on configurable cron schedules. Its primary responsibilities are to ensure data consistency, perform regular maintenance, and trigger agentic processes that operate on the entire knowledge base.

Key responsibilities:

-   **Vault Synchronization**: Maintains consistency between Markdown files and the Neo4j knowledge graph.
-   **Concept Embedding Refresh**: Updates vector embeddings for concept files that have been modified.
-   **Reprocessing Tasks**: Handles content marked for reprocessing by other parts of the system.
-   **Concept Hygiene**: (Future) Performs deduplication and refinement operations.
-   **Concept Highlighting**: Identifies and highlights the most central or trending concepts in the knowledge graph, generating summary pages like 'Top Concepts' and 'Trending Topics'.
-   **Concept Clustering**: Groups related concepts into thematic clusters based on their embeddings, providing input for other agents like the Subject Summary Agent.
-   **Concept Summary Generation**: Creates detailed Markdown pages for canonical concepts using RAG workflows to include relevant claims, summaries, and related concepts.

## Architecture

The scheduler system is a standalone service that runs within the Docker Compose stack. It follows aclarai architectural patterns:

-   **Structured Logging**: Adheres to `docs/arch/idea-logging.md` with service context and job IDs.
-   **Error Handling**: Implements `docs/arch/on-error-handling-and-resilience.md` patterns with retry logic for transient database or network issues.
-   **Neo4j Integration**: Uses `docs/arch/on-neo4j_interaction.md` patterns for direct Cypher execution.
-   **Configuration Management**: All parameters are sourced from `settings/aclarai.config.yaml`.

## Core Components

### SchedulerService

The main service class (`aclarai_scheduler.main`) that:

-   Initializes APScheduler with a thread pool executor.
-   Registers jobs based on the central configuration file.
-   Handles graceful shutdown and system signals (SIGINT/SIGTERM).
-   Provides centralized logging and error handling for all jobs.

### VaultSyncJob

The core synchronization job (`aclarai_scheduler.vault_sync`) that implements the periodic vault-to-graph sync loop as defined in `docs/arch/on-graph_vault_synchronization.md`.

-   **Block Extraction**: Parses Markdown files for `aclarai:id` blocks.
-   **Change Detection**: Uses SHA-256 hashes of semantic text content to detect changes.
-   **Graph Synchronization**: Creates/updates `:Block` nodes with version tracking.
-   **Statistics Tracking**: Provides detailed metrics on sync operations.

### ConceptEmbeddingRefreshJob

A dedicated job (`aclarai_scheduler.concept_refresh`) for maintaining the concept vector store, as defined in `docs/arch/on-refreshing_concept_embeddings.md`.

-   **File Scanning**: Iterates through all Tier 3 concept files.
-   **Hash Comparison**: Compares the hash of the file's semantic content with the hash stored in the corresponding `:Concept` node in Neo4j.
-   **Conditional Updates**: If hashes differ, it re-embeds the concept and updates both the vector store and the Neo4j node.

### ConceptHighlightRefreshJob

A combined job (`aclarai_scheduler.concept_highlight_refresh`) that executes both top concepts and trending topics analysis in a single scheduled execution.

-   **Top Concepts Generation**: Runs PageRank analysis to identify the most important concepts and generates `Top Concepts.md`.
-   **Trending Topics Generation**: Analyzes claim-concept relationships to identify concepts with high growth in mentions and generates `Trending Topics - <date>.md`.
-   **Atomic Writes**: Ensures both files are written atomically to prevent corruption.
-   **Vault Sync Support**: Generated files include proper `aclarai:id` and `ver=` markers for sync detection.

### ConceptSummaryRefreshJob

A job (`aclarai_scheduler.concept_summary_refresh`) that generates detailed Markdown pages for all canonical concepts in the knowledge graph.

-   **Concept Querying**: Retrieves all canonical concepts from the Neo4j knowledge graph.
-   **RAG Processing**: Uses the ConceptSummaryAgent to generate structured content including relevant claims, summaries, and related concepts.
-   **Conditional Processing**: Can skip concepts with insufficient claims based on configuration.
-   **Atomic Writes**: Ensures all generated `[[Concept]]` pages are written atomically with proper vault sync markers.

## Configuration

Jobs are configured via `settings/aclarai.config.yaml` under the `scheduler.jobs` section. This allows for granular control over each job's behavior.

```yaml
scheduler:
  jobs:
    vault_sync:
      enabled: true
      manual_only: false
      cron: "*/30 * * * *"
      description: "Sync vault files with knowledge graph"
    concept_embedding_refresh:
      enabled: true
      manual_only: false
      cron: "0 3 * * *" # Daily at 3 AM
      description: "Refresh concept embeddings from Tier 3 pages"
```

## Environment Controls

-   **Global Automation Pause**: The scheduler respects the `.aclarai_pause` file in the vault root. If this file exists, all automatic job executions are skipped.
-   **Individual Job Toggles**: Each job can be individually enabled, disabled, or set to `manual_only` via the configuration file.

## Synchronization Logic

The vault sync job implements the specification from `docs/arch/on-graph_vault_synchronization.md`:

### Block Types Supported

-   **Inline blocks**: Individual sentences/claims with `<!-- aclarai:id=blk_abc123 ver=1 -->`
-   **File-level blocks**: Agent-generated content with file-scope IDs

### Change Detection

-   Calculates SHA-256 hashes of visible content (excluding metadata comments).
-   Compares with stored hashes in Neo4j `:Block` nodes.
-   Increments version numbers (`ver=N`) and sets `needs_reprocessing: true` flags for changes.

## Logging Format

All operations use structured logging with required context:

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

## Error Handling

The system implements resilient patterns:

-   **Retry Logic**: Exponential backoff for transient Neo4j connection errors.
-   **Graceful Degradation**: Jobs continue processing other files if individual files fail.
-   **Atomic Operations**: Database updates use transactions to prevent partial state.
-   **Signal Handling**: Graceful shutdown on SIGINT/SIGTERM.

## Performance Considerations

-   **Thread Pool Execution**: Jobs run in isolated threads to prevent blocking.
-   **Batch Processing**: Neo4j operations use efficient batch patterns.
-   **Memory Management**: Large vaults processed incrementally.
-   **Connection Pooling**: Neo4j driver handles connection lifecycle.

## Monitoring and Observability

Each job execution provides:

-   **Start/completion timestamps**
-   **Processing statistics** (files scanned, blocks updated, errors)
-   **Performance metrics** (execution duration, throughput)
-   **Error details** with stack traces for debugging

## Dependencies

The scheduler system builds on:

-   **APScheduler**: Job scheduling and execution framework.
-   **aclarai_shared**: Configuration, logging, and Neo4j components.
-   **Neo4j Python Driver**: Database operations.
-   **Standard Library**: Signal handling, hashing, file operations.

## Future Enhancements

Planned extensions include:

-   Concept hygiene and cleanup jobs.
-   Advanced scheduling with dependency-based job execution.
-   Performance optimizations for very large vaults.
