# Concept Subject Linking System

This document describes the concept subject linking system implementation in aclarai, which automatically adds bidirectional links between concepts and their parent subjects.

## Overview

The concept subject linking system creates footer links in [[Concept]] Markdown files that point to their corresponding [[Subject:XYZ]] pages, establishing bidirectional navigation between concepts and subjects. The system optionally creates explicit Neo4j graph edges for these relationships.

## Architecture

### Components

1. **ConceptSubjectLinkingJob** (`services/scheduler/aclarai_scheduler/concept_subject_linking_job.py`)
   - Main job class that orchestrates the linking process
   - Integrates with the APScheduler system
   - Provides comprehensive logging and error handling

2. **Configuration** (`shared/aclarai_shared/config.py`)
   - `ConceptSubjectLinkingJobConfig` class for job-specific settings
   - Integrated into the main scheduler configuration
   - Environment variable overrides supported

3. **Scheduler Integration** (`services/scheduler/aclarai_scheduler/main.py`)
   - Registered as a scheduled job in the main scheduler
   - Runs daily at 8 AM (after subject summary generation at 6 AM)
   - Respects automation pause functionality

## Process Flow

The linking job follows this process:

1. **Get Cluster Assignments**: Retrieves concept-to-cluster mappings from the concept clustering job cache
2. **Find Concept Files**: Scans the `vault/concepts/` directory for all `.md` files containing concept metadata
3. **Subject Resolution**: For each concept cluster, queries Neo4j to find the corresponding subject name
4. **Footer Link Addition**: Adds "Part of Subjects" section to concept files with links to parent subjects
5. **Version Increment**: Updates the `ver=N` number in concept metadata
6. **Atomic Write**: Uses the standard atomic write pattern (`.tmp` → `fsync` → `rename`)
7. **Optional Neo4j Edges**: Creates `(:Concept)-[:PART_OF]->(:Subject)` relationships if enabled

## Configuration

The job is configured through the `concept_subject_linking` section in the scheduler configuration:

```yaml
scheduler:
  jobs:
    concept_subject_linking:
      enabled: true
      manual_only: false
      cron: "0 8 * * *"  # 8 AM daily
      description: "Link concepts to their subjects with footer links"
      create_neo4j_edges: false  # Optional Neo4j edge creation
      batch_size: 50
      footer_section_title: "Part of Subjects"
```

### Configuration Parameters

- **`create_neo4j_edges`**: Whether to create explicit `(:Concept)-[:PART_OF]->(:Subject)` edges in Neo4j
- **`batch_size`**: Number of concepts to process in one batch (for future pagination support)
- **`footer_section_title`**: Title for the footer section in concept files

## Output Format

The system adds a footer section to concept files following this format:

```markdown
# Original Concept Content

...existing content...

## Part of Subjects

- [[Subject:AI and Machine Learning]]
- [[Subject:Data Science]]

<!-- aclarai:id=concept_example ver=2 -->
^concept_example
```

### Key Features

- **Idempotent**: Won't add duplicate footer sections if they already exist
- **Version Increment**: Automatically increments the `ver=` number in metadata
- **Atomic Writes**: Uses safe file operations to prevent corruption
- **Multiple Subjects**: Supports concepts that belong to multiple subjects

## Dependencies

The system depends on:

1. **Concept Clustering Job**: Provides the concept-to-cluster mappings via cached assignments
2. **Subject Summary Agent**: Creates the `[[Subject:XYZ]]` pages and `(:Subject)` nodes in Neo4j
3. **Neo4j**: For subject name resolution and optional edge creation
4. **Atomic Write System**: For safe file updates

## Error Handling

The system includes comprehensive error handling:

- **Missing Cluster Assignments**: Logs warning and skips job execution
- **Missing Concept Files**: Logs warning and skips job execution
- **Individual File Errors**: Logs errors but continues processing other files
- **Neo4j Errors**: Logs errors but doesn't fail the entire job
- **File Write Errors**: Uses atomic writes to prevent corruption

## Monitoring and Logging

The job provides detailed logging with structured context:

```python
logger.info(
    f"Successfully completed linking with {stats['files_updated']} files updated",
    extra={
        "service": "aclarai-scheduler",
        "filename.function_name": "concept_subject_linking_job.run_job",
        "concepts_processed": stats["concepts_processed"],
        "concepts_linked": stats["concepts_linked"],
        "files_updated": stats["files_updated"],
        "neo4j_edges_created": stats["neo4j_edges_created"],
    },
)
```

### Job Statistics

The job returns detailed statistics:

- `success`: Whether the job completed successfully
- `concepts_processed`: Total number of concept files found
- `concepts_linked`: Number of concepts successfully linked to subjects
- `concepts_skipped`: Number of concepts skipped (no cluster assignment or subject)
- `files_updated`: Number of files actually modified
- `neo4j_edges_created`: Number of Neo4j edges created (if enabled)
- `duration`: Total job execution time
- `error_details`: List of any errors encountered

## Integration Points

### With Concept Clustering

The job retrieves cluster assignments using:

```python
assignments = self.concept_clustering_job.get_cluster_assignments()
```

This relies on the clustering job's cache, which has a configurable TTL.

### With Subject Summary Agent

The job queries Neo4j to find subject names by cluster ID:

```cypher
MATCH (s:Subject {cluster_id: $cluster_id})
RETURN s.name as name
```

### With Neo4j (Optional)

When `create_neo4j_edges` is enabled, the job creates explicit relationships:

```cypher
MATCH (c:Concept {name: $concept_name})
MATCH (s:Subject {name: $subject_name})
MERGE (c)-[:PART_OF]->(s)
```

## Usage Examples

### Manual Execution

For testing purposes, the job can be executed manually:

```python
from services.scheduler.aclarai_scheduler.concept_subject_linking_job import ConceptSubjectLinkingJob
from shared.aclarai_shared import load_config

config = load_config()
job = ConceptSubjectLinkingJob(config)
result = job.run_job()
print(result)
```

### Environment Variable Overrides

The job supports environment variable overrides:

```bash
export CONCEPT_SUBJECT_LINKING_ENABLED=true
export CONCEPT_SUBJECT_LINKING_CRON="0 9 * * *"  # Run at 9 AM instead
```

## Testing

The system includes comprehensive tests covering:

- Job initialization and configuration
- File processing and concept name extraction
- Footer link addition and formatting
- Version increment logic
- Neo4j edge creation
- Error handling scenarios
- Integration scenarios

Run tests with:

```bash
uv run python -m pytest services/scheduler/tests/test_concept_subject_linking_job.py -v
```

## Future Enhancements

Potential improvements include:

1. **Batch Processing**: Process concepts in configurable batches for large vaults
2. **Subject Conflict Resolution**: Handle concepts that could belong to multiple subjects
3. **Link Validation**: Verify that linked subjects actually exist
4. **Performance Optimization**: Cache subject lookups to reduce Neo4j queries
5. **Link Customization**: Allow custom link formats or additional metadata

## Related Documentation

- [Concept Clustering System](concept_clustering_system.md)
- [Subject Summary Agent](../arch/on-writing_vault_documents.md)
- [Atomic File Operations](../arch/on-filehandle_conflicts.md)
- [Graph Vault Synchronization](../arch/on-graph_vault_synchronization.md)