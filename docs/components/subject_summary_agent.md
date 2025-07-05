# Subject Summary Agent

## Configuration
The Subject Summary Agent's behavior can be customized through the Aclarai UI panel under the "Highlight & Summary" section:

### Clustering Parameters
- **Similarity Threshold** (0.0-1.0): Controls how similar concepts must be to be grouped into a subject. Higher values create more focused but fewer subjects.
- **Min Concepts** (1-100): Minimum concepts needed to form a subject cluster.
- **Max Concepts** (1-100): Maximum concepts allowed in a subject cluster.

### Generation Options
- **Allow Web Search**: When enabled, allows the agent to search the web for additional context when generating subject summaries.
- **Skip If Incoherent**: When enabled, skips generating summaries for concept clusters that lack clear thematic connections.

All parameters can be adjusted in real-time through the UI.

The Subject Summary Agent generates `[[Subject:XYZ]]` Markdown pages for concept clusters, following the architecture specified in `docs/arch/on-writing_vault_documents.md`.

## Overview

The Subject Summary Agent works with the concept clustering job to generate thematic subject pages from groups of related concepts. It provides a higher-level view of the knowledge graph by identifying and summarizing themes that span multiple concepts.

## Architecture

The agent follows the same patterns as the Concept Summary Agent but operates on clusters of concepts rather than individual concepts:

1. **Cluster Processing**: Gets cluster assignments from the concept clustering job
2. **Context Retrieval**: Retrieves shared claims and common summaries for concepts in each cluster  
3. **Content Generation**: Uses LLM or template fallback to generate subject content
4. **Web Search Integration**: Optionally enhances content with web search when enabled
5. **Atomic File Writing**: Uses safe atomic write patterns for vault compatibility

## Configuration

The agent is configured through the `subject_summaries` section of the aclarai configuration:

```yaml
subject_summaries:
  model: "gpt-3.5-turbo"        # LLM model for content generation
  similarity_threshold: 0.92    # Threshold for concept clustering
  min_concepts: 3               # Minimum concepts per cluster
  max_concepts: 15              # Maximum concepts per cluster
  allow_web_search: true        # Enable web search for additional context
  skip_if_incoherent: false     # Skip clusters with no shared elements
```

## Output Format

The agent generates Markdown files following this structure:

```markdown
## Subject: <name or synthesized theme>

<summary paragraph>

### Included Concepts
- [[Concept A]] — short internal blurb
- [[Concept B]] — short internal blurb

### Common Threads
- Summary of shared topics from claims
- Optional inline links to related subject pages

<!-- aclarai:id=subject_<slug> ver=1 -->
^subject_<slug>
```

## Integration

### Scheduler Integration

The agent runs as a scheduled job (`subject_summary_refresh`) that executes daily at 6 AM, after the concept clustering job runs at 2 AM:

```yaml
scheduler:
  jobs:
    subject_summary_refresh:
      enabled: true
      manual_only: false
      cron: "0 6 * * *"
      description: "Generate [[Subject:XYZ]] pages from concept clusters"
```

### Dependency Chain

1. **Concept Clustering Job** (2 AM) - Groups related concepts into clusters
2. **Subject Summary Agent** (6 AM) - Generates subject pages from clusters

## Usage

### Via Python API

```python
from aclarai_shared.subject_summary_agent import SubjectSummaryAgent
from aclarai_shared.config import load_config

config = load_config()
agent = SubjectSummaryAgent(config=config)
result = agent.run_agent()
```

### Via Scheduler

The agent runs automatically via the scheduler service. You can also trigger it manually by running the scheduler job.

## Key Features

### Intelligent Subject Naming

The agent generates meaningful subject names from concept clusters:
- Single concept: Uses the concept name
- 2-3 concepts: Combines with "&" (e.g., "Machine Learning & AI")
- 4+ concepts: Uses lead concept + "Related Topics"

### Quality Filtering

- Respects `min_concepts` and `max_concepts` constraints
- Optionally skips incoherent clusters (no shared claims/summaries)
- Handles outliers and edge cases gracefully

### Error Resilience

- Continues processing if individual clusters fail
- Falls back to template generation if LLM fails
- Provides detailed error reporting and statistics

### Atomic File Writing

All file operations use the safe atomic pattern:
1. Write to temporary file (`.tmp` extension)
2. Force sync to disk (`fsync`)
3. Atomic rename to final location

This prevents corruption and ensures vault watchers see complete files.

## Implementation Notes

The agent is implemented in `shared/aclarai_shared/subject_summary_agent/` and follows the same patterns as other aclarai agents. It includes comprehensive test coverage and proper error handling.

For detailed implementation information, see the module documentation and test files.