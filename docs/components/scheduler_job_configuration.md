# Scheduler Job Configuration

This document describes the configuration options available for scheduled jobs in the aclarai system.

## Overview

The aclarai scheduler supports configurable job execution with granular controls over when and how jobs are executed. Each job can be individually configured through the `settings/aclarai.config.yaml` file.

## Configuration Structure

Scheduler configuration is located under the `scheduler.jobs` section in the configuration file:

```yaml
scheduler:
  jobs:
    job_name:
      enabled: true|false
      manual_only: true|false  
      cron: "cron_expression"
      description: "Job description"
```

## Configuration Options

### `enabled`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Controls whether the job is enabled for execution. When `false`, the job will not be scheduled or executed in any way.

### `manual_only`
- **Type**: Boolean  
- **Default**: `false`
- **Description**: When `true`, the job will not be automatically scheduled even if `enabled` is `true`. The job can only be triggered manually. This is useful for jobs that should run on-demand rather than on a schedule.

### `cron`
- **Type**: String
- **Default**: Varies by job
- **Description**: Standard cron expression defining when the job should run automatically. Only applies when `enabled` is `true` and `manual_only` is `false`.

### `description`
- **Type**: String
- **Default**: Varies by job
- **Description**: Human-readable description of what the job does.

## Available Jobs

### `concept_embedding_refresh`
Refreshes concept embeddings from Tier 3 pages.

**Default Configuration:**
```yaml
concept_embedding_refresh:
  enabled: true
  manual_only: false
  cron: "0 3 * * *"  # Daily at 3 AM
  description: "Refresh concept embeddings from Tier 3 pages"
```

### `trending_topics`
Generates a `Trending Topics - <date>.md` file by analyzing the growth of concept mentions over a configurable time window.

**Default Configuration:**
```yaml
trending_topics:
  enabled: true
  manual_only: false
  cron: "0 5 * * *"  # 5 AM daily
  description: "Generate Trending Topics - {date}.md from concept mention deltas"
  window_days: 7
  count: null
  percent: 5
  min_mentions: 2
  target_file: "Trending Topics - {date}.md"
```

### `vault_sync`
Synchronizes vault files with the knowledge graph.

**Default Configuration:**
```yaml
vault_sync:
  enabled: true
  manual_only: false
  cron: "*/30 * * * *"  # Every 30 minutes
  description: "Sync vault files with knowledge graph"
```

### `top_concepts`
Generates a `Top Concepts.md` file by running PageRank analysis on the knowledge graph to identify the most central concepts.

**Default Configuration:**
```yaml
top_concepts:
  enabled: true
  manual_only: false
  cron: "0 4 * * *" # 4 AM daily
  description: "Generate Top Concepts.md from PageRank analysis"
  metric: "pagerank"
  count: 25
  percent: null
  target_file: "Top Concepts.md"
```
The `metric` parameter specifies the algorithm to use for identifying top concepts (e.g., `pagerank`). `count` determines the top N concepts to include, while `percent` specifies the top N% of concepts. `target_file` defines the output filename for the generated Markdown file.

## Job Execution Logic

The scheduler evaluates job configuration as follows:

1. **Disabled jobs** (`enabled: false`): Never executed, regardless of other settings
2. **Manual-only jobs** (`enabled: true`, `manual_only: true`): Never automatically scheduled, but can be triggered manually
3. **Automatic jobs** (`enabled: true`, `manual_only: false`): Scheduled according to their cron expression
4. **Global automation pause**: When `AUTOMATION_PAUSE=true` environment variable is set, no jobs are automatically scheduled regardless of individual job settings

## Configuration Examples

### Example 1: Disable a job completely
```yaml
scheduler:
  jobs:
    concept_embedding_refresh:
      enabled: false
      manual_only: false
      cron: "0 3 * * *"
      description: "Refresh concept embeddings from Tier 3 pages"
```

### Example 2: Enable a job for manual execution only
```yaml
scheduler:
  jobs:
    vault_sync:
      enabled: true
      manual_only: true
      cron: "*/30 * * * *"
      description: "Sync vault files with knowledge graph"
```

### Example 3: Custom cron schedule
```yaml
scheduler:
  jobs:
    concept_embedding_refresh:
      enabled: true
      manual_only: false
      cron: "0 2 * * 1"  # Weekly on Monday at 2 AM
      description: "Refresh concept embeddings from Tier 3 pages"
```

## Logging

The scheduler logs job configuration decisions with structured logging:

- When jobs are successfully scheduled
- When jobs are skipped due to `manual_only` setting
- When jobs are skipped due to being disabled
- When all jobs are skipped due to global automation pause

Example log entries:
```json
{
  "level": "INFO",
  "service": "aclarai-scheduler", 
  "filename.function_name": "scheduler.main._register_jobs",
  "job_id": "vault_sync",
  "manual_only": true,
  "message": "Vault sync job is enabled but set to manual_only, skipping automatic scheduling"
}
```

## Environment Variable Overrides

Some legacy environment variables can override job configuration:

- `CONCEPT_EMBEDDING_REFRESH_ENABLED`: Overrides the `enabled` setting for concept refresh job
- `CONCEPT_EMBEDDING_REFRESH_CRON`: Overrides the `cron` setting for concept refresh job
- `AUTOMATION_PAUSE`: When set to `true`, prevents all automatic job scheduling

## Related Documentation

- [Scheduler Setup Guide](../guides/scheduler_setup_guide.md) - How to add new scheduled jobs
- [Configuration Panel Design](../arch/design_config_panel.md) - UI design for job configuration
- [Review Panel Design](../arch/design_review_panel.md) - UI design for job monitoring