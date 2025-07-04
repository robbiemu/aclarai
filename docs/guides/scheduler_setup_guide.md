# Scheduler Job Setup Guide

This guide explains how to set up and configure periodic jobs using aclarai's scheduler service.

## Overview

The aclarai scheduler uses APScheduler (Advanced Python Scheduler) to run periodic background tasks. Jobs are configured through the central configuration file and can be enabled/disabled and customized as needed.

## Job Configuration

### Configuration File Structure

Jobs are defined in `settings/aclarai.config.yaml` under the `scheduler.jobs` section. Each job has its own configuration block.

```yaml
scheduler:
  jobs:
    job_name:
      enabled: true
      manual_only: false
      cron: "*/30 * * * *"
      description: "Job description"
      # Additional job-specific parameters
```

### Configuration Options

| Key           | Type    | Description                                                                                                                              |
| :------------ | :------ | :--------------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`     | Boolean | If `true`, the job is active. If `false`, it will never be scheduled or run.                                                              |
| `manual_only` | Boolean | If `true`, the job is active but will **not** be scheduled automatically. It can only be triggered manually (e.g., via a future UI or API). |
| `cron`        | String  | A standard cron expression defining the automatic schedule. This is only used if `enabled` is `true` and `manual_only` is `false`.          |
| `description` | String  | A human-readable description of what the job does.                                                                                       |

### Cron Schedule Format

The scheduler uses standard cron format for scheduling:

-   `* * * * *` - Minute, Hour, Day of Month, Month, Day of Week
-   Examples:
    -   `"*/30 * * * *"` - Every 30 minutes
    -   `"0 */6 * * *"` - Every 6 hours
    -   `"0 3 * * *"` - Daily at 3:00 AM
    -   `"0 0 * * 0"` - Weekly on Sunday at midnight

## Global Automation Control

### Pausing All Jobs

You can temporarily pause **all** scheduled jobs by creating a `.aclarai_pause` file in the root of your vault directory.

```bash
touch /path/to/your/vault/.aclarai_pause
```

When this file exists, the scheduler will skip all automatic job executions. To resume, simply delete the file:

```bash
rm /path/to/your/vault/.aclarai_pause
```

This is useful for maintenance, debugging, or when you need to prevent background tasks from running.

## Built-in Jobs

The following jobs are included with aclarai by default.

### Vault Sync Job

-   **ID:** `vault_sync`
-   **Description:** Periodically scans all Markdown files in the vault and synchronizes their block-level content with the Neo4j knowledge graph. This ensures that any manual edits made in Obsidian are reflected in the graph.
-   **Default `cron`:** `"*/30 * * * *"` (Every 30 minutes)

### Concept Embedding Refresh Job

-   **ID:** `concept_embedding_refresh`
-   **Description:** Scans all Tier 3 concept files (`[[Concept]]` pages) and updates their vector embeddings if the content has changed. This keeps the semantic search index for concepts up-to-date.
-   **Default `cron`:** `"0 3 * * *"` (Daily at 3:00 AM)

### Top Concepts Job

-   **ID:** `top_concepts`
-   **Description:** This job runs PageRank analysis on the knowledge graph's concepts to identify the most important ones and writes them to a `Top Concepts.md` file.
-   **Default `cron`:** `"0 4 * * *"` (Daily at 4 AM)

### Trending Topics Job

-   **ID:** `trending_topics`
-   **Description:** This job analyzes the creation timestamps of claim-concept relationships to identify concepts with the highest growth in mentions over a recent period. It writes the results to a `Trending Topics - <date>.md` file.
-   **Default `cron`:** `"0 5 * * *"` (Daily at 5:00 AM)

### Concept Highlight Refresh Job

-   **ID:** `concept_highlight_refresh`
-   **Description:** Combined job that executes both the Top Concepts and Trending Topics jobs in a single scheduled execution. This ensures both highlight files are generated together and reduces scheduler overhead.
-   **Default `cron`:** `"0 6 * * *"` (Daily at 6:00 AM)

### Concept Summary Refresh Job

-   **ID:** `concept_summary_refresh`
-   **Description:** Generates detailed Markdown pages for all canonical concepts in the knowledge graph, using RAG workflows to include relevant claims, summaries, and related concepts. Creates or updates `[[Concept]]` pages with structured content.
-   **Default `cron`:** `"0 7 * * *"` (Daily at 7:00 AM)

### Concept Clustering Job

-   **ID:** `concept_clustering`
-   **Description:** Groups related concepts into thematic clusters using their embeddings. Uses DBSCAN or hierarchical clustering algorithms to form semantically coherent groups based on configurable similarity thresholds and size constraints. Caches cluster assignments for use by the Subject Summary Agent.
-   **Default `cron`:** `"0 2 * * *"` (Daily at 2:00 AM)

## Adding a Custom Job

Follow these steps to add a new scheduled job to the system.

### Step 1: Implement the Job Class

Create a new Python file in `services/scheduler/aclarai_scheduler/` with a class that contains a `run_job` method.

```python
# services/scheduler/aclarai_scheduler/my_custom_job.py

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MyCustomJob:
    """Custom job implementation."""
    
    def __init__(self, config=None):
        """Initialize job with configuration."""
        self.config = config or load_config(validate=True)
    
    def run_job(self) -> Dict[str, Any]:
        """
        Execute the job logic.
        
        Returns:
            Dictionary with job results and statistics
        """
        logger.info("Starting custom job")
        
        try:
            # Your job logic here
            # ...
            
            logger.info("Custom job completed successfully")
            return {
                "success": True,
                "items_processed": 42,
                "duration": 5.5
            }
            
        except Exception as e:
            logger.error(f"Custom job failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
```

### Step 2: Register the Job in the Scheduler

Add the logic to register your new job in `services/scheduler/aclarai_scheduler/main.py`.

```python
# In services/scheduler/aclarai_scheduler/main.py

from .my_custom_job import MyCustomJob

class SchedulerService:
    def _register_jobs(self):
        # ... (existing job registrations) ...
        
        # Add your custom job
        job_config = self.config.scheduler.jobs.get("my_custom_job", {})
        if job_config.get("enabled", False) and not job_config.get("manual_only", False):
            self._register_my_custom_job(job_config)

    def _register_my_custom_job(self, job_config):
        """Register the custom job."""
        cron_schedule = job_config.get("cron", "0 4 * * *") # Default: 4 AM daily
        job = MyCustomJob(self.config)
        
        self.scheduler.add_job(
            func=self._run_job_with_logging,
            args=("my_custom_job", job.run_job),
            trigger="cron",
            **self._parse_cron(cron_schedule),
            id="my_custom_job",
            name="My Custom Job",
            replace_existing=True
        )
```

### Step 3: Add Configuration

Finally, add the default configuration for your new job to `settings/aclarai.config.yaml`.

```yaml
scheduler:
  jobs:
    # ... existing jobs ...
    my_custom_job:
      enabled: false  # Start disabled for safety
      manual_only: false
      cron: "0 4 * * *"
      description: "My custom periodic job."
```

## Troubleshooting

-   **Job Not Running:**
    -   Check that `enabled` is `true` and `manual_only` is `false` in your config.
    -   Verify that the global `.aclarai_pause` file does not exist.
    -   Inspect the scheduler logs for errors during registration or execution.
-   **Invalid Cron String:** Ensure your cron string has exactly 5 parts and uses valid characters (`*`, `-`, `,`, `/`, and numbers).
-   **Database/Service Errors:** Check the logs for the specific job to see if it failed to connect to a required service like Neo4j or Postgres.

## Related Documentation

-   [Scheduler System Component Overview](../components/scheduler_system.md)
-   [Graph Vault Synchronization](../arch/on-graph_vault_synchronization.md)
-   [Error Handling and Resilience](../arch/on-error-handling-and-resilience.md)
