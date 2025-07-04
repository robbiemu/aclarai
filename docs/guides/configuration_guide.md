# Configuration System Usage Guide

This guide describes how to use aclarai's configuration system, which provides both a user-friendly UI panel and persistent YAML configuration files.

## Overview

The configuration system allows you to customize aclarai's behavior by setting:
- **Model selections** for different processing stages (Claimify, concept linking, etc.)
- **Embedding models** for different content types
- **Processing thresholds** for similarity and quality metrics
- **Context window parameters** for claim extraction

## Configuration Files

### Default Configuration
- **File**: `shared/aclarai_shared/aclarai.config.default.yaml`
- **Purpose**: Contains all default values and serves as a reference
- **Usage**: Never edit this file directly - it's part of the codebase

### User Configuration  
- **File**: `settings/aclarai.config.yaml`
- **Purpose**: Your customizations that override the defaults
- **Usage**: Edit through the UI panel or manually (with caution)

### Configuration Merging
The system automatically merges your user settings over the defaults, so you only need to specify the values you want to change.

## Using the Configuration Panel

### Launching the Panel

```bash
cd services/aclarai-ui
uv run python aclarai_ui/config_launcher.py
```

This opens a web interface at http://localhost:7861

### Configuration Categories

#### 🤖 Model & Embedding Settings

**Claimify Models** - Control which LLMs process different stages of claim extraction:
- **Default Model**: Used for all Claimify stages unless overridden
- **Selection Model**: Identifies claims within text 
- **Disambiguation Model**: Resolves ambiguous references
- **Decomposition Model**: Breaks down complex claims

**Agent Models** - Control LLMs for specific aclarai agents:
- **Concept Linker**: Links claims to concepts
- **Concept Summary**: Generates `[[Concept]]` pages
- **Subject Summary**: Generates `[[Subject:XYZ]]` pages  
- **Trending Concepts Agent**: Writes trending topic summaries
- **Fallback Plugin**: Used when format detection fails

**Embedding Models** - Control embeddings for different content types:
- **Utterance Embeddings**: For Tier 1 conversation blocks
- **Concept Embeddings**: For Tier 3 concept files
- **Summary Embeddings**: For Tier 2 summaries
- **Fallback Embeddings**: Used when other configs fail

#### 📏 Thresholds & Parameters

**Similarity Thresholds**:
- **Concept Merge Threshold** (0.0-1.0): Cosine similarity required to merge concept candidates
- **Claim Link Strength** (0.0-1.0): Minimum strength to create claim→concept edges

**Context Window Parameters**:
- **Previous Sentences (p)** (0-10): How many sentences before target sentence to include
- **Following Sentences (f)** (0-10): How many sentences after target sentence to include

#### 🧠 Highlight & Summary

The Highlight & Summary section configures concept highlight jobs that generate vault-wide overview documents.

**Writing Agent**:
- **Model for Trending Concepts Agent**: LLM model used to generate concept highlight content
- This setting is synchronized with the "Trending Concepts Agent" field in the Model & Embedding Settings section

**Top Concepts Configuration**:
- **Ranking Metric**: Choose between "pagerank" (PageRank algorithm) or "degree" (simple degree centrality)
- **Count**: Fixed number of top concepts to include (mutually exclusive with Percent)
- **Percent**: Percentage of total concepts to include as top concepts (mutually exclusive with Count)
- **Target File**: Output filename for the top concepts document (e.g., "Top Concepts.md")

**Trending Topics Configuration**:
- **Window Days**: How many days to look back for trend analysis (e.g., 7 for weekly trends)
- **Count**: Fixed number of trending topics to include (mutually exclusive with Percent)
- **Percent**: Percentage of concepts to include as trending (mutually exclusive with Count)
- **Min Mentions**: Minimum number of mentions required for a concept to be considered trending
- **Target File**: Output filename pattern, supports `{date}` placeholder (e.g., "Trending Topics - {date}.md")

**Validation & Features**:
- Real-time filename previews show actual output filenames with date substitution
- Comprehensive validation prevents invalid configurations (mutual exclusivity, valid ranges, non-empty filenames)
- Clear error messages guide users to correct configuration issues

#### 🕒 Automation & Scheduler Control

**Concept Clustering Configuration**:
- **Similarity Threshold** (0.0-1.0): Minimum cosine similarity for concepts to be clustered together.
- **Min Concepts** (Integer): Minimum number of concepts required to form a cluster.
- **Max Concepts** (Integer): Maximum number of concepts allowed in a cluster.
- **Algorithm**: Clustering algorithm to use ("dbscan" or "hierarchical").
- **Cache TTL** (Seconds): Time-to-live for cached cluster assignments.
- **Use Persistent Cache** (true/false): Whether to use persistent storage for the cache.

**Concept Summary Configuration**:
- **Model**: LLM model for generating concept definitions (default: uses main LLM)
- **Max Examples** (1-20): Maximum number of examples to include per concept (default: 5)
- **Skip If No Claims** (true/false): Skip generating pages for concepts without supporting claims (default: true)
- **Include See Also** (true/false): Include related concepts section in generated pages (default: true)

### Making Changes

1. **Load Current Settings**: The panel automatically loads your current configuration
2. **Edit Values**: Modify any settings using the input fields
3. **Validation**: Invalid values are rejected with helpful error messages
4. **Save**: Click "Save Changes" to persist to `settings/aclarai.config.yaml`
5. **Reload**: Click "Reload from File" to discard unsaved changes

### Model Name Formats

The system accepts various model name formats:

**OpenAI Models**:
- `gpt-4`, `gpt-3.5-turbo`, `text-embedding-3-small`

**Anthropic Models**:
- `claude-3-opus`, `claude-3-sonnet`

**Open Source Models**:
- `mistral-7b`, `llama2-13b`

**Provider Prefixes**:
- `openrouter:gemma-2b` (OpenRouter)
- `ollama:llama2` (Ollama)
- `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)

## Manual Configuration

You can also edit `settings/aclarai.config.yaml` directly, but be careful with the format:

```yaml
model:
  claimify:
    default: "gpt-4"
    selection: null  # Use default
    disambiguation: "claude-3-opus"
    decomposition: null  # Use default
  concept_linker: "mistral-7b"
  trending_concepts_agent: "gpt-4"
  # ... other models

embedding:
  utterance: "sentence-transformers/all-MiniLM-L6-v2"
  concept: "text-embedding-3-small"
  # ... other embeddings

threshold:
  concept_merge: 0.90
  claim_link_strength: 0.60

window:
  claimify:
    p: 3  # Previous sentences
    f: 1  # Following sentences

concept_highlights:
  top_concepts:
    metric: "pagerank"        # pagerank | degree
    count: 25                 # number of top concepts (exclusive with percent)
    percent: null             # use top N% instead of fixed count
    target_file: "Top Concepts.md"
  
  trending_topics:
    window_days: 7            # How far back to look for change
    count: null
    percent: 5
    min_mentions: 2
    target_file: "Trending Topics - {date}.md"

scheduler:
  jobs:
    concept_clustering:
      enabled: true
      manual_only: false
      cron: "0 2 * * *"  # 2 AM daily
      description: "Group related concepts into thematic clusters"
      similarity_threshold: 0.92
      min_concepts: 3
      max_concepts: 15
      algorithm: "dbscan"
      cache_ttl: 3600
      use_persistent_cache: true
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

### Concept Highlight & Summary Jobs

The `top_concepts` job generates a `Top Concepts.md` file in your vault, summarizing the most central concepts identified through graph analysis.

-   **`enabled`**: (Boolean) Set to `true` to enable the job, `false` to disable it.
-   **`manual_only`**: (Boolean) If `true`, the job will only run when manually triggered, ignoring the `cron` schedule.
-   **`cron`**: (String) A standard cron expression defining the schedule for automatic execution (e.g., `"0 4 * * *"` for daily at 4 AM).
-   **`description`**: (String) A human-readable description of the job.
-   **`metric`**: (String) The graph algorithm to use for identifying top concepts. Currently, `"pagerank"` is supported.
-   **`count`**: (Integer) The number of top concepts to include in the `Top Concepts.md` file. If `percent` is also set, `count` takes precedence.
-   **`percent`**: (Float) The percentage of top concepts to include (e.g., `0.1` for the top 10%). If `count` is also set, `count` takes precedence.
-   **`target_file`**: (String) The name of the Markdown file where the top concepts summary will be written (e.g., `"Top Concepts.md"`).

### Trending Topics Job

The `trending_topics` job generates a `Trending Topics - <date>.md` file in your vault, summarizing concepts with the highest recent growth in mentions.

-   **`window_days`**: (Integer) The time window in days for trend analysis (e.g., `7` for weekly trends).
-   **`count`**: (Integer) The fixed number of top trending concepts to include.
-   **`percent`**: (Float) The percentage of top trending concepts to include (e.g., `0.05` for the top 5%). If `count` is set, `percent` is ignored.
-   **`min_mentions`**: (Integer) The minimum number of mentions a concept must have in the current period to be considered.
-   **`target_file`**: (String) The name of the output file. The `{date}` placeholder will be replaced with the current date (e.g., `YYYY-MM-DD`).

### Concept Highlight Refresh Job

The `concept_highlight_refresh` job combines the `top_concepts` and `trending_topics` jobs into a single, efficient execution.

-   **`enabled`**: (Boolean) Set to `true` to enable the job.
-   **`manual_only`**: (Boolean) If `true`, the job will only run when manually triggered.
-   **`cron`**: (String) The schedule for automatic execution (e.g., `"0 6 * * *"` for daily at 6 AM).
-   **`description`**: (String) A human-readable description of the job.

### Concept Summary Refresh Job

The `concept_summary_refresh` job generates detailed `[[Concept]]` pages for all canonical concepts.

-   **`enabled`**: (Boolean) Set to `true` to enable the job.
-   **`manual_only`**: (Boolean) If `true`, the job will only run when manually triggered.
-   **`cron`**: (String) The schedule for automatic execution (e.g., `"0 7 * * *"` for daily at 7 AM).
-   **`description`**: (String) A human-readable description of the job.


## Configuration Loading

aclarai services automatically load configuration on startup. After making changes:

1. **UI Changes**: Take effect immediately when saved
2. **Manual File Edits**: Require service restart to take effect

## Troubleshooting

### Invalid Configuration Values

The UI prevents most invalid values, but if you edit the file manually:
- **Model names**: Must match supported formats (see above)
- **Thresholds**: Must be numbers between 0.0 and 1.0
- **Window parameters**: Must be integers between 0 and 10

### Configuration Not Loading

1. Check file format with `yaml.safe_load()` in Python
2. Verify file permissions allow reading
3. Check logs for specific error messages
4. Use "Reload from File" in the UI to see current state

### Restoring Defaults

To restore all settings to defaults:
1. Delete or rename `settings/aclarai.config.yaml`
2. Restart services or reload configuration
3. The system will use only the default values

## Best Practices

1. **Test Changes Incrementally**: Change one setting at a time to isolate issues
2. **Document Custom Settings**: Keep notes on why you changed specific values
3. **Backup Configurations**: Save copies of working configurations before major changes
4. **Monitor Performance**: Track how model changes affect processing speed and quality
5. **Use Version Control**: Keep `settings/aclarai.config.yaml` in git to track changes

## Related Documentation

- [Configuration Panel Design](../arch/design_config_panel.md) - Technical design specification
- [UI System](components/ui_system.md) - Overall UI architecture
- [Model Configuration](../arch/on-evaluation_agents.md) - Model selection guidelines