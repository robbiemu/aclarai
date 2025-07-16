# Trending Topics Job

The Trending Topics Job is a scheduled component that analyzes concept mention patterns in the aclarai knowledge graph to identify trending concepts and generate automated reports.

## Overview

This job tracks the growth of concept mentions over configurable time windows by analyzing claim-concept relationships in the Neo4j graph database. It generates daily `Trending Topics - <date>.md` files that highlight concepts with the highest growth rates, helping users discover emerging themes in their knowledge base.

## Key Features

### 1. Scheduled Job Integration
- Fully integrated into the `aclarai-scheduler` service using APScheduler
- Configurable cron scheduling (default: 5 AM daily)
- Environment variable overrides for manual control
- Respects global automation pause settings

### 2. Neo4j Edge Tracking
- Tracks `SUPPORTS_CONCEPT` and `MENTIONS_CONCEPT` edge creation timestamps
- Uses `classified_at` property from existing claim-concept linking system
- Efficient Cypher queries for temporal analysis across configurable time windows

### 3. Delta Calculation Logic
- Configurable time windows (default: 7 days)
- Compares current period vs. previous period mentions
- Calculates growth rates with special handling for new concepts
- Respects `min_mentions` threshold to filter noise

### 4. Top Concepts Selection
- Supports both `count` (fixed number) and `percent` (percentage) selection modes
- Orders by growth rate with fallback to absolute mentions
- Graceful handling when no concepts meet criteria

### 5. Markdown File Generation
- Follows "Trending Concepts Agent" format from vault documentation standards
- Dynamic date in filename: `Trending Topics - {date}.md`
- Proper `[[wikilink]]` formatting for concepts
- Growth metrics display (percentage and absolute changes)
- `aclarai:id` metadata for vault synchronization

### 6. Atomic File Writing
- Implements `.tmp` → `fsync` → `rename` pattern
- Safe for concurrent access with Obsidian and vault watchers
- Follows established file handling guidelines

## Architecture

### Core Components

The job is implemented as a single class `TrendingTopicsJob` in `services/scheduler/aclarai_scheduler/trending_topics_job.py` with the following key methods:

- `_get_time_windows()`: Calculates analysis periods
- `_get_concept_mention_deltas()`: Queries Neo4j for concept mention data
- `_select_trending_concepts()`: Filters concepts based on configuration
- `_generate_markdown_content()`: Creates formatted output
- `_write_file_atomically()`: Safely writes to filesystem

### Data Flow

1. **Time Window Calculation**: Determines current and comparison periods
2. **Neo4j Query**: Retrieves concept mention counts for both periods
3. **Growth Analysis**: Calculates growth rates and filters by thresholds
4. **Concept Selection**: Applies count/percent limits to identify top concepts
5. **Content Generation**: Creates markdown with proper formatting
6. **Atomic Write**: Safely writes file to vault with metadata

## Configuration

### Job Configuration

Configure the job in `settings/aclarai.config.yaml`:

```yaml
scheduler:
  jobs:
    trending_topics:
      enabled: true
      manual_only: false
      cron: "0 5 * * *"  # 5 AM daily
      description: "Generate Trending Topics - <date>.md from concept mention deltas"
      window_days: 7      # Analysis time window
      count: null         # Fixed number of concepts (exclusive with percent)
      percent: 5          # Percentage of concepts to select (exclusive with count)
      min_mentions: 2     # Minimum mentions to be considered
      target_file: "Trending Topics - {date}.md"  # Output file with date placeholder
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | Boolean | `true` | Whether the job is active |
| `manual_only` | Boolean | `false` | Skip automatic scheduling |
| `cron` | String | `"0 5 * * *"` | Cron expression for scheduling |
| `window_days` | Integer | `7` | Analysis time window in days |
| `count` | Integer | `null` | Fixed number of concepts to select |
| `percent` | Integer | `5` | Percentage of concepts to select |
| `min_mentions` | Integer | `2` | Minimum mentions to be considered |
| `target_file` | String | `"Trending Topics - {date}.md"` | Output filename template |

## Implementation Details

### Files Created/Modified

#### Configuration Files
- `shared/aclarai_shared/aclarai.config.default.yaml`: Job configuration defaults
- `shared/aclarai_shared/config.py`: `TrendingTopicsJobConfig` class definition

#### Implementation Files
- `services/scheduler/aclarai_scheduler/trending_topics_job.py`: Complete job implementation (444 lines)
- `services/scheduler/aclarai_scheduler/main.py`: Job registration and execution logic

#### Test Files
- `services/scheduler/tests/test_trending_topics_job.py`: Comprehensive test suite (9 tests covering all functionality)

### Sample Output

```markdown
## Trending This Week

*Analysis period: 2025-06-26 to 2025-07-03*

- [[Python Debugging]] — Mentions up 88% (8 → 15)
- [[Docker Containers]] — New concept with 12 mentions
- [[API Development]] — Mentions up 25% (20 → 25)

<!-- aclarai:id=file_trending_topics_20250703 ver=1 -->
^file_trending_topics_20250703
```

### Neo4j Query

The job uses the following Cypher query to analyze concept mentions:

```cypher
MATCH (c:Concept)
OPTIONAL MATCH (c)<-[r_current:SUPPORTS_CONCEPT|MENTIONS_CONCEPT]-(claim_current:Claim)
WHERE r_current.classified_at >= $window_start AND r_current.classified_at <= $current_time
OPTIONAL MATCH (c)<-[r_prev:SUPPORTS_CONCEPT|MENTIONS_CONCEPT]-(claim_prev:Claim)
WHERE r_prev.classified_at >= $comparison_window_start AND r_prev.classified_at < $window_start

WITH c,
     count(DISTINCT r_current) as current_mentions,
     count(DISTINCT r_prev) as previous_mentions

WHERE current_mentions >= $min_mentions OR previous_mentions >= $min_mentions

WITH c, current_mentions, previous_mentions,
     CASE
        WHEN previous_mentions = 0 AND current_mentions > 0 THEN 1000.0  // New concept, high growth
        WHEN previous_mentions > 0 THEN toFloat(current_mentions - previous_mentions) / previous_mentions
        ELSE 0.0
     END as growth_rate

RETURN c.name as concept_name, current_mentions, previous_mentions, growth_rate
ORDER BY growth_rate DESC, current_mentions DESC
```

## Testing Coverage

The implementation includes comprehensive tests covering:

- ✅ Job initialization and configuration loading
- ✅ Time window calculation logic
- ✅ Concept selection algorithms (count vs. percent)
- ✅ Markdown content generation with proper formatting
- ✅ Dynamic file path generation with date substitution
- ✅ Atomic file writing with temp file handling
- ✅ Successful job execution with sample data
- ✅ Graceful handling of empty result sets
- ✅ Error recovery when database queries fail

## Integration Points

### Scheduler Service
- Fully integrated into existing APScheduler framework
- Follows established job registration patterns
- Respects automation pause mechanisms

### Neo4j Graph
- Uses existing claim-concept relationship data
- Leverages `classified_at` timestamps from claim processing
- Efficient querying with proper indexing considerations

### Configuration System
- Follows established configuration patterns
- Supports environment variable overrides
- Validates configuration on startup

### Vault Synchronization
- Compatible with existing vault sync infrastructure
- Generates proper `aclarai:id` metadata
- Supports atomic file operations

### Logging
- Structured logging following project standards
- Comprehensive error reporting and debugging information
- Performance metrics and execution statistics

## Usage

The job runs automatically based on the configured cron schedule. Users can:

1. **Configure the job** via `settings/aclarai.config.yaml`
2. **Monitor execution** through structured logs
3. **View results** in the generated `Trending Topics - <date>.md` files
4. **Enable/disable** via configuration or environment variables

## Performance Considerations

- **Efficient Queries**: Uses indexed Neo4j queries with proper time window filtering
- **Configurable Limits**: Supports both fixed count and percentage-based selection
- **Atomic Operations**: Ensures file writes don't interfere with other processes
- **Error Recovery**: Graceful degradation when database queries fail
- **Memory Usage**: Processes concepts incrementally to manage memory consumption

## Error Handling

The job implements resilient error handling:

- **Database Errors**: Graceful recovery when Neo4j queries fail
- **File System Errors**: Proper cleanup of temporary files
- **Configuration Errors**: Validation and sensible defaults
- **Empty Results**: Generates appropriate output when no trending concepts found

## Monitoring and Observability

Each job execution provides:

- **Start/completion timestamps**
- **Processing statistics** (concepts analyzed, concepts selected, errors)
- **Performance metrics** (execution duration, query performance)
- **Error details** with stack traces for debugging

Example log entry:
```json
{
  "level": "INFO",
  "service": "aclarai-scheduler",
  "filename.function_name": "trending_topics_job.run_job",
  "job_id": "trending_topics_1734264000",
  "concepts_analyzed": 42,
  "trending_concepts": 3,
  "duration": 2.1,
  "target_file": "Trending Topics - 2025-07-14.md"
}
```

## Dependencies

The trending topics job depends on:

- **APScheduler**: Job scheduling framework
- **Neo4j Python Driver**: Database connectivity
- **aclarai_shared**: Configuration and logging components
- **Standard Library**: File operations, datetime handling

## Future Enhancements

Potential improvements include:

- **Configurable Metrics**: Support for different trending algorithms
- **Multi-timeframe Analysis**: Compare trends across different time periods
- **Concept Categories**: Filter trending analysis by concept types
- **Notification Integration**: Alert mechanisms for significant trend changes
- **Historical Trending**: Track long-term concept popularity patterns

## Related Documentation

- [Scheduler System](scheduler_system.md) - Overall scheduler architecture
- [Scheduler Job Configuration](scheduler_job_configuration.md) - Job configuration details
- [Scheduler Setup Guide](../guides/scheduler_setup_guide.md) - Adding new jobs
