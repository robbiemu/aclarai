# Concept Summary Agent

## Overview

The Concept Summary Agent generates detailed Markdown pages for each canonical concept in the Neo4j knowledge graph. It follows the specifications from `docs/arch/on-writing_vault_documents.md` and uses the RAG (Retrieval-Augmented Generation) workflow described in `docs/arch/on-RAG_workflow.md`.

## Purpose

The agent creates `[[Concept]]` pages that serve as definitive reference documents for concepts identified and promoted through the aclarai pipeline. Each page provides:

- A clear definition of the concept
- Relevant examples from claims and summaries with `^aclarai:id` references
- Links to related concepts for navigation
- Consistent formatting following the established patterns

## Architecture

### Input Sources

The agent retrieves information using a hybrid approach:

| Source | Method | Purpose |
|--------|--------|---------|
| **Claims** | Neo4j Cypher queries | Pull direct supporting evidence via `SUPPORTS_CONCEPT`, `MENTIONS_CONCEPT`, `CONTRADICTS_CONCEPT` relationships |
| **Summaries** | Neo4j Cypher queries | Pull contextual summaries via `MENTIONS_CONCEPT`, `RELATES_TO` relationships |
| **Related Concepts** | Vector similarity search | Find semantically similar concepts (placeholder implementation) |
| **Utterances** | Vector similarity search | Find natural language usage examples (placeholder implementation) |

### Output Format

Generated files follow the exact format specified in `docs/arch/on-writing_vault_documents.md`:

```markdown
## Concept: <concept name>

<definition paragraph>

### Examples
- <claim or utterance> ^aclarai:id
- ...

### See Also
- [[Related Concept A]]
- [[Related Concept B]]

<!-- aclarai:id=concept_<slug> ver=N -->
^concept_<slug>
```

## Configuration

The agent is configured through the `concept_summaries` section of the aclarai configuration:

```yaml
concept_summaries:
  model: "gpt-4"              # LLM model for content generation
  max_examples: 5             # Maximum examples to include
  skip_if_no_claims: true     # Skip concepts with no supporting claims
  include_see_also: true      # Include "See Also" section
```

## Key Features

### Atomic File Writing

All file operations use the safe atomic pattern:
1. Write to temporary file (`.tmp` extension)
2. Force sync to disk (`fsync`)
3. Atomic rename to final location

This prevents corruption and ensures vault watchers see complete files.

### Quality Filtering

- Respects the `skip_if_no_claims` setting to avoid generating empty concept pages
- Prioritizes claims by relationship strength when selecting examples
- Orders content by relevance and recency

### Error Resilience

- Continues processing other concepts if individual concepts fail
- Provides detailed error reporting and statistics
- Uses retry logic for transient Neo4j connection issues

## Usage

### Via Python API

```python
from aclarai_shared.concept_summary_agent import ConceptSummaryAgent
from aclarai_shared.config import load_config

# Load configuration
config = load_config()

# Create and run agent
agent = ConceptSummaryAgent(config=config)
result = agent.run_agent()

print(f"Generated {result['concepts_generated']} concept pages")
```

### Via CLI

```bash
# Run with default configuration
python run_concept_summary_agent.py

# Run with custom config
python run_concept_summary_agent.py --config /path/to/config.yaml

# Dry run to see what would be generated
python run_concept_summary_agent.py --dry-run

# Enable debug logging
python run_concept_summary_agent.py --log-level DEBUG
```

### Via Scheduler

The agent can be integrated into the scheduler service by adding a job configuration:

```yaml
scheduler:
  jobs:
    concept_summary_generation:
      enabled: true
      manual_only: false
      cron: "0 4 * * *"  # Run daily at 4 AM
      description: "Generate concept summary pages"
```

## Implementation Notes

### Current Limitations

1. **LLM Integration**: Currently uses structured templates instead of LLM-generated content. The framework is in place for LLM integration.

2. **Vector Store Integration**: Related concepts and utterances retrieval uses placeholder implementations. The methods are designed for easy integration when vector stores are available.

3. **Incremental Updates**: Currently regenerates all concept pages. Future versions could implement change detection to only update modified concepts.

### Future Enhancements

1. **LLM Content Generation**: Replace template-based content with LLM-generated definitions and descriptions
2. **Vector Store Integration**: Implement full vector similarity search for related concepts and utterances
3. **Content Quality Assessment**: Add quality metrics and validation for generated content
4. **Incremental Processing**: Track concept modifications and only regenerate changed concepts

## File Locations

- **Agent Implementation**: `shared/aclarai_shared/concept_summary_agent/`
- **Configuration**: `shared/aclarai_shared/aclarai.config.default.yaml`
- **Tests**: `tests/test_concept_summary_agent.py`
- **CLI Entry Point**: `run_concept_summary_agent.py`
- **Generated Files**: `{vault_path}/{tier3_path}/*.md`

## Dependencies

- **Neo4j**: For graph queries and concept retrieval
- **aclarai Configuration System**: For settings and parameters
- **Python Standard Library**: For file operations and utilities

The agent is designed to work with minimal dependencies and gracefully handle missing optional components (like vector stores or LLM providers).

## Testing

The agent includes comprehensive tests covering:

- Initialization and configuration loading
- Content generation with various inputs
- File writing and atomic operations
- Error handling and edge cases
- Integration workflows

Run tests with:
```bash
python tests/test_concept_summary_agent.py
```