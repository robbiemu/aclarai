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
```

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