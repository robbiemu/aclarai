# Tier 1 Markdown Import System

The Tier 1 import system provides a complete pipeline for importing conversation files into the aclarai vault as standardized Tier 1 Markdown documents.

## Features

- **Hash-based duplicate detection**: SHA-256 hashing prevents re-importing the same content
- **Plugin-based format conversion**: Extensible system supports multiple conversation formats
- **Atomic file writing**: Safe `.tmp` → `fsync` → `rename` pattern prevents corruption
- **Proper Tier 1 format**: Generates compliant Markdown with metadata and block annotations
- **Configuration-driven**: Uses vault paths from central configuration
- **Import logging**: Tracks successful imports for auditing and duplicate detection

## Usage

For complete usage examples and step-by-step tutorials, see:
- **Tutorial**: `docs/tutorials/tier1_import_tutorial.md` - Complete guide with examples
- **CLI Reference**: `shared/aclarai_shared/scripts/import_cli.py --help` - Command line options

## Configuration

The import system uses the central `aclaraiConfig` system with vault path configuration. See `shared/aclarai_shared/config.py` for the `PathsConfig` dataclass implementation and `docs/ENVIRONMENT_CONFIGURATION.md` for environment variable details.

## Generated Format

The system generates Tier 1 Markdown files with proper file-level metadata, block annotations, and Obsidian-compatible anchors. For detailed format specifications and examples, see `docs/tutorials/tier1_import_tutorial.md`.

## Supported Input Formats

The system supports various conversation formats through the pluggable format conversion system:

- **Simple speaker format**: `alice: Hello\nbob: Hi there!`
- **ENTRY format**: `ENTRY [10:00] alice >> Hello`
- **With metadata**: Session IDs, topics, participants extraction
- **Graceful fallback**: Handles unknown formats and empty files

The plugin system uses a `MarkdownOutput` structure containing:
- `title`: Conversation title (auto-generated if missing)
- `markdown_text`: Full Markdown content to write
- `metadata`: Optional fields including `created_at`, `participants`, `message_count`, `duration_sec`, and `plugin_metadata`

## Duplicate Detection

The system uses SHA-256 content hashing for duplicate detection and maintains import logs in JSON format. For usage examples, see `docs/tutorials/tier1_import_tutorial.md`.

## File Naming

Output files use a canonical naming scheme:
- Format: `YYYY-MM-DD_{source_name}_{conversation_title}.md`
- Special characters in filenames are converted to underscores
- Date is extracted from conversation metadata or file modification time

## Error Handling

The system implements robust error handling with specific exception types for different failure modes. For usage examples and error handling patterns, see `docs/tutorials/tier1_import_tutorial.md`.

### Atomic Write Safety

The system uses a `write-temp → fsync → rename` pattern to ensure file integrity. Since Obsidian doesn't place exclusive locks on Markdown files and allows concurrent access from other programs, atomic writes prevent corruption during concurrent edits. The pattern ensures that other watchers (including Obsidian) either see the old version or the fully written new one, never a half-written file.

## Testing

The system includes comprehensive test coverage. See test files in `shared/tests/` for implementation details and `shared/tests/test_tier1_*.py` for running specific test suites.

## Architecture Integration

The import system integrates with aclarai's broader architecture:

- **Plugin System**: Uses existing plugin interface and default plugin
- **Configuration**: Leverages central configuration system
- **Vault Structure**: Respects configured directory layout
- **Graph Sync**: Generated files are ready for graph synchronization
- **Obsidian Compatibility**: Safe concurrent access with Obsidian

### Generated Tier 1 Format

The system generates Tier 1 Markdown files where each conversation utterance becomes a block with:

- **Speaker format**: `speaker: text` blocks
- **Unique identifiers**: Each utterance gets a `<!-- aclarai:id=blk_xyz ver=1 -->` comment
- **Obsidian anchors**: Corresponding `^blk_xyz` anchors for block references
- **File metadata**: HTML comments at the top with conversation details

This format enables:
- Precise linking between Markdown and graph nodes
- File-safe sync and version tracking
- Native compatibility with Obsidian block references