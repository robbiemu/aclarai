"""
Tier 1 Markdown import system for aclarai.
This module implements the complete pipeline for importing conversation files
into the vault as Tier 1 Markdown documents, including:
- Hash-based duplicate detection
- Plugin-based format conversion
- Atomic file writing
- Proper Tier 1 Markdown formatting with metadata and block annotations
"""

import contextlib
import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import aclaraiConfig, load_config
from .plugin_interface import MarkdownOutput, Plugin, UnknownFormatError
from .plugin_manager import ImportOrchestrator, ImportStatus, PluginManager

logger = logging.getLogger(__name__)


class DuplicateDetectionError(Exception):
    """Raised when a file is detected as a duplicate and should be skipped."""

    pass


class ImportSystemError(Exception):
    """Base exception for import system errors."""

    pass


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA-256 hash of file content for duplicate detection.
    Args:
        file_path: Path to the file to hash
    Returns:
        Hex string of SHA-256 hash
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
    except Exception as e:
        logger.error(f"Failed to calculate hash for {file_path}: {e}")
        raise ImportSystemError(f"Cannot calculate hash for {file_path}: {e}") from e
    return sha256_hash.hexdigest()


def is_duplicate_import(
    _file_path: Path, hash_value: str, import_log_dir: Path
) -> bool:
    """
    Check if a file has already been imported based on its hash.
    Args:
        file_path: Path to the file being imported
        hash_value: SHA-256 hash of the file content
        import_log_dir: Directory where import logs are stored
    Returns:
        True if file is a duplicate, False otherwise
    """
    import_log_file = import_log_dir / "imported_files.json"
    if not import_log_file.exists():
        return False
    try:
        with open(import_log_file, "r", encoding="utf-8") as f:
            imported_files: Dict[str, Any] = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read import log {import_log_file}: {e}")
        return False
    # Check if hash exists in imported files
    return hash_value in imported_files.get("hashes", {})


def record_import(
    file_path: Path, hash_value: str, import_log_dir: Path, output_files: List[Path]
):
    """
    Record a successful import in the import log.
    Args:
        file_path: Path to the imported file
        hash_value: SHA-256 hash of the file content
        import_log_dir: Directory where import logs are stored
        output_files: List of output files created during import
    """
    import_log_file = import_log_dir / "imported_files.json"
    # Create directory if it doesn't exist
    import_log_dir.mkdir(parents=True, exist_ok=True)
    # Read existing log or create new one
    imported_files: Dict[str, Dict] = {"hashes": {}, "files": {}}
    if import_log_file.exists():
        try:
            with open(import_log_file, "r", encoding="utf-8") as f:
                imported_files = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read existing import log: {e}")
    # Record the import
    timestamp = datetime.now().isoformat()
    imported_files["hashes"][hash_value] = {
        "source_file": str(file_path),
        "imported_at": timestamp,
        "output_files": [str(f) for f in output_files],
    }
    # Also record by filename for easier lookup
    imported_files["files"][str(file_path)] = {
        "hash": hash_value,
        "imported_at": timestamp,
        "output_files": [str(f) for f in output_files],
    }
    # Write atomically
    try:
        write_file_atomically(import_log_file, json.dumps(imported_files, indent=2))
        logger.debug(f"Recorded import of {file_path} with hash {hash_value}")
    except Exception as e:
        logger.error(f"Failed to record import: {e}")
        # Don't raise - import succeeded even if logging failed


def write_file_atomically(target_path: Path, content: str):
    """
    Write file content atomically using .tmp -> fsync -> rename pattern.
    This follows the atomic write pattern specified in the architecture docs
    to prevent corruption and partial reads by other processes like Obsidian.
    Args:
        target_path: Final path where file should be written
        content: Content to write to the file
    """
    target_path = Path(target_path)
    # Create parent directory if it doesn't exist
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # Create temporary file in the same directory
    temp_file = target_path.parent / f".{target_path.name}.tmp"
    try:
        # Write to temporary file
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()  # Ensure content is written to OS buffer
            os.fsync(f.fileno())  # Force write to disk
        # Atomic rename
        temp_file.rename(target_path)
        logger.debug(f"Atomically wrote file: {target_path}")
    except Exception as e:
        # Clean up temporary file if it exists
        with contextlib.suppress(OSError):
            temp_file.unlink()
        raise ImportSystemError(f"Failed to write file {target_path}: {e}") from e


def ensure_defaults(md: MarkdownOutput, path: Path) -> MarkdownOutput:
    """
    Ensure MarkdownOutput has all required default values.
    This implements the ensure_defaults function from the architecture docs.
    Args:
        md: Original MarkdownOutput from plugin
        path: Path to the source file
    Returns:
        MarkdownOutput with all defaults filled in
    """
    meta = md.metadata or {}
    # Get file modification time for created_at default
    try:
        file_mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    except OSError:
        file_mtime = datetime.now().isoformat()
    created = meta.get("created_at") or file_mtime
    title = md.title or f"Conversation last modified at {created}"
    # Replace placeholder in markdown text if present
    markdown_text = md.markdown_text
    if "PLACEHOLDER" in markdown_text:
        markdown_text = markdown_text.replace("PLACEHOLDER", created)
    return MarkdownOutput(
        title=title,
        markdown_text=markdown_text,
        metadata={
            "created_at": created,
            "participants": meta.get("participants", ["user", "assistant"]),
            "message_count": meta.get("message_count", 0),
            "duration_sec": meta.get("duration_sec"),
            "plugin_metadata": meta.get("plugin_metadata", {}),
        },
    )


def convert_file_to_markdowns(
    input_path: Path, registry: List[Plugin]
) -> List[MarkdownOutput]:
    """
    Convert a file to MarkdownOutput objects using the plugin system.
    This implements the main execution flow from the architecture docs.
    Args:
        input_path: Path to the input file
        registry: List of plugins to try
    Returns:
        List of MarkdownOutput objects
    Raises:
        UnknownFormatError: If no plugin can handle the file
    """
    try:
        raw_input = input_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try with different encoding
        raw_input = input_path.read_text(encoding="latin-1")
    except Exception as e:
        raise ImportSystemError(f"Cannot read file {input_path}: {e}") from e
    for plugin in registry:
        if plugin.can_accept(raw_input):
            try:
                raw_outputs = plugin.convert(raw_input, input_path)
                return [ensure_defaults(md, input_path) for md in raw_outputs]
            except Exception as e:
                logger.warning(f"Plugin {type(plugin).__name__} failed: {e}")
                continue
    raise UnknownFormatError(f"No plugin could handle file: {input_path}")


def format_tier1_markdown(md: MarkdownOutput) -> str:
    """
    Format a MarkdownOutput as complete Tier 1 Markdown with metadata.
    This adds the file-level HTML metadata comments at the top of the file
    as specified in the architecture documentation, but only if they're not
    already present in the markdown_text.
    Args:
        md: MarkdownOutput to format
    Returns:
        Complete Tier 1 Markdown content with metadata headers
    """
    lines = []
    # Check if metadata comments are already present
    if "<!-- aclarai:title=" in md.markdown_text:
        # Metadata already included, just return the content
        return md.markdown_text
    # Add file-level metadata as HTML comments
    metadata = md.metadata or {}
    lines.extend(
        [
            f"<!-- aclarai:title={md.title} -->",
            f"<!-- aclarai:created_at={metadata.get('created_at', '')} -->",
            f"<!-- aclarai:participants={json.dumps(metadata.get('participants', []))} -->",
            f"<!-- aclarai:message_count={metadata.get('message_count', 0)} -->",
            f"<!-- aclarai:plugin_metadata={json.dumps(metadata.get('plugin_metadata', {}))} -->",
            "",  # Empty line after metadata
        ]
    )
    # Add the main content (which should already include block-level annotations)
    lines.append(md.markdown_text)
    return "\n".join(lines)


def generate_output_filename(md: MarkdownOutput, source_path: Path) -> str:
    """
    Generate a canonical filename for the Tier 1 Markdown file.
    Args:
        md: MarkdownOutput containing the conversation
        source_path: Path to the original source file
    Returns:
        Filename string in format YYYY-MM-DD_Source_Title.md
    """
    # Extract date from created_at or use current date
    metadata = md.metadata or {}
    created_at = metadata.get("created_at", datetime.now().isoformat())
    try:
        date_part = datetime.fromisoformat(created_at.replace("Z", "+00:00")).strftime(
            "%Y-%m-%d"
        )
    except (ValueError, AttributeError):
        date_part = datetime.now().strftime("%Y-%m-%d")
    # Clean up title for filename
    title = md.title.replace(" ", "_")
    # Remove chars that are problematic in filenames
    title = "".join(c for c in title if c.isalnum() or c in "._-")
    # Clean up source name
    source_name = source_path.stem
    # Replace problematic characters in source name
    source_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in source_name)
    return f"{date_part}_{source_name}_{title}.md"


class Tier1ImportSystem:
    """
    Main import system for creating Tier 1 Markdown files.
    This class orchestrates the complete import pipeline:
    1. Hash calculation and duplicate detection
    2. Plugin-based format conversion via ImportOrchestrator
    3. Tier 1 Markdown formatting
    4. Atomic file writing to vault
    5. Import logging with enhanced status tracking
    """

    def __init__(self, config: Optional[aclaraiConfig] = None):
        """
        Initialize the import system.
        Args:
            config: aclarai configuration (will load from env if not provided)
        """
        self.config = config or load_config(validate=False)
        self.vault_path = Path(self.config.vault_path)
        self.import_log_dir = self.vault_path / self.config.paths.logs

        # Initialize plugin manager and orchestrator
        self.plugin_manager = PluginManager()
        self.orchestrator = ImportOrchestrator(self.plugin_manager)

        logger.info(f"Initialized Tier1ImportSystem with vault: {self.vault_path}")
        logger.info(
            f"Plugin manager loaded {self.plugin_manager.get_plugin_count()} plugins"
        )

    def add_plugin(self, plugin: Plugin):
        """
        Add a plugin to the registry.
        Args:
            plugin: Plugin instance to add
        """
        self.plugin_manager.register_plugin(plugin)
        logger.debug(f"Added plugin: {type(plugin).__name__}")

    def get_plugin_info(self) -> Dict[str, Any]:
        """
        Get information about loaded plugins.
        Returns:
            Dictionary with plugin information
        """
        return self.orchestrator.get_plugin_info()

    def import_file(self, source_path: Path) -> List[Path]:
        """
        Import a single file into the vault as Tier 1 Markdown.

        This method now uses the ImportOrchestrator for plugin management
        and enhanced status tracking, while maintaining the same interface.

        Args:
            source_path: Path to the file to import
        Returns:
            List of paths to created Tier 1 Markdown files
        Raises:
            DuplicateDetectionError: If file is a duplicate
            ImportSystemError: If import fails
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise ImportSystemError(f"Source file does not exist: {source_path}")

        logger.info(f"Starting import of {source_path}")

        # Step 1: Calculate hash for duplicate detection
        file_hash = calculate_file_hash(source_path)
        logger.debug(f"File hash: {file_hash}")

        # Step 2: Check for duplicates
        if is_duplicate_import(source_path, file_hash, self.import_log_dir):
            logger.info(f"Skipping duplicate file: {source_path}")
            raise DuplicateDetectionError(
                f"File {source_path} is a duplicate (hash: {file_hash})"
            )

        # Step 3: Use orchestrator for plugin-based conversion
        import_result = self.orchestrator.import_file(source_path)

        # Handle different import statuses
        if import_result.status == ImportStatus.ERROR:
            logger.error(f"Import failed for {source_path}: {import_result.message}")
            raise ImportSystemError(f"Import failed: {import_result.message}")

        elif import_result.status == ImportStatus.SKIPPED:
            logger.error(
                f"No plugin could handle {source_path}: {import_result.message}"
            )
            raise ImportSystemError(f"Unknown format: {source_path}")

        elif import_result.status == ImportStatus.IGNORED:
            logger.info(f"No conversations found in {source_path}")
            return []

        elif import_result.status == ImportStatus.SUCCESS:
            # Use conversion outputs from orchestrator (no need to re-run conversion)
            markdown_outputs = import_result.conversion_outputs

            if not markdown_outputs:
                logger.info(f"No conversations found in {source_path}")
                return []

            logger.info(
                f"Found {len(markdown_outputs)} conversation(s) in {source_path}"
            )
            logger.info(f"Used plugin: {import_result.plugin_used}")

            # Step 4: Write Tier 1 Markdown files
            output_files = []
            tier1_dir = self.vault_path / self.config.paths.tier1
            for i, md in enumerate(markdown_outputs):
                # Generate filename
                if len(markdown_outputs) > 1:
                    # Multiple conversations - add index
                    base_filename = generate_output_filename(md, source_path)
                    name, ext = base_filename.rsplit(".", 1)
                    filename = f"{name}_{i + 1}.{ext}"
                else:
                    filename = generate_output_filename(md, source_path)

                output_path = tier1_dir / filename
                # Format as complete Tier 1 Markdown
                content = format_tier1_markdown(md)
                # Write atomically
                write_file_atomically(output_path, content)
                output_files.append(output_path)
                logger.info(f"Created Tier 1 file: {output_path}")

            # Step 5: Record successful import with enhanced metadata
            record_import(source_path, file_hash, self.import_log_dir, output_files)
            logger.info(
                f"Successfully imported {source_path} -> {len(output_files)} files"
            )
            return output_files

        else:
            # Unexpected status
            raise ImportSystemError(f"Unexpected import status: {import_result.status}")

    def import_directory(
        self, source_dir: Path, recursive: bool = True
    ) -> Dict[Path, List[Path]]:
        """
        Import all files from a directory.
        Args:
            source_dir: Directory to import from
            recursive: Whether to search subdirectories
        Returns:
            Dictionary mapping source files to their output files
        """
        source_dir = Path(source_dir)
        if not source_dir.is_dir():
            raise ImportSystemError(f"Source is not a directory: {source_dir}")
        logger.info(
            f"Starting directory import from {source_dir} (recursive={recursive})"
        )
        # Find all files to import
        files = list(source_dir.rglob("*")) if recursive else list(source_dir.iterdir())
        # Filter to only files (not directories)
        files = [f for f in files if f.is_file()]
        logger.info(f"Found {len(files)} files to process")
        results = {}
        duplicates = 0
        errors = 0
        for file_path in files:
            try:
                output_files = self.import_file(file_path)
                results[file_path] = output_files
            except DuplicateDetectionError:
                duplicates += 1
                logger.debug(f"Skipped duplicate: {file_path}")
            except Exception as e:
                errors += 1
                logger.error(f"Failed to import {file_path}: {e}")
                results[file_path] = []
        logger.info(
            f"Directory import complete: {len(results)} processed, {duplicates} duplicates, {errors} errors"
        )
        return results
