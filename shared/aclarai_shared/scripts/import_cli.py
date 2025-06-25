#!/usr/bin/env python3
"""
Command-line interface for aclarai Tier 1 import system.
This script provides a simple way to import conversation files into the vault
as Tier 1 Markdown documents.
"""

import argparse
import logging
import sys
from pathlib import Path

from aclarai_shared.config import load_config
from aclarai_shared.import_system import (
    DuplicateDetectionError,
    ImportSystemError,
    Tier1ImportSystem,
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=level, format=format_str)


def import_file(
    system: Tier1ImportSystem, file_path: Path, force: bool = False
) -> bool:
    """
    Import a single file.
    Args:
        system: Import system instance
        file_path: Path to file to import
        force: Whether to skip duplicate detection
    Returns:
        True if successful, False otherwise
    """
    try:
        output_files = system.import_file(file_path)
        if output_files:
            print(f"✓ Imported {file_path} -> {len(output_files)} Tier 1 file(s)")
            for f in output_files:
                print(f"  Created: {f}")
        else:
            print(f"⚠ No conversations found in {file_path}")
        return True
    except DuplicateDetectionError:
        if force:
            print(f"⚠ Duplicate detected but force=True not implemented: {file_path}")
            return False
        else:
            print(f"⚠ Skipping duplicate: {file_path}")
            return True
    except ImportSystemError as e:
        print(f"✗ Import failed: {file_path} - {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error importing {file_path}: {e}")
        return False


def import_directory(system: Tier1ImportSystem, dir_path: Path, recursive: bool = True):
    """
    Import all files from a directory.
    Args:
        system: Import system instance
        dir_path: Directory to import from
        recursive: Whether to search subdirectories
    """
    try:
        results = system.import_directory(dir_path, recursive=recursive)
        total_files = len(results)
        successful = sum(1 for files in results.values() if files)
        print("\n📊 Directory import summary:")
        print(f"  Total files processed: {total_files}")
        print(f"  Successful imports: {successful}")
        print(f"  Failed/skipped: {total_files - successful}")
    except Exception as e:
        print(f"✗ Directory import failed: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Import conversation files into aclarai vault as Tier 1 Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import a single file
  python import_cli.py --file chat_export.txt
  # Import all files from a directory
  python import_cli.py --directory /path/to/chat/exports
  # Import with verbose logging
  python import_cli.py --file chat.txt --verbose
  # Import without recursive directory search
  python import_cli.py --directory chats --no-recursive
        """,
    )
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", "-f", type=Path, help="Import a single file")
    input_group.add_argument(
        "--directory", "-d", type=Path, help="Import all files from a directory"
    )
    # Options
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        default=True,
        help="Search directories recursively (default: True)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Don't search directories recursively",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force import even if duplicates are detected (not implemented)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--vault-path", type=Path, help="Override vault path from config"
    )
    args = parser.parse_args()
    # Setup logging
    setup_logging(args.verbose)
    # Load configuration
    try:
        config = load_config(validate=False)
        if args.vault_path:
            config.vault_path = str(args.vault_path)
        print(f"📁 Using vault: {config.vault_path}")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        sys.exit(1)
    # Initialize import system
    try:
        system = Tier1ImportSystem(config)
        print("🚀 aclarai Tier 1 Import System initialized")
    except Exception as e:
        print(f"✗ Failed to initialize import system: {e}")
        sys.exit(1)
    # Perform import
    success = True
    if args.file:
        if not args.file.exists():
            print(f"✗ File not found: {args.file}")
            sys.exit(1)
        print(f"📄 Importing file: {args.file}")
        success = import_file(system, args.file, args.force)
    elif args.directory:
        if not args.directory.is_dir():
            print(f"✗ Directory not found: {args.directory}")
            sys.exit(1)
        print(f"📁 Importing directory: {args.directory} (recursive={args.recursive})")
        import_directory(system, args.directory, args.recursive)
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
