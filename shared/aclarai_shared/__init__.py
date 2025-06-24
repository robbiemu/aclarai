"""
Shared utilities and configuration for aclarai services.
"""

from .config import DatabaseConfig, PathsConfig, aclaraiConfig, load_config
from .plugin_interface import MarkdownOutput, Plugin, UnknownFormatError

try:
    from .import_system import (
        DuplicateDetectionError,
        ImportSystemError,
        Tier1ImportSystem,
    )
    from .plugins import DefaultPlugin, convert_file_to_markdowns, ensure_defaults

    __all__ = [
        "aclaraiConfig",
        "DatabaseConfig",
        "PathsConfig",
        "load_config",
        "Plugin",
        "MarkdownOutput",
        "UnknownFormatError",
        "ensure_defaults",
        "convert_file_to_markdowns",
        "DefaultPlugin",
        "Tier1ImportSystem",
        "DuplicateDetectionError",
        "ImportSystemError",
    ]
except ImportError:
    # LlamaIndex dependencies not available
    __all__ = [
        "aclaraiConfig",
        "DatabaseConfig",
        "PathsConfig",
        "load_config",
        "Plugin",
        "MarkdownOutput",
        "UnknownFormatError",
    ]
