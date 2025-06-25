"""Plugins package for aclarai format conversion."""

from typing import TYPE_CHECKING, List

# The TYPE_CHECKING block is only evaluated by static type checkers like mypy.
# It allows us to import types for type hinting without causing circular imports
# or runtime errors if optional dependencies are not installed.
if TYPE_CHECKING:
    from .default_plugin import DefaultPlugin
    from .utils import convert_file_to_markdowns, ensure_defaults
else:
    # Only define these at runtime, not during type checking
    try:
        from .default_plugin import DefaultPlugin
        from .utils import convert_file_to_markdowns, ensure_defaults
    except ImportError:
        # This block runs only if `llama_index` or its dependencies are not installed.
        # We define dummy (no-op) versions of the classes and functions so that the
        # rest of the application can still import them without crashing.
        from pathlib import Path

        from ..plugin_interface import MarkdownOutput, Plugin

        class DefaultPlugin(Plugin):  # type: ignore[misc]
            """Dummy DefaultPlugin for when llama_index is not installed."""

            def can_accept(self, _raw_input: str) -> bool:
                """This dummy plugin should not be used."""
                return False

            def convert(self, _raw_input: str, _path: Path) -> List[MarkdownOutput]:
                """This dummy plugin should not be used."""
                return []

        def ensure_defaults(_md: MarkdownOutput, _path: Path) -> MarkdownOutput:
            """Dummy ensure_defaults function."""
            return _md

        def convert_file_to_markdowns(
            _input_path: Path, _registry: List[Plugin]
        ) -> List[MarkdownOutput]:
            """Dummy convert_file_to_markdowns function."""
            return []


__all__ = ["DefaultPlugin", "ensure_defaults", "convert_file_to_markdowns"]
