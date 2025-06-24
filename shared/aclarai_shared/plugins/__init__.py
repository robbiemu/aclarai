"""Plugins package for aclarai format conversion."""

from typing import List


try:
    from .default_plugin import DefaultPlugin
    from .utils import convert_file_to_markdowns, ensure_defaults

    __all__ = ["DefaultPlugin", "ensure_defaults", "convert_file_to_markdowns"]
except ImportError:
    # LlamaIndex not available, provide dummy implementations for testing
    class DefaultPlugin:
        pass

    def ensure_defaults(*args, **kwargs):
        pass

    def convert_file_to_markdowns(*_args, **_kwargs) -> List[str]:
        return []

    __all__ = ["DefaultPlugin", "ensure_defaults", "convert_file_to_markdowns"]
