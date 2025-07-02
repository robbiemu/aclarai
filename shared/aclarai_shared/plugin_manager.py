"""
Plugin manager and import orchestrator for aclarai.

This module implements the centralized plugin manager and import orchestrator
as specified in docs/project/epic_1/sprint_8-Create_plugin_manager.md.

Key Features:
- Centralized plugin discovery and management
- Import orchestration with detailed status tracking
- Plugin ordering with fallback plugin as last resort
- Integration with existing Plugin interface
- Structured import status for UI consumption
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from importlib.metadata import entry_points, EntryPoint
except ImportError:
    # Python < 3.8 fallback
    from importlib_metadata import entry_points, EntryPoint  # type: ignore

from .plugin_interface import MarkdownOutput, Plugin
from .plugins.default_plugin import DefaultPlugin
from .plugins.utils import ensure_defaults

logger = logging.getLogger(__name__)


class ImportStatus(Enum):
    """Status of an import operation for UI consumption."""

    SUCCESS = "success"
    IGNORED = "ignored"  # e.g., duplicate file
    ERROR = "error"  # e.g., conversion failure
    SKIPPED = "skipped"  # e.g., no plugin could handle


@dataclass
class ImportResult:
    """Result of a single file import operation."""

    file_path: Path
    status: ImportStatus
    message: str
    plugin_used: Optional[str] = None
    output_files: Optional[List[Path]] = None
    error_details: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    conversion_outputs: Optional[List[MarkdownOutput]] = (
        None  # Store actual conversion results
    )


class PluginManager:
    """
    Centralized plugin manager for discovering, managing, and ordering plugins.

    This class handles:
    - Plugin discovery from known locations
    - Plugin registry management
    - Plugin ordering (specific plugins first, fallback last)
    - Plugin introspection and metadata
    """

    def __init__(self):
        """Initialize the plugin manager with discovered plugins."""
        self._plugins: List[Plugin] = []
        self._plugin_metadata: Dict[str, Dict[str, Any]] = {}
        self._discover_plugins()
        logger.info(f"Initialized PluginManager with {len(self._plugins)} plugins")

    def _discover_plugins(self) -> None:
        """
        Discover and register all available plugins.

        This method:
        1. Discovers plugins via entry_points (aclarai.plugins group)
        2. Registers discovered plugins with proper ordering
        3. Always registers DefaultPlugin as fallback (lowest priority)
        """
        # Step 1: Discover plugins via entry_points
        discovered_plugins: List[Plugin] = []
        try:
            # Get entry points for the 'aclarai.plugins' group
            eps = entry_points()
            plugin_entries: List[EntryPoint] = []
            
            if hasattr(eps, "select"):
                # Python 3.10+ style
                plugin_entries = list(eps.select(group="aclarai.plugins"))
            else:
                # Python 3.8-3.9 style - eps is a dict  
                entries: Any = eps.get("aclarai.plugins") if "aclarai.plugins" in eps else []
                plugin_entries = list(entries) if entries else []

            for entry_point in plugin_entries:
                try:
                    # Load the plugin class from entry point
                    plugin_class = entry_point.load()

                    # Instantiate the plugin
                    plugin = plugin_class()

                    # Validate it's a proper plugin
                    if hasattr(plugin, "can_accept") and hasattr(plugin, "convert"):
                        discovered_plugins.append(plugin)
                        logger.info(
                            f"Discovered plugin via entry_points: {type(plugin).__name__}"
                        )
                    else:
                        logger.warning(
                            f"Entry point {entry_point.name} does not implement Plugin interface"
                        )

                except Exception as e:
                    logger.warning(
                        f"Failed to load plugin from entry point {entry_point.name}: {e}"
                    )

        except Exception as e:
            logger.warning(f"Failed to discover plugins via entry_points: {e}")

        # Step 2: Register discovered plugins (they get priority over fallback)
        for plugin in discovered_plugins:
            self.register_plugin(plugin, is_fallback=False)

        # Step 3: Always register the default plugin as fallback (lowest priority)
        default_plugin = DefaultPlugin()
        self.register_plugin(default_plugin, is_fallback=True)

        logger.info(
            f"Plugin discovery complete: {len(discovered_plugins)} from entry_points + 1 default plugin"
        )

    def register_plugin(self, plugin: Plugin, is_fallback: bool = False) -> None:
        """
        Register a plugin with the manager.

        Args:
            plugin: Plugin instance to register
            is_fallback: If True, plugin is added as fallback (last priority)
        """
        plugin_name = type(plugin).__name__

        if is_fallback:
            # Fallback plugins go at the end
            self._plugins.append(plugin)
            logger.debug(f"Registered fallback plugin: {plugin_name}")
        else:
            # Specific plugins go before any fallback plugins
            # Find insertion point (before any fallback plugins)
            fallback_index = len(self._plugins)
            for i, existing_plugin in enumerate(self._plugins):
                if isinstance(existing_plugin, DefaultPlugin):
                    fallback_index = i
                    break

            self._plugins.insert(fallback_index, plugin)
            logger.debug(
                f"Registered plugin: {plugin_name} at position {fallback_index}"
            )

        # Store plugin metadata
        self._plugin_metadata[plugin_name] = {
            "class_name": plugin_name,
            "is_fallback": is_fallback,
            "module": type(plugin).__module__,
        }

    def get_plugins(self) -> List[Plugin]:
        """Get the ordered list of plugins."""
        return self._plugins.copy()

    def get_plugin_count(self) -> int:
        """Get the total number of registered plugins."""
        return len(self._plugins)

    def get_plugin_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all registered plugins."""
        return self._plugin_metadata.copy()

    def find_accepting_plugin(self, raw_input: str) -> Optional[Plugin]:
        """
        Find the first plugin that can accept the given input.

        Args:
            raw_input: Raw text content to test

        Returns:
            First plugin that accepts the input, or None if none accept
        """
        for plugin in self._plugins:
            try:
                if plugin.can_accept(raw_input):
                    plugin_name = type(plugin).__name__
                    logger.debug(f"Plugin {plugin_name} accepted input")
                    return plugin
            except Exception as e:
                plugin_name = type(plugin).__name__
                logger.warning(f"Plugin {plugin_name} failed during can_accept(): {e}")
                continue

        logger.debug("No plugin accepted the input")
        return None


class ImportOrchestrator:
    """
    High-level import orchestrator that coordinates the import process.

    This class provides the main entry point for importing files and handles:
    - Plugin selection and execution
    - Import status tracking
    - Error handling and recovery
    - Result aggregation for UI consumption
    """

    def __init__(self, plugin_manager: Optional[PluginManager] = None):
        """
        Initialize the import orchestrator.

        Args:
            plugin_manager: PluginManager instance (creates new one if not provided)
        """
        self.plugin_manager = plugin_manager or PluginManager()
        logger.info("Initialized ImportOrchestrator")

    def import_file(self, file_path: Path) -> ImportResult:
        """
        Import a single file using the plugin system.

        This method implements the core orchestration logic:
        1. Read file content
        2. Find accepting plugin
        3. Execute plugin conversion
        4. Apply defaults
        5. Return structured result

        Args:
            file_path: Path to the file to import

        Returns:
            ImportResult with detailed status and information
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return ImportResult(
                file_path=file_path,
                status=ImportStatus.ERROR,
                message=f"File does not exist: {file_path}",
                error_details="File not found",
            )

        logger.info(f"Starting import orchestration for: {file_path}")

        try:
            # Step 1: Read file content
            try:
                raw_input = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Try with different encoding
                raw_input = file_path.read_text(encoding="latin-1")
            except Exception as e:
                return ImportResult(
                    file_path=file_path,
                    status=ImportStatus.ERROR,
                    message=f"Cannot read file: {e}",
                    error_details=str(e),
                )

            # Step 2: Find accepting plugin
            accepting_plugin = self.plugin_manager.find_accepting_plugin(raw_input)

            if accepting_plugin is None:
                return ImportResult(
                    file_path=file_path,
                    status=ImportStatus.SKIPPED,
                    message="No plugin could handle the file",
                    error_details="Unknown format - no accepting plugin found",
                )

            plugin_name = type(accepting_plugin).__name__
            logger.info(f"Using plugin {plugin_name} for {file_path}")

            # Step 3: Execute plugin conversion
            try:
                raw_outputs = accepting_plugin.convert(raw_input, file_path)

                if not raw_outputs:
                    # Plugin executed successfully but found no conversations
                    return ImportResult(
                        file_path=file_path,
                        status=ImportStatus.IGNORED,
                        message="No conversations found in file",
                        plugin_used=plugin_name,
                        metadata={"reason": "no_conversations"},
                    )

                # Step 4: Apply defaults
                processed_outputs = [
                    ensure_defaults(md, file_path) for md in raw_outputs
                ]

                # Step 5: Return success result
                return ImportResult(
                    file_path=file_path,
                    status=ImportStatus.SUCCESS,
                    message=f"Successfully processed {len(processed_outputs)} conversation(s)",
                    plugin_used=plugin_name,
                    output_files=None,  # Will be set by higher-level import system
                    conversion_outputs=processed_outputs,  # Store the actual conversion outputs
                    metadata={
                        "conversation_count": len(processed_outputs),
                        "plugin_metadata": (processed_outputs[0].metadata or {}).get(
                            "plugin_metadata", {}
                        )
                        if processed_outputs
                        else {},
                    },
                )

            except Exception as e:
                logger.warning(
                    f"Plugin {plugin_name} failed to convert {file_path}: {e}"
                )
                return ImportResult(
                    file_path=file_path,
                    status=ImportStatus.ERROR,
                    message=f"Plugin conversion failed: {e}",
                    plugin_used=plugin_name,
                    error_details=str(e),
                )

        except Exception as e:
            logger.error(
                f"Unexpected error during import orchestration for {file_path}: {e}"
            )
            return ImportResult(
                file_path=file_path,
                status=ImportStatus.ERROR,
                message=f"Unexpected error: {e}",
                error_details=str(e),
            )

    def import_files(self, file_paths: List[Path]) -> List[ImportResult]:
        """
        Import multiple files and return aggregated results.

        Args:
            file_paths: List of file paths to import

        Returns:
            List of ImportResult objects
        """
        logger.info(f"Starting batch import of {len(file_paths)} files")
        results = []

        for file_path in file_paths:
            result = self.import_file(file_path)
            results.append(result)

        # Log summary
        success_count = len([r for r in results if r.status == ImportStatus.SUCCESS])
        ignored_count = len([r for r in results if r.status == ImportStatus.IGNORED])
        error_count = len([r for r in results if r.status == ImportStatus.ERROR])
        skipped_count = len([r for r in results if r.status == ImportStatus.SKIPPED])

        logger.info(
            f"Batch import complete: {success_count} success, {ignored_count} ignored, "
            f"{error_count} error, {skipped_count} skipped"
        )

        return results

    def get_plugin_info(self) -> Dict[str, Any]:
        """
        Get information about available plugins for debugging/UI.

        Returns:
            Dictionary with plugin information
        """
        return {
            "plugin_count": self.plugin_manager.get_plugin_count(),
            "plugins": self.plugin_manager.get_plugin_metadata(),
            "plugin_order": [
                type(p).__name__ for p in self.plugin_manager.get_plugins()
            ],
        }


# Convenience function for backward compatibility
def orchestrate_file_import(file_path: Path) -> ImportResult:
    """
    Convenience function to import a single file using the orchestrator.

    Args:
        file_path: Path to the file to import

    Returns:
        ImportResult with detailed status
    """
    orchestrator = ImportOrchestrator()
    return orchestrator.import_file(file_path)
