"""
File watcher for monitoring vault Markdown files.
This module implements file system monitoring with batching and throttling
to efficiently handle multiple file changes, following the requirements
in sprint_4-Vault_file_watcher.md.
"""

import logging
from pathlib import Path
from threading import Lock, Timer
from typing import Callable, Optional, Set

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class BatchedFileWatcher:
    """
    File watcher with batching and throttling capabilities.
    Monitors Markdown files for changes and batches events to avoid
    overwhelming the system during bulk operations like git pulls.
    """

    def __init__(
        self,
        vault_path: str,
        batch_interval: float = 2.0,
        max_batch_size: int = 100,
        callback: Optional[Callable[[Set[Path], Set[Path], Set[Path]], None]] = None,
    ):
        """
        Initialize the file watcher.
        Args:
            vault_path: Root path of the vault to monitor
            batch_interval: Seconds to wait before processing batched events
            max_batch_size: Maximum events in a batch before forcing processing
            callback: Function to call with (created, modified, deleted) file sets
        """
        self.vault_path = Path(vault_path)
        self.batch_interval = batch_interval
        self.max_batch_size = max_batch_size
        self.callback = callback
        # Event batching state
        self._pending_created: Set[Path] = set()
        self._pending_modified: Set[Path] = set()
        self._pending_deleted: Set[Path] = set()
        self._batch_timer: Optional[Timer] = None
        self._lock = Lock()
        # Watchdog components
        self._observer: Optional[Observer] = None
        self._event_handler = VaultFileEventHandler(self)
        logger.info(
            "vault_watcher.BatchedFileWatcher: Initialized file watcher",
            extra={
                "service": "vault-watcher",
                "filename.function_name": "file_watcher.__init__",
                "vault_path": str(self.vault_path),
                "batch_interval": self.batch_interval,
                "max_batch_size": self.max_batch_size,
            },
        )

    def start(self):
        """Start monitoring the vault directory."""
        if self._observer is not None:
            logger.warning(
                "vault_watcher.BatchedFileWatcher: Watcher already started",
                extra={
                    "service": "vault-watcher",
                    "filename.function_name": "file_watcher.start",
                },
            )
            return
        self._observer = Observer()
        # Monitor the entire vault directory recursively
        self._observer.schedule(
            self._event_handler, str(self.vault_path), recursive=True
        )
        self._observer.start()
        logger.info(
            "vault_watcher.BatchedFileWatcher: Started monitoring vault directory",
            extra={
                "service": "vault-watcher",
                "filename.function_name": "file_watcher.start",
                "vault_path": str(self.vault_path),
            },
        )

    def stop(self):
        """Stop monitoring and clean up resources."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        # Cancel any pending batch timer
        with self._lock:
            if self._batch_timer is not None:
                self._batch_timer.cancel()
                self._batch_timer = None
        # Process any remaining events
        self._process_batch()
        logger.info(
            "vault_watcher.BatchedFileWatcher: Stopped monitoring",
            extra={
                "service": "vault-watcher",
                "filename.function_name": "file_watcher.stop",
            },
        )

    def _add_event(self, event_type: str, file_path: Path):
        """Add an event to the pending batch."""
        with self._lock:
            # Remove from other event types (file can only have one final state)
            if event_type == "created":
                self._pending_modified.discard(file_path)
                self._pending_deleted.discard(file_path)
                self._pending_created.add(file_path)
            elif event_type == "modified":
                # Don't add to modified if it's already created in this batch
                if file_path not in self._pending_created:
                    self._pending_deleted.discard(file_path)
                    self._pending_modified.add(file_path)
            elif event_type == "deleted":
                self._pending_created.discard(file_path)
                self._pending_modified.discard(file_path)
                self._pending_deleted.add(file_path)
            total_events = (
                len(self._pending_created)
                + len(self._pending_modified)
                + len(self._pending_deleted)
            )
            logger.debug(
                f"vault_watcher.BatchedFileWatcher: Added {event_type} event",
                extra={
                    "service": "vault-watcher",
                    "filename.function_name": "file_watcher._add_event",
                    "file_path": str(file_path),
                    "event_type": event_type,
                    "total_pending": total_events,
                },
            )
            # Check if we should process the batch immediately
            if total_events >= self.max_batch_size:
                self._cancel_timer()
                # Don't process batch while holding the lock - schedule it to run
                # after we release the lock to avoid deadlock
                should_process = True
            else:
                self._reset_timer()
                should_process = False
        # Process batch outside of lock to avoid deadlock
        if should_process:
            self._process_batch()

    def _reset_timer(self):
        """Reset the batch processing timer."""
        self._cancel_timer()
        self._batch_timer = Timer(self.batch_interval, self._process_batch)
        self._batch_timer.start()

    def _cancel_timer(self):
        """Cancel the current batch timer if it exists."""
        if self._batch_timer is not None:
            self._batch_timer.cancel()
            self._batch_timer = None

    def _process_batch(self):
        """Process the current batch of events."""
        with self._lock:
            # Capture current events and clear the pending sets
            created = self._pending_created.copy()
            modified = self._pending_modified.copy()
            deleted = self._pending_deleted.copy()
            self._pending_created.clear()
            self._pending_modified.clear()
            self._pending_deleted.clear()
            self._cancel_timer()
        if not (created or modified or deleted):
            return  # No events to process
        total_events = len(created) + len(modified) + len(deleted)
        logger.info(
            "vault_watcher.BatchedFileWatcher: Processing batch of file events",
            extra={
                "service": "vault-watcher",
                "filename.function_name": "file_watcher._process_batch",
                "created_count": len(created),
                "modified_count": len(modified),
                "deleted_count": len(deleted),
                "total_events": total_events,
            },
        )
        # Call the callback if it's set
        if self.callback is not None:
            try:
                self.callback(created, modified, deleted)
            except Exception as e:
                logger.error(
                    f"vault_watcher.BatchedFileWatcher: Error in callback: {e}",
                    extra={
                        "service": "vault-watcher",
                        "filename.function_name": "file_watcher._process_batch",
                        "error": str(e),
                    },
                )


class VaultFileEventHandler(FileSystemEventHandler):
    """Event handler for vault file system events."""

    def __init__(self, watcher: BatchedFileWatcher):
        """
        Initialize the event handler.
        Args:
            watcher: The BatchedFileWatcher instance to notify of events
        """
        self.watcher = watcher
        super().__init__()

    def on_created(self, event: FileSystemEvent):
        """Handle file creation events."""
        if self._should_process_event(event):
            self.watcher._add_event("created", Path(str(event.src_path)))

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events."""
        if self._should_process_event(event):
            self.watcher._add_event("modified", Path(str(event.src_path)))

    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion events."""
        if self._should_process_event(event):
            self.watcher._add_event("deleted", Path(str(event.src_path)))

    def _should_process_event(self, event: FileSystemEvent) -> bool:
        """
        Determine if an event should be processed.
        Args:
            event: The file system event
        Returns:
            True if the event should be processed, False otherwise
        """
        # Only process file events (not directories)
        if event.is_directory:
            return False
        # Only process Markdown files
        path = Path(str(event.src_path))
        if path.suffix.lower() != ".md":
            return False
        # Ignore temporary files and hidden files
        if path.name.startswith(".") or path.name.endswith(".tmp"):
            return False
        # Check if the file is within the vault directory
        try:
            path.relative_to(self.watcher.vault_path)
            return True
        except ValueError:
            # File is not within the vault directory
            return False
