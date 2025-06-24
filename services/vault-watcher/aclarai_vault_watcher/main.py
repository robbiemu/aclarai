"""
aclarai Vault Watcher Service
This service monitors the vault directory for changes in Markdown files
and emits dirty block IDs for processing.
"""

import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Set

from aclarai_shared import aclaraiConfig, load_config

from .block_parser import BlockParser, aclaraiBlock
from .file_watcher import BatchedFileWatcher
from .rabbitmq_publisher import DirtyBlockPublisher


class VaultWatcherService:
    """Main service class that orchestrates file watching and block processing."""

    def __init__(self, config: aclaraiConfig):
        """
        Initialize the vault watcher service.
        Args:
            config: aclarai configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Initialize components
        self.block_parser = BlockParser()
        self.file_watcher = BatchedFileWatcher(
            vault_path=config.paths.vault,
            batch_interval=config.vault_watcher.batch_interval,
            max_batch_size=config.vault_watcher.max_batch_size,
            callback=self._handle_file_changes,
        )
        # Initialize RabbitMQ publisher
        self.publisher = DirtyBlockPublisher(
            rabbitmq_host=config.rabbitmq_host,
            rabbitmq_port=config.rabbitmq_port,
            rabbitmq_user=os.getenv("RABBITMQ_USER"),
            rabbitmq_password=os.getenv("RABBITMQ_PASSWORD"),
            queue_name=config.vault_watcher.queue_name,
        )
        # State management for tracking known blocks
        self._known_blocks: Dict[str, List[aclaraiBlock]] = {}
        self._lock = Lock()
        self.logger.info(
            "vault_watcher.VaultWatcherService: Initialized service",
            extra={
                "service": "vault-watcher",
                "filename.function_name": "main.VaultWatcherService.__init__",
                "vault_path": config.paths.vault,
            },
        )

    def start(self):
        """Start the vault watcher service."""
        try:
            # Connect to RabbitMQ
            self.publisher.connect()
            # Perform initial scan of existing files
            self._perform_initial_scan()
            # Start file watcher
            self.file_watcher.start()
            self.logger.info(
                "vault_watcher.VaultWatcherService: Service started successfully",
                extra={
                    "service": "vault-watcher",
                    "filename.function_name": "main.VaultWatcherService.start",
                },
            )
        except Exception as e:
            self.logger.error(
                f"vault_watcher.VaultWatcherService: Failed to start service: {e}",
                extra={
                    "service": "vault-watcher",
                    "filename.function_name": "main.VaultWatcherService.start",
                    "error": str(e),
                },
            )
            raise

    def stop(self):
        """Stop the vault watcher service."""
        try:
            # Stop file watcher
            self.file_watcher.stop()
            # Disconnect from RabbitMQ
            self.publisher.disconnect()
            self.logger.info(
                "vault_watcher.VaultWatcherService: Service stopped",
                extra={
                    "service": "vault-watcher",
                    "filename.function_name": "main.VaultWatcherService.stop",
                },
            )
        except Exception as e:
            self.logger.error(
                f"vault_watcher.VaultWatcherService: Error stopping service: {e}",
                extra={
                    "service": "vault-watcher",
                    "filename.function_name": "main.VaultWatcherService.stop",
                    "error": str(e),
                },
            )

    def _perform_initial_scan(self):
        """Scan existing files to build initial state."""
        vault_path = Path(self.config.paths.vault)
        if not vault_path.exists():
            self.logger.warning(
                f"vault_watcher.VaultWatcherService: Vault path does not exist: {vault_path}",
                extra={
                    "service": "vault-watcher",
                    "filename.function_name": "main.VaultWatcherService._perform_initial_scan",
                    "vault_path": str(vault_path),
                },
            )
            return
        # Find all Markdown files
        md_files = list(vault_path.rglob("*.md"))
        self.logger.info(
            f"vault_watcher.VaultWatcherService: Performing initial scan of {len(md_files)} files",
            extra={
                "service": "vault-watcher",
                "filename.function_name": "main.VaultWatcherService._perform_initial_scan",
                "file_count": len(md_files),
            },
        )
        # Parse each file and store the blocks
        for file_path in md_files:
            try:
                blocks = self.block_parser.parse_file(file_path)
                with self._lock:
                    self._known_blocks[str(file_path)] = blocks
                if blocks:
                    self.logger.debug(
                        f"vault_watcher.VaultWatcherService: Found {len(blocks)} blocks in {file_path.name}",
                        extra={
                            "service": "vault-watcher",
                            "filename.function_name": "main.VaultWatcherService._perform_initial_scan",
                            "file_path": str(file_path),
                            "block_count": len(blocks),
                        },
                    )
            except Exception as e:
                self.logger.error(
                    f"vault_watcher.VaultWatcherService: Error scanning file {file_path}: {e}",
                    extra={
                        "service": "vault-watcher",
                        "filename.function_name": "main.VaultWatcherService._perform_initial_scan",
                        "file_path": str(file_path),
                        "error": str(e),
                    },
                )

    def _handle_file_changes(
        self, created: Set[Path], modified: Set[Path], deleted: Set[Path]
    ):
        """
        Handle batched file change events.
        Args:
            created: Set of newly created file paths
            modified: Set of modified file paths
            deleted: Set of deleted file paths
        """
        self.logger.info(
            "vault_watcher.VaultWatcherService: Processing file changes",
            extra={
                "service": "vault-watcher",
                "filename.function_name": "main.VaultWatcherService._handle_file_changes",
                "created_count": len(created),
                "modified_count": len(modified),
                "deleted_count": len(deleted),
            },
        )
        # Process each type of change
        for file_path in created:
            self._handle_file_created(file_path)
        for file_path in modified:
            self._handle_file_modified(file_path)
        for file_path in deleted:
            self._handle_file_deleted(file_path)

    def _handle_file_created(self, file_path: Path):
        """Handle a newly created file."""
        try:
            blocks = self.block_parser.parse_file(file_path)
            with self._lock:
                self._known_blocks[str(file_path)] = blocks
            # All blocks in a new file are considered "added"
            if blocks:
                dirty_blocks = {
                    "added": [
                        {
                            "aclarai_id": block.aclarai_id,
                            "version": block.version,
                            "content_hash": block.content_hash,
                            "block_type": block.block_type,
                        }
                        for block in blocks
                    ],
                    "modified": [],
                    "deleted": [],
                }
                self.publisher.publish_dirty_blocks(file_path, dirty_blocks)
                self.logger.info(
                    f"vault_watcher.VaultWatcherService: Created file with {len(blocks)} blocks",
                    extra={
                        "service": "vault-watcher",
                        "filename.function_name": "main.VaultWatcherService._handle_file_created",
                        "file_path": str(file_path),
                        "block_count": len(blocks),
                    },
                )
        except Exception as e:
            self.logger.error(
                f"vault_watcher.VaultWatcherService: Error handling created file {file_path}: {e}",
                extra={
                    "service": "vault-watcher",
                    "filename.function_name": "main.VaultWatcherService._handle_file_created",
                    "file_path": str(file_path),
                    "error": str(e),
                },
            )

    def _handle_file_modified(self, file_path: Path):
        """Handle a modified file."""
        try:
            new_blocks = self.block_parser.parse_file(file_path)
            with self._lock:
                old_blocks = self._known_blocks.get(str(file_path), [])
                self._known_blocks[str(file_path)] = new_blocks
            # Compare old and new blocks to find differences
            dirty_blocks = self.block_parser.compare_blocks(old_blocks, new_blocks)
            # Only publish if there are actual changes
            if any(dirty_blocks[key] for key in ["added", "modified", "deleted"]):
                self.publisher.publish_dirty_blocks(file_path, dirty_blocks)
                total_changes = sum(
                    len(dirty_blocks[key]) for key in ["added", "modified", "deleted"]
                )
                self.logger.info(
                    f"vault_watcher.VaultWatcherService: Modified file with {total_changes} block changes",
                    extra={
                        "service": "vault-watcher",
                        "filename.function_name": "main.VaultWatcherService._handle_file_modified",
                        "file_path": str(file_path),
                        "added_count": len(dirty_blocks["added"]),
                        "modified_count": len(dirty_blocks["modified"]),
                        "deleted_count": len(dirty_blocks["deleted"]),
                    },
                )
        except Exception as e:
            self.logger.error(
                f"vault_watcher.VaultWatcherService: Error handling modified file {file_path}: {e}",
                extra={
                    "service": "vault-watcher",
                    "filename.function_name": "main.VaultWatcherService._handle_file_modified",
                    "file_path": str(file_path),
                    "error": str(e),
                },
            )

    def _handle_file_deleted(self, file_path: Path):
        """Handle a deleted file."""
        try:
            with self._lock:
                old_blocks = self._known_blocks.pop(str(file_path), [])
            # All blocks in the deleted file are considered "deleted"
            if old_blocks:
                dirty_blocks = {
                    "added": [],
                    "modified": [],
                    "deleted": [
                        {
                            "aclarai_id": block.aclarai_id,
                            "version": block.version,
                            "content_hash": block.content_hash,
                            "block_type": block.block_type,
                        }
                        for block in old_blocks
                    ],
                }
                self.publisher.publish_dirty_blocks(file_path, dirty_blocks)
                self.logger.info(
                    f"vault_watcher.VaultWatcherService: Deleted file with {len(old_blocks)} blocks",
                    extra={
                        "service": "vault-watcher",
                        "filename.function_name": "main.VaultWatcherService._handle_file_deleted",
                        "file_path": str(file_path),
                        "block_count": len(old_blocks),
                    },
                )
        except Exception as e:
            self.logger.error(
                f"vault_watcher.VaultWatcherService: Error handling deleted file {file_path}: {e}",
                extra={
                    "service": "vault-watcher",
                    "filename.function_name": "main.VaultWatcherService._handle_file_deleted",
                    "file_path": str(file_path),
                    "error": str(e),
                },
            )


def main():
    """Main entry point for the Vault Watcher service."""
    service = None
    try:
        # Load configuration with validation
        config = load_config(validate=True)
        logger = logging.getLogger(__name__)
        logger.info(
            "vault_watcher.main: Starting aclarai Vault Watcher service",
            extra={"service": "vault-watcher", "filename.function_name": "main.main"},
        )
        # Log configuration details
        logger.info(
            "vault_watcher.main: Configuration loaded",
            extra={
                "service": "vault-watcher",
                "filename.function_name": "main.main",
                "vault_path": config.paths.vault,
                "rabbitmq_host": config.rabbitmq_host,
            },
        )
        # Ensure vault directory exists
        if not os.path.exists(config.paths.vault):
            logger.warning(
                f"vault_watcher.main: Vault path does not exist, creating: {config.paths.vault}",
                extra={
                    "service": "vault-watcher",
                    "filename.function_name": "main.main",
                    "vault_path": config.paths.vault,
                },
            )
            os.makedirs(config.paths.vault, exist_ok=True)
        # Initialize and start the service
        service = VaultWatcherService(config)
        service.start()
        logger.info(
            "vault_watcher.main: Vault Watcher service is monitoring for changes",
            extra={"service": "vault-watcher", "filename.function_name": "main.main"},
        )
        # Keep the service running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info(
                "vault_watcher.main: Received shutdown signal",
                extra={
                    "service": "vault-watcher",
                    "filename.function_name": "main.main",
                },
            )
    except ValueError as e:
        # Configuration validation error
        logging.error(
            f"vault_watcher.main: Configuration error: {e}",
            extra={
                "service": "vault-watcher",
                "filename.function_name": "main.main",
                "error": str(e),
            },
        )
        logging.error(
            "Please check your .env file and ensure all required variables are set."
        )
        raise
    except KeyboardInterrupt:
        logger.info(
            "vault_watcher.main: Shutting down Vault Watcher service",
            extra={"service": "vault-watcher", "filename.function_name": "main.main"},
        )
    except Exception as e:
        logging.error(
            f"vault_watcher.main: Error in Vault Watcher service: {e}",
            extra={
                "service": "vault-watcher",
                "filename.function_name": "main.main",
                "error": str(e),
            },
        )
        raise
    finally:
        # Ensure clean shutdown
        if service is not None:
            try:
                service.stop()
            except Exception as e:
                logging.error(
                    f"vault_watcher.main: Error during service shutdown: {e}",
                    extra={
                        "service": "vault-watcher",
                        "filename.function_name": "main.main",
                        "error": str(e),
                    },
                )


if __name__ == "__main__":
    main()
