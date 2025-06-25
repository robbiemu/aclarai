"""
Pause controller for system-wide automation control.

This module provides a simple file-based mechanism to pause/resume all automated jobs
in the aclarai system. The state is determined by the presence/absence of a
.aclarai_pause file in the vault root.
"""

import logging
from pathlib import Path

from aclarai_shared.config import load_config

logger = logging.getLogger(__name__)

PAUSE_FILE = ".aclarai_pause"


def get_pause_file_path() -> Path:
    """Get the absolute path to the pause file in the vault root."""
    config = load_config(validate=False)  # validate=False is safe here
    vault_root = Path(config.vault_path)
    return vault_root / PAUSE_FILE


def is_paused() -> bool:
    """Check if automation is currently paused.

    Returns:
        bool: True if automation is paused, False otherwise.
    """
    return get_pause_file_path().exists()


def pause() -> None:
    """Pause all automation by creating the pause file."""
    pause_file = get_pause_file_path()
    try:
        pause_file.touch()
        logger.info("Automation paused.")
    except Exception as e:
        logger.error(f"Failed to pause automation: {e}")
        raise


def resume() -> None:
    """Resume automation by removing the pause file."""
    pause_file = get_pause_file_path()
    try:
        if pause_file.exists():
            pause_file.unlink()
            logger.info("Automation resumed.")
    except Exception as e:
        logger.error(f"Failed to resume automation: {e}")
        raise
