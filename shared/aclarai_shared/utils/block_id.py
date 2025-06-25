"""
Block ID generation utilities for aclarai plugins.
Provides shared functionality for generating unique block identifiers
that can be used across different plugins and components.
"""

import random
import string
from typing import Callable


def generate_unique_block_id(used_ids: set) -> str:
    """
    Generate a unique block ID that hasn't been used.
    Args:
        used_ids: Set of already used block IDs to avoid duplicates
    Returns:
        Unique block ID in format 'blk_xxxxxx' where x is alphanumeric
    Raises:
        RuntimeError: If unable to generate unique ID after max attempts
    """
    chars = string.ascii_lowercase + string.digits
    max_attempts = 100  # Prevent infinite loop
    for _ in range(max_attempts):
        # Using random for non-cryptographic block ID generation
        suffix = "".join(random.choices(chars, k=6))  # nosec B311
        block_id = f"blk_{suffix}"
        if block_id not in used_ids:
            return block_id
    # Fallback - this should be extremely rare
    raise RuntimeError(
        f"Unable to generate unique block ID after {max_attempts} attempts"
    )


def create_block_id_generator() -> Callable[[], str]:
    """
    Create a generator function for block IDs that maintains state.
    Returns:
        Generator function that yields unique block IDs
    """
    used_ids: set[str] = set()

    def generate() -> str:
        """Generate next unique block ID."""
        block_id = generate_unique_block_id(used_ids)
        used_ids.add(block_id)
        return block_id

    return generate
