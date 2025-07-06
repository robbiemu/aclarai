"""
Test runtime utilities for detecting test environment conditions.
"""

import os


def is_running_under_pytest():
    """
    Check if code is currently running under pytest.

    Returns:
        bool: True if running under pytest, False otherwise

    This function can be used to modify behavior during test runs, such as
    allowing certain operations that would normally be restricted in production.
    """
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))
