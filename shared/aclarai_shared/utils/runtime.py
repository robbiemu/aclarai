"""Runtime detection utilities."""

import os


def is_running_under_pytest() -> bool:
    """Detect if code is running under pytest.
    
    This utility function checks if the code is currently being executed
    by pytest by looking for the PYTEST_CURRENT_TEST environment variable.
    
    Returns:
        bool: True if running under pytest, False otherwise
    """
    return "PYTEST_CURRENT_TEST" in os.environ
