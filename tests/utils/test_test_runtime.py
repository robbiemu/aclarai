"""
Tests for test runtime detection utilities.
"""
import os

import pytest

from tests.utils.test_runtime import is_running_under_pytest


def test_is_running_under_pytest():
    """Test that the pytest detection function works correctly."""
    # Since this test is running under pytest, it should return True
    assert is_running_under_pytest() is True

    # Test with environment variable explicitly unset
    original_value = os.environ.pop('PYTEST_CURRENT_TEST', None)
    try:
        assert is_running_under_pytest() is False
    finally:
        # Restore the environment variable
        if original_value is not None:
            os.environ['PYTEST_CURRENT_TEST'] = original_value
