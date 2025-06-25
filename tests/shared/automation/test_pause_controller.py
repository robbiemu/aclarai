"""
Unit tests for the pause_controller module.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

from aclarai_shared.automation.pause_controller import (
    is_paused,
    pause,
    resume,
)


class TestPauseController(unittest.TestCase):
    @patch("aclarai_shared.automation.pause_controller.get_pause_file_path")
    def test_is_paused(self, mock_get_pause_file_path):
        """Test is_paused function."""
        # Mock the file path and return a Path object
        mock_get_pause_file_path.return_value = Path("/tmp/.aclarai_pause")

        # Test when the pause file does not exist
        with patch.object(Path, "exists", return_value=False):
            self.assertFalse(is_paused())

        # Test when the pause file exists
        with patch.object(Path, "exists", return_value=True):
            self.assertTrue(is_paused())

    @patch("aclarai_shared.automation.pause_controller.get_pause_file_path")
    def test_pause(self, mock_get_pause_file_path):
        """Test pause function."""
        pause_file_path = Path("/tmp/.aclarai_pause")
        mock_get_pause_file_path.return_value = pause_file_path

        # Test creating the pause file
        with patch("pathlib.Path.touch") as mock_touch:
            pause()
            mock_touch.assert_called_once()

    @patch("aclarai_shared.automation.pause_controller.get_pause_file_path")
    def test_resume(self, mock_get_pause_file_path):
        """Test resume function."""
        pause_file_path = Path("/tmp/.aclarai_pause")
        mock_get_pause_file_path.return_value = pause_file_path

        # Test removing the pause file
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "unlink") as mock_unlink,
        ):
            resume()
            mock_unlink.assert_called_once()


if __name__ == "__main__":
    unittest.main()
