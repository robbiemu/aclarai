"""Tests for configuration functionality."""

import os
import sys
from pathlib import Path

# Add the service directory to the path for testing
service_dir = Path(__file__).parent.parent
sys.path.insert(0, str(service_dir))
from aclarai_ui.config import UIConfig


class TestUIConfig:
    """Test the UIConfig class functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UIConfig()
        assert config.tier1_path == "vault/tier1"
        assert config.summaries_path == "vault"
        assert config.concepts_path == "vault"
        assert config.logs_path == ".aclarai/import_logs"
        assert config.server_host == "0.0.0.0"
        assert config.server_port == 7860
        assert config.debug_mode is False

    def test_from_env_with_defaults(self):
        """Test configuration from environment with default values."""
        # Clear any existing environment variables
        env_vars = [
            "ACLARAI_TIER1_PATH",
            "ACLARAI_SUMMARIES_PATH",
            "ACLARAI_CONCEPTS_PATH",
            "ACLARAI_LOGS_PATH",
            "ACLARAI_UI_HOST",
            "ACLARAI_UI_PORT",
            "ACLARAI_UI_DEBUG",
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]
        config = UIConfig.from_env()
        assert config.tier1_path == "vault/tier1"
        assert config.summaries_path == "vault"
        assert config.concepts_path == "vault"
        assert config.logs_path == ".aclarai/import_logs"
        assert config.server_host == "0.0.0.0"
        assert config.server_port == 7860
        assert config.debug_mode is False

    def test_from_env_with_custom_values(self):
        """Test configuration from environment with custom values."""
        # Set custom environment variables
        os.environ["ACLARAI_TIER1_PATH"] = "custom/tier1"
        os.environ["ACLARAI_SUMMARIES_PATH"] = "custom/summaries"
        os.environ["ACLARAI_CONCEPTS_PATH"] = "custom/concepts"
        os.environ["ACLARAI_LOGS_PATH"] = "custom/logs"
        os.environ["ACLARAI_UI_HOST"] = "127.0.0.1"
        os.environ["ACLARAI_UI_PORT"] = "8080"
        os.environ["ACLARAI_UI_DEBUG"] = "true"
        try:
            config = UIConfig.from_env()
            assert config.tier1_path == "custom/tier1"
            assert config.summaries_path == "custom/summaries"
            assert config.concepts_path == "custom/concepts"
            assert config.logs_path == "custom/logs"
            assert config.server_host == "127.0.0.1"
            assert config.server_port == 8080
            assert config.debug_mode is True
        finally:
            # Clean up environment variables
            for var in [
                "ACLARAI_TIER1_PATH",
                "ACLARAI_SUMMARIES_PATH",
                "ACLARAI_CONCEPTS_PATH",
                "ACLARAI_LOGS_PATH",
                "ACLARAI_UI_HOST",
                "ACLARAI_UI_PORT",
                "ACLARAI_UI_DEBUG",
            ]:
                if var in os.environ:
                    del os.environ[var]

    def test_get_next_steps_links(self):
        """Test generation of next steps links."""
        config = UIConfig(tier1_path="custom/tier1", logs_path="custom/logs")
        links = config.get_next_steps_links()
        assert links["vault"] == "./custom/tier1/"
        assert links["logs"] == "./custom/logs/"

    def test_debug_mode_false_variations(self):
        """Test that debug mode handles various false values correctly."""
        false_values = ["false", "False", "FALSE", "0", "no", ""]
        for value in false_values:
            os.environ["ACLARAI_UI_DEBUG"] = value
            try:
                config = UIConfig.from_env()
                assert config.debug_mode is False, f"Failed for value: {value}"
            finally:
                if "aclarai_UI_DEBUG" in os.environ:
                    del os.environ["ACLARAI_UI_DEBUG"]

    def test_port_number_conversion(self):
        """Test that port numbers are correctly converted to integers."""
        os.environ["ACLARAI_UI_PORT"] = "9000"
        try:
            config = UIConfig.from_env()
            assert config.server_port == 9000
            assert isinstance(config.server_port, int)
        finally:
            if "aclarai_UI_PORT" in os.environ:
                del os.environ["ACLARAI_UI_PORT"]
