"""Configuration management for ClarifAI UI service.

This module handles configuration settings for the UI service, including
paths and other configurable parameters. Settings can be provided via
environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass
from typing import Dict
from clarifai_shared import ClarifAIConfig


@dataclass
class UIConfig:
    """Configuration settings for the ClarifAI UI service."""

    # Vault paths configuration (following docs/arch/design_config_panel.md)
    tier1_path: str = "vault/tier1"
    summaries_path: str = "vault"
    concepts_path: str = "vault"
    logs_path: str = ".clarifai/import_logs"

    # UI-specific settings
    server_host: str = "0.0.0.0"
    server_port: int = 7860
    debug_mode: bool = False

    # Shared configuration access
    _shared_config: ClarifAIConfig = None

    @classmethod
    def from_env(cls) -> "UIConfig":
        """Create configuration from environment variables with defaults."""
        return cls(
            # Vault paths (following documented patterns)
            tier1_path=os.getenv("CLARIFAI_TIER1_PATH", "vault/tier1"),
            summaries_path=os.getenv("CLARIFAI_SUMMARIES_PATH", "vault"),
            concepts_path=os.getenv("CLARIFAI_CONCEPTS_PATH", "vault"),
            logs_path=os.getenv("CLARIFAI_LOGS_PATH", ".clarifai/import_logs"),
            # Server configuration
            server_host=os.getenv("CLARIFAI_UI_HOST", "0.0.0.0"),
            server_port=int(os.getenv("CLARIFAI_UI_PORT", "7860")),
            debug_mode=os.getenv("CLARIFAI_UI_DEBUG", "false").lower() == "true",
        )

    @property
    def shared_config(self) -> ClarifAIConfig:
        """Get shared configuration (lazy loaded)."""
        if self._shared_config is None:
            from clarifai_shared import load_config

            self._shared_config = load_config(
                validate=False
            )  # UI can work without all DB credentials
        return self._shared_config

    def get_next_steps_links(self) -> Dict[str, str]:
        """Generate next steps links based on configured paths."""
        return {"vault": f"./{self.tier1_path}/", "logs": f"./{self.logs_path}/"}


# Global configuration instance (loaded once at startup)
config = UIConfig.from_env()
