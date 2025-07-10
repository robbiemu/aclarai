#!/usr/bin/env python3
"""Standalone configuration panel launcher for aclarai.
This script launches the configuration panel interface independently,
allowing users to configure aclarai system parameters.
Usage:
    python -m aclarai_ui.config_launcher
"""

import logging
import sys
from pathlib import Path

# Add the services directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from aclarai_ui.config_panel import create_configuration_panel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("aclarai-ui.config_launcher")


def main():
    """Launch the configuration panel."""
    try:
        logger.info("Starting aclarai Configuration Panel")
        interface = create_configuration_panel()
        logger.info("Launching configuration panel on http://localhost:7861")
        interface.launch(
            server_name="0.0.0.0", server_port=7861, share=False, debug=True
        )
    except Exception as e:
        logger.error(f"Failed to start configuration panel: {e}")
        raise


if __name__ == "__main__":
    main()
