"""
Configuration integration for Claimify pipeline.
This module extracts and applies configuration options for the Claimify pipeline
from the main aclarai configuration file.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .data_models import ClaimifyConfig

logger = logging.getLogger(__name__)


def load_claimify_config_from_yaml(config_data: Dict[str, Any]) -> ClaimifyConfig:
    """
    Load ClaimifyConfig from YAML configuration data.
    Args:
        config_data: Dictionary containing parsed YAML configuration
    Returns:
        ClaimifyConfig instance with settings from YAML
    """
    # Context window settings (following design_config_panel.md structure)
    window_config = config_data.get("window", {})
    claimify_window = window_config.get("claimify", {})
    context_window_p = claimify_window.get("p", 3)
    context_window_f = claimify_window.get("f", 1)
    # Model settings (following design_config_panel.md structure)
    model_config = config_data.get("model", {})
    claimify_models = model_config.get("claimify", {})
    selection_model = claimify_models.get("selection")
    disambiguation_model = claimify_models.get("disambiguation")
    decomposition_model = claimify_models.get("decomposition")
    claimify_default = claimify_models.get("default")
    # Fall back to global processing config if claimify-specific values not set
    if not claimify_default:
        claimify_default = model_config.get("fallback_plugin", "gpt-3.5-turbo")
    # Processing settings (check global processing first, then claimify-specific)
    processing_config = config_data.get("processing", {})
    claimify_processing = processing_config.get("claimify", {})
    max_retries = claimify_processing.get("max_retries", 3)
    timeout_seconds = processing_config.get("timeout_seconds", 30)
    temperature = processing_config.get("temperature", 0.1)
    max_tokens = processing_config.get("max_tokens", 1000)
    # Threshold settings (to be implemented when threshold evaluation is added)
    # For now, these return default values as the thresholds are not yet configured
    selection_confidence_threshold = 0.5
    disambiguation_confidence_threshold = 0.5
    decomposition_confidence_threshold = 0.5
    # Logging settings
    logging_config = claimify_processing.get("logging", {})
    log_decisions = logging_config.get("log_decisions", True)
    log_transformations = logging_config.get("log_transformations", True)
    log_timing = logging_config.get("log_timing", True)
    return ClaimifyConfig(
        context_window_p=context_window_p,
        context_window_f=context_window_f,
        selection_model=selection_model,
        disambiguation_model=disambiguation_model,
        decomposition_model=decomposition_model,
        default_model=claimify_default,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
        selection_confidence_threshold=selection_confidence_threshold,
        disambiguation_confidence_threshold=disambiguation_confidence_threshold,
        decomposition_confidence_threshold=decomposition_confidence_threshold,
        log_decisions=log_decisions,
        log_transformations=log_transformations,
        log_timing=log_timing,
    )


def load_claimify_config_from_file(config_file: Optional[str] = None) -> ClaimifyConfig:
    """
    Load ClaimifyConfig from YAML configuration file.
    Args:
        config_file: Path to YAML config file (optional, will search for default)
    Returns:
        ClaimifyConfig instance
    """
    try:
        import yaml
    except ImportError:
        logger.warning(
            "[config_integration.load_claimify_config_from_file] PyYAML not available, using default Claimify configuration"
        )
        return ClaimifyConfig()
    if config_file is None:
        # Look for default config file
        current_path = Path.cwd()
        search_paths = []
        # Priority 1: settings directory
        for path in [current_path] + list(current_path.parents):
            search_paths.append(path / "settings" / "aclarai.config.yaml")
        # Priority 2: root level in current and parent directories
        for path in [current_path] + list(current_path.parents):
            search_paths.append(path / "aclarai.config.yaml")
        for config_path in search_paths:
            if config_path.exists():
                config_file = str(config_path)
                break
    if config_file and Path(config_file).exists():
        logger.info(
            f"[config_integration.load_claimify_config_from_file] Loading Claimify configuration from {config_file}"
        )
        try:
            with open(config_file, "r") as f:
                config_data = yaml.safe_load(f) or {}
            return load_claimify_config_from_yaml(config_data)
        except Exception as e:
            logger.error(
                f"[config_integration.load_claimify_config_from_file] Failed to load Claimify config from {config_file}: {e}"
            )
            return ClaimifyConfig()
    else:
        logger.info(
            "[config_integration.load_claimify_config_from_file] No configuration file found, using default Claimify configuration"
        )
        return ClaimifyConfig()


def get_model_config_for_stage(config_data: Dict[str, Any], stage: str) -> str:
    """
    Get the configured model for a specific Claimify stage.
    Args:
        config_data: Configuration dictionary
        stage: Stage name ("selection", "disambiguation", "decomposition")
    Returns:
        Model name for the stage
    """
    model_config = config_data.get("model", {})
    claimify_models = model_config.get("claimify", {})

    stage_model = claimify_models.get(stage)
    if isinstance(stage_model, str):
        return stage_model

    claimify_default = claimify_models.get("default")
    if isinstance(claimify_default, str):
        return claimify_default

    fallback_plugin = model_config.get("fallback_plugin")
    if isinstance(fallback_plugin, str):
        return fallback_plugin

    raise ValueError(f"No valid model configuration found for stage '{stage}'")
