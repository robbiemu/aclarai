"""
Utility for installing and restoring default prompt YAML files.

This module provides functions to install default prompt templates to the user's
prompts directory, making them easily customizable while providing a way to
restore defaults when needed.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def install_default_prompt(
    template_name: str = "conversation_extraction",
    force: bool = False,
    prompts_dir: Optional[Path] = None,
) -> bool:
    """
    Install or restore a default prompt YAML file to the user's prompts directory.

    This function copies the built-in prompt template to the user's prompts directory,
    making it available for customization. In Docker environments, this ensures the
    prompt file is always present and writable.

    Args:
        template_name: Name of the template to install (without .yaml extension)
        force: If True, overwrites existing file; if False, only creates if missing
        prompts_dir: Target directory for user prompts. Defaults to config.paths.prompts

    Returns:
        bool: True if file was installed/updated, False if it already existed and force=False

    Raises:
        FileNotFoundError: If the built-in template doesn't exist
        PermissionError: If unable to create the target directory or file
    """
    if prompts_dir is None:
        # Use explicit path from configuration
        try:
            # Try relative import first (when imported as part of package)
            from ..config import load_config

            config = load_config(validate=False)
            prompts_dir = Path(config.paths.prompts)
        except (ImportError, ValueError, AttributeError):
            # Fallback to original behavior if config loading fails
            prompts_dir = Path.cwd() / "prompts"

    # Ensure prompts directory exists
    prompts_dir.mkdir(exist_ok=True)

    target_file = prompts_dir / f"{template_name}.yaml"

    # Check if file already exists and force=False
    if target_file.exists() and not force:
        logger.debug(f"Prompt file already exists: {target_file}")
        return False

    # Find the built-in template using robust path handling
    current_file = Path(__file__)
    # First try: prompts at shared package level (shared/prompts/)
    builtin_template = current_file.parent.parent / "prompts" / f"{template_name}.yaml"

    # If not found, try project root level (for development/testing)
    if not builtin_template.exists():
        project_root = current_file.parent.parent.parent
        builtin_template = project_root / "prompts" / f"{template_name}.yaml"

    if not builtin_template.exists():
        raise FileNotFoundError(f"Built-in template not found: {builtin_template}")

    try:
        # Copy the built-in template to user directory
        shutil.copy2(builtin_template, target_file)
        logger.info(f"Installed default prompt: {target_file}")
        return True

    except (OSError, PermissionError) as e:
        logger.error(f"Failed to install prompt {template_name}: {e}")
        raise


def install_all_default_prompts(
    force: bool = False, prompts_dir: Optional[Path] = None
) -> int:
    """
    Install all available default prompt templates.

    Args:
        force: If True, overwrites existing files; if False, only creates missing files
        prompts_dir: Target directory for user prompts. Defaults to config.paths.prompts

    Returns:
        int: Number of prompt files installed/updated
    """
    if prompts_dir is None:
        # Use explicit path from configuration
        try:
            # Try relative import first (when imported as part of package)
            from ..config import load_config

            config = load_config(validate=False)
            prompts_dir = Path(config.paths.prompts)
        except (ImportError, ValueError, AttributeError):
            # Fallback to original behavior if config loading fails
            prompts_dir = Path.cwd() / "prompts"

    # Find all built-in templates
    current_file = Path(__file__)
    builtin_prompts_dir = current_file.parent.parent / "prompts"

    if not builtin_prompts_dir.exists():
        logger.warning(f"Built-in prompts directory not found: {builtin_prompts_dir}")
        return 0

    installed_count = 0
    for template_file in builtin_prompts_dir.glob("*.yaml"):
        template_name = template_file.stem
        try:
            if install_default_prompt(
                template_name, force=force, prompts_dir=prompts_dir
            ):
                installed_count += 1
        except Exception as e:
            logger.error(f"Failed to install template {template_name}: {e}")

    logger.info(f"Installed {installed_count} default prompt templates")
    return installed_count


def ensure_prompt_exists(template_name: str = "conversation_extraction") -> Path:
    """
    Ensure a prompt file exists in the user's prompts directory.

    If the file doesn't exist, installs the default version. This is useful
    for ensuring prompts are available without overwriting user customizations.

    Args:
        template_name: Name of the template (without .yaml extension)

    Returns:
        Path: Path to the prompt file in user's prompts directory

    Raises:
        FileNotFoundError: If the built-in template doesn't exist
        PermissionError: If unable to create the file
    """
    # Default to prompts directory in settings (accessible to users in Docker)
    # Fallback to project root for local development
    try:
        # Try relative import first (when imported as part of package)
        from ..config import load_config

        config = load_config(validate=False)
        settings_prompts_dir = Path(config.settings_path) / "prompts"
        # Use settings/prompts if vault path exists, otherwise fallback to ./prompts
        if Path(config.settings_path).exists():
            prompts_dir = settings_prompts_dir
        else:
            prompts_dir = Path.cwd() / "prompts"
    except (ImportError, ValueError):
        # Try absolute import (when imported directly)
        try:
            import importlib.util

            # Find the config module
            current_file = Path(__file__)
            config_path = current_file.parent.parent / "config.py"
            if config_path.exists():
                spec = importlib.util.spec_from_file_location("config", config_path)
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)

                config = config_module.load_config(validate=False)
                settings_prompts_dir = Path(config.settings_path) / "prompts"
                # Use settings/prompts if vault path exists, otherwise fallback to ./prompts
                if Path(config.settings_path).exists():
                    prompts_dir = settings_prompts_dir
                else:
                    prompts_dir = Path.cwd() / "prompts"
            else:
                raise ImportError("Config module not found")
        except Exception:
            # Fallback to original behavior if config loading fails
            prompts_dir = Path.cwd() / "prompts"

    prompt_file = prompts_dir / f"{template_name}.yaml"

    if not prompt_file.exists():
        install_default_prompt(template_name, force=False, prompts_dir=prompts_dir)

    return prompt_file
