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


def _get_user_prompts_dir() -> Path:
    """
    Determines the correct directory for user-customizable prompts.

    It first tries to use the path defined in the main configuration.
    If that fails or is not found, it falls back to a relative path
    from the current working directory, which is useful for testing
    and local development.
    """
    try:
        # Try relative import first (when imported as part of package)
        from ..config import load_config

        config = load_config(validate=False)
        # This is the ideal path, defined by the central config
        settings_prompts_dir = Path(config.settings_path) / "prompts"

        # Use the configured path if it exists, otherwise fall back.
        # This handles cases where the settings/ dir might not be created yet.
        if Path(config.settings_path).exists():
            return settings_prompts_dir
        else:
            # Fallback for environments where settings/ might not be present
            # but we still want a predictable location.
            return Path.cwd() / "settings" / "prompts"
    except (ImportError, ValueError, FileNotFoundError):
        # This except block catches:
        # - ImportError: if aclarai_shared is not installed correctly.
        # - ValueError/FileNotFoundError: if load_config fails.
        # In all these cases, we fall back to a predictable relative path.
        return Path.cwd() / "settings" / "prompts"


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
        prompts_dir: Target directory for user prompts. Defaults to the configured path.
    Returns:
        bool: True if file was installed/updated, False if it already existed and force=False
    Raises:
        FileNotFoundError: If the built-in template doesn't exist
        PermissionError: If unable to create the target directory or file
    """
    # Use the helper function to resolve the directory if not provided
    if prompts_dir is None:
        prompts_dir = _get_user_prompts_dir()

    # Ensure prompts directory exists
    prompts_dir.mkdir(parents=True, exist_ok=True)
    target_file = prompts_dir / f"{template_name}.yaml"
    # Check if file already exists and force=False
    if target_file.exists() and not force:
        logger.debug(f"Prompt file already exists: {target_file}")
        return False
    # Find the built-in template using robust path handling
    current_file = Path(__file__)
    # First try: prompts at shared package level (shared/aclarai_shared/prompts/)
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
        prompts_dir: Target directory for user prompts. Defaults to the configured path.
    Returns:
        int: Number of prompt files installed/updated
    """
    # Use the helper function to resolve the directory if not provided
    if prompts_dir is None:
        prompts_dir = _get_user_prompts_dir()

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
    # Use the helper function to resolve the directory
    prompts_dir = _get_user_prompts_dir()
    prompt_file = prompts_dir / f"{template_name}.yaml"
    if not prompt_file.exists():
        install_default_prompt(template_name, force=False, prompts_dir=prompts_dir)
    return prompt_file
