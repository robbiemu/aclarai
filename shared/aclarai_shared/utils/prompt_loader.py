"""
Prompt template loader utility for externalized YAML templates.
Implements the LLM interaction strategy for loading prompt templates
from external YAML files at runtime, as defined in docs/arch/on-llm_interaction_strategy.md
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Container for a loaded prompt template with metadata."""

    role: str
    description: str
    template: str
    variables: Dict[str, Any]
    system_prompt: Optional[str] = None
    instructions: Optional[list] = None
    output_format: Optional[str] = None
    rules: Optional[list] = None


class PromptLoader:
    """Utility class for loading externalized prompt templates from YAML files."""

    def __init__(
        self,
        prompts_dir: Optional[Path] = None,
        user_prompts_dir: Optional[Path] = None,
    ):
        """
        Initialize the prompt loader.
        Args:
            prompts_dir: Directory containing built-in prompt YAML files.
                        Defaults to shared/aclarai_shared/prompts/
            user_prompts_dir: Directory containing user-customized prompt files.
                             Defaults to settings/prompts/
        """
        if prompts_dir is None:
            # Default to the prompts directory in aclarai_shared
            current_file = Path(__file__)
            self.prompts_dir = current_file.parent.parent / "prompts"
        else:
            self.prompts_dir = Path(prompts_dir)
        if user_prompts_dir is None:
            # Default to prompts directory in settings (accessible to users in Docker)
            # Fallback to relative settings path for local development
            try:
                # Try relative import first (when imported as part of package)
                from ..config import load_config

                config = load_config(validate=False)
                settings_prompts_dir = Path(config.settings_path) / "prompts"
                # Use settings/prompts if vault path exists, otherwise fallback to relative path
                if Path(config.settings_path).exists():
                    self.user_prompts_dir = settings_prompts_dir
                else:
                    # Fallback to relative settings path for local development
                    self.user_prompts_dir = Path.cwd() / "settings" / "prompts"
            except (ImportError, ValueError):
                # Try absolute import (when imported directly)
                try:
                    import importlib.util

                    # Find the config module
                    current_file = Path(__file__)
                    config_path = current_file.parent.parent / "config.py"
                    if config_path.exists():
                        spec = importlib.util.spec_from_file_location(
                            "config", config_path
                        )
                        if spec is None or spec.loader is None:
                            raise ImportError(f"Could not load spec for {config_path}")
                        config_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(config_module)
                        config = config_module.load_config(validate=False)
                        settings_prompts_dir = Path(config.settings_path) / "prompts"
                        # Use settings/prompts if vault path exists, otherwise fallback to relative path
                        if Path(config.settings_path).exists():
                            self.user_prompts_dir = settings_prompts_dir
                        else:
                            # Fallback to relative settings path for local development
                            self.user_prompts_dir = Path.cwd() / "settings" / "prompts"
                    else:
                        raise ImportError("Config module not found")
                except Exception:
                    # Fallback to relative settings path if config loading fails
                    self.user_prompts_dir = Path.cwd() / "settings" / "prompts"
        else:
            self.user_prompts_dir = Path(user_prompts_dir)
        if not self.prompts_dir.exists():
            logger.warning(
                f"Built-in prompts directory does not exist: {self.prompts_dir}"
            )
        # User prompts directory is optional
        if self.user_prompts_dir.exists():
            logger.debug(f"User prompts directory found: {self.user_prompts_dir}")

    def _find_template_files(self, template_name: str) -> tuple[Path, Path | None]:
        """
        Find the default and user template files for deep merging.
        Args:
            template_name: Name of the template file (without .yaml extension)
        Returns:
            Tuple of (default_path, user_path) where user_path may be None
        Raises:
            FileNotFoundError: If default template file doesn't exist
        """
        # Default template must exist
        builtin_template_path = self.prompts_dir / f"{template_name}.yaml"
        if not builtin_template_path.exists():
            raise FileNotFoundError(
                f"Default prompt template '{template_name}' not found in built-in prompts ({builtin_template_path})"
            )
        # User template is optional for deep merge
        user_template_path = self.user_prompts_dir / f"{template_name}.yaml"
        if user_template_path.exists():
            logger.debug(f"Found user customization: {user_template_path}")
            return builtin_template_path, user_template_path
        else:
            logger.debug(f"Using default prompt only: {builtin_template_path}")
            return builtin_template_path, None

    @staticmethod
    def _deep_merge_configs(
        default: Dict[str, Any], user: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge user configuration over default configuration.
        This reuses the same deep merge logic as the main config system.
        Args:
            default: Default configuration dictionary
            user: User configuration dictionary
        Returns:
            Merged configuration dictionary
        """
        import copy

        result = copy.deepcopy(default)

        def _merge_recursive(base_dict: Dict[str, Any], override_dict: Dict[str, Any]):
            for key, value in override_dict.items():
                if (
                    key in base_dict
                    and isinstance(base_dict[key], dict)
                    and isinstance(value, dict)
                ):
                    _merge_recursive(base_dict[key], value)
                else:
                    base_dict[key] = value

        _merge_recursive(result, user)
        return result

    def load_template(self, template_name: str) -> PromptTemplate:
        """
        Load a prompt template from YAML files with deep merge support.
        Loads the default prompt and merges any user customizations over it.
        This allows users to override specific keys without copying the entire file.
        Args:
            template_name: Name of the template file (without .yaml extension)
        Returns:
            PromptTemplate instance with merged data
        Raises:
            FileNotFoundError: If default template file doesn't exist
            ValueError: If template is malformed or missing required fields
        """
        default_path, user_path = self._find_template_files(template_name)
        # Load default template
        try:
            with open(default_path, "r", encoding="utf-8") as f:
                default_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(
                f"Invalid YAML in default template {template_name}: {e}"
            ) from e
        # Load user template if it exists
        user_data: Dict[str, Any] = {}
        if user_path:
            try:
                with open(user_path, "r", encoding="utf-8") as f:
                    user_data = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(
                    f"Invalid YAML in user template {template_name}: {e}"
                ) from e
        # Deep merge user data over default data
        merged_data = self._deep_merge_configs(default_data, user_data)
        # Validate required fields in merged result
        required_fields = ["role", "description", "template", "variables"]
        missing_fields = [
            field for field in required_fields if field not in merged_data
        ]
        if missing_fields:
            raise ValueError(
                f"Template {template_name} missing required fields after merge: {missing_fields}"
            )
        logger.debug(
            f"Loaded prompt template: {template_name} from {default_path}"
            + (f" with user customizations from {user_path}" if user_path else "")
        )
        return PromptTemplate(
            role=merged_data["role"],
            description=merged_data["description"],
            template=merged_data["template"],
            variables=merged_data["variables"],
            system_prompt=merged_data.get("system_prompt"),
            instructions=merged_data.get("instructions"),
            output_format=merged_data.get("output_format"),
            rules=merged_data.get("rules"),
        )

    def format_template(self, template: PromptTemplate, **kwargs) -> str:
        """
        Format a template with provided variables.
        Args:
            template: The loaded template
            **kwargs: Variable values to inject into the template
        Returns:
            Formatted prompt string
        Raises:
            ValueError: If required variables are missing
        """
        # Check for required variables
        required_vars = [
            var_name
            for var_name, var_config in template.variables.items()
            if var_config.get("required", False)
        ]
        missing_vars = [var for var in required_vars if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required template variables: {missing_vars}")
        try:
            return template.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(
                f"Template formatting error - missing variable: {e}"
            ) from e

    def load_and_format(self, template_name: str, **kwargs) -> str:
        """
        Convenience method to load a template and format it in one call.
        Args:
            template_name: Name of the template file (without .yaml extension)
            **kwargs: Variable values to inject into the template
        Returns:
            Formatted prompt string
        """
        template = self.load_template(template_name)
        return self.format_template(template, **kwargs)


# Global instance for convenience
_default_loader = None


def get_prompt_loader() -> PromptLoader:
    """Get the default prompt loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = PromptLoader()
    return _default_loader


def load_prompt_template(template_name: str, **kwargs) -> str:
    """
    Convenience function to load and format a prompt template.
    Args:
        template_name: Name of the template file (without .yaml extension)
        **kwargs: Variable values to inject into the template
    Returns:
        Formatted prompt string
    """
    loader = get_prompt_loader()
    return loader.load_and_format(template_name, **kwargs)
