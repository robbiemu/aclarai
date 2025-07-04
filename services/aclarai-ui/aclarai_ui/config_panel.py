"""Configuration Panel for aclarai UI.
This module implements the configuration panel that allows users to:
- View and edit model selections for different aclarai agents
- Adjust processing thresholds and parameters
- Configure context window settings
- Persist changes to settings/aclarai.config.yaml
Follows the design specification from docs/arch/design_config_panel.md
"""

import copy
import logging
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import gradio as gr
import yaml

logger = logging.getLogger("aclarai-ui.config_panel")


class ConfigurationManager:
    """Manages reading and writing configuration to/from YAML files."""

    def __init__(
        self,
        config_path: str = "/home/runner/work/aclarai/aclarai/settings/aclarai.config.yaml",
    ):
        self.config_path = Path(config_path)
        self.default_config_path = Path(
            "/home/runner/work/aclarai/aclarai/shared/aclarai_shared/aclarai.config.default.yaml"
        )

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with defaults merge."""
        try:
            # Load default configuration first
            default_config: Dict[str, Any] = {}
            if self.default_config_path.exists():
                with open(self.default_config_path, "r") as f:
                    default_config = yaml.safe_load(f) or {}
            # Load user configuration
            user_config: Dict[str, Any] = {}
            if self.config_path.exists():
                with open(self.config_path, "r") as f:
                    user_config = yaml.safe_load(f) or {}
            # Deep merge user config over default config
            merged_config = self._deep_merge_configs(default_config, user_config)
            logger.info(
                "Configuration loaded successfully",
                extra={
                    "service": "aclarai-ui",
                    "component": "config_panel",
                    "action": "load_config",
                    "config_file": str(self.config_path),
                },
            )
            return merged_config
        except Exception as e:
            logger.error(
                "Failed to load configuration",
                extra={
                    "service": "aclarai-ui",
                    "component": "config_panel",
                    "action": "load_config",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Return default configuration as fallback
            return self._get_default_fallback_config()

    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to YAML file atomically."""
        try:
            # Ensure the settings directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            # Write to temporary file first (atomic write pattern)
            temp_path = self.config_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
            # Atomic rename
            temp_path.rename(self.config_path)
            logger.info(
                "Configuration saved successfully",
                extra={
                    "service": "aclarai-ui",
                    "component": "config_panel",
                    "action": "save_config",
                    "config_file": str(self.config_path),
                },
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to save configuration",
                extra={
                    "service": "aclarai-ui",
                    "component": "config_panel",
                    "action": "save_config",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return False

    @staticmethod
    def _deep_merge_configs(
        default: Dict[str, Any], user: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge user configuration over default configuration."""
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

    def _get_default_fallback_config(self) -> Dict[str, Any]:
        """Return minimal fallback configuration."""
        return {
            "model": {
                "claimify": {
                    "default": "gpt-3.5-turbo",
                    "selection": None,
                    "disambiguation": None,
                    "decomposition": None,
                },
                "concept_linker": "gpt-3.5-turbo",
                "concept_summary": "gpt-4",
                "subject_summary": "gpt-3.5-turbo",
                "trending_concepts_agent": "gpt-4",
                "fallback_plugin": "gpt-3.5-turbo",
            },
            "embedding": {
                "utterance": "sentence-transformers/all-MiniLM-L6-v2",
                "concept": "text-embedding-3-small",
                "summary": "sentence-transformers/all-MiniLM-L6-v2",
                "fallback": "sentence-transformers/all-mpnet-base-v2",
            },
            "threshold": {
                "concept_merge": 0.90,
                "claim_link_strength": 0.60,
            },
            "window": {
                "claimify": {
                    "p": 3,
                    "f": 1,
                }
            },
        }


def validate_model_name(model_name: str) -> Tuple[bool, str]:
    """Validate model name format."""
    if not model_name or not model_name.strip():
        return False, "Model name cannot be empty"
    model_name = model_name.strip()
    # Allow common model formats
    valid_patterns = [
        lambda x: x.startswith("gpt-"),  # OpenAI GPT models
        lambda x: x.startswith("claude-"),  # Anthropic Claude models
        lambda x: x.startswith("mistral"),  # Mistral models
        lambda x: x.startswith("openrouter:"),  # OpenRouter models
        lambda x: x.startswith("ollama:"),  # Ollama models
        lambda x: x.startswith(
            "sentence-transformers/"
        ),  # HuggingFace sentence transformers
        lambda x: x.startswith("text-embedding-"),  # OpenAI embeddings
        lambda x: "/" in x and not x.startswith("/"),  # HuggingFace models
    ]
    if any(pattern(model_name) for pattern in valid_patterns):
        return True, ""
    return False, f"Invalid model name format: {model_name}"


def validate_threshold(
    value: float, min_val: float = 0.0, max_val: float = 1.0
) -> Tuple[bool, str]:
    """Validate threshold value is within expected range."""
    try:
        if not isinstance(value, (int, float)):
            return False, "Threshold must be a number"
        if value < min_val or value > max_val:
            return False, f"Threshold must be between {min_val} and {max_val}"
        return True, ""
    except Exception:
        return False, "Invalid threshold value"


def validate_window_param(
    value: int, min_val: int = 0, max_val: int = 10
) -> Tuple[bool, str]:
    """Validate window parameter value."""
    try:
        if not isinstance(value, int):
            return False, "Window parameter must be an integer"
        if value < min_val or value > max_val:
            return False, f"Window parameter must be between {min_val} and {max_val}"
        return True, ""
    except Exception:
        return False, "Invalid window parameter value"


def validate_cron_expression(cron: str) -> Tuple[bool, str]:
    """Validate cron expression format."""
    if not cron or not cron.strip():
        return False, "Cron expression cannot be empty"

    cron = cron.strip()
    # Basic cron validation: 5 fields separated by spaces
    fields = cron.split()
    if len(fields) != 5:
        return (
            False,
            "Cron expression must have exactly 5 fields (minute hour day month weekday)",
        )

    # Validate each field has valid characters
    valid_chars_pattern = r"^[0-9\*\-,/]+$"
    field_names = ["minute", "hour", "day", "month", "weekday"]

    for i, field in enumerate(fields):
        if not re.match(valid_chars_pattern, field):
            return False, f"Invalid characters in {field_names[i]} field: {field}"

    return True, ""


def create_configuration_panel() -> gr.Blocks:
    """Create the configuration panel interface."""
    config_manager = ConfigurationManager()

    def load_current_config() -> Tuple[
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,  # summary_embedding
        str,  # fallback_embedding
        float,
        float,
        int,
        int,
        bool,
        bool,
        str,
        bool,
        bool,
        str,
        # Concept highlights configuration
        str,   # top_concepts_metric
        int,   # top_concepts_count
        float, # top_concepts_percent
        str,   # top_concepts_target_file
        int,   # trending_topics_window_days
        int,   # trending_topics_count
        float, # trending_topics_percent
        int,   # trending_topics_min_mentions
        str,   # trending_topics_target_file
    ]:
        """Load current configuration values for UI display."""
        try:
            config = config_manager.load_config()
            # Extract model configurations
            model_config = config.get("model", {})
            claimify_config = model_config.get("claimify", {})
            claimify_default = claimify_config.get("default", "gpt-3.5-turbo")
            claimify_selection = claimify_config.get("selection") or claimify_default
            claimify_disambiguation = (
                claimify_config.get("disambiguation") or claimify_default
            )
            claimify_decomposition = (
                claimify_config.get("decomposition") or claimify_default
            )
            concept_linker = model_config.get("concept_linker", "gpt-3.5-turbo")
            concept_summary = model_config.get("concept_summary", "gpt-4")
            subject_summary = model_config.get("subject_summary", "gpt-3.5-turbo")
            trending_concepts_agent = model_config.get(
                "trending_concepts_agent", "gpt-4"
            )
            fallback_plugin = model_config.get("fallback_plugin", "gpt-3.5-turbo")
            # Extract embedding configurations
            embedding_config = config.get("embedding", {})
            utterance_embedding = embedding_config.get(
                "utterance", "sentence-transformers/all-MiniLM-L6-v2"
            )
            concept_embedding = embedding_config.get(
                "concept", "text-embedding-3-small"
            )
            summary_embedding = embedding_config.get(
                "summary", "sentence-transformers/all-MiniLM-L6-v2"
            )
            fallback_embedding = embedding_config.get(
                "fallback", "sentence-transformers/all-mpnet-base-v2"
            )
            # Extract threshold configurations
            threshold_config = config.get("threshold", {})
            concept_merge = threshold_config.get("concept_merge", 0.90)
            claim_link_strength = threshold_config.get("claim_link_strength", 0.60)
            # Extract window configurations
            window_config = config.get("window", {})
            claimify_window = window_config.get("claimify", {})
            window_p = claimify_window.get("p", 3)
            window_f = claimify_window.get("f", 1)

            # Extract scheduler configurations
            scheduler_config = config.get("scheduler", {})
            jobs_config = scheduler_config.get("jobs", {})

            # concept_embedding_refresh job
            concept_refresh_config = jobs_config.get("concept_embedding_refresh", {})
            concept_refresh_enabled = concept_refresh_config.get("enabled", True)
            concept_refresh_manual_only = concept_refresh_config.get(
                "manual_only", False
            )
            concept_refresh_cron = concept_refresh_config.get("cron", "0 3 * * *")

            # vault_sync job
            vault_sync_config = jobs_config.get("vault_sync", {})
            vault_sync_enabled = vault_sync_config.get("enabled", True)
            vault_sync_manual_only = vault_sync_config.get("manual_only", False)
            vault_sync_cron = vault_sync_config.get("cron", "*/30 * * * *")

            # Extract concept highlights configurations
            concept_highlights_config = config.get("concept_highlights", {})

            # Top concepts configuration
            top_concepts_config = concept_highlights_config.get("top_concepts", {})
            top_concepts_metric = top_concepts_config.get("metric", "pagerank")
            top_concepts_count = top_concepts_config.get("count", 25)
            top_concepts_percent = top_concepts_config.get("percent", 0.0)
            top_concepts_target_file = top_concepts_config.get("target_file", "Top Concepts.md")

            # Trending topics configuration
            trending_topics_config = concept_highlights_config.get("trending_topics", {})
            trending_topics_window_days = trending_topics_config.get("window_days", 7)
            trending_topics_count = trending_topics_config.get("count", 0)
            trending_topics_percent = trending_topics_config.get("percent", 5.0)
            trending_topics_min_mentions = trending_topics_config.get("min_mentions", 2)
            trending_topics_target_file = trending_topics_config.get("target_file", "Trending Topics - {date}.md")

            return (
                claimify_default,
                claimify_selection,
                claimify_disambiguation,
                claimify_decomposition,
                concept_linker,
                concept_summary,
                subject_summary,
                trending_concepts_agent,
                fallback_plugin,
                utterance_embedding,
                concept_embedding,
                summary_embedding,
                fallback_embedding,
                concept_merge,
                claim_link_strength,
                window_p,
                window_f,
                concept_refresh_enabled,
                concept_refresh_manual_only,
                concept_refresh_cron,
                vault_sync_enabled,
                vault_sync_manual_only,
                vault_sync_cron,
                # Concept highlights values
                top_concepts_metric,
                top_concepts_count,
                top_concepts_percent,
                top_concepts_target_file,
                trending_topics_window_days,
                trending_topics_count,
                trending_topics_percent,
                trending_topics_min_mentions,
                trending_topics_target_file,
            )
        except Exception as e:
            logger.error(
                "Failed to load current configuration for UI",
                extra={
                    "service": "aclarai-ui",
                    "component": "config_panel",
                    "action": "load_current_config",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Return defaults
            return (
                "gpt-3.5-turbo",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
                "sentence-transformers/all-MiniLM-L6-v2",
                "text-embedding-3-small",
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                0.90,
                0.60,
                3,
                1,
                True,
                False,
                "0 3 * * *",
                True,
                False,
                "*/30 * * * *",
                # Concept highlights defaults
                "pagerank",
                25,
                0.0,
                "Top Concepts.md",
                7,
                0,
                5.0,
                2,
                "Trending Topics - {date}.md",
            )

    def save_configuration(
        claimify_default: str,
        claimify_selection: str,
        claimify_disambiguation: str,
        claimify_decomposition: str,
        concept_linker: str,
        concept_summary: str,
        subject_summary: str,
        trending_concepts_agent: str,
        fallback_plugin: str,
        utterance_embedding: str,
        concept_embedding: str,
        summary_embedding: str,
        fallback_embedding: str,
        concept_merge: float,
        claim_link_strength: float,
        window_p: int,
        window_f: int,
        concept_refresh_enabled: bool,
        concept_refresh_manual_only: bool,
        concept_refresh_cron: str,
        vault_sync_enabled: bool,
        vault_sync_manual_only: bool,
        vault_sync_cron: str,
        # Concept highlights parameters
        top_concepts_metric: str,
        top_concepts_count: int,
        top_concepts_percent: float,
        top_concepts_target_file: str,
        trending_topics_window_days: int,
        trending_topics_count: int,
        trending_topics_percent: float,
        trending_topics_min_mentions: int,
        trending_topics_target_file: str,
    ) -> str:
        """Save configuration changes to YAML file."""
        try:
            # Validate all inputs
            validation_errors = []
            # Validate model names
            models_to_validate = [
                ("Claimify Default", claimify_default),
                ("Claimify Selection", claimify_selection),
                ("Claimify Disambiguation", claimify_disambiguation),
                ("Claimify Decomposition", claimify_decomposition),
                ("Concept Linker", concept_linker),
                ("Concept Summary", concept_summary),
                ("Subject Summary", subject_summary),
                ("Trending Concepts Agent", trending_concepts_agent),
                ("Fallback Plugin", fallback_plugin),
                ("Utterance Embedding", utterance_embedding),
                ("Concept Embedding", concept_embedding),
                ("Summary Embedding", summary_embedding),
                ("Fallback Embedding", fallback_embedding),
            ]
            for name, model in models_to_validate:
                is_valid, error = validate_model_name(model)
                if not is_valid:
                    validation_errors.append(f"{name}: {error}")
            # Validate thresholds
            is_valid, error = validate_threshold(concept_merge)
            if not is_valid:
                validation_errors.append(f"Concept Merge Threshold: {error}")
            is_valid, error = validate_threshold(claim_link_strength)
            if not is_valid:
                validation_errors.append(f"Claim Link Strength: {error}")
            # Validate window parameters
            is_valid, error = validate_window_param(window_p)
            if not is_valid:
                validation_errors.append(f"Window Previous (p): {error}")
            is_valid, error = validate_window_param(window_f)
            if not is_valid:
                validation_errors.append(f"Window Following (f): {error}")
            # Validate cron expressions
            is_valid, error = validate_cron_expression(concept_refresh_cron)
            if not is_valid:
                validation_errors.append(f"Concept Refresh Cron: {error}")
            is_valid, error = validate_cron_expression(vault_sync_cron)
            if not is_valid:
                validation_errors.append(f"Vault Sync Cron: {error}")

            # Validate concept highlights parameters
            # Validate top concepts metric
            if top_concepts_metric not in ["pagerank", "degree"]:
                validation_errors.append(f"Top Concepts Metric: Must be 'pagerank' or 'degree', got '{top_concepts_metric}'")

            # Validate top concepts count/percent (should be mutually exclusive)
            if top_concepts_count > 0 and top_concepts_percent > 0:
                validation_errors.append("Top Concepts: Cannot use both count and percent - choose one")

            if top_concepts_count < 0:
                validation_errors.append("Top Concepts Count: Must be non-negative")

            if top_concepts_percent < 0 or top_concepts_percent > 100:
                validation_errors.append("Top Concepts Percent: Must be between 0 and 100")

            # Validate trending topics parameters
            if trending_topics_window_days < 1:
                validation_errors.append("Trending Topics Window Days: Must be at least 1")

            if trending_topics_count < 0:
                validation_errors.append("Trending Topics Count: Must be non-negative")

            if trending_topics_percent < 0 or trending_topics_percent > 100:
                validation_errors.append("Trending Topics Percent: Must be between 0 and 100")

            if trending_topics_min_mentions < 0:
                validation_errors.append("Trending Topics Min Mentions: Must be non-negative")

            # Validate target file names
            if not top_concepts_target_file.strip():
                validation_errors.append("Top Concepts Target File: Cannot be empty")

            if not trending_topics_target_file.strip():
                validation_errors.append("Trending Topics Target File: Cannot be empty")

            if validation_errors:
                error_msg = "❌ **Validation Errors:**\n" + "\n".join(
                    f"- {error}" for error in validation_errors
                )
                logger.warning(
                    "Configuration validation failed",
                    extra={
                        "service": "aclarai-ui",
                        "component": "config_panel",
                        "action": "save_configuration",
                        "validation_errors": validation_errors,
                    },
                )
                return error_msg
            # Load current configuration to preserve other settings
            current_config = config_manager.load_config()
            # Update with new values
            if "model" not in current_config:
                current_config["model"] = {}
            current_config["model"]["claimify"] = {
                "default": claimify_default.strip(),
                "selection": claimify_selection.strip()
                if claimify_selection.strip() != claimify_default.strip()
                else None,
                "disambiguation": claimify_disambiguation.strip()
                if claimify_disambiguation.strip() != claimify_default.strip()
                else None,
                "decomposition": claimify_decomposition.strip()
                if claimify_decomposition.strip() != claimify_default.strip()
                else None,
            }
            current_config["model"]["concept_linker"] = concept_linker.strip()
            current_config["model"]["concept_summary"] = concept_summary.strip()
            current_config["model"]["subject_summary"] = subject_summary.strip()
            current_config["model"]["trending_concepts_agent"] = (
                trending_concepts_agent.strip()
            )
            current_config["model"]["fallback_plugin"] = fallback_plugin.strip()
            if "embedding" not in current_config:
                current_config["embedding"] = {}
            current_config["embedding"]["utterance"] = utterance_embedding.strip()
            current_config["embedding"]["concept"] = concept_embedding.strip()
            current_config["embedding"]["summary"] = summary_embedding.strip()
            current_config["embedding"]["fallback"] = fallback_embedding.strip()
            if "threshold" not in current_config:
                current_config["threshold"] = {}
            current_config["threshold"]["concept_merge"] = concept_merge
            current_config["threshold"]["claim_link_strength"] = claim_link_strength
            if "window" not in current_config:
                current_config["window"] = {}
            if "claimify" not in current_config["window"]:
                current_config["window"]["claimify"] = {}
            current_config["window"]["claimify"]["p"] = window_p
            current_config["window"]["claimify"]["f"] = window_f

            # Update scheduler configuration
            if "scheduler" not in current_config:
                current_config["scheduler"] = {}
            if "jobs" not in current_config["scheduler"]:
                current_config["scheduler"]["jobs"] = {}

            # concept_embedding_refresh job
            if "concept_embedding_refresh" not in current_config["scheduler"]["jobs"]:
                current_config["scheduler"]["jobs"]["concept_embedding_refresh"] = {}
            current_config["scheduler"]["jobs"]["concept_embedding_refresh"][
                "enabled"
            ] = concept_refresh_enabled
            current_config["scheduler"]["jobs"]["concept_embedding_refresh"][
                "manual_only"
            ] = concept_refresh_manual_only
            current_config["scheduler"]["jobs"]["concept_embedding_refresh"]["cron"] = (
                concept_refresh_cron.strip()
            )

            # vault_sync job
            if "vault_sync" not in current_config["scheduler"]["jobs"]:
                current_config["scheduler"]["jobs"]["vault_sync"] = {}
            current_config["scheduler"]["jobs"]["vault_sync"]["enabled"] = (
                vault_sync_enabled
            )
            current_config["scheduler"]["jobs"]["vault_sync"]["manual_only"] = (
                vault_sync_manual_only
            )
            current_config["scheduler"]["jobs"]["vault_sync"]["cron"] = (
                vault_sync_cron.strip()
            )

            # Update concept highlights configuration
            if "concept_highlights" not in current_config:
                current_config["concept_highlights"] = {}

            # Top concepts configuration
            if "top_concepts" not in current_config["concept_highlights"]:
                current_config["concept_highlights"]["top_concepts"] = {}
            current_config["concept_highlights"]["top_concepts"]["metric"] = top_concepts_metric.strip()
            current_config["concept_highlights"]["top_concepts"]["count"] = top_concepts_count if top_concepts_count > 0 else None
            current_config["concept_highlights"]["top_concepts"]["percent"] = top_concepts_percent if top_concepts_percent > 0 else None
            current_config["concept_highlights"]["top_concepts"]["target_file"] = top_concepts_target_file.strip()

            # Trending topics configuration
            if "trending_topics" not in current_config["concept_highlights"]:
                current_config["concept_highlights"]["trending_topics"] = {}
            current_config["concept_highlights"]["trending_topics"]["window_days"] = trending_topics_window_days
            current_config["concept_highlights"]["trending_topics"]["count"] = trending_topics_count if trending_topics_count > 0 else None
            current_config["concept_highlights"]["trending_topics"]["percent"] = trending_topics_percent if trending_topics_percent > 0 else None
            current_config["concept_highlights"]["trending_topics"]["min_mentions"] = trending_topics_min_mentions
            current_config["concept_highlights"]["trending_topics"]["target_file"] = trending_topics_target_file.strip()

            # Save to file
            success = config_manager.save_config(current_config)
            if success:
                logger.info(
                    "Configuration saved successfully",
                    extra={
                        "service": "aclarai-ui",
                        "component": "config_panel",
                        "action": "save_configuration",
                    },
                )
                return "✅ **Configuration saved successfully!**\n\nChanges have been written to `settings/aclarai.config.yaml`."
            else:
                return "❌ **Failed to save configuration.** Please check the logs for details."
        except Exception as e:
            logger.error(
                "Failed to save configuration",
                extra={
                    "service": "aclarai-ui",
                    "component": "config_panel",
                    "action": "save_configuration",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return f"❌ **Error saving configuration:** {str(e)}"

    # Create the Gradio interface
    with gr.Blocks(
        title="aclarai - Configuration Panel", theme=gr.themes.Soft()
    ) as interface:
        gr.Markdown("# ⚙️ aclarai Configuration Panel")
        gr.Markdown(
            """Configure aclarai's core system parameters including model selections, processing thresholds,
            and context window settings. Changes are automatically saved to `settings/aclarai.config.yaml`."""
        )
        # Load initial values
        initial_values = load_current_config()
        # Model & Embedding Settings Section
        with gr.Group():
            gr.Markdown("## 🤖 Model & Embedding Settings")
            with gr.Group():
                gr.Markdown("### 🔮 Claimify Models")
                with gr.Row():
                    claimify_default_input = gr.Textbox(
                        label="Default Model",
                        value=initial_values[0],
                        placeholder="gpt-4",
                        info="Default model for all Claimify stages",
                    )
                    claimify_selection_input = gr.Textbox(
                        label="Selection Model",
                        value=initial_values[1],
                        placeholder="claude-3-opus",
                        info="Model for Claimify selection stage",
                    )
                with gr.Row():
                    claimify_disambiguation_input = gr.Textbox(
                        label="Disambiguation Model",
                        value=initial_values[2],
                        placeholder="mistral-7b",
                        info="Model for Claimify disambiguation",
                    )
                    claimify_decomposition_input = gr.Textbox(
                        label="Decomposition Model",
                        value=initial_values[3],
                        placeholder="gpt-4",
                        info="Model for Claimify decomposition",
                    )
            with gr.Group():
                gr.Markdown("### 🧠 Agent Models")
                with gr.Row():
                    concept_linker_input = gr.Textbox(
                        label="Concept Linker",
                        value=initial_values[4],
                        placeholder="mistral-7b",
                        info="Used to classify Claim→Concept relationships",
                    )
                    concept_summary_input = gr.Textbox(
                        label="Concept Summary",
                        value=initial_values[5],
                        placeholder="gpt-4",
                        info="Generates individual [[Concept]] Markdown pages",
                    )
                with gr.Row():
                    subject_summary_input = gr.Textbox(
                        label="Subject Summary",
                        value=initial_values[6],
                        placeholder="mistral-7b",
                        info="Generates [[Subject:XYZ]] pages from concept clusters",
                    )
                    trending_concepts_agent_input = gr.Textbox(
                        label="Trending Concepts Agent",
                        value=initial_values[7],
                        placeholder="gpt-4",
                        info="Writes newsletter-style blurbs for Top/Trending Concepts",
                    )
                fallback_plugin_input = gr.Textbox(
                    label="Fallback Plugin",
                    value=initial_values[8],
                    placeholder="openrouter:gemma-2b",
                    info="Used when format detection fails",
                )
            with gr.Group():
                gr.Markdown("### 🧬 Embedding Models")
                with gr.Row():
                    utterance_embedding_input = gr.Textbox(
                        label="Utterance Embeddings",
                        value=initial_values[9],
                        placeholder="all-MiniLM-L6-v2",
                        info="Embeddings for Tier 1 utterance blocks",
                    )
                    concept_embedding_input = gr.Textbox(
                        label="Concept Embeddings",
                        value=initial_values[10],
                        placeholder="text-embedding-3-small",
                        info="Embeddings for Tier 3 concept files",
                    )
                with gr.Row():
                    summary_embedding_input = gr.Textbox(
                        label="Summary Embeddings",
                        value=initial_values[11],
                        placeholder="sentence-transformers/all-MiniLM-L6-v2",
                        info="Embeddings for Tier 2 summaries",
                    )
                    fallback_embedding_input = gr.Textbox(
                        label="Fallback Embeddings",
                        value=initial_values[12],
                        placeholder="sentence-transformers/all-mpnet-base-v2",
                        info="Used if other embedding configs fail or for general purpose",
                    )
        # Thresholds & Parameters Section
        with gr.Group():
            gr.Markdown("## 📏 Thresholds & Parameters")
            with gr.Row():
                concept_merge_input = gr.Number(
                    label="Concept Merge Threshold",
                    value=initial_values[13],
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    info="Cosine similarity threshold for merging candidates",
                )
                claim_link_strength_input = gr.Number(
                    label="Claim Link Strength",
                    value=initial_values[14],
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    info="Minimum link strength to create graph edge",
                )
            with gr.Group():
                gr.Markdown("### 🪟 Context Window Parameters")
                gr.Markdown("Configure context window size for Claimify processing.")
                with gr.Row():
                    window_p_input = gr.Number(
                        label="Previous Sentences (p)",
                        value=initial_values[15],
                        minimum=0,
                        maximum=10,
                        step=1,
                        info="Number of previous sentences to include in context",
                    )
                    window_f_input = gr.Number(
                        label="Following Sentences (f)",
                        value=initial_values[16],
                        minimum=0,
                        maximum=10,
                        step=1,
                        info="Number of following sentences to include in context",
                    )

        # Scheduler Job Controls Section
        with gr.Group():
            gr.Markdown("## ⏰ Scheduler Job Controls")
            gr.Markdown(
                "Configure scheduled job execution settings. Jobs can be enabled/disabled, set to manual-only mode, or scheduled with custom cron expressions."
            )

            with gr.Group():
                gr.Markdown("### 🔄 Concept Embedding Refresh")
                gr.Markdown("Refreshes concept embeddings from Tier 3 pages")
                with gr.Row():
                    concept_refresh_enabled_input = gr.Checkbox(
                        label="Enabled",
                        value=initial_values[17],
                        info="Enable or disable this job completely",
                    )
                    concept_refresh_manual_only_input = gr.Checkbox(
                        label="Manual Only",
                        value=initial_values[18],
                        info="Job can only be triggered manually (not automatically scheduled)",
                    )
                concept_refresh_cron_input = gr.Textbox(
                    label="Cron Schedule",
                    value=initial_values[19],
                    placeholder="0 3 * * *",
                    info="Cron expression for automatic scheduling (only applies when enabled and not manual-only)",
                )

            with gr.Group():
                gr.Markdown("### 📁 Vault Sync")
                gr.Markdown("Synchronizes vault files with knowledge graph")
                with gr.Row():
                    vault_sync_enabled_input = gr.Checkbox(
                        label="Enabled",
                        value=initial_values[20],
                        info="Enable or disable this job completely",
                    )
                    vault_sync_manual_only_input = gr.Checkbox(
                        label="Manual Only",
                        value=initial_values[21],
                        info="Job can only be triggered manually (not automatically scheduled)",
                    )
                vault_sync_cron_input = gr.Textbox(
                    label="Cron Schedule",
                    value=initial_values[22],
                    placeholder="*/30 * * * *",
                    info="Cron expression for automatic scheduling (only applies when enabled and not manual-only)",
                )

        # Highlight & Summary Section
        with gr.Group():
            gr.Markdown("## 🧠 Highlight & Summary")
            gr.Markdown(
                "Configure concept highlight jobs that generate global summary pages for your vault."
            )

            # Model selection for concept highlights
            with gr.Group():
                gr.Markdown("### 🤖 Writing Agent")
                trending_concepts_agent_summary_input = gr.Textbox(
                    label="Model for Trending Concepts Agent",
                    value=initial_values[7],  # trending_concepts_agent from initial_values
                    placeholder="gpt-4",
                    info="LLM model used to generate concept highlight content (also configured in Model & Embedding Settings)",
                )

            with gr.Group():
                gr.Markdown("### 🏆 Top Concepts")
                gr.Markdown("Generate ranked lists of most important concepts")
                with gr.Row():
                    top_concepts_metric_input = gr.Dropdown(
                        label="Ranking Metric",
                        value=initial_values[23],
                        choices=["pagerank", "degree"],
                        info="Algorithm for ranking concepts (PageRank or simple degree centrality)",
                    )
                    top_concepts_count_input = gr.Number(
                        label="Count",
                        value=initial_values[24],
                        minimum=0,
                        maximum=1000,
                        step=1,
                        info="Number of top concepts to include (0 to use percent instead)",
                    )
                with gr.Row():
                    top_concepts_percent_input = gr.Number(
                        label="Percent",
                        value=initial_values[25],
                        minimum=0.0,
                        maximum=100.0,
                        step=0.1,
                        info="Percentage of top concepts to include (0 to use count instead)",
                    )
                    top_concepts_target_file_input = gr.Textbox(
                        label="Target File",
                        value=initial_values[26],
                        placeholder="Top Concepts.md",
                        info="Output filename for the top concepts page",
                    )

                # Preview for top concepts filename
                with gr.Row():
                    top_concepts_preview = gr.Markdown(
                        value=f"**Preview:** `{initial_values[26]}`",
                        label="Filename Preview",
                    )

            with gr.Group():
                gr.Markdown("### 📈 Trending Topics")
                gr.Markdown("Track concepts with recent activity increases")
                with gr.Row():
                    trending_topics_window_days_input = gr.Number(
                        label="Window Days",
                        value=initial_values[27],
                        minimum=1,
                        maximum=365,
                        step=1,
                        info="Number of days to look back for trend analysis",
                    )
                    trending_topics_count_input = gr.Number(
                        label="Count",
                        value=initial_values[28],
                        minimum=0,
                        maximum=1000,
                        step=1,
                        info="Number of trending topics to include (0 to use percent instead)",
                    )
                with gr.Row():
                    trending_topics_percent_input = gr.Number(
                        label="Percent",
                        value=initial_values[29],
                        minimum=0.0,
                        maximum=100.0,
                        step=0.1,
                        info="Percentage of trending topics to include (0 to use count instead)",
                    )
                    trending_topics_min_mentions_input = gr.Number(
                        label="Min Mentions",
                        value=initial_values[30],
                        minimum=0,
                        maximum=100,
                        step=1,
                        info="Minimum mentions required for a concept to be considered trending",
                    )
                trending_topics_target_file_input = gr.Textbox(
                    label="Target File",
                    value=initial_values[31],
                    placeholder="Trending Topics - {date}.md",
                    info="Output filename pattern for trending topics (use {date} for current date)",
                )

                # Preview for trending topics filename
                with gr.Row():
                    trending_topics_preview = gr.Markdown(
                        value=f"**Preview:** `{initial_values[31].replace('{date}', '2024-01-01')}`",
                        label="Filename Preview",
                    )

                # Function to update filename previews
                def update_top_concepts_preview(filename: str) -> str:
                    return f"**Preview:** `{filename}`"

                def update_trending_topics_preview(filename: str) -> str:
                    from datetime import date
                    preview_filename = filename.replace("{date}", date.today().strftime("%Y-%m-%d"))
                    return f"**Preview:** `{preview_filename}`"

                # Connect preview updates
                top_concepts_target_file_input.change(
                    fn=update_top_concepts_preview,
                    inputs=[top_concepts_target_file_input],
                    outputs=[top_concepts_preview],
                )

                trending_topics_target_file_input.change(
                    fn=update_trending_topics_preview,
                    inputs=[trending_topics_target_file_input],
                    outputs=[trending_topics_preview],
                )

                # Synchronize trending concepts agent inputs
                def sync_trending_agent_to_summary(value: str) -> str:
                    return value

                def sync_trending_agent_to_main(value: str) -> str:
                    return value

                trending_concepts_agent_input.change(
                    fn=sync_trending_agent_to_summary,
                    inputs=[trending_concepts_agent_input],
                    outputs=[trending_concepts_agent_summary_input],
                )

                trending_concepts_agent_summary_input.change(
                    fn=sync_trending_agent_to_main,
                    inputs=[trending_concepts_agent_summary_input],
                    outputs=[trending_concepts_agent_input],
                )

        # Save Section
        with gr.Group():
            gr.Markdown("## 💾 Save Configuration")
            with gr.Row():
                save_btn = gr.Button("💾 Save Changes", variant="primary", size="lg")
                reload_btn = gr.Button("🔄 Reload from File", variant="secondary")
            save_status = gr.Markdown(
                value="Make changes above and click **Save Changes** to persist to `settings/aclarai.config.yaml`.",
                label="Status",
            )

        # Event handlers
        def reload_configuration() -> Tuple[Any, ...]:
            """Reload configuration from file."""
            try:
                values = load_current_config()
                # Insert the trending_concepts_agent value again after the original one for the summary input
                values_list = list(values)
                # Insert at position 8 (after trending_concepts_agent which is at position 7)
                values_list.insert(8, values[7])  # Duplicate trending_concepts_agent value
                return tuple(values_list) + ("🔄 **Configuration reloaded from file.**",)
            except Exception as e:
                logger.error(
                    "Failed to reload configuration",
                    extra={
                        "service": "aclarai-ui",
                        "component": "config_panel",
                        "action": "reload_configuration",
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                # For initial_values, also insert the duplicate value
                initial_list = list(initial_values)
                initial_list.insert(8, initial_values[7])
                return tuple(initial_list) + (
                    f"❌ **Error reloading configuration:** {str(e)}",
                )

        # Save button click handler
        save_btn.click(
            fn=save_configuration,
            inputs=[
                claimify_default_input,
                claimify_selection_input,
                claimify_disambiguation_input,
                claimify_decomposition_input,
                concept_linker_input,
                concept_summary_input,
                subject_summary_input,
                trending_concepts_agent_input,
                fallback_plugin_input,
                utterance_embedding_input,
                concept_embedding_input,
                summary_embedding_input,
                fallback_embedding_input,
                concept_merge_input,
                claim_link_strength_input,
                window_p_input,
                window_f_input,
                concept_refresh_enabled_input,
                concept_refresh_manual_only_input,
                concept_refresh_cron_input,
                vault_sync_enabled_input,
                vault_sync_manual_only_input,
                vault_sync_cron_input,
                # Concept highlights inputs
                top_concepts_metric_input,
                top_concepts_count_input,
                top_concepts_percent_input,
                top_concepts_target_file_input,
                trending_topics_window_days_input,
                trending_topics_count_input,
                trending_topics_percent_input,
                trending_topics_min_mentions_input,
                trending_topics_target_file_input,
            ],
            outputs=[save_status],
        )
        # Reload button click handler
        reload_btn.click(
            fn=reload_configuration,
            outputs=[
                claimify_default_input,
                claimify_selection_input,
                claimify_disambiguation_input,
                claimify_decomposition_input,
                concept_linker_input,
                concept_summary_input,
                subject_summary_input,
                trending_concepts_agent_input,
                trending_concepts_agent_summary_input,  # Add the new input
                fallback_plugin_input,
                utterance_embedding_input,
                concept_embedding_input,
                summary_embedding_input,
                fallback_embedding_input,
                concept_merge_input,
                claim_link_strength_input,
                window_p_input,
                window_f_input,
                concept_refresh_enabled_input,
                concept_refresh_manual_only_input,
                concept_refresh_cron_input,
                vault_sync_enabled_input,
                vault_sync_manual_only_input,
                vault_sync_cron_input,
                # Concept highlights inputs
                top_concepts_metric_input,
                top_concepts_count_input,
                top_concepts_percent_input,
                top_concepts_target_file_input,
                trending_topics_window_days_input,
                trending_topics_count_input,
                trending_topics_percent_input,
                trending_topics_min_mentions_input,
                trending_topics_target_file_input,
                save_status,
            ],
        )
    return gr.Blocks.from_config(interface.config, interface.fns)  # type: ignore


if __name__ == "__main__":
    # For testing the configuration panel standalone
    interface = create_configuration_panel()
    interface.launch(debug=True)
