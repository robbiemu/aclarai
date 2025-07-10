"""Configuration Panel for aclarai UI.
This module implements the configuration panel that allows users to:
- View and edit model selections for different aclarai agents
- Adjust processing thresholds and parameters
- Configure context window settings
- Persist changes to settings/aclarai.config.yaml
Follows the design specification from docs/arch/design_config_panel.md
"""

import logging
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import yaml
from aclarai_shared.config import aclaraiConfig

logger = logging.getLogger("aclarai-ui.config_panel")


@dataclass
class ConfigData:
    """A structured container for all UI configuration values."""

    # --- Section: Model & Embedding Settings ---
    claimify_default: str
    claimify_selection: str
    claimify_disambiguation: str
    claimify_decomposition: str
    concept_linker: str
    concept_summary: str
    subject_summary: str
    trending_concepts_agent: str
    fallback_plugin: str
    utterance_embedding: str
    concept_embedding: str
    summary_embedding: str
    fallback_embedding: str

    # --- Section: Thresholds & Parameters ---
    concept_merge: float
    claim_link_strength: float
    window_p: int
    window_f: int

    # --- Section: Automation & Scheduler Control ---
    concept_refresh_enabled: bool
    concept_refresh_manual_only: bool
    concept_refresh_cron: Union[str, float]
    vault_sync_enabled: bool
    vault_sync_manual_only: bool
    vault_sync_cron: Union[str, float]

    # --- Section: Highlight & Summary ---
    top_concepts_metric: str
    top_concepts_count: Optional[int]
    top_concepts_percent: Optional[float]
    top_concepts_target_file: str
    trending_topics_window_days: int
    trending_topics_count: Optional[int]
    trending_topics_percent: Optional[float]
    trending_topics_min_mentions: int
    trending_topics_target_file: str

    # --- Section: Subject Summary & Concept Summary Agents ---
    subject_summary_similarity_threshold: float
    subject_summary_min_concepts: int
    subject_summary_max_concepts: int
    subject_summary_allow_web_search: bool
    subject_summary_skip_if_incoherent: bool
    concept_summary_max_examples: int
    concept_summary_skip_if_no_claims: bool
    concept_summary_include_see_also: bool


class ConfigurationManager:
    """Manages reading and writing configuration to/from YAML files."""

    def __init__(
        self,
        config_path: str = "settings/aclarai.config.yaml",
    ):
        self.config_path = Path(config_path).absolute()
        self.default_config_path = Path(
            "shared/aclarai_shared/aclarai.config.default.yaml"
        ).absolute()

    def load_config(self) -> aclaraiConfig:
        """Load configuration using the shared loader."""
        try:
            config = aclaraiConfig.from_env(config_file=str(self.config_path))
            logger.info(
                "Configuration loaded successfully via shared loader",
                extra={
                    "service": "aclarai-ui",
                    "component": "config_panel",
                    "action": "load_config",
                },
            )
            return config
        except Exception as e:
            logger.error(
                "Failed to load configuration via shared loader",
                extra={
                    "service": "aclarai-ui",
                    "component": "config_panel",
                    "action": "load_config",
                    "error": str(e),
                },
            )
            # Return default configuration as fallback
            return aclaraiConfig()

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


def validate_summary_agents_config(
    subject_summary_similarity_threshold: float,
    subject_summary_min_concepts: int,
    subject_summary_max_concepts: int,
    concept_summary_max_examples: int,
) -> Tuple[bool, List[str]]:
    """Validate Summary agent parameters. Assumes inputs are not None."""
    errors = []

    if not (0.0 <= subject_summary_similarity_threshold <= 1.0):
        errors.append("Similarity threshold must be between 0.0 and 1.0")

    if not (1 <= subject_summary_min_concepts <= 100):
        errors.append("Minimum concepts must be between 1 and 100")

    if not (1 <= subject_summary_max_concepts <= 100):
        errors.append("Maximum concepts must be between 1 and 100")

    if subject_summary_min_concepts > subject_summary_max_concepts:
        errors.append("Minimum concepts cannot be greater than maximum concepts")

    if not (0 <= concept_summary_max_examples <= 20):
        errors.append("Maximum examples must be between 0 and 20")

    return len(errors) == 0, errors


def validate_concept_highlights_config(
    top_concepts_metric: str,
    top_concepts_count: Optional[int],
    top_concepts_percent: Optional[float],
    top_concepts_target_file: str,
    trending_topics_window_days: int,
    trending_topics_count: Optional[int],
    trending_topics_percent: Optional[float],
    trending_topics_min_mentions: int,
    trending_topics_target_file: str,
) -> Tuple[bool, List[str]]:
    """Validate concept highlights parameters. Assumes inputs are not None."""
    validation_errors = []
    top_concepts_count = top_concepts_count or 0
    top_concepts_percent = top_concepts_percent or 0.0
    trending_topics_count = trending_topics_count or 0
    trending_topics_percent = trending_topics_percent or 0.0

    if top_concepts_metric not in ["pagerank", "degree"]:
        validation_errors.append(
            "Top Concepts metric must be either 'pagerank' or 'degree'"
        )

    if top_concepts_count > 0 and top_concepts_percent > 0:
        validation_errors.append(
            "Top Concepts count and percent cannot both be used (set one to 0)"
        )

    if not (0 <= top_concepts_percent <= 100):
        validation_errors.append("Top Concepts percent must be between 0 and 100")

    if not top_concepts_target_file or not top_concepts_target_file.strip():
        validation_errors.append("Top Concepts target_file cannot be empty")

    if trending_topics_count > 0 and trending_topics_percent > 0:
        validation_errors.append(
            "Trending Topics count and percent cannot both be used (set one to 0)"
        )

    if not (0 <= trending_topics_percent <= 100):
        validation_errors.append("Trending Topics percent must be between 0 and 100")

    if trending_topics_window_days < 1:
        validation_errors.append("Trending Topics window_days must be at least 1")

    if trending_topics_min_mentions < 0:
        validation_errors.append("Trending Topics min_mentions must be non-negative")

    if not trending_topics_target_file or not trending_topics_target_file.strip():
        validation_errors.append("Trending Topics target_file cannot be empty")

    return len(validation_errors) == 0, validation_errors


def validate_cron_expression(cron: str) -> Tuple[bool, str]:
    """Validate a cron expression.

    Args:
        cron: A string or numeric value representing a cron expression.

    Returns:
        A tuple of (is_valid, error_message).
    """
    if cron is None:
        return False, "Cron expression cannot be empty"

    # Convert input to string, handling both str and float inputs
    if isinstance(cron, (float, int)):
        cron_str = str(int(cron)) if float(cron).is_integer() else str(cron)
    else:
        cron_str = str(cron).strip()

    if not cron_str:
        return False, "Cron expression cannot be empty"

    # Basic cron validation: 5 fields separated by spaces
    fields = cron_str.split()
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

    def load_current_config() -> ConfigData:
        """Load current configuration values into a structured dataclass."""
        config = config_manager.load_config()

        # Extract values from the nested aclaraiConfig object
        model = config.model
        claimify = model.claimify
        embedding = config.embedding
        threshold = config.threshold
        window = config.window.get("claimify")
        scheduler_jobs = config.scheduler.jobs
        concept_refresh = scheduler_jobs.concept_embedding_refresh
        vault_sync = scheduler_jobs.vault_sync
        top_concepts = scheduler_jobs.top_concepts
        trending_topics = scheduler_jobs.trending_topics
        subject_summaries = config.subject_summaries
        concept_summaries = config.concept_summaries

        return ConfigData(
            # Model & Embedding Settings
            claimify_default=claimify.default,
            claimify_selection=claimify.selection or claimify.default,
            claimify_disambiguation=claimify.disambiguation or claimify.default,
            claimify_decomposition=claimify.decomposition or claimify.default,
            concept_linker=model.concept_linker,
            concept_summary=model.concept_summary,
            subject_summary=model.subject_summary,
            trending_concepts_agent=model.trending_concepts_agent,
            fallback_plugin=model.fallback_plugin,
            utterance_embedding=embedding.utterance or "",
            concept_embedding=embedding.concept or "",
            summary_embedding=embedding.summary or "",
            fallback_embedding=embedding.fallback or "",
            # Thresholds & Parameters
            concept_merge=threshold.concept_merge,
            claim_link_strength=threshold.claim_link_strength,
            window_p=window.p if window else 3,
            window_f=window.f if window else 1,
            # Automation & Scheduler Control
            concept_refresh_enabled=concept_refresh.enabled,
            concept_refresh_manual_only=concept_refresh.manual_only,
            concept_refresh_cron=concept_refresh.cron,
            vault_sync_enabled=vault_sync.enabled,
            vault_sync_manual_only=vault_sync.manual_only,
            vault_sync_cron=vault_sync.cron,
            # Highlight & Summary
            top_concepts_metric=top_concepts.metric,
            top_concepts_count=top_concepts.count,
            top_concepts_percent=top_concepts.percent,
            top_concepts_target_file=top_concepts.target_file,
            trending_topics_window_days=trending_topics.window_days,
            trending_topics_count=trending_topics.count,
            trending_topics_percent=trending_topics.percent,
            trending_topics_min_mentions=trending_topics.min_mentions,
            trending_topics_target_file=trending_topics.target_file,
            # Subject Summary & Concept Summary Agents
            subject_summary_similarity_threshold=subject_summaries.similarity_threshold,
            subject_summary_min_concepts=subject_summaries.min_concepts,
            subject_summary_max_concepts=subject_summaries.max_concepts,
            subject_summary_allow_web_search=subject_summaries.allow_web_search,
            subject_summary_skip_if_incoherent=subject_summaries.skip_if_incoherent,
            concept_summary_max_examples=concept_summaries.max_examples,
            concept_summary_skip_if_no_claims=concept_summaries.skip_if_no_claims,
            concept_summary_include_see_also=concept_summaries.include_see_also,
        )

    def save_configuration(*args) -> str:
        """Save configuration changes to YAML file after validation."""
        # 1. Unpack all arguments into the ConfigData structure for clarity.
        ui_data = ConfigData(*args)

        # 2. Coalesce None to zero-values for validation where needed.
        concept_merge = ui_data.concept_merge or 0.0
        claim_link_strength = ui_data.claim_link_strength or 0.0
        window_p = ui_data.window_p or 0
        window_f = ui_data.window_f or 0
        subject_summary_similarity_threshold = (
            ui_data.subject_summary_similarity_threshold or 0.0
        )
        subject_summary_min_concepts = ui_data.subject_summary_min_concepts or 0
        subject_summary_max_concepts = ui_data.subject_summary_max_concepts or 0
        concept_summary_max_examples = ui_data.concept_summary_max_examples or 0

        # 3. Perform all input validation
        validation_errors = []
        models_to_validate = {
            "Claimify Default": ui_data.claimify_default,
            "Claimify Selection": ui_data.claimify_selection,
            "Claimify Disambiguation": ui_data.claimify_disambiguation,
            "Claimify Decomposition": ui_data.claimify_decomposition,
            "Concept Linker": ui_data.concept_linker,
            "Concept Summary": ui_data.concept_summary,
            "Subject Summary": ui_data.subject_summary,
            "Trending Concepts Agent": ui_data.trending_concepts_agent,
            "Fallback Plugin": ui_data.fallback_plugin,
            "Utterance Embedding": ui_data.utterance_embedding,
            "Concept Embedding": ui_data.concept_embedding,
            "Summary Embedding": ui_data.summary_embedding,
            "Fallback Embedding": ui_data.fallback_embedding,
        }
        for name, model in models_to_validate.items():
            is_valid, error = validate_model_name(model)
            if not is_valid:
                validation_errors.append(f"{name}: {error}")

        for desc, value in [
            ("Concept Merge Threshold", concept_merge),
            ("Claim Link Strength", claim_link_strength),
        ]:
            is_valid, error = validate_threshold(value)
            if not is_valid:
                validation_errors.append(f"{desc}: {error}")

        for desc, value in [("Window P", window_p), ("Window F", window_f)]:
            is_valid, error = validate_window_param(value)
            if not is_valid:
                validation_errors.append(f"{desc}: {error}")

        for desc, value_str in [
            ("Concept Refresh Cron", str(ui_data.concept_refresh_cron).strip()),
            ("Vault Sync Cron", str(ui_data.vault_sync_cron).strip()),
        ]:
            is_valid, error = validate_cron_expression(value_str)
            if not is_valid:
                validation_errors.append(f"{desc}: {error}")

        is_valid, highlights_errors = validate_concept_highlights_config(
            ui_data.top_concepts_metric,
            ui_data.top_concepts_count,
            ui_data.top_concepts_percent,
            ui_data.top_concepts_target_file,
            ui_data.trending_topics_window_days,
            ui_data.trending_topics_count,
            ui_data.trending_topics_percent,
            ui_data.trending_topics_min_mentions,
            ui_data.trending_topics_target_file,
        )
        if not is_valid:
            validation_errors.extend(highlights_errors)

        is_valid, summary_errors = validate_summary_agents_config(
            subject_summary_similarity_threshold,
            subject_summary_min_concepts,
            subject_summary_max_concepts,
            concept_summary_max_examples,
        )
        if not is_valid:
            validation_errors.extend(summary_errors)

        if validation_errors:
            error_msg = "❌ **Validation Errors:**\n" + "\n".join(
                f"- {error}" for error in validation_errors
            )
            logger.warning(
                "Configuration validation failed",
                extra={"validation_errors": validation_errors},
            )
            return error_msg

        # 4. Build the configuration dictionary for saving.
        # Start with a deepcopy of the current config to preserve sections not edited by the UI.
        current_config_obj = config_manager.load_config()
        config_dict = asdict(current_config_obj)

        # Update sections from UI data
        config_dict["model"]["claimify"]["default"] = ui_data.claimify_default.strip()
        config_dict["model"]["claimify"]["selection"] = (
            ui_data.claimify_selection.strip()
            if ui_data.claimify_selection.strip() != ui_data.claimify_default.strip()
            else None
        )
        config_dict["model"]["claimify"]["disambiguation"] = (
            ui_data.claimify_disambiguation.strip()
            if ui_data.claimify_disambiguation.strip()
            != ui_data.claimify_default.strip()
            else None
        )
        config_dict["model"]["claimify"]["decomposition"] = (
            ui_data.claimify_decomposition.strip()
            if ui_data.claimify_decomposition.strip()
            != ui_data.claimify_default.strip()
            else None
        )
        config_dict["model"]["concept_linker"] = ui_data.concept_linker.strip()
        config_dict["model"]["concept_summary"] = ui_data.concept_summary.strip()
        config_dict["model"]["subject_summary"] = ui_data.subject_summary.strip()
        config_dict["model"]["trending_concepts_agent"] = (
            ui_data.trending_concepts_agent.strip()
        )
        config_dict["model"]["fallback_plugin"] = ui_data.fallback_plugin.strip()

        config_dict["embedding"]["utterance"] = ui_data.utterance_embedding.strip()
        config_dict["embedding"]["concept"] = ui_data.concept_embedding.strip()
        config_dict["embedding"]["summary"] = ui_data.summary_embedding.strip()
        config_dict["embedding"]["fallback"] = ui_data.fallback_embedding.strip()

        config_dict["threshold"]["concept_merge"] = concept_merge
        config_dict["threshold"]["claim_link_strength"] = claim_link_strength

        config_dict["window"]["claimify"]["p"] = window_p
        config_dict["window"]["claimify"]["f"] = window_f

        # Correctly update scheduler jobs
        jobs = config_dict["scheduler"]["jobs"]
        jobs["concept_embedding_refresh"]["enabled"] = ui_data.concept_refresh_enabled
        jobs["concept_embedding_refresh"]["manual_only"] = (
            ui_data.concept_refresh_manual_only
        )
        jobs["concept_embedding_refresh"]["cron"] = str(
            ui_data.concept_refresh_cron
        ).strip()
        jobs["vault_sync"]["enabled"] = ui_data.vault_sync_enabled
        jobs["vault_sync"]["manual_only"] = ui_data.vault_sync_manual_only
        jobs["vault_sync"]["cron"] = str(ui_data.vault_sync_cron).strip()

        top_concepts_job = jobs["top_concepts"]
        top_concepts_job["metric"] = ui_data.top_concepts_metric.strip()
        top_concepts_job["count"] = (
            ui_data.top_concepts_count
            if (ui_data.top_concepts_count or 0) > 0
            else None
        )
        top_concepts_job["percent"] = (
            ui_data.top_concepts_percent
            if (ui_data.top_concepts_percent or 0) > 0
            else None
        )
        top_concepts_job["target_file"] = ui_data.top_concepts_target_file.strip()

        trending_topics_job = jobs["trending_topics"]
        trending_topics_job["window_days"] = ui_data.trending_topics_window_days
        trending_topics_job["count"] = (
            ui_data.trending_topics_count
            if (ui_data.trending_topics_count or 0) > 0
            else None
        )
        trending_topics_job["percent"] = (
            ui_data.trending_topics_percent
            if (ui_data.trending_topics_percent or 0) > 0
            else None
        )
        trending_topics_job["min_mentions"] = ui_data.trending_topics_min_mentions
        trending_topics_job["target_file"] = ui_data.trending_topics_target_file.strip()

        config_dict["subject_summaries"]["similarity_threshold"] = (
            subject_summary_similarity_threshold
        )
        config_dict["subject_summaries"]["min_concepts"] = subject_summary_min_concepts
        config_dict["subject_summaries"]["max_concepts"] = subject_summary_max_concepts
        config_dict["subject_summaries"]["allow_web_search"] = (
            ui_data.subject_summary_allow_web_search
        )
        config_dict["subject_summaries"]["skip_if_incoherent"] = (
            ui_data.subject_summary_skip_if_incoherent
        )

        config_dict["concept_summaries"]["max_examples"] = concept_summary_max_examples
        config_dict["concept_summaries"]["skip_if_no_claims"] = (
            ui_data.concept_summary_skip_if_no_claims
        )
        config_dict["concept_summaries"]["include_see_also"] = (
            ui_data.concept_summary_include_see_also
        )

        # 5. Save the configuration
        try:
            success = config_manager.save_config(config_dict)
            if success:
                logger.info("Configuration saved successfully")
                return "✅ **Configuration saved successfully!**"
            return "❌ **Failed to save configuration.** Please check the logs for details."
        except Exception as e:
            logger.error(
                f"Failed to save configuration: {str(e)}",
                extra={
                    "service": "aclarai-ui",
                    "component": "config_panel",
                    "action": "save_configuration",
                    "error": str(e),
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
        initial_config = load_current_config()
        # Model & Embedding Settings Section
        with gr.Group(elem_id="model_embedding_settings_group"):
            gr.Markdown("## 🤖 Model & Embedding Settings")
            with gr.Group():
                gr.Markdown("### 🔮 Claimify Models")
                with gr.Row():
                    claimify_default_input = gr.Textbox(
                        label="Default Model",
                        value=initial_config.claimify_default,
                        placeholder="gpt-4",
                        info="Default model for all Claimify stages",
                        lines=1,
                    )
                    claimify_selection_input = gr.Textbox(
                        label="Selection Model",
                        value=initial_config.claimify_selection,
                        placeholder="claude-3-opus",
                        info="Model for Claimify selection stage",
                        lines=1,
                    )
                with gr.Row():
                    claimify_disambiguation_input = gr.Textbox(
                        label="Disambiguation Model",
                        value=initial_config.claimify_disambiguation,
                        placeholder="mistral-7b",
                        info="Model for Claimify disambiguation",
                        lines=1,
                    )
                    claimify_decomposition_input = gr.Textbox(
                        label="Decomposition Model",
                        value=initial_config.claimify_decomposition,
                        placeholder="gpt-4",
                        info="Model for Claimify decomposition",
                        lines=1,
                    )
            with gr.Group():
                gr.Markdown("### 🧠 Agent Models")
                with gr.Row():
                    concept_linker_input = gr.Textbox(
                        label="Concept Linker",
                        value=initial_config.concept_linker,
                        placeholder="mistral-7b",
                        info="Used to classify Claim→Concept relationships",
                        lines=1,
                    )
                    concept_summary_input = gr.Textbox(
                        label="Concept Summary",
                        value=initial_config.concept_summary,
                        placeholder="gpt-4",
                        info="Generates individual [[Concept]] Markdown pages",
                        lines=1,
                    )
                with gr.Row():
                    subject_summary_input = gr.Textbox(
                        label="Subject Summary",
                        value=initial_config.subject_summary,
                        placeholder="mistral-7b",
                        info="Generates [[Subject:XYZ]] pages from concept clusters",
                        lines=1,
                    )
                    trending_concepts_agent_input = gr.Textbox(
                        label="Trending Concepts Agent",
                        value=initial_config.trending_concepts_agent,
                        placeholder="gpt-4",
                        info="Writes newsletter-style blurbs for Top/Trending Concepts",
                        lines=1,
                    )
                fallback_plugin_input = gr.Textbox(
                    label="Fallback Plugin",
                    value=initial_config.fallback_plugin,
                    placeholder="openrouter:gemma-2b",
                    info="Used when format detection fails",
                    lines=1,
                )
            with gr.Group():
                gr.Markdown("### 🧬 Embedding Models")
                with gr.Row():
                    utterance_embedding_input = gr.Textbox(
                        label="Utterance Embeddings",
                        value=initial_config.utterance_embedding,
                        placeholder="all-MiniLM-L6-v2",
                        info="Embeddings for Tier 1 utterance blocks",
                        lines=1,
                    )
                    concept_embedding_input = gr.Textbox(
                        label="Concept Embeddings",
                        value=initial_config.concept_embedding,
                        placeholder="text-embedding-3-small",
                        info="Embeddings for Tier 3 concept files",
                        lines=1,
                    )
                with gr.Row():
                    summary_embedding_input = gr.Textbox(
                        label="Summary Embeddings",
                        value=initial_config.summary_embedding,
                        placeholder="sentence-transformers/all-MiniLM-L6-v2",
                        info="Embeddings for Tier 2 summaries",
                        lines=1,
                    )
                    fallback_embedding_input = gr.Textbox(
                        label="Fallback Embeddings",
                        value=initial_config.fallback_embedding,
                        placeholder="sentence-transformers/all-mpnet-base-v2",
                        info="Used if other embedding configs fail or for general purpose",
                        lines=1,
                    )
        # Thresholds & Parameters Section
        with gr.Group(elem_id="thresholds_parameters_group"):
            gr.Markdown("## 📏 Thresholds & Parameters")
            with gr.Row():
                concept_merge_input = gr.Number(
                    label="Concept Merge Threshold",
                    value=initial_config.concept_merge,
                    step=0.01,
                    info="Cosine similarity threshold for merging candidates",
                )
                claim_link_strength_input = gr.Number(
                    label="Claim Link Strength",
                    value=initial_config.claim_link_strength,
                    step=0.01,
                    info="Minimum link strength to create graph edge",
                )
            with gr.Group():
                gr.Markdown("### 🪟 Context Window Parameters")
                gr.Markdown("Configure context window size for Claimify processing.")
                with gr.Row():
                    window_p_input = gr.Number(
                        label="Previous Sentences (p)",
                        value=initial_config.window_p,
                        step=1,
                        info="Number of previous sentences to include in context",
                    )
                    window_f_input = gr.Number(
                        label="Following Sentences (f)",
                        value=initial_config.window_f,
                        step=1,
                        info="Number of following sentences to include in context",
                    )

        # Scheduler Job Controls Section
        with gr.Group(elem_id="scheduler_controls_group"):
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
                        value=initial_config.concept_refresh_enabled,
                        info="Enable or disable this job completely",
                    )
                    concept_refresh_manual_only_input = gr.Checkbox(
                        label="Manual Only",
                        value=initial_config.concept_refresh_manual_only,
                        info="Job can only be triggered manually (not automatically scheduled)",
                    )
                concept_refresh_cron_input = gr.Textbox(
                    label="Cron Schedule",
                    value=str(initial_config.concept_refresh_cron),
                    placeholder="0 3 * * *",
                    info="Cron expression for automatic scheduling (only applies when enabled and not manual-only)",
                    lines=1,
                )

            with gr.Group():
                gr.Markdown("### 📁 Vault Sync")
                gr.Markdown("Synchronizes vault files with knowledge graph")
                with gr.Row():
                    vault_sync_enabled_input = gr.Checkbox(
                        label="Enabled",
                        value=initial_config.vault_sync_enabled,
                        info="Enable or disable this job completely",
                    )
                    vault_sync_manual_only_input = gr.Checkbox(
                        label="Manual Only",
                        value=initial_config.vault_sync_manual_only,
                        info="Job can only be triggered manually (not automatically scheduled)",
                    )
                vault_sync_cron_input = gr.Textbox(
                    label="Cron Schedule",
                    value=str(initial_config.vault_sync_cron),
                    placeholder="*/30 * * * *",
                    info="Cron expression for automatic scheduling (only applies when enabled and not manual-only)",
                    lines=1,
                )

        # Highlight & Summary Section
        with gr.Group(elem_id="highlight_summary_group"):
            gr.Markdown("## 🧠 Highlight & Summary")
            gr.Markdown(
                "Configure concept highlight jobs that generate global summary pages for your vault."
            )

            with gr.Group():
                gr.Markdown("### 🏆 Top Concepts")
                gr.Markdown("Generate ranked lists of most important concepts")
                with gr.Row():
                    top_concepts_metric_input = gr.Dropdown(
                        label="Ranking Metric",
                        value=initial_config.top_concepts_metric,
                        choices=["pagerank", "degree"],
                        info="Algorithm for ranking concepts (PageRank or simple degree centrality)",
                    )
                    top_concepts_count_input = gr.Number(
                        label="Count",
                        value=initial_config.top_concepts_count,
                        step=1,
                        info="Number of top concepts to include (0 or empty to use percent instead)",
                    )
                with gr.Row():
                    top_concepts_percent_input = gr.Number(
                        label="Percent",
                        value=initial_config.top_concepts_percent,
                        step=0.1,
                        info="Percentage of top concepts to include (0 or empty to use count instead)",
                    )
                    top_concepts_target_file_input = gr.Textbox(
                        label="Target File",
                        value=initial_config.top_concepts_target_file,
                        placeholder="Top Concepts.md",
                        info="Output filename for the top concepts page",
                        lines=1,
                    )
                top_concepts_preview = gr.Markdown(
                    value=f"Preview: `{initial_config.top_concepts_target_file}`"
                )

            with gr.Group():
                gr.Markdown("### 📈 Trending Topics")
                gr.Markdown("Track concepts with recent activity increases")
                with gr.Row():
                    trending_topics_window_days_input = gr.Number(
                        label="Window Days",
                        value=initial_config.trending_topics_window_days,
                        step=1,
                        info="Number of days to look back for trend analysis",
                    )
                    trending_topics_count_input = gr.Number(
                        label="Count",
                        value=initial_config.trending_topics_count,
                        step=1,
                        info="Number of trending topics to include (0 or empty to use percent instead)",
                    )
                with gr.Row():
                    trending_topics_percent_input = gr.Number(
                        label="Percent",
                        value=initial_config.trending_topics_percent,
                        step=0.1,
                        info="Percentage of trending topics to include (0 or empty to use count instead)",
                    )
                    trending_topics_min_mentions_input = gr.Number(
                        label="Min Mentions",
                        value=initial_config.trending_topics_min_mentions,
                        step=1,
                        info="Minimum mentions required for a concept to be considered trending",
                    )
                trending_topics_target_file_input = gr.Textbox(
                    label="Target File",
                    value=initial_config.trending_topics_target_file,
                    placeholder="Trending Topics - {date}.md",
                    info="Output filename pattern for trending topics (use {date} for current date)",
                    lines=1,
                )
                trending_topics_preview = gr.Markdown(
                    value=f"Preview: `{initial_config.trending_topics_target_file.replace('{date}', date.today().strftime('%Y-%m-%d'))}`"
                )

            with gr.Group():
                gr.Markdown("### 🎯 Subject Summary Agent")
                gr.Markdown(
                    "Configure the agent that generates [[Subject:XYZ]] pages from concept clusters"
                )
                with gr.Row():
                    subject_summary_similarity_threshold_input = gr.Slider(
                        label="Similarity Threshold",
                        value=initial_config.subject_summary_similarity_threshold,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        info="Cosine similarity threshold for concept clustering (higher = more similar concepts required)",
                    )
                    subject_summary_min_concepts_input = gr.Number(
                        label="Min Concepts",
                        value=initial_config.subject_summary_min_concepts,
                        step=1,
                        info="Minimum number of concepts required to form a subject cluster",
                    )
                with gr.Row():
                    subject_summary_max_concepts_input = gr.Number(
                        label="Max Concepts",
                        value=initial_config.subject_summary_max_concepts,
                        step=1,
                        info="Maximum number of concepts to include in a subject cluster",
                    )
                    subject_summary_allow_web_search_input = gr.Checkbox(
                        label="Allow Web Search",
                        value=initial_config.subject_summary_allow_web_search,
                        info="Allow the agent to use web search for additional context",
                    )
                with gr.Row():
                    subject_summary_skip_if_incoherent_input = gr.Checkbox(
                        label="Skip If Incoherent",
                        value=initial_config.subject_summary_skip_if_incoherent,
                        info="Skip generating subjects for clusters with no shared elements",
                    )

            with gr.Group():
                gr.Markdown("### 📄 Concept Summary Agent")
                gr.Markdown(
                    "Configure the agent that generates [[Concept]] pages for individual concepts"
                )
                with gr.Row():
                    concept_summary_max_examples_input = gr.Number(
                        label="Max Examples",
                        value=initial_config.concept_summary_max_examples,
                        step=1,
                        info="Maximum number of examples to include in concept summaries",
                    )
                    concept_summary_skip_if_no_claims_input = gr.Checkbox(
                        label="Skip If No Claims",
                        value=initial_config.concept_summary_skip_if_no_claims,
                        info="Skip generating summaries for concepts with no associated claims",
                    )
                with gr.Row():
                    concept_summary_include_see_also_input = gr.Checkbox(
                        label="Include See Also",
                        value=initial_config.concept_summary_include_see_also,
                        info="Include 'See Also' sections with related concepts",
                    )

        # Save Section
        with gr.Group(elem_id="save_configuration_group"):
            gr.Markdown("## 💾 Save Configuration")
            with gr.Row():
                save_btn = gr.Button("💾 Save Changes", variant="primary", size="lg")
                reload_btn = gr.Button("🔄 Reload from File", variant="secondary")
            save_status = gr.Markdown(
                value="Make changes above and click **Save Changes** to persist to `settings/aclarai.config.yaml`.",
                label="Status",
                elem_id="config_save_status",
            )

        # Event handlers
        all_inputs = [
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
            top_concepts_metric_input,
            top_concepts_count_input,
            top_concepts_percent_input,
            top_concepts_target_file_input,
            trending_topics_window_days_input,
            trending_topics_count_input,
            trending_topics_percent_input,
            trending_topics_min_mentions_input,
            trending_topics_target_file_input,
            subject_summary_similarity_threshold_input,
            subject_summary_min_concepts_input,
            subject_summary_max_concepts_input,
            subject_summary_allow_web_search_input,
            subject_summary_skip_if_incoherent_input,
            concept_summary_max_examples_input,
            concept_summary_skip_if_no_claims_input,
            concept_summary_include_see_also_input,
        ]

        def reload_configuration() -> Tuple[Any, ...]:
            """Reload configuration from file and return values for all UI components."""
            try:
                config = load_current_config()
                status_message = "🔄 **Configuration reloaded.**"
                gr.Info("Configuration reloaded")
                # Also update the preview markdown components
                top_preview = f"Preview: `{config.top_concepts_target_file}`"
                trending_preview = f"Preview: `{config.trending_topics_target_file.replace('{date}', date.today().strftime('%Y-%m-%d'))}`"

                return (
                    *asdict(config).values(),
                    top_preview,
                    trending_preview,
                    status_message,
                )
            except Exception as e:
                logger.error(
                    "Failed to reload configuration",
                    extra={
                        "service": "aclarai-ui",
                        "component": "config_panel",
                        "action": "reload_configuration",
                        "error": str(e),
                    },
                )
                config = initial_config
                status_message = f"❌ **Error reloading configuration:** {str(e)}"
                gr.Error("Error reloading configuration")
                top_preview = f"Preview: `{config.top_concepts_target_file}`"
                trending_preview = f"Preview: `{config.trending_topics_target_file.replace('{date}', date.today().strftime('%Y-%m-%d'))}`"
                return (
                    *asdict(config).values(),
                    top_preview,
                    trending_preview,
                    status_message,
                )

        save_btn.click(
            fn=save_configuration,
            inputs=all_inputs,
            outputs=[save_status],
        )

        reload_btn.click(
            fn=reload_configuration,
            outputs=all_inputs
            + [top_concepts_preview, trending_topics_preview, save_status],
        )

        # --- Functions and Handlers for Live Previews ---
        def update_top_concepts_preview(filename: str) -> str:
            return f"Preview: `{filename}`"

        def update_trending_topics_preview(filename: str) -> str:
            preview_filename = filename.replace(
                "{date}", date.today().strftime("%Y-%m-%d")
            )
            return f"Preview: `{preview_filename}`"

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

    if not isinstance(interface, gr.Blocks):
        interface = gr.Blocks()
    return interface


if __name__ == "__main__":
    # For testing the configuration panel standalone
    interface = create_configuration_panel()
    interface.launch(debug=True)
