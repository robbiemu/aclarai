"""Tests for the configuration panel functionality."""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml
from aclarai_shared.config import aclaraiConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aclarai_ui.config_panel import (
    ConfigurationManager,
    validate_model_name,
    validate_threshold,
    validate_window_param,
)


class TestConfigurationManager:
    """Test the ConfigurationManager class."""

    def test_load_config_with_user_file(self):
        """Test loading configuration when user file exists."""
        default_config = {
            "model": {
                "claimify": {"default": "gpt-3.5-turbo"},
                "fallback_plugin": "gpt-3.5-turbo",
            },
            "threshold": {"concept_merge": 0.90},
        }
        user_config = {
            "model": {"claimify": {"default": "gpt-4"}},
            "threshold": {"concept_merge": 0.95},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "aclarai.config.yaml"
            default_path = Path(temp_dir) / "aclarai.config.default.yaml"
            # Write test files
            with open(default_path, "w") as f:
                yaml.safe_dump(default_config, f)
            with open(config_path, "w") as f:
                yaml.safe_dump(user_config, f)
            # Test loading
            manager = ConfigurationManager(str(config_path))
            manager.default_config_path = default_path
            config = manager.load_config()
            # Should merge user over default
            assert config.model.claimify.default == "gpt-4"  # From user
            assert config.model.fallback_plugin == "gpt-3.5-turbo"  # From default
            assert config.threshold.concept_merge == 0.95  # From user

    def test_load_config_default_only(self):
        """Test loading configuration when only default file exists."""

        default_config = {
            "model": {"claimify": {"default": "gpt-3.5-turbo"}},
            "threshold": {"concept_merge": 0.90},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "nonexistent.yaml"
            default_path = Path(temp_dir) / "aclarai.config.default.yaml"
            # Only write default file
            with open(default_path, "w") as f:
                yaml.safe_dump(default_config, f)
            manager = ConfigurationManager(str(config_path))
            manager.default_config_path = default_path
            config = manager.load_config()
            # Should use defaults
            assert config.model.claimify.default == "gpt-3.5-turbo"
            assert config.threshold.concept_merge == 0.90

    def test_save_config(self):
        """Test saving configuration to file."""

        test_config = {
            "model": {"claimify": {"default": "gpt-4"}},
            "threshold": {"concept_merge": 0.95},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            manager = ConfigurationManager(str(config_path))
            success = manager.save_config(test_config)
            assert success is True
            assert config_path.exists()
            # Verify content
            with open(config_path, "r") as f:
                saved_config = yaml.safe_load(f)
            assert saved_config == test_config

    def test_deep_merge_configs(self):
        """Test the deep merge functionality."""
        default = {"a": {"b": {"c": 1, "d": 2}, "e": 3}, "f": 4}
        user = {"a": {"b": {"c": 10}, "g": 5}, "h": 6}
        result = aclaraiConfig._deep_merge_configs(default, user)
        expected = {"a": {"b": {"c": 10, "d": 2}, "e": 3, "g": 5}, "f": 4, "h": 6}
        assert result == expected


class TestConceptHighlightsValidation:
    """Test validation functions for concept highlights configuration."""

    def test_concept_highlights_mutual_exclusivity_validation(self):
        """Test that count and percent are mutually exclusive."""
        from aclarai_ui.config_panel import validate_concept_highlights_config

        # Test that using both count and percent triggers validation error
        is_valid, errors = validate_concept_highlights_config(
            "pagerank",
            25,
            5.0,
            "Top Concepts.md",  # Both count and percent non-zero
            7,
            0,
            5.0,
            2,
            "Trending Topics - {date}.md",
        )

        assert not is_valid
        assert any(
            "count and percent cannot both be used" in error.lower() for error in errors
        )

    def test_concept_highlights_metric_validation(self):
        """Test that metric is validated correctly."""
        from aclarai_ui.config_panel import validate_concept_highlights_config

        # Test invalid metric
        is_valid, errors = validate_concept_highlights_config(
            "invalid_metric",
            25,
            0.0,
            "Top Concepts.md",
            7,
            0,
            5.0,
            2,
            "Trending Topics - {date}.md",
        )

        assert not is_valid
        assert any(
            "metric must be either 'pagerank' or 'degree'" in error.lower()
            for error in errors
        )

    def test_concept_highlights_filename_validation(self):
        """Test that target filenames are validated."""
        from aclarai_ui.config_panel import validate_concept_highlights_config

        # Test empty filename
        is_valid, errors = validate_concept_highlights_config(
            "pagerank",
            25,
            0.0,
            "",  # Empty filename
            7,
            0,
            5.0,
            2,
            "Trending Topics - {date}.md",
        )

        assert not is_valid
        assert any("target_file cannot be empty" in error.lower() for error in errors)

    def test_concept_highlights_percentage_range_validation(self):
        """Test that percentages are within valid range."""
        from aclarai_ui.config_panel import validate_concept_highlights_config

        # Test percentage > 100
        is_valid, errors = validate_concept_highlights_config(
            "pagerank",
            0,
            150.0,
            "Top Concepts.md",  # Percentage > 100
            7,
            0,
            5.0,
            2,
            "Trending Topics - {date}.md",
        )

        assert not is_valid
        assert any(
            "percent must be between 0 and 100" in error.lower() for error in errors
        )

    def test_concept_highlights_valid_configuration(self):
        """Test that valid concept highlights configuration passes validation."""
        from aclarai_ui.config_panel import validate_concept_highlights_config

        # Test valid configuration
        is_valid, errors = validate_concept_highlights_config(
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

        assert is_valid
        assert len(errors) == 0

    def test_concept_highlights_window_days_validation(self):
        """Test that window_days is validated correctly."""
        from aclarai_ui.config_panel import validate_concept_highlights_config

        # Test window_days < 1
        is_valid, errors = validate_concept_highlights_config(
            "pagerank",
            25,
            0.0,
            "Top Concepts.md",
            0,
            0,
            5.0,
            2,
            "Trending Topics - {date}.md",  # window_days = 0
        )

        assert not is_valid
        assert any(
            "window_days must be at least 1" in error.lower() for error in errors
        )

    def test_concept_highlights_min_mentions_validation(self):
        """Test that min_mentions is validated correctly."""
        from aclarai_ui.config_panel import validate_concept_highlights_config

        # Test negative min_mentions
        is_valid, errors = validate_concept_highlights_config(
            "pagerank",
            25,
            0.0,
            "Top Concepts.md",
            7,
            0,
            5.0,
            -1,
            "Trending Topics - {date}.md",  # min_mentions = -1
        )

        assert not is_valid
        assert any(
            "min_mentions must be non-negative" in error.lower() for error in errors
        )


class TestSummaryAgentsValidation:
    """Test validation functions for Subject Summary and Concept Summary configurations."""

    def test_validate_summary_agents_config_valid_values(self):
        """Test validation with valid parameter values."""
        from aclarai_ui.config_panel import validate_summary_agents_config

        is_valid, errors = validate_summary_agents_config(
            subject_summary_similarity_threshold=0.92,
            subject_summary_min_concepts=3,
            subject_summary_max_concepts=15,
            concept_summary_max_examples=5,
        )
        assert is_valid
        assert len(errors) == 0

    def test_validate_summary_agents_config_invalid_threshold(self):
        """Test validation with invalid similarity threshold."""
        from aclarai_ui.config_panel import validate_summary_agents_config

        is_valid, errors = validate_summary_agents_config(
            subject_summary_similarity_threshold=1.5,  # Invalid: > 1.0
            subject_summary_min_concepts=3,
            subject_summary_max_concepts=15,
            concept_summary_max_examples=5,
        )
        assert not is_valid
        assert "Similarity threshold must be between 0.0 and 1.0" in errors

    def test_validate_summary_agents_config_invalid_min_concepts(self):
        """Test validation with invalid minimum concepts."""
        from aclarai_ui.config_panel import validate_summary_agents_config

        is_valid, errors = validate_summary_agents_config(
            subject_summary_similarity_threshold=0.92,
            subject_summary_min_concepts=0,  # Invalid: < 1
            subject_summary_max_concepts=15,
            concept_summary_max_examples=5,
        )
        assert not is_valid
        assert "Minimum concepts must be between 1 and 100" in errors

    def test_validate_summary_agents_config_invalid_max_concepts(self):
        """Test validation with invalid maximum concepts."""
        from aclarai_ui.config_panel import validate_summary_agents_config

        is_valid, errors = validate_summary_agents_config(
            subject_summary_similarity_threshold=0.92,
            subject_summary_min_concepts=3,
            subject_summary_max_concepts=101,  # Invalid: > 100
            concept_summary_max_examples=5,
        )
        assert not is_valid
        assert "Maximum concepts must be between 1 and 100" in errors

    def test_validate_summary_agents_config_min_greater_than_max(self):
        """Test validation when min concepts is greater than max concepts."""
        from aclarai_ui.config_panel import validate_summary_agents_config

        is_valid, errors = validate_summary_agents_config(
            subject_summary_similarity_threshold=0.92,
            subject_summary_min_concepts=10,
            subject_summary_max_concepts=5,  # Invalid: < min_concepts
            concept_summary_max_examples=5,
        )
        assert not is_valid
        assert "Minimum concepts cannot be greater than maximum concepts" in errors

    def test_validate_summary_agents_config_invalid_max_examples(self):
        """Test validation with invalid maximum examples."""
        from aclarai_ui.config_panel import validate_summary_agents_config

        is_valid, errors = validate_summary_agents_config(
            subject_summary_similarity_threshold=0.92,
            subject_summary_min_concepts=3,
            subject_summary_max_concepts=15,
            concept_summary_max_examples=21,  # Invalid: > 20
        )
        assert not is_valid
        assert "Maximum examples must be between 0 and 20" in errors


class TestValidationFunctions:
    """Test validation functions."""

    def test_validate_model_name_valid(self):
        """Test validation of valid model names."""
        valid_models = [
            "gpt-4",
            "gpt-3.5-turbo",
            "claude-3-opus",
            "mistral-7b",
            "openrouter:gemma-2b",
            "ollama:llama2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "text-embedding-3-small",
            "huggingface/model-name",
        ]
        for model in valid_models:
            is_valid, error = validate_model_name(model)
            assert is_valid, f"Model {model} should be valid but got error: {error}"
            assert error == ""

    def test_validate_model_name_invalid(self):
        """Test validation of invalid model names."""
        invalid_models = [
            "",
            "   ",
            "invalid-model",
            "/starts-with-slash",
            "spaces in name",
        ]
        for model in invalid_models:
            is_valid, error = validate_model_name(model)
            if model.strip():  # Only test non-empty strings for format validation
                assert not is_valid, f"Model {model} should be invalid"
                assert error != ""

    def test_validate_threshold_valid(self):
        """Test validation of valid threshold values."""
        valid_thresholds = [0.0, 0.5, 0.90, 1.0, 0.123]
        for threshold in valid_thresholds:
            is_valid, error = validate_threshold(threshold)
            assert is_valid, (
                f"Threshold {threshold} should be valid but got error: {error}"
            )
            assert error == ""

    def test_validate_threshold_invalid(self):
        """Test validation of invalid threshold values."""
        invalid_thresholds = [-0.1, 1.1, "string", None]
        for threshold in invalid_thresholds:
            is_valid, error = validate_threshold(threshold)
            assert not is_valid, f"Threshold {threshold} should be invalid"
            assert error != ""

    def test_validate_window_param_valid(self):
        """Test validation of valid window parameters."""
        valid_params = [0, 1, 3, 5, 10]
        for param in valid_params:
            is_valid, error = validate_window_param(param)
            assert is_valid, (
                f"Window param {param} should be valid but got error: {error}"
            )
            assert error == ""

    def test_validate_window_param_invalid(self):
        """Test validation of invalid window parameters."""
        invalid_params = [-1, 11, 3.5, "string", None]
        for param in invalid_params:
            is_valid, error = validate_window_param(param)
            assert not is_valid, f"Window param {param} should be invalid"
            assert error != ""


if __name__ == "__main__":
    pytest.main([__file__])
