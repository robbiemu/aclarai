"""Integration tests for the configuration panel UI functionality."""

import os
import sys
from datetime import date
from typing import Generator

import pytest
from playwright.sync_api import Page, expect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestConfigurationPanelIntegration:
    """Integration tests for configuration panel UI using Playwright."""

    @pytest.fixture(scope="class")
    def gradio_app(self) -> str:
        """Return the URL of the running Gradio app for integration testing."""
        return "http://127.0.0.1:7860"

    @pytest.fixture(scope="function")
    def reset_to_default_config(
        self, page: Page, gradio_app: str
    ) -> Generator[Page, None, None]:
        """
        Fixture to ensure each test starts with a clean, default configuration.
        This runs before each test function. It works by "dirtying" the form
        and then reloading to guarantee the reload action and confirmation occur.
        """
        page.goto(gradio_app, wait_until="domcontentloaded")
        config_tab = page.get_by_role("tab", name="⚙️ Configuration")
        expect(config_tab).to_be_visible(timeout=10000)
        config_tab.click()

        expect(
            page.get_by_role("heading", name="⚙️ aclarai Configuration Panel", level=1)
        ).to_be_visible()

        page.get_by_label("Concept Linker").fill("dirty-the-form-for-reset")
        page.get_by_role("button", name="Reload from File").click()

        reloaded_toast = page.get_by_test_id("toast-body").get_by_text(
            "Configuration reloaded"
        )
        expect(reloaded_toast).to_be_visible(timeout=5000)

        page.locator("button.toast-close").first.click()
        expect(reloaded_toast).to_be_hidden()

        yield page

    @pytest.mark.integration
    def test_configuration_panel_loads(self, reset_to_default_config: Page):
        """Test that the configuration panel loads correctly."""
        page = reset_to_default_config
        expect(
            page.get_by_role("heading", name="🤖 Model & Embedding Settings", level=2)
        ).to_be_visible()
        expect(
            page.get_by_role("heading", name="📏 Thresholds & Parameters", level=2)
        ).to_be_visible()
        expect(
            page.get_by_role("heading", name="🧠 Highlight & Summary", level=2)
        ).to_be_visible()

    @pytest.mark.integration
    def test_model_input_validation(self, reset_to_default_config: Page):
        """Test that model input validation works in the UI."""
        page = reset_to_default_config
        page.get_by_label("Default Model").fill("invalid-model-name")
        page.get_by_role("button", name="Save Changes").click()

        # 1. Locate the status area directly by its unique element ID.
        #    This is the most robust selector possible.
        status_area = page.locator("#config_save_status")

        # 2. Assert that this specific component contains our error messages.
        expect(status_area).to_contain_text("❌ Validation Errors:", timeout=5000)
        expect(status_area).to_contain_text("Invalid model name format")

    @pytest.mark.integration
    def test_threshold_input_validation(self, reset_to_default_config: Page):
        """Test that threshold input validation works in the UI."""
        page = reset_to_default_config
        page.get_by_label("Concept Merge Threshold").fill("1.5")
        page.get_by_role("button", name="Save Changes").click()

        # Use the successful pattern: locate the status area by its unique ID.
        status_area = page.locator("#config_save_status")

        # Assert that the status area contains the expected error texts.
        expect(status_area).to_contain_text("❌ Validation Errors:", timeout=5000)
        expect(status_area).to_contain_text("Threshold must be between 0.0 and 1.0")

    @pytest.mark.integration
    def test_window_parameter_validation(self, reset_to_default_config: Page):
        """Test that window parameter validation works in the UI."""
        page = reset_to_default_config

        # Use the correct label from the UI
        page.get_by_label("Previous Sentences (p)").fill("15")
        page.get_by_role("button", name="Save Changes").click()

        # Use the successful pattern: locate the status area by its unique ID.
        status_area = page.locator("#config_save_status")

        # Assert that the status area contains the expected error texts.
        expect(status_area).to_contain_text("❌ Validation Errors:", timeout=5000)
        expect(status_area).to_contain_text("Window parameter must be between 0 and 10")

    @pytest.mark.integration
    def test_successful_configuration_save(self, reset_to_default_config: Page):
        """Test that valid configuration can be saved successfully."""
        page = reset_to_default_config
        page.get_by_label("Concept Linker").fill("claude-3-opus-20240229")
        page.get_by_role("button", name="Save Changes").click()

        # Use the successful pattern: locate the status area by its unique ID.
        status_area = page.locator("#config_save_status")

        # Assert that the status area contains the expected success message.
        expect(status_area).to_contain_text(
            "✅ Configuration saved successfully!", timeout=5000
        )

    @pytest.mark.integration
    def test_reload_configuration(self, reset_to_default_config: Page):
        """Test that configuration can be reloaded from file, discarding changes."""
        page = reset_to_default_config
        concept_linker_input = page.get_by_label("Concept Linker")
        initial_value = concept_linker_input.input_value()
        new_value = "a-completely-new-value"
        concept_linker_input.fill(new_value)
        expect(concept_linker_input).to_have_value(new_value)
        page.get_by_role("button", name="Reload from File").click()
        expect(concept_linker_input).to_have_value(initial_value, timeout=5000)
        expect(
            page.get_by_test_id("toast-body").get_by_text("Configuration reloaded")
        ).to_be_visible()

    @pytest.mark.integration
    def test_all_input_fields_present(self, reset_to_default_config: Page):
        """Test that all expected input fields are present in the UI."""
        page = reset_to_default_config

        # 1. Test all the truly UNAMBIGUOUS labels in a simple loop.
        # "Similarity Threshold" has been removed from this list.
        unambiguous_labels = [
            "Default Model",
            "Concept Linker",
            "Concept Summary",
            "Subject Summary",
            "Fallback Plugin",
            "Utterance Embeddings",
            "Concept Embeddings",
            "Summary Embeddings",
            "Fallback Embeddings",
            "Concept Merge Threshold",
            "Claim Link Strength",
            "Previous Sentences (p)",
            "Following Sentences (f)",
            "Ranking Metric",
            "Window Days",
            "Min Mentions",
            "Min Concepts",
            "Max Concepts",
            "Max Examples",
        ]
        for label in unambiguous_labels:
            expect(page.get_by_label(label)).to_be_visible()

        # 2. Handle the ambiguous "Trending Concepts Agent" label by scoping the search.
        agent_models_group = page.locator("div.gr-group").filter(
            has=page.get_by_role("heading", name="🧠 Agent Models")
        )
        expect(
            agent_models_group.get_by_label("Trending Concepts Agent")
        ).to_be_visible()

        writing_agent_group = page.locator("div.gr-group").filter(
            has=page.get_by_role("heading", name="🤖 Writing Agent")
        )
        expect(
            writing_agent_group.get_by_label("Model for Trending Concepts Agent")
        ).to_be_visible()

        # 3. Handle the ambiguous "Similarity Threshold" label from the gr.Slider.
        # We can assert that both the number input and the slider role are visible.
        # The label is associated with both, so we need to be more specific.
        expect(
            page.get_by_role("spinbutton", name="Similarity Threshold")
        ).to_be_visible()
        expect(page.get_by_role("slider", name="Similarity Threshold")).to_be_visible()

    @pytest.mark.integration
    def test_filename_preview_functionality(self, reset_to_default_config: Page):
        """Test that filename previews update correctly."""
        page = reset_to_default_config

        # 1. Locate and fill the input field. This part is correct.
        trending_file_input = page.get_by_placeholder("Trending Topics - {date}.md")
        trending_file_input.fill("My-Topics-{date}.md")

        # 2. Locate the specific preview element we want to observe.
        # The error message shows two "Preview:" paragraphs are found.
        # The first is for "Top Concepts", the second is for "Trending Topics".
        # We use .nth(1) to select the second one (zero-indexed).
        preview_element = page.locator("p:has-text('Preview:')").nth(1)

        # 3. Assert that this uniquely located element is visible and contains the pattern.
        # Instead of checking for an exact date, verify the pattern substitution worked
        expect(preview_element).to_be_visible(timeout=3000)
        expect(preview_element).to_contain_text("My-Topics-")
        expect(preview_element).to_contain_text(".md")
        # Verify the date pattern was replaced with an actual date (YYYY-MM-DD format)
        preview_text = preview_element.text_content()
        import re
        date_pattern = r"My-Topics-(\d{4}-\d{2}-\d{2})\.md"
        assert re.search(date_pattern, preview_text), f"Expected date pattern in preview text: {preview_text}"

    @pytest.mark.integration
    def test_subject_summary_validation(self, reset_to_default_config: Page):
        """Test validation for the Subject Summary agent settings."""
        page = reset_to_default_config
        page.get_by_label("Min Concepts").fill("10")
        page.get_by_label("Max Concepts").fill("5")
        page.get_by_role("button", name="Save Changes").click()

        # Use the successful pattern: locate the status area by its unique ID.
        status_area = page.locator("#config_save_status")

        # Assert that the status area contains the expected error texts.
        expect(status_area).to_contain_text("❌ Validation Errors:", timeout=5000)
        expect(status_area).to_contain_text(
            "Minimum concepts cannot be greater than maximum concepts"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
