"""Integration tests for the configuration panel UI functionality."""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml
from playwright.sync_api import Page, expect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aclarai_ui.config_panel import create_configuration_panel


class TestConfigurationPanelIntegration:
    """Integration tests for configuration panel UI using Playwright."""

    @pytest.fixture(scope="class")
    def gradio_app(self):
        """Create and launch Gradio app for testing."""
        # Create configuration panel interface
        interface = create_configuration_panel()
        # Launch with a free port
        interface.launch(
            server_name="127.0.0.1",
            server_port=0,  # Let Gradio choose a free port
            share=False,
            debug=False,
            quiet=True,
            prevent_thread_lock=True,
        )
        # Get the actual port and URL
        url = interface.local_url
        yield url
        # Cleanup: close the interface
        interface.close()

    @pytest.fixture
    def temp_config_files(self):
        """Create temporary configuration files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "aclarai.config.yaml"
            default_path = Path(temp_dir) / "aclarai.config.default.yaml"
            # Create default configuration
            default_config = {
                "model": {
                    "claimify": {"default": "gpt-3.5-turbo"},
                    "concept_linker": "gpt-3.5-turbo",
                    "concept_summary": "gpt-4",
                    "subject_summary": "claude-3-sonnet",
                    "trending_concepts_agent": "gpt-3.5-turbo",
                    "fallback_plugin": "gpt-3.5-turbo",
                },
                "embedding": {
                    "utterance": "text-embedding-3-small",
                    "concept": "sentence-transformers/all-MiniLM-L6-v2",
                    "summary": "text-embedding-3-small",
                    "fallback": "text-embedding-3-small",
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
            with open(default_path, "w") as f:
                yaml.safe_dump(default_config, f)
            yield config_path, default_path, default_config

    @pytest.mark.integration
    def test_configuration_panel_loads(self, page: Page, gradio_app):
        """Test that the configuration panel loads correctly."""
        page.goto(gradio_app)
        # Wait for the page to load
        page.wait_for_selector("h1", timeout=10000)
        # Check that the main heading is present
        expect(page.locator("h1")).to_contain_text("Configuration Panel")
        # Check that key sections are present
        expect(page.locator("text=Model & Embedding Settings")).to_be_visible()
        expect(page.locator("text=Thresholds & Parameters")).to_be_visible()

    @pytest.mark.integration
    def test_model_input_validation(self, page: Page, gradio_app):
        """Test that model input validation works in the UI."""
        page.goto(gradio_app)
        page.wait_for_selector("h1", timeout=10000)
        # Find the claimify default model input
        claimify_default_input = page.locator(
            'input[placeholder*="gpt-3.5-turbo"]'
        ).first
        # Enter an invalid model name
        claimify_default_input.fill("invalid-model-name")
        # Find and click the save button
        save_button = page.locator('button:has-text("Save Changes")')
        save_button.click()
        # Wait for validation response
        page.wait_for_timeout(1000)
        # Check that validation error appears
        expect(page.locator("text=Validation Errors")).to_be_visible()
        expect(page.locator("text=Claimify Default")).to_be_visible()

    @pytest.mark.integration
    def test_threshold_input_validation(self, page: Page, gradio_app):
        """Test that threshold input validation works in the UI."""
        page.goto(gradio_app)
        page.wait_for_selector("h1", timeout=10000)
        # Find the concept merge threshold input
        threshold_input = page.locator('input[type="number"]').first
        # Enter an invalid threshold value (> 1.0)
        threshold_input.fill("1.5")
        # Find and click the save button
        save_button = page.locator('button:has-text("Save Changes")')
        save_button.click()
        # Wait for validation response
        page.wait_for_timeout(1000)
        # Check that validation error appears
        expect(page.locator("text=Validation Errors")).to_be_visible()

    @pytest.mark.integration
    def test_window_parameter_validation(self, page: Page, gradio_app):
        """Test that window parameter validation works in the UI."""
        page.goto(gradio_app)
        page.wait_for_selector("h1", timeout=10000)
        # Find the window parameter inputs and enter an invalid window value (> 10)
        # Find the "Previous Sentences" input specifically
        page.locator("text=Previous Sentences").locator("..//input").fill("15")
        # Find and click the save button
        save_button = page.locator('button:has-text("Save Changes")')
        save_button.click()
        # Wait for validation response
        page.wait_for_timeout(1000)
        # Check that validation error appears
        expect(page.locator("text=Validation Errors")).to_be_visible()

    @pytest.mark.integration
    def test_successful_configuration_save(self, page: Page, gradio_app):
        """Test that valid configuration can be saved successfully."""
        page.goto(gradio_app)
        page.wait_for_selector("h1", timeout=10000)
        # Fill in valid configuration values
        claimify_default_input = page.locator(
            'input[placeholder*="gpt-3.5-turbo"]'
        ).first
        claimify_default_input.fill("gpt-4")
        # Find concept linker input and update it
        concept_linker_input = page.locator("text=Concept Linker").locator("..//input")
        concept_linker_input.fill("claude-3-opus")
        # Find and click the save button
        save_button = page.locator('button:has-text("Save Changes")')
        save_button.click()
        # Wait for save response
        page.wait_for_timeout(2000)
        # Check that success message appears
        expect(page.locator("text=Configuration saved successfully")).to_be_visible()

    @pytest.mark.integration
    def test_reload_configuration(self, page: Page, gradio_app):
        """Test that configuration can be reloaded from file."""
        page.goto(gradio_app)
        page.wait_for_selector("h1", timeout=10000)
        # Change a value
        claimify_default_input = page.locator(
            'input[placeholder*="gpt-3.5-turbo"]'
        ).first
        claimify_default_input.fill("changed-value")
        # Find and click the reload button
        reload_button = page.locator('button:has-text("Reload from File")')
        reload_button.click()
        # Wait for reload response
        page.wait_for_timeout(2000)
        # Check that value is restored
        # Note: This depends on having a valid config file, so may restore to default
        current_value = claimify_default_input.input_value()
        # The value should either be the original or a default from file
        assert current_value != "changed-value"
        # Check that reload message appears
        expect(page.locator("text=Configuration reloaded")).to_be_visible()

    @pytest.mark.integration
    def test_all_input_fields_present(self, page: Page, gradio_app):
        """Test that all expected input fields are present in the UI."""
        page.goto(gradio_app)
        page.wait_for_selector("h1", timeout=10000)
        # Check that all expected input labels are present
        expected_labels = [
            "Default Model",
            "Selection Model",
            "Disambiguation Model",
            "Decomposition Model",
            "Concept Linker",
            "Concept Summary",
            "Subject Summary",
            "Trending Concepts Agent",
            "Fallback Plugin",
            "Utterance Embeddings",
            "Concept Embeddings",
            "Summary Embeddings",
            "Fallback Embeddings",
            "Concept Merge Threshold",
            "Claim Link Strength",
            "Previous Sentences",
            "Following Sentences",
        ]
        for label in expected_labels:
            expect(page.locator(f"text={label}")).to_be_visible()

    @pytest.mark.integration
    def test_buttons_present_and_functional(self, page: Page, gradio_app):
        """Test that save and reload buttons are present and clickable."""
        page.goto(gradio_app)
        page.wait_for_selector("h1", timeout=10000)
        # Check that buttons are present
        save_button = page.locator('button:has-text("Save Changes")')
        reload_button = page.locator('button:has-text("Reload from File")')
        expect(save_button).to_be_visible()
        expect(reload_button).to_be_visible()
        # Test that buttons are clickable
        expect(save_button).to_be_enabled()
        expect(reload_button).to_be_enabled()

    @pytest.mark.integration
    def test_concept_highlights_section_present(self, page: Page, gradio_app):
        """Test that the Highlight & Summary section is present."""
        page.goto(gradio_app)
        page.wait_for_selector("h1", timeout=10000)

        # Check that the Highlight & Summary section is present
        expect(page.locator("text=🧠 Highlight & Summary")).to_be_visible()
        expect(page.locator("text=Writing Agent")).to_be_visible()
        expect(page.locator("text=Top Concepts")).to_be_visible()
        expect(page.locator("text=Trending Topics")).to_be_visible()

    @pytest.mark.integration
    def test_concept_highlights_inputs_present(self, page: Page, gradio_app):
        """Test that all concept highlights input fields are present."""
        page.goto(gradio_app)
        page.wait_for_selector("h1", timeout=10000)

        # Check Top Concepts inputs
        expect(page.locator("text=Ranking Metric")).to_be_visible()
        expect(page.locator("text=Count").nth(0)).to_be_visible()  # Top concepts count
        expect(
            page.locator("text=Percent").nth(0)
        ).to_be_visible()  # Top concepts percent
        expect(
            page.locator("text=Target File").nth(0)
        ).to_be_visible()  # Top concepts file

        # Check Trending Topics inputs
        expect(page.locator("text=Window Days")).to_be_visible()
        expect(page.locator("text=Min Mentions")).to_be_visible()
        expect(
            page.locator("text=Count").nth(1)
        ).to_be_visible()  # Trending topics count
        expect(
            page.locator("text=Percent").nth(1)
        ).to_be_visible()  # Trending topics percent
        expect(
            page.locator("text=Target File").nth(1)
        ).to_be_visible()  # Trending topics file

    @pytest.mark.integration
    def test_trending_concepts_agent_synchronization(self, page: Page, gradio_app):
        """Test that trending concepts agent inputs are synchronized."""
        page.goto(gradio_app)
        page.wait_for_selector("h1", timeout=10000)

        # Find the trending concepts agent input in the Model section
        main_trending_agent_input = page.locator(
            "text=Trending Concepts Agent"
        ).locator("..//input")

        # Find the trending concepts agent input in the Highlight section
        summary_trending_agent_input = page.locator(
            "text=Model for Trending Concepts Agent"
        ).locator("..//input")

        # Change value in main input
        main_trending_agent_input.fill("gpt-4-custom")
        page.wait_for_timeout(500)  # Wait for synchronization

        # Check that summary input is updated
        summary_value = summary_trending_agent_input.input_value()
        assert summary_value == "gpt-4-custom"

        # Change value in summary input
        summary_trending_agent_input.fill("claude-3-opus")
        page.wait_for_timeout(500)  # Wait for synchronization

        # Check that main input is updated
        main_value = main_trending_agent_input.input_value()
        assert main_value == "claude-3-opus"

    @pytest.mark.integration
    def test_concept_highlights_validation_messages(self, page: Page, gradio_app):
        """Test that concept highlights validation shows appropriate messages."""
        page.goto(gradio_app)
        page.wait_for_selector("h1", timeout=10000)

        # Fill in invalid values
        # Set both count and percent for top concepts (should be mutually exclusive)
        count_input = page.locator("text=Count").nth(0).locator("..//input")
        percent_input = page.locator("text=Percent").nth(0).locator("..//input")

        count_input.fill("25")
        percent_input.fill("10")

        # Set invalid metric
        metric_dropdown = page.locator("text=Ranking Metric").locator("..//select")
        # Note: We can't actually set an invalid option in a dropdown, so we'll test this differently

        # Click save to trigger validation
        save_button = page.locator('button:has-text("Save Changes")')
        save_button.click()
        page.wait_for_timeout(2000)

        # Check that validation error appears
        expect(page.locator("text=Validation Errors")).to_be_visible()

    @pytest.mark.integration
    def test_filename_preview_functionality(self, page: Page, gradio_app):
        """Test that filename previews update correctly."""
        page.goto(gradio_app)
        page.wait_for_selector("h1", timeout=10000)

        # Find the trending topics target file input
        trending_file_input = (
            page.locator("text=Target File").nth(1).locator("..//input")
        )

        # Change the filename pattern
        trending_file_input.fill("My Topics - {date}.md")
        page.wait_for_timeout(500)  # Wait for preview update

        # Check that preview is updated (should contain current date)
        from datetime import date

        expected_date = date.today().strftime("%Y-%m-%d")
        expect(page.locator(f"text=My Topics - {expected_date}.md")).to_be_visible()

    @pytest.mark.integration
    def test_subject_summary_ui_elements(self, page: Page, gradio_app):
        """Test that Subject Summary UI elements are present and interactive."""
        page.goto(gradio_app)
        page.wait_for_selector("h1", timeout=10000)
        # Check Subject Summary section header
        expect(page.locator("text=🎯 Subject Summary Agent")).to_be_visible()
        # Test Similarity Threshold slider
        slider = page.locator("text=Similarity Threshold").locator("..//input")
        expect(slider).to_be_visible()
        expect(slider).to_have_attribute("min", "0.0")
        expect(slider).to_have_attribute("max", "1.0")
        expect(slider).to_have_value("0.92")  # Default value
        # Test Min/Max Concepts inputs
        min_concepts = page.locator("text=Min Concepts").locator("..//input")
        max_concepts = page.locator("text=Max Concepts").locator("..//input")
        expect(min_concepts).to_be_visible()
        expect(max_concepts).to_be_visible()
        expect(min_concepts).to_have_value("3")  # Default value
        expect(max_concepts).to_have_value("15")  # Default value
        # Test checkboxes
        web_search = page.locator("text=Allow Web Search").locator("..//input")
        skip_incoherent = page.locator("text=Skip If Incoherent").locator("..//input")
        expect(web_search).to_be_visible()
        expect(skip_incoherent).to_be_visible()

    @pytest.mark.integration
    def test_concept_summary_ui_elements(self, page: Page, gradio_app):
        """Test that Concept Summary UI elements are present and interactive."""
        page.goto(gradio_app)
        page.wait_for_selector("h1", timeout=10000)
        # Check Concept Summary section header
        expect(page.locator("text=📄 Concept Summary Agent")).to_be_visible()
        # Test Max Examples input
        max_examples = page.locator("text=Max Examples").locator("..//input")
        expect(max_examples).to_be_visible()
        expect(max_examples).to_have_attribute("min", "0")
        expect(max_examples).to_have_attribute("max", "20")
        expect(max_examples).to_have_value("5")  # Default value
        # Test checkboxes
        skip_no_claims = page.locator("text=Skip If No Claims").locator("..//input")
        include_see_also = page.locator("text=Include See Also").locator("..//input")
        expect(skip_no_claims).to_be_visible()
        expect(include_see_also).to_be_visible()

    @pytest.mark.integration
    def test_subject_summary_configuration_save_reload(self, page: Page, gradio_app):
        """Test saving and reloading Subject Summary configuration changes."""
        page.goto(gradio_app)
        page.wait_for_selector("h1", timeout=10000)
        # Make changes to Subject Summary settings
        page.locator("text=Similarity Threshold").locator("..//input").fill("0.85")
        page.locator("text=Min Concepts").locator("..//input").fill("5")
        page.locator("text=Max Concepts").locator("..//input").fill("12")
        page.locator("text=Allow Web Search").locator("..//input").check()
        page.locator("text=Skip If Incoherent").locator("..//input").check()

        # Save changes
        page.locator('button:has-text("Save Changes")').click()
        page.wait_for_timeout(2000)

        # Verify success message
        expect(page.locator("text=Configuration saved successfully")).to_be_visible()

        # Change some values
        page.locator("text=Min Concepts").locator("..//input").fill("8")
        page.locator("text=Max Concepts").locator("..//input").fill("20")

        # Reload configuration
        page.locator('button:has-text("Reload from File")').click()
        page.wait_for_timeout(2000)

        # Verify original values are restored
        expect(page.locator("text=Min Concepts").locator("..//input")).to_have_value(
            "5"
        )
        expect(page.locator("text=Max Concepts").locator("..//input")).to_have_value(
            "12"
        )

    @pytest.mark.integration
    def test_concept_summary_configuration_save_reload(self, page: Page, gradio_app):
        """Test saving and reloading Concept Summary configuration changes."""
        page.goto(gradio_app)
        page.wait_for_selector("h1", timeout=10000)

        # Make changes to Concept Summary settings
        page.locator("text=Max Examples").locator("..//input").fill("10")
        page.locator("text=Skip If No Claims").locator("..//input").check()
        page.locator("text=Include See Also").locator("..//input").uncheck()

        # Save changes
        page.locator('button:has-text("Save Changes")').click()
        page.wait_for_timeout(2000)

        # Verify success message
        expect(page.locator("text=Configuration saved successfully")).to_be_visible()

        # Change some values
        page.locator("text=Max Examples").locator("..//input").fill("15")
        page.locator("text=Include See Also").locator("..//input").check()

        # Reload configuration
        page.locator('button:has-text("Reload from File")').click()
        page.wait_for_timeout(2000)

        # Verify original values are restored
        expect(page.locator("text=Max Examples").locator("..//input")).to_have_value(
            "10"
        )
        expect(
            page.locator("text=Include See Also").locator("..//input")
        ).not_to_be_checked()

    @pytest.mark.integration
    def test_summary_agents_validation_messages(self, page: Page, gradio_app):
        """Test that validation errors are displayed correctly for Summary agents."""
        page.goto(gradio_app)
        page.wait_for_selector("h1", timeout=10000)

        # Set invalid values
        page.locator("text=Similarity Threshold").locator("..//input").fill(
            "1.5"
        )  # Invalid: > 1.0
        page.locator("text=Min Concepts").locator("..//input").fill("10")
        page.locator("text=Max Concepts").locator("..//input").fill(
            "5"
        )  # Invalid: < min_concepts
        page.locator("text=Max Examples").locator("..//input").fill(
            "25"
        )  # Invalid: > 20

        # Try to save
        page.locator('button:has-text("Save Changes")').click()
        page.wait_for_timeout(2000)

        # Verify error messages
        expect(
            page.locator("text=Similarity threshold must be between 0.0 and 1.0")
        ).to_be_visible()
        expect(
            page.locator(
                "text=Minimum concepts cannot be greater than maximum concepts"
            )
        ).to_be_visible()
        expect(
            page.locator("text=Maximum examples must be between 0 and 20")
        ).to_be_visible()

    @pytest.mark.integration
    def test_concept_highlights_save_and_reload(self, page: Page, gradio_app):
        """Test that concept highlights configuration can be saved and reloaded."""
        page.goto(gradio_app)
        page.wait_for_selector("h1", timeout=10000)

        # Fill in valid concept highlights configuration
        metric_dropdown = page.locator("text=Ranking Metric").locator("..//select")
        metric_dropdown.select_option("degree")

        count_input = page.locator("text=Count").nth(0).locator("..//input")
        count_input.fill("30")

        percent_input = page.locator("text=Percent").nth(0).locator("..//input")
        percent_input.fill("0")

        target_file_input = page.locator("text=Target File").nth(0).locator("..//input")
        target_file_input.fill("My Top Concepts.md")

        window_days_input = page.locator("text=Window Days").locator("..//input")
        window_days_input.fill("14")

        # Save configuration
        save_button = page.locator('button:has-text("Save Changes")')
        save_button.click()
        page.wait_for_timeout(2000)

        # Check for success message
        expect(page.locator("text=Configuration saved successfully")).to_be_visible()

        # Change values to test reload
        count_input.fill("50")
        target_file_input.fill("Changed File.md")

        # Reload configuration
        reload_button = page.locator('button:has-text("Reload from File")')
        reload_button.click()
        page.wait_for_timeout(2000)

        # Check that values are restored
        assert count_input.input_value() == "30"
        assert target_file_input.input_value() == "My Top Concepts.md"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
