"""
Test that Tier 1 conversion examples are properly formatted and accessible.
"""

import csv
import json
from pathlib import Path


def test_tier1_fixtures_exist():
    """Test that all Tier 1 conversion fixture files exist and are readable."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "tier1_conversion_examples"
    # Check that the main directory exists
    assert fixtures_dir.exists(), "Tier 1 conversion fixtures directory should exist"
    # Check inputs directory
    inputs_dir = fixtures_dir / "inputs"
    assert inputs_dir.exists(), "Inputs directory should exist"
    # Check expected_outputs directory
    outputs_dir = fixtures_dir / "expected_outputs"
    assert outputs_dir.exists(), "Expected outputs directory should exist"
    # Check README
    readme_file = fixtures_dir / "README.md"
    assert readme_file.exists(), "README.md should exist"


def test_input_files_are_valid():
    """Test that input files are properly formatted and readable."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "tier1_conversion_examples"
    inputs_dir = fixtures_dir / "inputs"
    # Test JSON file
    json_file = inputs_dir / "chatgpt_export.json"
    assert json_file.exists(), "ChatGPT JSON file should exist"
    with open(json_file, "r") as f:
        data = json.load(f)
        assert "title" in data, "JSON should have title field"
        assert "mapping" in data, "JSON should have mapping field"
        assert len(data["mapping"]) > 0, "JSON should have conversation messages"
    # Test CSV file
    csv_file = inputs_dir / "generic_tabular_export.csv"
    assert csv_file.exists(), "Generic tabular CSV file should exist"
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) > 0, "CSV should have data rows"
        assert "user_name" in rows[0], "CSV should have user_name column"
        assert "message" in rows[0], "CSV should have message column"
    # Test Slack JSON file
    slack_file = inputs_dir / "slack_export.json"
    assert slack_file.exists(), "Slack JSON file should exist"
    with open(slack_file, "r") as f:
        data = json.load(f)
        assert isinstance(data, list), "Slack JSON should be a list of messages"
        assert len(data) > 0, "Slack JSON should have messages"
        assert "type" in data[0], "Slack messages should have type field"
        assert "user" in data[0], "Slack messages should have user field"
        assert "text" in data[0], "Slack messages should have text field"
        assert "ts" in data[0], "Slack messages should have timestamp field"
    # Test plain text file
    txt_file = inputs_dir / "plain_text_chat.txt"
    assert txt_file.exists(), "Plain text file should exist"
    with open(txt_file, "r") as f:
        content = f.read()
        assert len(content) > 0, "Text file should have content"
        assert ":" in content, "Text file should have speaker: message format"
    # Test unrecognized format file
    xyz_file = inputs_dir / "unrecognized_format.xyz"
    assert xyz_file.exists(), "Unrecognized format file should exist"
    with open(xyz_file, "r") as f:
        content = f.read()
        assert len(content) > 0, "XYZ file should have content"
        assert "ENTRY" in content, "XYZ file should have ENTRY markers"


def test_expected_output_format():
    """Test that expected output files follow the correct Tier 1 Markdown format."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "tier1_conversion_examples"
    outputs_dir = fixtures_dir / "expected_outputs"
    output_files = [
        "chatgpt_export.md",
        "slack_export.md",
        "generic_tabular_export.md",
        "plain_text_chat.md",
        "unrecognized_format.md",
    ]
    for filename in output_files:
        file_path = outputs_dir / filename
        assert file_path.exists(), f"Output file {filename} should exist"
        with open(file_path, "r") as f:
            content = f.read()
            # Check for required metadata comments at the start
            assert "<!-- aclarai:title=" in content, (
                f"{filename} should have title metadata"
            )
            assert "<!-- aclarai:created_at=" in content, (
                f"{filename} should have created_at metadata"
            )
            assert "<!-- aclarai:participants=" in content, (
                f"{filename} should have participants metadata"
            )
            assert "<!-- aclarai:message_count=" in content, (
                f"{filename} should have message_count metadata"
            )
            assert "<!-- aclarai:plugin_metadata=" in content, (
                f"{filename} should have plugin_metadata"
            )
            # Check for block IDs and anchors
            assert "<!-- aclarai:id=blk_" in content, (
                f"{filename} should have aclarai:id comments"
            )
            assert "^blk_" in content, f"{filename} should have anchor references"
            assert " ver=1 -->" in content, f"{filename} should have version numbers"
            # Check for evaluation scores (at least one file should have them)
            if "entailed_score" in content:
                assert "<!-- aclarai:entailed_score=" in content, (
                    f"{filename} should have properly formatted entailed_score"
                )
                assert "<!-- aclarai:coverage_score=" in content, (
                    f"{filename} should have properly formatted coverage_score"
                )
                assert "<!-- aclarai:decontextualization_score=" in content, (
                    f"{filename} should have properly formatted decontextualization_score"
                )
            # Check speaker: message format
            lines = content.split("\n")
            has_speaker_format = any(
                ":" in line and not line.startswith("<!--") and not line.startswith("^")
                for line in lines
                if line.strip()
            )
            assert has_speaker_format, (
                f"{filename} should have 'speaker: message' format"
            )


def test_consistent_block_id_format():
    """Test that block IDs follow consistent format across all outputs."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "tier1_conversion_examples"
    outputs_dir = fixtures_dir / "expected_outputs"
    import re

    # Pattern for aclarai:id - should be blk_ followed by 6 alphanumeric characters
    id_pattern = r"<!-- aclarai:id=blk_([a-z0-9]{6}) ver=1 -->"
    anchor_pattern = r"\^blk_([a-z0-9]{6})"
    for md_file in outputs_dir.glob("*.md"):
        with open(md_file, "r") as f:
            content = f.read()
            # Find all IDs and anchors
            ids = re.findall(id_pattern, content)
            anchors = re.findall(anchor_pattern, content)
            # Should have same number of IDs and anchors
            assert len(ids) == len(anchors), (
                f"{md_file.name} should have matching IDs and anchors"
            )
            # IDs and anchors should match
            assert set(ids) == set(anchors), (
                f"{md_file.name} should have matching ID and anchor sets"
            )
            # All IDs should be unique within the file
            assert len(ids) == len(set(ids)), (
                f"{md_file.name} should have unique block IDs"
            )


def test_metadata_format_consistency():
    """Test that metadata format is consistent across all output files."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "tier1_conversion_examples"
    outputs_dir = fixtures_dir / "expected_outputs"
    required_metadata_fields = [
        "aclarai:title=",
        "aclarai:created_at=",
        "aclarai:participants=",
        "aclarai:message_count=",
        "aclarai:plugin_metadata=",
    ]
    for md_file in outputs_dir.glob("*.md"):
        with open(md_file, "r") as f:
            content = f.read()
            for field in required_metadata_fields:
                assert f"<!-- {field}" in content, (
                    f"{md_file.name} should have {field} metadata"
                )
            # Metadata should be at the top of the file
            lines = content.split("\n")
            metadata_lines = [
                line for line in lines[:10] if line.startswith("<!-- aclarai:")
            ]
            assert len(metadata_lines) >= 5, (
                f"{md_file.name} should have metadata at the top"
            )
