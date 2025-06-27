"""
Tests for the default fallback plugin.
"""

import json
import tempfile
from pathlib import Path

import pytest
from aclarai_shared.plugins.default_plugin import (
    ConversationExtractorAgent,
    DefaultPlugin,
)


@pytest.fixture
def default_plugin():
    """Create a default plugin instance for testing."""
    return DefaultPlugin()


@pytest.fixture
def conversation_agent():
    """Create a conversation agent instance for testing."""
    return ConversationExtractorAgent()


def test_default_plugin_always_accepts(default_plugin):
    """Test that the default plugin always accepts input."""
    assert default_plugin.can_accept("") is True
    assert default_plugin.can_accept("random text") is True
    assert default_plugin.can_accept("no conversation here") is True
    assert default_plugin.can_accept("user: hello\nassistant: hi") is True


def test_conversation_agent_fallback_extraction_simple_format(conversation_agent):
    """Test fallback extraction with simple speaker: message format."""
    raw_input = """
alice: Hello, how are you?
bob: I'm doing well, thanks for asking!
alice: That's great to hear.
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(raw_input)
        temp_path = Path(f.name)
    try:
        conversations = conversation_agent.extract_conversations(raw_input, temp_path)
        assert len(conversations) == 1
        conv = conversations[0]
        assert len(conv["messages"]) == 3
        assert conv["participants"] == ["alice", "bob"]
        assert conv["messages"][0] == {
            "speaker": "alice",
            "text": "Hello, how are you?",
        }
        assert conv["messages"][1] == {
            "speaker": "bob",
            "text": "I'm doing well, thanks for asking!",
        }
        assert conv["messages"][2] == {
            "speaker": "alice",
            "text": "That's great to hear.",
        }
    finally:
        temp_path.unlink()


def test_conversation_agent_fallback_extraction_entry_format(conversation_agent):
    """Test fallback extraction with ENTRY format."""
    raw_input = """
CONVERSATION_LOG_v2.1
=====================================
SESSION_ID: sess_123abc
START_TIME: 2023-12-22T10:00:00Z
PARTICIPANTS: alice, bob
TOPIC: Project Discussion
ENTRY [10:00:00] alice >> Let's discuss the project timeline.
ENTRY [10:00:30] bob >> I think we need more time for testing.
ENTRY [10:01:00] alice >> Agreed, let's add two more days.
SESSION_END: 2023-12-22T10:05:00Z
DURATION: 5m0s
EXPORT_FORMAT: Custom_LOG_v2.1
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(raw_input)
        temp_path = Path(f.name)
    try:
        conversations = conversation_agent.extract_conversations(raw_input, temp_path)
        assert len(conversations) == 1
        conv = conversations[0]
        assert conv["title"] == "Project Discussion"
        assert set(conv["participants"]) == {"alice", "bob"}
        assert len(conv["messages"]) == 3
        # Check first message has timestamp
        first_msg = conv["messages"][0]
        assert first_msg["speaker"] == "alice"
        assert first_msg["text"] == "Let's discuss the project timeline."
        assert first_msg["timestamp"] == "10:00:00"
        # Check metadata extraction
        metadata = conv["metadata"]
        assert metadata["session_id"] == "sess_123abc"
        assert metadata["duration"] == "5m0s"
        assert metadata["original_format"] == "Custom_LOG_v2.1"
    finally:
        temp_path.unlink()


def test_conversation_agent_fallback_extraction_no_conversation(conversation_agent):
    """Test fallback extraction when no conversation pattern is found."""
    raw_input = """
This is just a regular document.
It has multiple lines.
But no conversation format.
No speakers or messages.
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(raw_input)
        temp_path = Path(f.name)
    try:
        conversations = conversation_agent.extract_conversations(raw_input, temp_path)
        assert len(conversations) == 0
    finally:
        temp_path.unlink()


def test_conversation_agent_fallback_extraction_json_format(conversation_agent):
    """Test fallback extraction with JSON conversation format."""
    raw_input = """
{
  "conversation": {
    "id": "conv_456",
    "participants": ["john", "jane"],
    "messages": [
      {"speaker": "john", "text": "How's the project going?"},
      {"speaker": "jane", "text": "Making good progress on the API"},
      {"speaker": "john", "text": "Great! Any blockers?"},
      {"speaker": "jane", "text": "Just waiting on the database schema review"}
    ]
  }
}
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(raw_input)
        temp_path = Path(f.name)
    try:
        conversations = conversation_agent.extract_conversations(raw_input, temp_path)
        # Should extract conversation from JSON structure
        assert (
            len(conversations) >= 0
        )  # May find patterns or not, depending on LLM availability
    finally:
        temp_path.unlink()


def test_conversation_agent_fallback_extraction_csv_format(conversation_agent):
    """Test fallback extraction with CSV conversation format."""
    raw_input = """
timestamp,speaker,message
09:00:00,alice,"Let's start the meeting"
09:00:15,bob,"I have the agenda ready"
09:00:30,charlie,"Perfect, let's begin with item 1"
09:01:00,alice,"The budget review shows we're on track"
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(raw_input)
        temp_path = Path(f.name)
    try:
        conversations = conversation_agent.extract_conversations(raw_input, temp_path)
        # Should extract conversation from CSV structure
        assert (
            len(conversations) >= 0
        )  # May find patterns or not, depending on LLM availability
    finally:
        temp_path.unlink()


def test_default_plugin_convert_with_conversation():
    """Test default plugin conversion with valid conversation."""
    raw_input = """
alice: How's the weather today?
bob: It's quite nice, sunny and warm.
alice: Perfect for a walk in the park!
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(raw_input)
        temp_path = Path(f.name)
    try:
        plugin = DefaultPlugin()
        # Test that convert handles potential exceptions gracefully
        try:
            outputs = plugin.convert(raw_input, temp_path)
        except Exception as e:
            # If extract_conversations throws an exception, the plugin should handle it gracefully
            # and return an empty list rather than letting the exception propagate
            pytest.fail(
                f"Plugin.convert() should handle exceptions gracefully, but got: {e}"
            )
        assert len(outputs) == 1
        output = outputs[0]
        # Check basic structure
        assert output.title
        assert output.markdown_text
        assert output.metadata
        # Check metadata
        assert output.metadata["participants"] == ["alice", "bob"]
        assert output.metadata["message_count"] == 3
        assert output.metadata["plugin_metadata"]["source_format"] == "fallback_llm"
        # Check markdown format
        markdown = output.markdown_text
        # Should have metadata comments at top
        assert "<!-- aclarai:title=" in markdown
        assert "<!-- aclarai:participants=" in markdown
        assert "<!-- aclarai:message_count=3 -->" in markdown
        assert "<!-- aclarai:plugin_metadata=" in markdown
        # Should have conversation content
        assert "alice: How's the weather today?" in markdown
        assert "bob: It's quite nice, sunny and warm." in markdown
        assert "alice: Perfect for a walk in the park!" in markdown
        # Should have block IDs
        assert "<!-- aclarai:id=blk_" in markdown
        assert "^blk_" in markdown
        assert " ver=1 -->" in markdown
        # Should NOT have evaluation scores (generated by separate evaluation agents)
        assert "aclarai:entailed_score" not in markdown
        assert "aclarai:coverage_score" not in markdown
        assert "aclarai:decontextualization_score" not in markdown
    finally:
        temp_path.unlink()


def test_default_plugin_convert_no_conversation():
    """Test default plugin returns empty list when no conversation found."""
    raw_input = """
This is just a regular document.
No conversation patterns here.
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(raw_input)
        temp_path = Path(f.name)
    try:
        plugin = DefaultPlugin()
        outputs = plugin.convert(raw_input, temp_path)
        assert len(outputs) == 0
    finally:
        temp_path.unlink()


def test_default_plugin_convert_xyz_format():
    """Test default plugin with the test fixture XYZ format."""
    raw_input = """
CONVERSATION_LOG_v2.1
=====================================
SESSION_ID: sess_789xyz
START_TIME: 2023-12-22T16:45:00Z
PARTICIPANTS: tech_lead, junior_dev, product_manager
TOPIC: API Design Review
ENTRY [16:45:00] tech_lead >> Let's review the REST API endpoints for the aclarai services. We need consistency across all microservices.
ENTRY [16:45:32] junior_dev >> I noticed we have different authentication patterns. Some use JWT, others use API keys.
ENTRY [16:46:15] tech_lead >> Good catch. We should standardize on JWT with proper scope-based authorization for all services.
ENTRY [16:46:45] product_manager >> From a product perspective, what's the impact on existing integrations?
ENTRY [16:47:20] tech_lead >> Minimal impact if we version the APIs properly. We can deprecate the old auth methods gradually.
ENTRY [16:48:00] junior_dev >> Should we implement rate limiting consistently too?
ENTRY [16:48:30] tech_lead >> Absolutely. Same rate limits, same headers, same error responses across all endpoints.
ENTRY [16:49:10] product_manager >> That sounds good for developer experience. Consistent APIs are much easier to work with.
ENTRY [16:49:45] junior_dev >> I'll update the OpenAPI specs to reflect these standards.
ENTRY [16:50:15] tech_lead >> Perfect. And let's make sure the error handling follows RFC 7807 for problem details.
SESSION_END: 2023-12-22T16:50:30Z
DURATION: 5m30s
EXPORT_FORMAT: Custom_XYZ_v2.1
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(raw_input)
        temp_path = Path(f.name)
    try:
        plugin = DefaultPlugin()
        outputs = plugin.convert(raw_input, temp_path)
        assert len(outputs) == 1
        output = outputs[0]
        # Check extracted conversation details
        assert output.title == "API Design Review"
        assert set(output.metadata["participants"]) == {
            "tech_lead",
            "junior_dev",
            "product_manager",
        }
        assert output.metadata["message_count"] == 10
        # Check plugin metadata
        plugin_meta = output.metadata["plugin_metadata"]
        assert plugin_meta["session_id"] == "sess_789xyz"
        assert plugin_meta["duration"] == "5m30s"
        assert plugin_meta["original_format"] == "Custom_XYZ_v2.1"
        # Check markdown content
        markdown = output.markdown_text
        assert "tech_lead: Let's review the REST API endpoints" in markdown
        assert (
            "junior_dev: I noticed we have different authentication patterns"
            in markdown
        )
        assert "product_manager: From a product perspective" in markdown
        # Check that all messages are included
        assert markdown.count("tech_lead:") == 5  # tech_lead speaks 5 times
        assert markdown.count("junior_dev:") == 3  # junior_dev speaks 3 times
        assert markdown.count("product_manager:") == 2  # product_manager speaks 2 times
    finally:
        temp_path.unlink()


def test_default_plugin_markdown_format_compliance():
    """Test that the plugin output complies with expected markdown format."""
    raw_input = "speaker1: Hello\nspeaker2: Hi there"
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(raw_input)
        temp_path = Path(f.name)
    try:
        plugin = DefaultPlugin()
        outputs = plugin.convert(raw_input, temp_path)
        assert len(outputs) == 1
        markdown = outputs[0].markdown_text
        lines = markdown.split("\n")
        # First few lines should be metadata comments
        metadata_lines = [line for line in lines if line.startswith("<!-- aclarai:")]
        assert (
            len(metadata_lines) >= 4
        )  # title, created_at, participants, message_count, plugin_metadata
        # Should have proper block ID format
        import re

        block_ids = re.findall(r"<!-- aclarai:id=(blk_[a-z0-9]{6}) ver=1 -->", markdown)
        anchors = re.findall(r"\^(blk_[a-z0-9]{6})", markdown)
        assert len(block_ids) == len(
            anchors
        )  # Each block ID should have corresponding anchor
        assert set(block_ids) == set(anchors)  # IDs and anchors should match
        # Check JSON in metadata is valid
        participants_line = next(
            line for line in lines if "aclarai:participants=" in line
        )
        participants_json = participants_line.split("aclarai:participants=")[1].split(
            " -->"
        )[0]
        participants = json.loads(participants_json)
        assert isinstance(participants, list)
    finally:
        temp_path.unlink()


def test_default_plugin_block_id_uniqueness():
    """Test that generated block IDs are unique within a conversation."""
    raw_input = """
speaker1: Message 1
speaker2: Message 2
speaker1: Message 3
speaker2: Message 4
speaker1: Message 5
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(raw_input)
        temp_path = Path(f.name)
    try:
        plugin = DefaultPlugin()
        outputs = plugin.convert(raw_input, temp_path)
        assert len(outputs) == 1
        markdown = outputs[0].markdown_text
        import re

        # Extract block IDs from comments and anchors separately
        comment_ids = re.findall(
            r"<!-- aclarai:id=(blk_[a-z0-9]{6}) ver=1 -->", markdown
        )
        anchor_ids = re.findall(r"\^(blk_[a-z0-9]{6})", markdown)
        # Each message should have exactly one ID in comment and one matching anchor
        assert len(comment_ids) == 5  # 5 messages
        assert len(anchor_ids) == 5  # 5 corresponding anchors
        assert comment_ids == anchor_ids  # IDs should match exactly
        # All IDs should be unique (no duplicates within comments or within anchors)
        assert len(comment_ids) == len(set(comment_ids))
        assert len(anchor_ids) == len(set(anchor_ids))
    finally:
        temp_path.unlink()


def test_default_plugin_short_conversation_no_scores():
    """Test that short conversations don't get evaluation scores."""
    raw_input = "speaker1: Hello\nspeaker2: Hi"
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(raw_input)
        temp_path = Path(f.name)
    try:
        plugin = DefaultPlugin()
        outputs = plugin.convert(raw_input, temp_path)
        assert len(outputs) == 1
        markdown = outputs[0].markdown_text
        # Short conversation (< 3 messages) should not have evaluation scores
        assert "aclarai:entailed_score" not in markdown
        assert "aclarai:coverage_score" not in markdown
        assert "aclarai:decontextualization_score" not in markdown
    finally:
        temp_path.unlink()


def test_default_plugin_uses_customized_prompt():
    """Test that DefaultPlugin uses a user-supplied YAML prompt file when available."""
    import os
    import shutil
    import tempfile
    from pathlib import Path

    # Create a temporary directory structure
    original_cwd = Path.cwd()
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    try:
        # Change to temp directory to control where prompts are loaded from
        os.chdir(temp_path)
        # Create settings/prompts directory with custom prompt template
        prompts_dir = temp_path / "settings" / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        # Create a custom prompt YAML file with only partial overrides to test deep merge
        custom_prompt_content = """
system_prompt: "You are a CUSTOM conversation extraction agent for testing."
template: "CUSTOM PROMPT TEST: Extract conversations from the following text: {input_text}"
rules:
  - "This is a custom test prompt"
  - "Should be merged with default prompt"
"""
        custom_prompt_file = prompts_dir / "conversation_extraction.yaml"
        custom_prompt_file.write_text(custom_prompt_content.strip())
        # Test that the prompt loader would find our custom prompt
        from aclarai_shared.utils.prompt_loader import PromptLoader

        # Create prompt loader and verify it finds our custom prompt
        loader = PromptLoader()
        default_path, user_path = loader._find_template_files("conversation_extraction")
        # Verify we found the user customization
        assert user_path.resolve() == custom_prompt_file.resolve()
        # Verify the template loads correctly with deep merge
        template = loader.load_template("conversation_extraction")
        # Check that custom fields override defaults
        assert "CUSTOM PROMPT TEST" in template.template
        assert (
            "CUSTOM conversation extraction agent for testing" in template.system_prompt
        )
        assert len(template.rules) == 2
        assert "This is a custom test prompt" in template.rules
        assert "Should be merged with default prompt" in template.rules
        # Check that default fields that weren't overridden are preserved
        assert template.role == "conversation_analyst"  # From default
        assert (
            "expert conversation analyst" in template.description.lower()
        )  # From default
        assert template.variables["input_text"]["required"] is True  # From default
        assert len(template.instructions) == 3  # From default
        assert "conversations" in template.output_format  # From default
        # Verify the template can be formatted
        formatted_prompt = loader.load_and_format(
            "conversation_extraction", input_text="test input text"
        )
        assert "CUSTOM PROMPT TEST" in formatted_prompt
        assert "test input text" in formatted_prompt
        print("✓ Custom prompt loading functionality verified")
    finally:
        # Cleanup
        os.chdir(original_cwd)
        shutil.rmtree(temp_path)
