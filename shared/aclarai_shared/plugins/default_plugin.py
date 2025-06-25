"""
Default fallback plugin for aclarai format conversion.
This plugin implements the fallback/default plugin as specified in the architecture docs:
- docs/arch/idea-default_plugin_task.md
- docs/arch/on-pluggable_formats.md
The plugin always accepts input and uses an LLM agent to interpret and format
unstructured text, extracting conversations and outputting standard aclarai Tier 1 Markdown.
Key Features:
- Always returns True from can_accept() (fallback behavior)
- Uses LLM agent for conversation extraction from unstructured text
- Supports multiple conversation formats via pattern matching
- Generates aclarai-compliant Markdown with block IDs and metadata
- Graceful fallback when LLM is unavailable
- Structured logging for debugging and monitoring
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI

from ..config import load_config
from ..plugin_interface import MarkdownOutput, Plugin
from ..utils.block_id import generate_unique_block_id
from ..utils.prompt_installer import ensure_prompt_exists
from ..utils.prompt_loader import load_prompt_template

logger = logging.getLogger(__name__)


class ConversationExtractorAgent:
    """LLM-powered agent for extracting conversations from unstructured text."""

    llm: Optional[LLM]

    def __init__(self, llm: Optional[LLM] = None):
        """Initialize the agent with an LLM instance."""
        if llm is None:
            # Load model configuration from YAML following design_config_panel.md
            config = load_config(validate=False)
            # Try to load model configuration from YAML
            fallback_model = self._get_fallback_model_from_config()
            api_key = getattr(config, "openai_api_key", None)
            if api_key and fallback_model:
                # Use configured model if available
                self.llm = OpenAI(api_key=api_key, model=fallback_model)
                logger.info(
                    f"Initialized ConversationExtractorAgent with configured model: {fallback_model}"
                )
            elif api_key:
                # No specific model configuration found - this indicates a configuration issue
                logger.warning(
                    "No fallback_plugin model configured, initializing without LLM"
                )
                self.llm = None
            else:
                # Graceful fallback for testing/development when no LLM is available
                self.llm = None
                logger.info(
                    "ConversationExtractorAgent initialized without LLM (fallback mode)"
                )
        else:
            self.llm = llm
            logger.info(
                f"ConversationExtractorAgent initialized with provided LLM: {type(llm).__name__}"
            )

    def _get_fallback_model_from_config(self) -> Optional[str]:
        """
        Get the fallback_plugin model configuration from YAML config.
        Returns:
            Model name for fallback plugin or None if not configured
        """
        try:
            # Import here to avoid circular imports
            from ..config import aclaraiConfig

            # Load YAML config using the existing utility
            yaml_config = aclaraiConfig._load_yaml_config()
            # Get model.fallback_plugin configuration following design_config_panel.md
            model_config = yaml_config.get("model", {})
            if not isinstance(model_config, dict):
                logger.warning("Model configuration in YAML is not a dictionary.")
                return None

            fallback_model = model_config.get("fallback_plugin")

            # Explicitly check if the retrieved value is a string.
            if isinstance(fallback_model, str):
                logger.debug(
                    f"Found fallback_plugin model configuration: {fallback_model}"
                )
                return fallback_model

            if fallback_model is not None:
                # Log a warning if the value is present but not a string.
                logger.warning(
                    "Expected 'fallback_plugin' to be a string, but got "
                    f"{type(fallback_model).__name__}"
                )

            logger.debug("No fallback_plugin model configuration found or it's invalid")
            return None
        except Exception as e:
            logger.warning(f"Failed to load model configuration: {e}")
            return None

    def extract_conversations(self, raw_input: str, path: Path) -> List[Dict[str, Any]]:
        """
        Extract conversations from raw input using procedural approach first, then LLM if needed.
        This method implements the agent responsibilities from docs/arch/idea-default_plugin_task.md:
        - Determines if the input contains conversations
        - Returns empty list if none found
        - Splits multiple conversations if found
        - Formats as structured data for Markdown conversion
        The extraction approach prioritizes procedural methods over LLM for efficiency:
        1. Try procedural pattern matching first
        2. If procedural succeeds, return results
        3. If procedural fails and LLM is available, try LLM
        4. If LLM also fails or unavailable, return empty list
        Args:
            raw_input: The raw text content
            path: Path to the original file
        Returns:
            List of conversation dictionaries with metadata
        """
        logger.debug(f"Extracting conversations from file: {path}")
        # Always try procedural approach first (faster, more reliable, no cost)
        logger.debug("Trying procedural extraction first")
        conversations = self._fallback_extraction(raw_input, path)
        if conversations:
            # Procedural extraction succeeded
            logger.info(
                f"Procedural extraction found {len(conversations)} conversation(s)"
            )
            return conversations
        # Procedural extraction found no conversations
        if self.llm is None:
            # No LLM available, return empty results
            logger.debug(
                "No conversations found via procedural extraction and no LLM available"
            )
            return []
        # Try LLM as last resort
        logger.debug("Procedural extraction found no conversations, trying LLM")
        prompt = self._build_extraction_prompt(raw_input)
        try:
            logger.debug("Calling LLM for conversation extraction")
            response = self.llm.complete(prompt)
            llm_conversations = self._parse_llm_response(response.text, raw_input, path)
            logger.info(f"LLM extracted {len(llm_conversations)} conversation(s)")
            return llm_conversations
        except Exception as e:
            # LLM failed, return empty results
            logger.warning(f"LLM extraction failed: {e}")
            return []

    def _build_extraction_prompt(self, raw_input: str) -> str:
        """Build the prompt for conversation extraction using externalized YAML template."""
        try:
            # Ensure the prompt file exists in user's prompts directory
            ensure_prompt_exists("conversation_extraction")
            return load_prompt_template("conversation_extraction", input_text=raw_input)
        except Exception as e:
            logger.error(f"Failed to load conversation extraction template: {e}")
            # Fallback to simple hardcoded prompt if template loading fails
            return f"""
You are an expert conversation analyst. Analyze the following text and extract conversations.
If no conversation is found, respond with "NO_CONVERSATION".
If conversations are found, extract them in JSON format with title, participants, and messages.
INPUT TEXT:
{raw_input}"""

    def _parse_llm_response(
        self, response: str, raw_input: str, path: Path
    ) -> List[Dict[str, Any]]:
        """Parse the LLM response into conversation data."""
        if "NO_CONVERSATION" in response:
            return []
        try:
            # Try to extract JSON from the response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                conversations = data.get("conversations", [])
                if isinstance(conversations, list):
                    return conversations
        except json.JSONDecodeError:
            pass
        # Fallback to simple extraction if JSON parsing fails
        return self._fallback_extraction(raw_input, path)

    def _fallback_extraction(self, raw_input: str, path: Path) -> List[Dict[str, Any]]:
        """
        Simple fallback extraction using pattern matching.
        This method handles basic conversation formats when LLM is unavailable,
        following the graceful degradation pattern from docs/arch/on-error-handling-and-resilience.md
        """
        logger.debug("Using pattern-based fallback extraction")
        conversations = []
        # Look for common conversation patterns
        lines = raw_input.strip().split("\n")
        messages = []
        participants = set()
        # Pattern 1: Custom format like "ENTRY [timestamp] speaker >> message" (check first)
        entry_pattern = r"^ENTRY\s*\[([^\]]+)\]\s*([^>]+)\s*>>\s*(.+)$"
        # Pattern 2: "speaker: message" format (but exclude metadata lines)
        # Updated to handle speakers with spaces like "Dr. Smith"
        speaker_pattern = r"^([a-zA-Z_][a-zA-Z0-9_\s\.]*[a-zA-Z0-9_])\s*:\s*(.+)$"
        # Metadata fields to exclude from speaker detection
        metadata_fields = {
            "SESSION_ID",
            "START_TIME",
            "PARTICIPANTS",
            "TOPIC",
            "SESSION_END",
            "DURATION",
            "EXPORT_FORMAT",
            "CONVERSATION_LOG",
        }
        for line in lines:
            line = line.strip()
            if not line or line.startswith("<!--") or line.startswith("="):
                continue
            # Try ENTRY format first (more specific)
            match = re.match(entry_pattern, line)
            if match:
                timestamp = match.group(1).strip()
                speaker = match.group(2).strip()
                text = match.group(3).strip()
                messages.append(
                    {"speaker": speaker, "text": text, "timestamp": timestamp}
                )
                participants.add(speaker)
                continue
            # Try speaker: message format (but exclude metadata)
            match = re.match(speaker_pattern, line)
            if match:
                speaker = match.group(1).strip()
                text = match.group(2).strip()
                # Skip if this looks like metadata
                if speaker.upper() in metadata_fields:
                    continue
                messages.append({"speaker": speaker, "text": text})
                participants.add(speaker)
                continue
        if messages:
            # Extract metadata from the raw input
            metadata = self._extract_metadata(raw_input, path)
            conversation = {
                "title": self._generate_title(raw_input, list(participants)),
                "participants": sorted(participants),  # Sort for deterministic order
                "messages": messages,
                "metadata": metadata,
            }
            conversations.append(conversation)
            logger.info(
                f"Fallback extraction found conversation with {len(messages)} messages and {len(participants)} participants"
            )
        else:
            logger.info("No conversation patterns detected in fallback extraction")
        return conversations

    def _extract_metadata(self, raw_input: str, _path: Path) -> Dict[str, Any]:
        """Extract metadata from the raw input."""
        metadata = {"source_format": "fallback_llm", "original_format": "unknown"}
        # Look for common metadata patterns
        lines = raw_input.split("\n")
        for line in lines:
            line = line.strip()
            # Session ID
            if "SESSION_ID:" in line:
                metadata["session_id"] = line.split("SESSION_ID:")[1].strip()
            # Duration
            if "DURATION:" in line:
                metadata["duration"] = line.split("DURATION:")[1].strip()
            # Format info
            if "EXPORT_FORMAT:" in line:
                metadata["original_format"] = line.split("EXPORT_FORMAT:")[1].strip()
        return metadata

    def _generate_title(self, raw_input: str, participants: List[str]) -> str:
        """Generate a title for the conversation."""
        # Look for topic or title in the input
        lines = raw_input.split("\n")
        for line in lines:
            if "TOPIC:" in line:
                return line.split("TOPIC:")[1].strip()
        # Fallback to generic title
        if len(participants) > 1:
            return f"Conversation between {', '.join(participants)}"
        elif participants:
            return f"Conversation with {participants[0]}"
        else:
            return "Conversation"


class DefaultPlugin(Plugin):
    """
    Default/fallback plugin that always accepts input and uses LLM-based conversion.
    This plugin implements the requirements from docs/arch/idea-default_plugin_task.md:
    - Always returns True from can_accept()
    - Uses an LLM agent to interpret and format unstructured text
    - Extracts conversations and outputs standard aclarai Markdown
    - Includes appropriate metadata with plugin_metadata field
    """

    def __init__(self, llm: Optional[LLM] = None):
        """Initialize the plugin with an optional LLM instance."""
        self.agent = ConversationExtractorAgent(llm)

    def can_accept(self, _raw_input: str) -> bool:
        """Always accept input - this is the fallback plugin."""
        return True

    def convert(self, raw_input: str, path: Path) -> List[MarkdownOutput]:
        """
        Convert raw input to aclarai Markdown format using LLM analysis.
        This method implements the plugin behavior from docs/arch/idea-default_plugin_task.md:
        - Calls the agent for conversation extraction
        - Wraps results as MarkdownOutput objects
        - Returns empty list if no conversations found (plugin skips file)
        Args:
            raw_input: The raw text content
            path: Path to the input file
        Returns:
            List of MarkdownOutput objects (may be empty if no conversations found)
        """
        logger.debug(f"Converting file: {path}")
        conversations = self.agent.extract_conversations(raw_input, path)
        if not conversations:
            # No conversations found - return empty list (plugin skips file)
            logger.info(f"No conversations found in {path}, skipping file")
            return []
        logger.info(f"Converting {len(conversations)} conversation(s) to Markdown")
        outputs = []
        for i, conv in enumerate(conversations):
            markdown_text = self._format_conversation_as_markdown(conv)
            metadata = {
                "created_at": None,  # Will be filled by ensure_defaults
                "participants": conv["participants"],
                "message_count": len(conv["messages"]),
                "plugin_metadata": conv.get("metadata", {}),
            }
            outputs.append(
                MarkdownOutput(
                    title=conv["title"], markdown_text=markdown_text, metadata=metadata
                )
            )
            logger.debug(
                f"Converted conversation {i + 1}/{len(conversations)}: '{conv['title']}'"
            )
        return outputs

    def _format_conversation_as_markdown(self, conversation: Dict[str, Any]) -> str:
        """
        Format a conversation as aclarai Tier 1 Markdown.
        This follows the format specified in docs/arch/idea-creating_tier1_documents.md:
        - Metadata comments at the top
        - speaker: text blocks
        - <!-- aclarai:id=blk_xyz ver=1 --> comments
        - ^blk_xyz anchors
        - Evaluation scores for substantial conversations
        """
        lines = []
        # Add metadata comments at the top
        participants_json = json.dumps(conversation["participants"])
        message_count = len(conversation["messages"])
        plugin_metadata = json.dumps(conversation.get("metadata", {}))
        lines.extend(
            [
                f"<!-- aclarai:title={conversation['title']} -->",
                "<!-- aclarai:created_at=PLACEHOLDER -->",  # Will be replaced by ensure_defaults
                f"<!-- aclarai:participants={participants_json} -->",
                f"<!-- aclarai:message_count={message_count} -->",
                f"<!-- aclarai:plugin_metadata={plugin_metadata} -->",
                "",  # Empty line after metadata
            ]
        )
        # Track used block IDs to ensure uniqueness
        used_block_ids: Set[str] = set()
        # Add conversation messages with block IDs
        for _i, message in enumerate(conversation["messages"]):
            speaker = message["speaker"]
            text = message["text"]
            # Generate unique block ID
            block_id = generate_unique_block_id(used_block_ids)
            used_block_ids.add(block_id)
            lines.extend(
                [
                    f"{speaker}: {text}",
                    f"<!-- aclarai:id={block_id} ver=1 -->",
                    f"^{block_id}",
                    "",  # Empty line between messages
                ]
            )
        # Note: Evaluation scores (entailed_score, coverage_score, decontextualization_score)
        # are generated by separate evaluation agents in later processing phases,
        # not by the default plugin during ingestion
        return "\n".join(lines)
