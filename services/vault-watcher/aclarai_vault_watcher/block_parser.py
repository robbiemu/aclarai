"""
Block parser for extracting and managing aclarai blocks from Markdown files.
This module implements parsing logic to extract blocks marked with aclarai:id
comments from Markdown files, following the specifications in
on-graph_vault_synchronization.md.
"""

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional


class aclaraiBlock(NamedTuple):
    """Represents a aclarai block with its metadata."""

    aclarai_id: str
    version: int
    content: str
    content_hash: str
    start_pos: int
    end_pos: int
    block_type: str  # 'inline' or 'file'


class BlockParser:
    """Parser for extracting aclarai blocks from Markdown content."""

    # Regex patterns for different block types
    INLINE_BLOCK_PATTERN = re.compile(
        r"<!--\s*aclarai:id=([^\s]+)\s+ver=(\d+)\s*-->", re.IGNORECASE
    )
    FILE_BLOCK_PATTERN = re.compile(
        r"<!--\s*aclarai:id=([^\s]+)\s+ver=(\d+)\s*-->\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    def __init__(self):
        """Initialize the block parser."""
        pass

    def parse_file(self, file_path: Path) -> List[aclaraiBlock]:
        """
        Parse a Markdown file and extract all aclarai blocks.
        Args:
            file_path: Path to the Markdown file
        Returns:
            List of aclaraiBlock objects found in the file
        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: If there's an error reading the file
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            return self.parse_content(content)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}") from None
        except IOError as e:
            raise IOError(f"Error reading file {file_path}: {e}") from e

    def parse_content(self, content: str) -> List[aclaraiBlock]:
        """
        Parse Markdown content and extract all aclarai blocks.
        Args:
            content: The Markdown content as a string
        Returns:
            List of aclaraiBlock objects found in the content
        """
        blocks = []
        # First, check for file-level block (at the end of file)
        file_block = self._extract_file_block(content)
        if file_block:
            blocks.append(file_block)
        # Then, extract inline blocks
        inline_blocks = self._extract_inline_blocks(content)
        blocks.extend(inline_blocks)
        return blocks

    def _extract_file_block(self, content: str) -> Optional[aclaraiBlock]:
        """Extract file-level block from the end of the content."""
        # File-level blocks should be at the end with just whitespace after
        # and should be on their own line
        lines = content.split("\n")
        # Look for the comment in the last few lines
        for i in range(len(lines) - 1, max(-1, len(lines) - 5), -1):
            line = lines[i].strip()
            match = self.INLINE_BLOCK_PATTERN.match(line)
            if match:
                # Check if this is truly a file-level block
                # (i.e., there's no content after it except whitespace)
                remaining_content = "\n".join(lines[i + 1 :]).strip()
                if not remaining_content:  # Only whitespace after the comment
                    # Also check that the comment line has no content before it
                    content_before_comment = line[: line.find("<!--")].strip()
                    if not content_before_comment:  # Comment is alone on the line
                        aclarai_id = match.group(1)
                        version = int(match.group(2))
                        # For file-level blocks, the content is everything before the comment
                        file_content = "\n".join(lines[:i]).rstrip()
                        content_hash = self._compute_content_hash(file_content)
                        return aclaraiBlock(
                            aclarai_id=aclarai_id,
                            version=version,
                            content=file_content,
                            content_hash=content_hash,
                            start_pos=0,
                            end_pos=len(file_content),
                            block_type="file",
                        )
        return None

    def _extract_inline_blocks(self, content: str) -> List[aclaraiBlock]:
        """Extract inline blocks from the content."""
        blocks = []
        for match in self.INLINE_BLOCK_PATTERN.finditer(content):
            # Check if this is a file-level block by seeing if it's at the very end
            # with only whitespace after it AND on its own line
            comment_end = match.end()
            after_comment = content[comment_end:].strip()
            # If there's nothing after the comment except whitespace,
            # and the comment is at the end of its line, it might be file-level
            if not after_comment:
                # Check if the comment is on a line by itself (or just with whitespace)
                comment_line_start = content.rfind("\n", 0, match.start()) + 1
                comment_line = content[comment_line_start : match.end()]
                content_before_comment = comment_line[
                    : comment_line.find("<!--")
                ].strip()
                # If there's no content before the comment on the same line,
                # this is likely a file-level block
                if not content_before_comment:
                    continue  # Skip, will be handled as file-level block
            aclarai_id = match.group(1)
            version = int(match.group(2))
            # Find the content that belongs to this block
            # Look backwards from the comment to find the content
            block_content = self._extract_block_content(content, match)
            if block_content:
                content_hash = self._compute_content_hash(block_content["content"])
                blocks.append(
                    aclaraiBlock(
                        aclarai_id=aclarai_id,
                        version=version,
                        content=block_content["content"],
                        content_hash=content_hash,
                        start_pos=block_content["start_pos"],
                        end_pos=match.end(),
                        block_type="inline",
                    )
                )
        return blocks

    def _extract_block_content(
        self, content: str, comment_match: re.Match
    ) -> Optional[Dict]:
        """
        Extract the content that belongs to an inline block comment.
        This looks backwards from the comment to find the associated content,
        typically the paragraph or sentence preceding the comment.
        """
        comment_start = comment_match.start()
        # Look backwards to find the start of the content
        # We'll look for paragraph breaks (double newlines) or start of file
        before_comment = content[:comment_start].rstrip()
        # Find the last paragraph or sentence before the comment
        # First try paragraph breaks
        paragraphs = before_comment.split("\n\n")
        if len(paragraphs) > 0:
            last_paragraph = paragraphs[-1].strip()
            if last_paragraph:
                # Handle single line content that might be on the same line as the comment
                # Split by newlines to get the last line/sentence
                lines = last_paragraph.split("\n")
                last_line = lines[-1].strip()
                if last_line:
                    # Calculate the actual start position in the full content
                    # Find where this content starts in the original text
                    start_pos = content.rfind(last_line, 0, comment_start)
                    return {
                        "content": last_line,
                        "start_pos": start_pos if start_pos >= 0 else 0,
                    }
        return None

    def _compute_content_hash(self, content: str) -> str:
        """
        Compute SHA-256 hash of the content for change detection.
        Args:
            content: The content to hash
        Returns:
            Hexadecimal string representation of the hash
        """
        # Normalize whitespace for consistent hashing
        normalized_content = " ".join(content.split())
        return hashlib.sha256(normalized_content.encode("utf-8")).hexdigest()

    def compare_blocks(
        self, old_blocks: List[aclaraiBlock], new_blocks: List[aclaraiBlock]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Compare two sets of blocks and return the differences.
        Args:
            old_blocks: Previous state of blocks
            new_blocks: Current state of blocks
        Returns:
            Dictionary with keys 'added', 'modified', 'deleted' containing
            lists of relevant block IDs and change information
        """
        old_block_map = {block.aclarai_id: block for block in old_blocks}
        new_block_map = {block.aclarai_id: block for block in new_blocks}
        added = []
        modified = []
        deleted = []
        # Find added and modified blocks
        for block_id, new_block in new_block_map.items():
            if block_id not in old_block_map:
                added.append(
                    {
                        "aclarai_id": block_id,
                        "version": new_block.version,
                        "content_hash": new_block.content_hash,
                        "block_type": new_block.block_type,
                    }
                )
            else:
                old_block = old_block_map[block_id]
                if (
                    old_block.content_hash != new_block.content_hash
                    or old_block.version != new_block.version
                ):
                    modified.append(
                        {
                            "aclarai_id": block_id,
                            "old_version": old_block.version,
                            "new_version": new_block.version,
                            "old_hash": old_block.content_hash,
                            "new_hash": new_block.content_hash,
                            "block_type": new_block.block_type,
                        }
                    )
        # Find deleted blocks
        for block_id in old_block_map:
            if block_id not in new_block_map:
                deleted.append(
                    {
                        "aclarai_id": block_id,
                        "version": old_block_map[block_id].version,
                        "content_hash": old_block_map[block_id].content_hash,
                        "block_type": old_block_map[block_id].block_type,
                    }
                )
        return {"added": added, "modified": modified, "deleted": deleted}
