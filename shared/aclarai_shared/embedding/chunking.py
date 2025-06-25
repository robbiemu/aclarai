"""
Text chunking module for aclarai utterance processing.
This module implements the chunking strategy from docs/arch/on-sentence_splitting.md
using LlamaIndex SentenceSplitter with post-processing rules for coherent chunks.
Key Features:
- Uses LlamaIndex SentenceSplitter as base layer
- Post-processing rules for semantic coherence
- Configurable chunk size and overlap
- Preserves metadata and traceability
- Handles Tier 1 Markdown blocks with aclarai:id references
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode

from ..config import aclaraiConfig

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a processed chunk."""

    aclarai_block_id: str  # ID of the parent Tier 1 block
    chunk_index: int  # Ordinal within block
    original_text: str  # Original text before chunking
    text: str  # The chunked text
    token_count: Optional[int] = None
    offset_start: Optional[int] = None
    offset_end: Optional[int] = None


class UtteranceChunker:
    """
    Chunker for Tier 1 Markdown utterance blocks using LlamaIndex SentenceSplitter.
    Implements the hybrid strategy from docs/arch/on-sentence_splitting.md:
    1. Base layer: LlamaIndex SentenceSplitter with token-aware splitting
    2. Post-processing: Merge colon-ended lead-ins and short prefixes
    3. No discards: All chunks go to Claimify pipeline
    """

    def __init__(self, config: Optional[aclaraiConfig] = None):
        """
        Initialize the chunker with configuration.
        Args:
            config: aclarai configuration (loads default if None)
        """
        if config is None:
            from ..config import load_config

            config = load_config(validate=False)
        self.config = config
        # Initialize LlamaIndex SentenceSplitter
        self.splitter = SentenceSplitter(
            chunk_size=config.embedding.chunk_size,
            chunk_overlap=config.embedding.chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^,.;。]+[,.;。]?",
            tokenizer=None,  # Use default tokenizer
        )
        logger.info(
            f"Initialized UtteranceChunker with chunk_size={config.embedding.chunk_size}, "
            f"chunk_overlap={config.embedding.chunk_overlap}"
        )

    def chunk_tier1_blocks(self, tier1_content: str) -> List[ChunkMetadata]:
        """
        Chunk Tier 1 Markdown content into utterance chunks.
        Args:
            tier1_content: Raw Tier 1 Markdown content with aclarai:id blocks
        Returns:
            List of ChunkMetadata objects for each chunk
        """
        logger.debug("Parsing Tier 1 Markdown content for utterance blocks")
        # Parse Tier 1 content to extract utterance blocks
        utterance_blocks = self._parse_tier1_blocks(tier1_content)
        logger.info(f"Found {len(utterance_blocks)} utterance blocks to chunk")
        all_chunks = []
        for block in utterance_blocks:
            chunks = self.chunk_utterance_block(block["text"], block["aclarai_id"])
            all_chunks.extend(chunks)
        logger.info(f"Generated {len(all_chunks)} total chunks from Tier 1 content")
        return all_chunks

    def chunk_utterance_block(
        self, text: str, aclarai_block_id: str
    ) -> List[ChunkMetadata]:
        """
        Chunk a single utterance block into coherent segments.
        Args:
            text: The utterance text to chunk
            aclarai_block_id: The aclarai:id of the source block
        Returns:
            List of ChunkMetadata objects
        """
        logger.debug(f"Chunking utterance block: {aclarai_block_id}")
        # Step 1: Use LlamaIndex SentenceSplitter as base layer
        document = Document(text=text, metadata={"source_block_id": aclarai_block_id})
        base_chunks = self.splitter.get_nodes_from_documents([document])
        logger.debug(f"Base splitter generated {len(base_chunks)} initial chunks")
        # Step 2: Apply post-processing rules for semantic coherence
        processed_chunks = self._apply_postprocessing_rules(
            cast(List[TextNode], base_chunks)
        )
        # Step 3: Create ChunkMetadata objects
        chunk_metadata = []
        for i, chunk_node in enumerate(processed_chunks):
            metadata = ChunkMetadata(
                aclarai_block_id=aclarai_block_id,
                chunk_index=i,
                original_text=text,
                text=chunk_node.get_content(),
                # Note: token_count and offsets could be added later if needed
            )
            chunk_metadata.append(metadata)
        logger.debug(
            f"Generated {len(chunk_metadata)} final chunks for block {aclarai_block_id}"
        )
        return chunk_metadata

    def _parse_tier1_blocks(self, tier1_content: str) -> List[Dict[str, Any]]:
        """
        Parse Tier 1 Markdown content to extract utterance blocks.
        Expected format from docs/arch/idea-creating_tier1_documents.md:
        ```
        speaker: utterance text
        <!-- aclarai:id=blk_xyz ver=1 -->
        ^blk_xyz
        ```
        Args:
            tier1_content: Raw Tier 1 Markdown content
        Returns:
            List of utterance block dictionaries
        """
        blocks = []
        lines = tier1_content.split("\n")
        current_utterance = None
        current_speaker = None
        current_text = ""
        # Pattern to match aclarai:id comments
        id_pattern = re.compile(r"<!-- aclarai:id=([a-z0-9_]+) ver=\d+ -->")
        # Pattern to match speaker: text format (but not HTML comments)
        speaker_pattern = re.compile(r"^([^:<]+):\s*(.+)$")
        for line in lines:
            line = line.strip()
            # Skip empty lines and metadata comments
            if (
                not line
                or line.startswith("<!-- aclarai:title=")
                or line.startswith("<!-- aclarai:created_at=")
            ):
                continue
            # Check for aclarai:id comment FIRST (before speaker pattern)
            id_match = id_pattern.match(line)
            if id_match:
                current_utterance = id_match.group(1)
                continue
            # Check for speaker: text pattern (only if not an HTML comment)
            if not line.startswith("<!--"):
                speaker_match = speaker_pattern.match(line)
                if speaker_match:
                    # Save previous utterance if exists
                    if current_utterance and current_text:
                        blocks.append(
                            {
                                "aclarai_id": current_utterance,
                                "speaker": current_speaker,
                                "text": current_text.strip(),
                            }
                        )
                    # Start new utterance
                    current_speaker = speaker_match.group(1).strip()
                    current_text = speaker_match.group(2).strip()
                    continue
            # Check for anchor (^blk_xyz) - marks end of utterance
            if line.startswith("^") and current_utterance:
                # Save the utterance
                if current_text:
                    blocks.append(
                        {
                            "aclarai_id": current_utterance,
                            "speaker": current_speaker or "Unknown",
                            "text": current_text.strip(),
                        }
                    )
                # Reset for next utterance
                current_utterance = None
                current_speaker = None
                current_text = ""
                continue
            # Continuation of current utterance text
            if current_speaker and line:
                current_text += " " + line
        # Handle last utterance if no final anchor
        if current_utterance and current_text:
            blocks.append(
                {
                    "aclarai_id": current_utterance,
                    "speaker": current_speaker or "Unknown",
                    "text": current_text.strip(),
                }
            )
        logger.debug(f"Parsed {len(blocks)} utterance blocks from Tier 1 content")
        return blocks

    def _apply_postprocessing_rules(
        self, base_chunks: List[TextNode]
    ) -> List[TextNode]:
        """
        Apply post-processing rules for semantic coherence.
        Rules from docs/arch/on-sentence_splitting.md:
        1. Merge colon-ended lead-ins with next chunk
        2. Merge short prefixes (< 5 tokens) with next chunk
        3. No discards - preserve all content
        Args:
            base_chunks: Initial chunks from SentenceSplitter
        Returns:
            Post-processed chunks
        """
        if not base_chunks:
            return base_chunks
        processed: List[TextNode] = []
        i = 0
        while i < len(base_chunks):
            current_chunk = base_chunks[i]
            current_text = current_chunk.text.strip()
            # Rule 1: Merge colon-ended lead-ins
            if (
                self.config.embedding.merge_colon_endings
                and current_text.endswith(":")
                and i + 1 < len(base_chunks)
            ):
                next_chunk = base_chunks[i + 1]
                next_text = next_chunk.text.strip()
                # Check if next chunk starts with lowercase or quote/code
                if next_text and (
                    next_text[0].islower() or next_text.startswith(("`", '"', "'"))
                ):
                    # Merge the chunks
                    merged_text = f"{current_text} {next_text}"
                    merged_chunk = TextNode(
                        text=merged_text, metadata=current_chunk.metadata
                    )
                    processed.append(merged_chunk)
                    i += 2  # Skip both chunks
                    logger.debug("Merged colon-ended lead-in with continuation")
                    continue
            # Rule 2: Merge short prefixes
            if (
                self.config.embedding.merge_short_prefixes
                and self._count_tokens(current_text)
                < self.config.embedding.min_chunk_tokens
                and i + 1 < len(base_chunks)
            ):
                next_chunk = base_chunks[i + 1]
                merged_text = f"{current_text} {next_chunk.text.strip()}"
                merged_chunk = TextNode(
                    text=merged_text, metadata=current_chunk.metadata
                )
                processed.append(merged_chunk)
                i += 2  # Skip both chunks
                logger.debug("Merged short prefix with next chunk")
                continue
            # No rule applied, keep chunk as-is
            processed.append(current_chunk)
            i += 1
        logger.debug(f"Post-processing: {len(base_chunks)} -> {len(processed)} chunks")
        return processed

    def _count_tokens(self, text: str) -> int:
        """
        Simple token counting (word-based approximation).
        Args:
            text: Text to count tokens for
        Returns:
            Approximate token count
        """
        # Simple whitespace-based token counting
        # In production, could use tiktoken or similar for more accuracy
        return len(text.split())
