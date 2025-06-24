"""
Data models for aclarai knowledge graph nodes.
This module defines the data structures for Claims, Sentences, Blocks, and other
graph entities, following the schema from technical_overview.md and
graph_schema.cypher.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class BlockInput:
    """Input data for creating a Block node."""

    text: str
    block_id: str  # aclarai:id from markdown
    content_hash: str
    source_file: str
    version: int = 1

    @property
    def id(self) -> str:
        """Alias for block_id for backward compatibility."""
        return self.block_id


@dataclass
class ClaimInput:
    """Input data for creating a Claim node."""

    text: str
    block_id: str  # ID of the originating Block node
    entailed_score: Optional[float] = None
    coverage_score: Optional[float] = None
    decontextualization_score: Optional[float] = None
    claim_id: Optional[str] = None  # Will generate if not provided
    # Properties derived from ClaimCandidate for testing/validation
    verifiable: bool = True
    self_contained: bool = True
    context_complete: bool = True

    def __post_init__(self):
        """Generate claim_id if not provided."""
        if self.claim_id is None:
            self.claim_id = f"claim_{uuid.uuid4().hex[:12]}"

    @property
    def id(self) -> str:
        """Alias for claim_id for backward compatibility."""
        assert self.claim_id is not None

        return self.claim_id


@dataclass
class SentenceInput:
    """Input data for creating a Sentence node."""

    text: str
    block_id: str  # ID of the originating Block node
    ambiguous: Optional[bool] = None
    verifiable: Optional[bool] = None
    sentence_id: Optional[str] = None  # Will generate if not provided
    # Additional properties for testing/validation
    failed_decomposition: bool = False
    rejection_reason: Optional[str] = None

    def __post_init__(self):
        """Generate sentence_id if not provided."""
        if self.sentence_id is None:
            self.sentence_id = f"sentence_{uuid.uuid4().hex[:12]}"

    @property
    def id(self) -> str:
        """Alias for sentence_id for backward compatibility."""
        assert self.sentence_id is not None

        return self.sentence_id


@dataclass
class Block:
    """Represents a Block node in the knowledge graph."""

    block_id: str
    text: str
    content_hash: str
    source_file: str
    version: int
    timestamp: datetime
    needs_reprocessing: bool = True
    last_updated: Optional[datetime] = None

    @classmethod
    def from_input(cls, block_input: BlockInput) -> "Block":
        """Create a Block from BlockInput."""
        current_time = datetime.now(timezone.utc)
        return cls(
            block_id=block_input.block_id,
            text=block_input.text,
            content_hash=block_input.content_hash,
            source_file=block_input.source_file,
            version=block_input.version,
            timestamp=current_time,
            needs_reprocessing=True,
            last_updated=current_time,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j storage."""
        return {
            "id": self.block_id,  # Use 'id' as primary key in Neo4j
            "text": self.text,
            "hash": self.content_hash,
            "source_file": self.source_file,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "needs_reprocessing": self.needs_reprocessing,
            "last_updated": self.last_updated.isoformat()
            if self.last_updated
            else None,
        }


@dataclass
class Claim:
    """Represents a Claim node in the knowledge graph."""

    claim_id: str
    text: str
    entailed_score: Optional[float]
    coverage_score: Optional[float]
    decontextualization_score: Optional[float]
    version: int
    timestamp: datetime

    @classmethod
    def from_input(cls, claim_input: ClaimInput, version: int = 1) -> "Claim":
        """Create a Claim from ClaimInput."""
        assert claim_input.claim_id is not None

        return cls(
            claim_id=claim_input.claim_id,
            text=claim_input.text,
            entailed_score=claim_input.entailed_score,
            coverage_score=claim_input.coverage_score,
            decontextualization_score=claim_input.decontextualization_score,
            version=version,
            timestamp=datetime.now(timezone.utc),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j storage."""
        return {
            "id": self.claim_id,  # Use 'id' as primary key in Neo4j
            "text": self.text,
            "entailed_score": self.entailed_score,
            "coverage_score": self.coverage_score,
            "decontextualization_score": self.decontextualization_score,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Sentence:
    """Represents a Sentence node in the knowledge graph."""

    sentence_id: str
    text: str
    ambiguous: Optional[bool]
    verifiable: Optional[bool]
    failed_decomposition: bool
    rejection_reason: Optional[str]
    version: int
    timestamp: datetime

    @classmethod
    def from_input(cls, sentence_input: SentenceInput, version: int = 1) -> "Sentence":
        """Create a Sentence from SentenceInput."""
        assert sentence_input.sentence_id is not None

        return cls(
            sentence_id=sentence_input.sentence_id,
            text=sentence_input.text,
            ambiguous=sentence_input.ambiguous,
            verifiable=sentence_input.verifiable,
            failed_decomposition=sentence_input.failed_decomposition,
            rejection_reason=sentence_input.rejection_reason,
            version=version,
            timestamp=datetime.now(timezone.utc),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j storage."""
        return {
            "id": self.sentence_id,  # Use 'id' as primary key in Neo4j
            "text": self.text,
            "ambiguous": self.ambiguous,
            "verifiable": self.verifiable,
            "failed_decomposition": self.failed_decomposition,
            "rejection_reason": self.rejection_reason,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConceptInput:
    """Input data for creating a Concept node."""

    text: str  # The canonical concept text
    source_candidate_id: str  # ID of the concept candidate that was promoted
    source_node_id: str  # ID of the source node (Claim/Summary)
    source_node_type: str  # Type of source node
    aclarai_id: str  # aclarai_id of the source document
    concept_id: Optional[str] = None  # Will generate if not provided

    def __post_init__(self):
        """Generate concept_id if not provided."""
        if self.concept_id is None:
            self.concept_id = f"concept_{uuid.uuid4().hex[:12]}"

    @property
    def id(self) -> str:
        """Alias for concept_id for backward compatibility."""
        assert self.concept_id is not None

        return self.concept_id


@dataclass
class Concept:
    """Represents a Concept node in the knowledge graph."""

    concept_id: str
    text: str
    source_candidate_id: str
    source_node_id: str
    source_node_type: str
    aclarai_id: str
    version: int
    timestamp: datetime

    @classmethod
    def from_input(cls, concept_input: ConceptInput, version: int = 1) -> "Concept":
        """Create a Concept from ConceptInput."""
        assert concept_input.concept_id is not None

        return cls(
            concept_id=concept_input.concept_id,
            text=concept_input.text,
            source_candidate_id=concept_input.source_candidate_id,
            source_node_id=concept_input.source_node_id,
            source_node_type=concept_input.source_node_type,
            aclarai_id=concept_input.aclarai_id,
            version=version,
            timestamp=datetime.now(timezone.utc),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j storage."""
        return {
            "id": self.concept_id,  # Use 'id' as primary key in Neo4j
            "text": self.text,
            "source_candidate_id": self.source_candidate_id,
            "source_node_id": self.source_node_id,
            "source_node_type": self.source_node_type,
            "aclarai_id": self.aclarai_id,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
        }
