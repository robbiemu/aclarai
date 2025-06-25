"""
Data models for claim-concept linking.
This module defines the data structures used for linking claims to concepts,
following the schema from docs/arch/on-linking_claims_to_concepts.md.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class RelationshipType(Enum):
    """Types of relationships between claims and concepts."""

    SUPPORTS_CONCEPT = "SUPPORTS_CONCEPT"
    MENTIONS_CONCEPT = "MENTIONS_CONCEPT"
    CONTRADICTS_CONCEPT = "CONTRADICTS_CONCEPT"


@dataclass
class ClaimConceptPair:
    """Represents a claim-concept pair for linking analysis."""

    claim_id: str
    claim_text: str
    concept_id: str
    concept_text: str
    # Optional context for better classification
    source_sentence: Optional[str] = None
    summary_block: Optional[str] = None
    # Claim properties (may be null during Sprint 5)
    entailed_score: Optional[float] = None
    coverage_score: Optional[float] = None
    decontextualization_score: Optional[float] = None


@dataclass
class ClaimConceptLinkResult:
    """Result of claim-concept linking classification."""

    claim_id: str
    concept_id: str
    relationship: RelationshipType
    strength: float  # 0.0 to 1.0
    # Scores from the claim (copied during linking)
    entailed_score: Optional[float] = None
    coverage_score: Optional[float] = None
    # Metadata
    classified_at: Optional[datetime] = None
    agent_model: Optional[str] = None

    def __post_init__(self):
        """Set classified_at if not provided."""
        if self.classified_at is None:
            self.classified_at = datetime.now(timezone.utc)

    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j relationship properties."""
        return {
            "strength": self.strength,
            "entailed_score": self.entailed_score,
            "coverage_score": self.coverage_score,
            "classified_at": self.classified_at.isoformat()
            if self.classified_at
            else None,
            "agent_model": self.agent_model,
        }


@dataclass
class LinkingError:
    """Represents an error during claim-concept linking."""

    claim_id: str
    concept_id: Optional[str] = None
    error_type: str = "unknown"
    error_message: str = ""
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class AgentClassificationResult:
    """Raw result from the LLM agent classification."""

    relation: str  # Raw string from agent
    strength: float
    entailed_score: Optional[float] = None
    coverage_score: Optional[float] = None

    def to_relationship_type(self) -> Optional[RelationshipType]:
        """Convert string relation to RelationshipType enum."""
        try:
            return RelationshipType(self.relation)
        except ValueError:
            return None


@dataclass
class ConceptCandidate:
    """Represents a concept candidate for linking."""

    concept_id: str
    concept_text: str
    similarity_score: float  # Cosine similarity from vector search
    # Metadata from the concept
    source_node_id: Optional[str] = None
    source_node_type: Optional[str] = None
    aclarai_id: Optional[str] = None
