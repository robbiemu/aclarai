# Claim-Concept Linking System

This document describes the implementation of the claim-concept linking system for aclarai, following the requirements in `docs/project/epic_1/sprint_5-Link_claims_to_concepts.md` and the architecture defined in `docs/arch/on-linking_claims_to_concepts.md`.

## Overview

The claim-concept linking system establishes semantic relationships between `(:Claim)` nodes and `(:Concept)` nodes in the knowledge graph using LLM-based classification. It supports three types of relationships:

-   `SUPPORTS_CONCEPT`: The claim directly supports or affirms the concept.
-   `MENTIONS_CONCEPT`: The claim is related to the concept but does not affirm or refute it.
-   `CONTRADICTS_CONCEPT`: The claim contradicts the concept.

### Quality Filtering for Concept Linking

To ensure the integrity of the knowledge graph, the linking process incorporates a quality filtering step based on claim evaluation scores. This gatekeeps the creation of strong semantic relationships.

-   **Score Location:** The evaluation scores for a claim are stored on the `[:ORIGINATES_FROM]` relationship that links the `(:Claim)` node to its source `(:Block)` node.
-   **Geometric Mean:** The system first calculates a geometric mean from the claim's `entailed_score`, `coverage_score`, and `decontextualization_score` to produce a single quality metric.
-   **Quality Threshold:** This mean is compared against a configurable `threshold.claim_quality` (default: 0.7).
-   **Relationship-Specific Logic:**
    -   **`SUPPORTS_CONCEPT` / `CONTRADICTS_CONCEPT`:** These strong semantic links are only created if the claim's quality score meets or exceeds the threshold.
    -   **`MENTIONS_CONCEPT`:** This weaker, associative link is permitted even if a claim's quality score is below the threshold, as long as its evaluation scores are not `null`.
-   **Null Score Handling:** Any claim with a `null` value for any of the three evaluation scores is automatically excluded from all types of concept linking.

This ensures that only high-quality, well-vetted claims form the strong evidentiary backbone of the knowledge graph.

### Quality Filtering for Concept Linking

To ensure the integrity of the knowledge graph, the linking process incorporates a quality filtering step based on claim evaluation scores. This gatekeeps the creation of strong semantic relationships.

-   **Geometric Mean:** The system first calculates a geometric mean from the claim's `entailed_score`, `coverage_score`, and `decontextualization_score` to produce a single quality metric.
-   **Quality Threshold:** This mean is compared against a configurable `threshold.claim_quality` (default: 0.7).
-   **Relationship-Specific Logic:**
    -   **`SUPPORTS_CONCEPT` / `CONTRADICTS_CONCEPT`:** These strong semantic links are only created if the claim's quality score meets or exceeds the threshold.
    -   **`MENTIONS_CONCEPT`:** This weaker, associative link is permitted even if a claim's quality score is below the threshold, as long as its evaluation scores are not `null`.
-   **Null Score Handling:** Any claim with a `null` value for any of the three evaluation scores is automatically excluded from all types of concept linking.

This ensures that only high-quality, well-vetted claims form the strong evidentiary backbone of the knowledge graph.


## Architecture

The system consists of several key components:

### 1. ClaimConceptLinkerAgent

The LLM agent responsible for classifying relationships between claims and concepts.

**Key Features:**

-   Uses configurable LLM (currently supports OpenAI)
-   Structured prompt generation following architecture specifications
-   JSON response parsing with validation
-   Robust error handling for invalid responses

### 2. ClaimConceptNeo4jManager

Handles all Neo4j operations for claim-concept linking.

**Key Features:**

-   Fetches unlinked claims prioritized by recency
-   Fetches available concepts for linking
-   Creates relationships with proper metadata
-   Batch processing support
-   Context retrieval for improved classification

### 3. Tier2MarkdownUpdater

Updates Tier 2 Markdown files with concept wikilinks.

**Key Features:**

-   Atomic file writes using temp file + rename pattern
-   Preserves `aclarai:id` anchors and increments version numbers
-   Adds `[[concept]]` wikilinks to relevant sections
-   Batch processing of multiple files

### 4. ClaimConceptLinker (Orchestrator)

Main coordinator that manages the full linking process.

**Key Features:**

-   End-to-end workflow coordination
-   Configurable similarity and strength thresholds
-   Comprehensive statistics and error tracking
-   Fallback concept matching when vector store unavailable

## Data Models

### ClaimConceptPair

Represents a claim-concept pair for analysis:

```python
@dataclass
class ClaimConceptPair:
    claim_id: str
    claim_text: str
    concept_id: str
    concept_text: str
    source_sentence: Optional[str] = None  # Context
    summary_block: Optional[str] = None    # Context
    entailed_score: Optional[float] = None
    coverage_score: Optional[float] = None
    decontextualization_score: Optional[float] = None
```

### ClaimConceptLinkResult

Represents a successful linking result:

```python
@dataclass
class ClaimConceptLinkResult:
    claim_id: str
    concept_id: str
    relationship: RelationshipType
    strength: float
    entailed_score: Optional[float] = None
    coverage_score: Optional[float] = None
    decontextualization_score: Optional[float] = None
```

## Integration Instructions

### Prerequisites

1.  Concepts vector store populated by Tier 3 creation task
2.  LLM API configuration (OpenAI recommended)
3.  Neo4j database with `(:Claim)` and `(:Concept)` nodes
4.  Vault directory structure for Tier 2 files

### Configuration

Add LLM configuration to your aclarai config:

```yaml
llm:
  provider: "openai"
  model: "gpt-4o"
  api_key: "${OPENAI_API_KEY}"
```

### Usage Example

```python
from aclarai_shared.claim_concept_linking import ClaimConceptLinker

linker = ClaimConceptLinker()
results = linker.link_claims_to_concepts(
    max_claims=100,
    similarity_threshold=0.7,
    strength_threshold=0.5
)

print(f"Linked {results['links_created']} claim-concept pairs")
print(f"Updated {results['files_updated']} Tier 2 files")
```

## Testing

Run the test suite:

```bash
cd /home/runner/work/aclarai/aclarai
uv run python -m pytest shared/tests/claim_concept_linking/ -v
```

## Error Handling

The system includes comprehensive error handling:

-   **LLM Failures**: Graceful degradation with logging
-   **Invalid Responses**: JSON parsing validation with fallbacks
-   **Neo4j Errors**: Database operation retries and error reporting
-   **File System**: Atomic writes prevent corruption
-   **Null Values**: Explicit handling throughout the pipeline

## Logging

All components use structured logging with:

-   Service identification (`service: "aclarai"`)
-   Function-level context (`filename.function_name`)
-   Relevant IDs (claim_id, concept_id, aclarai_id)
-   Performance metrics and error details

## Next Steps

1.  **Integration**: Deploy to `aclarai-core` service for automated processing
2.  **Monitoring**: Set up alerts for relationship creation rates
3.  **Quality**: Review created relationships and adjust thresholds as needed