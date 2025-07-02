# Neo4j Graph System

The graph system provides functionality for creating and managing `:Claim` and `:Sentence` nodes in the aclarai Neo4j knowledge graph.

## Overview

The graph management module provides:

-   **Data Models**: `Claim`, `Sentence`, `ClaimInput`, `SentenceInput` for structured node data
-   **Neo4j Manager**: `Neo4jGraphManager` for database operations
-   **Schema Management**: Automatic constraint and index creation
-   **Batch Operations**: Efficient batch creation using Cypher UNWIND
-   **Relationship Management**: Automatic `ORIGINATES_FROM` relationships to Block nodes

## Architecture

This implementation follows the architectural guidelines from `docs/arch/idea-neo4J-ineteraction.md`, using the **Neo4j Python Driver (Direct Cypher)** for:

-   Precise control over node and relationship properties, including nullable evaluation scores.
-   Batch operations with UNWIND for performance.
-   Schema management with constraints and indexes.

## Data Models

### ClaimInput

Input data for creating Claim nodes:

```python
from aclarai_shared.graph import ClaimInput

claim_input = ClaimInput(
    text="The Earth orbits the Sun",
    block_id="block_conversation_123_chunk_5"
)
```

Properties:

-   `text`: The claim text content
-   `block_id`: ID of the originating `:Block` node
-   `claim_id`: Optional string - Auto-generated if not provided

### SentenceInput

Input data for creating Sentence nodes:

```python
from aclarai_shared.graph import SentenceInput

sentence_input = SentenceInput(
    text="This statement is ambiguous and unclear",
    block_id="block_conversation_123_chunk_8",
    ambiguous=True,
    verifiable=False
)
```

Properties:

-   `text`: The sentence text content
-   `block_id`: ID of the originating `:Block` node
-   `ambiguous`: Optional boolean - Whether the sentence is ambiguous
-   `verifiable`: Optional boolean - Whether the sentence is verifiable
-   `sentence_id`: Optional string - Auto-generated if not provided

## Schema

### Node Properties

#### Claim Nodes

```cypher
(:Claim {
    id: String,       // Unique identifier
    text: String,     // Claim text content
    version: Integer, // Version number (starts at 1)
    timestamp: DateTime // Creation timestamp
})
```

#### Sentence Nodes

```cypher
(:Sentence {
    id: String,         // Unique identifier
    text: String,       // Sentence text content
    ambiguous: Boolean, // Whether sentence is ambiguous (or null)
    verifiable: Boolean,// Whether sentence is verifiable (or null)
    version: Integer,   // Version number (starts at 1)
    timestamp: DateTime // Creation timestamp
})
```

### Relationships

Both Claims and Sentences have `ORIGINATES_FROM` relationships to their source Block nodes. **The relationship for Claims holds the evaluation scores.**

```cypher
(:Claim)-[:ORIGINATES_FROM {
    entailed_score: Float,
    coverage_score: Float,
    decontextualization_score: Float
}]->(:Block)

(:Sentence)-[:ORIGINATES_FROM]->(:Block)
```

### Constraints and Indexes

The schema setup creates:

**Constraints:**

-   `claim_id_unique`: Ensures Claim IDs are unique
-   `sentence_id_unique`: Ensures Sentence IDs are unique

**Indexes:**

-   `claim_text_index`: Index on Claim text for search performance
-   `sentence_text_index`: Index on Sentence text for search performance

## Integration with Claimify Pipeline

This module is designed to integrate with the Claimify claim extraction pipeline:

1.  **Claimify Output Processing**: Convert Claimify claim extraction results into `ClaimInput` objects.
2.  **Score Persistence**: The evaluation scores (`entailed_score`, `coverage_score`, `decontextualization_score`) are stored on the `[:ORIGINATES_FROM]` relationship, not the `:Claim` node itself.
3.  **Fallback Sentence Creation**: Create `SentenceInput` objects for utterances that didn't produce high-quality claims.
4.  **Batch Processing**: Use batch operations for efficient database writes.

## Error Handling

The module includes comprehensive error handling:

-   **Connection Errors**: Proper handling of Neo4j connection failures
-   **Authentication Errors**: Clear error messages for auth problems
-   **Validation**: Input validation for required fields
-   **Graceful Degradation**: Null score handling for failed evaluation agents

## Configuration

Neo4j connection settings are managed through the aclarai configuration system:

```yaml
databases:
  neo4j:
    host: "neo4j"      # NEO4J_HOST environment variable
    port: 7687         # NEO4J_BOLT_PORT environment variable
    # NEO4J_USER and NEO4J_PASSWORD from environment
```

## Implementation Details

The graph system is implemented in `shared/aclarai_shared/graph/`:

-   `models.py`: Data models for Claims and Sentences
-   `neo4j_manager.py`: Neo4j operations manager
-   `__init__.py`: Module initialization and exports

For detailed usage examples, see the [Neo4j Graph Tutorial](../tutorials/neo4j_graph_tutorial.md).