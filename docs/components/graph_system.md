# Neo4j Graph System

The graph system provides functionality for creating and managing (:Claim) and (:Sentence) nodes in the aclarai Neo4j knowledge graph.

## Overview

The graph management module provides:

- **Data Models**: `Claim`, `Sentence`, `ClaimInput`, `SentenceInput` for structured node data
- **Neo4j Manager**: `Neo4jGraphManager` for database operations 
- **Schema Management**: Automatic constraint and index creation
- **Batch Operations**: Efficient batch creation using Cypher UNWIND
- **Relationship Management**: Automatic `ORIGINATES_FROM` relationships to Block nodes

## Architecture

This implementation follows the architectural guidelines from `docs/arch/idea-neo4J-ineteraction.md`, using **Neo4j Python Driver (Direct Cypher)** for:

- Precise control over node properties including nullable evaluation scores
- Batch operations with UNWIND for performance
- Complex property types and relationships
- Schema management with constraints and indexes

## Data Models

### ClaimInput

Input data for creating Claim nodes:

```python
from aclarai_shared.graph import ClaimInput

claim_input = ClaimInput(
    text="The Earth orbits the Sun",
    block_id="block_conversation_123_chunk_5",
    entailed_score=0.95,
    coverage_score=0.88,
    decontextualization_score=0.92
)
```

Properties:
- `text`: The claim text content
- `block_id`: ID of the originating Block node 
- `entailed_score`: Optional float (0.0-1.0) - NLI entailment score
- `coverage_score`: Optional float (0.0-1.0) - Information completeness score
- `decontextualization_score`: Optional float (0.0-1.0) - Context independence score
- `claim_id`: Optional string - Auto-generated if not provided

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
- `text`: The sentence text content
- `block_id`: ID of the originating Block node
- `ambiguous`: Optional boolean - Whether the sentence is ambiguous
- `verifiable`: Optional boolean - Whether the sentence is verifiable
- `sentence_id`: Optional string - Auto-generated if not provided

## Claimify Pipeline Integration

The graph system integrates seamlessly with the Claimify pipeline to persist extracted claims and sentences:

### Integration Flow
1. **Claimify Processing**: Sentences are processed through Selection → Disambiguation → Decomposition
2. **Result Conversion**: `ClaimifyResult` objects are converted to graph input data
3. **Batch Persistence**: Claims and sentences are persisted in batches with relationships

### Node Creation Patterns
- **Valid Claims**: Claims meeting quality criteria (atomic, self-contained, verifiable) → `:Claim` nodes
- **Invalid Claims**: Claims failing criteria → `:Sentence` nodes with rejection reasons
- **Unprocessed Sentences**: Sentences not selected by pipeline → `:Sentence` nodes

### Relationship Management
All created nodes automatically get `ORIGINATES_FROM` relationships to their source `:Block` nodes, maintaining data lineage.

## Usage

For complete usage examples and step-by-step tutorials, see:
- **Tutorial**: `docs/tutorials/neo4j_graph_tutorial.md` - Complete guide with examples and integration patterns
- **Claimify Integration**: `docs/tutorials/claimify_integration_tutorial.md` - End-to-end Claimify pipeline integration
- **Architecture**: `docs/arch/idea-neo4J-ineteraction.md` - Technical implementation details

## Schema

### Node Properties

#### Claim Nodes
```cypher
(:Claim {
    id: String,                        // Unique identifier
    text: String,                      // Claim text content
    entailed_score: Float,             // 0.0-1.0 or null
    coverage_score: Float,             // 0.0-1.0 or null  
    decontextualization_score: Float,  // 0.0-1.0 or null
    version: Integer,                  // Version number (starts at 1)
    timestamp: DateTime                // Creation timestamp
})
```

#### Sentence Nodes
```cypher
(:Sentence {
    id: String,           // Unique identifier
    text: String,         // Sentence text content
    ambiguous: Boolean,   // Whether sentence is ambiguous (or null)
    verifiable: Boolean,  // Whether sentence is verifiable (or null)
    version: Integer,     // Version number (starts at 1)
    timestamp: DateTime   // Creation timestamp
})
```

### Relationships

Both Claims and Sentences have `ORIGINATES_FROM` relationships to their source Block nodes:

```cypher
(:Claim)-[:ORIGINATES_FROM]->(:Block)
(:Sentence)-[:ORIGINATES_FROM]->(:Block)
```

### Constraints and Indexes

The schema setup creates:

**Constraints:**
- `claim_id_unique`: Ensures Claim IDs are unique
- `sentence_id_unique`: Ensures Sentence IDs are unique

**Indexes:**
- `claim_text_index`: Index on Claim text for search performance
- `sentence_text_index`: Index on Sentence text for search performance  
- `claim_entailed_score_index`: Index on entailed_score for filtering
- `claim_coverage_score_index`: Index on coverage_score for filtering
- `claim_decontextualization_score_index`: Index on decontextualization_score for filtering

## Integration with Claimify Pipeline

This module is designed to integrate with the Claimify claim extraction pipeline:

1. **Claimify Output Processing**: Convert Claimify claim extraction results into `ClaimInput` objects
2. **Quality Score Handling**: Store evaluation scores (`entailed_score`, `coverage_score`, `decontextualization_score`) as nullable fields
3. **Fallback Sentence Creation**: Create `SentenceInput` objects for utterances that didn't produce high-quality claims
4. **Batch Processing**: Use batch operations for efficient database writes

## Error Handling

The module includes comprehensive error handling:

- **Connection Errors**: Proper handling of Neo4j connection failures
- **Authentication Errors**: Clear error messages for auth problems
- **Validation**: Input validation for required fields
- **Graceful Degradation**: Null score handling for failed evaluation agents

## Performance Considerations

- **Batch Operations**: All creation operations use Cypher UNWIND for efficient batch processing
- **Indexes**: Comprehensive indexing on frequently queried properties
- **Connection Management**: Context managers for proper resource cleanup
- **Transaction Safety**: All operations are transactional

## Configuration

Neo4j connection settings are managed through the aclarai configuration system:

```yaml
databases:
  neo4j:
    host: "neo4j"      # NEO4J_HOST environment variable
    port: 7687         # NEO4J_BOLT_PORT environment variable
    # NEO4J_USER and NEO4J_PASSWORD from environment
```

Environment variables:
- `NEO4J_HOST`: Neo4j server hostname
- `NEO4J_BOLT_PORT`: Neo4j bolt port (default: 7687)
- `NEO4J_USER`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password

## Implementation Details

The graph system is implemented in `shared/aclarai_shared/graph/`:

- `models.py`: Data models for Claims and Sentences
- `neo4j_manager.py`: Neo4j operations manager
- `__init__.py`: Module initialization and exports

For detailed usage examples, see the [Neo4j Graph Tutorial](../tutorials/neo4j_graph_tutorial.md).