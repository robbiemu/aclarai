# Neo4j Graph Management Tutorial

This tutorial shows you how to use the aclarai Neo4j graph system to create and manage `(:Claim)` and `(:Sentence)` nodes in your knowledge graph. We'll walk through everything from basic data models to real-world integration patterns.

## What You'll Learn

-   How to create and use `ClaimInput` and `SentenceInput` objects
-   How to connect to Neo4j and set up the schema
-   How to create Claims and Sentences in batch
-   How to query relationship properties like evaluation scores
-   How to integrate with the Claimify pipeline

## Prerequisites

Before starting this tutorial, make sure you have:

-   A running Neo4j database
-   aclarai shared library installed
-   Neo4j connection credentials (`NEO4J_HOST`, `NEO4J_USER`, `NEO4J_PASSWORD`)

## Part 1: Understanding the Data Models

Let's start by exploring the data models that represent Claims and Sentences.

### Creating ClaimInput Objects

`ClaimInput` objects represent claims that will be stored in the knowledge graph. They contain the claim text and metadata. The evaluation scores are handled separately when the relationship is created.

```python
from aclarai_shared.graph import ClaimInput

# Create a claim input
claim_input = ClaimInput(
    text="Python is a programming language developed by Guido van Rossum",
    block_id="block_conversation_001_chunk_05"
)

print(f"Claim ID: {claim_input.claim_id}")  # Auto-generated UUID
print(f"Text: {claim_input.text}")
print(f"Source: {claim_input.block_id}")
```

### Creating SentenceInput Objects

`SentenceInput` objects represent utterances that didn't produce high-quality claims but should still be stored:

```python
from aclarai_shared.graph import SentenceInput

# Ambiguous sentence that can't be verified
sentence_input = SentenceInput(
    text="Hmm, that's really interesting",
    block_id="block_conversation_001_chunk_08",
    ambiguous=True,
    verifiable=False
)

print(f"Sentence ID: {sentence_input.sentence_id}")  # Auto-generated UUID
print(f"Ambiguous: {sentence_input.ambiguous}")
print(f"Verifiable: {sentence_input.verifiable}")
```

## Part 2: Database Operations

Now let's connect to Neo4j and create nodes in the database.

### Setting Up the Connection

```python
from aclarai_shared.graph import Neo4jGraphManager
from aclarai_shared.config import aclaraiConfig

# Load configuration (uses environment variables)
config = aclaraiConfig.from_env()

# Create graph manager with context management for automatic cleanup
with Neo4jGraphManager(config) as graph:
    print("Connected to Neo4j successfully")

    # Setup schema (constraints and indexes)
    graph.setup_schema()
    print("Schema setup completed")

    # Database operations go here...
```

### Creating Claims and Sentences in Batch

Batch operations are efficient for creating multiple nodes:

```python
# Prepare multiple claims and sentences
claim_inputs = [
    ClaimInput(
        text="Python is a programming language developed by Guido van Rossum",
        block_id="block_conversation_001_chunk_05"
    ),
    ClaimInput(
        text="Neo4j is a graph database",
        block_id="block_conversation_001_chunk_18"
    )
]

sentence_inputs = [
    SentenceInput(
        text="What do you think we should do next?",
        block_id="block_conversation_001_chunk_25",
        ambiguous=False,
        verifiable=False
    )
]

# Create nodes in batch
with Neo4jGraphManager(config) as graph:
    claims = graph.create_claims(claim_inputs)
    sentences = graph.create_sentences(sentence_inputs)
    print(f"Successfully created {len(claims)} Claim and {len(sentences)} Sentence nodes")
```

### Updating Evaluation Scores on Relationships

After evaluation agents run, their scores are stored on the `[:ORIGINATES_FROM]` relationship.

```python
from aclarai_core.graph.claim_evaluation_graph_service import ClaimEvaluationGraphService

# Assume 'driver' is an initialized Neo4j driver
graph_service = ClaimEvaluationGraphService(driver, config)

# Update scores for one of the claims we created
claim_id_to_update = claims[0].claim_id
block_id_source = claim_inputs[0].block_id

graph_service.update_relationship_score(claim_id_to_update, block_id_source, "entailed_score", 0.98)
graph_service.update_relationship_score(claim_id_to_update, block_id_source, "coverage_score", 0.95)
graph_service.update_relationship_score(claim_id_to_update, block_id_source, "decontextualization_score", 0.87)

print(f"Updated evaluation scores for claim: {claim_id_to_update}")
```

### Retrieving Nodes and Scores

Once created, you can retrieve nodes and their relationship properties:

```python
with Neo4jGraphManager(config) as graph:
    # Get a specific claim and its evaluation scores
    query = """
    MATCH (c:Claim {id: $claim_id})-[r:ORIGINATES_FROM]->(b:Block)
    RETURN c.text AS text,
           r.entailed_score AS entailed,
           r.coverage_score AS coverage,
           r.decontextualization_score AS decon
    """
    result = graph.execute_query(query, parameters={"claim_id": claim_id_to_update})

    if result:
        claim_data = result[0]
        print(f"\nFound claim: '{claim_data['text']}'")
        print(f"Scores: E={claim_data['entailed']:.2f}, C={claim_data['coverage']:.2f}, D={claim_data['decon']:.2f}")

    # Get overall statistics
    counts = graph.count_nodes()
    print(f"\nGraph contains:")
    print(f"  Claims: {counts['claims']}")
    print(f"  Sentences: {counts['sentences']}")
    print(f"  Blocks: {counts['blocks']}")
```

## Part 3: Claimify Pipeline Integration

Here's how to integrate this system with the Claimify claim extraction pipeline.

### Processing Claimify Output

Simulate converting raw Claimify results into graph nodes:

```python
# Simulate raw output from Claimify pipeline
raw_claimify_results = [
    {
        "type": "claim",
        "text": "The Earth orbits the Sun in an elliptical path",
        "source_block": "block_astro_conv_chunk_7",
    },
    {
        "type": "sentence",
        "text": "This seems complex and has multiple interpretations",
        "source_block": "block_sys_conv_chunk_22",
        "analysis": {"ambiguous": True, "verifiable": True}
    }
]

# Convert to graph inputs
claim_inputs = [
    ClaimInput(text=r["text"], block_id=r["source_block"])
    for r in raw_claimify_results if r["type"] == "claim"
]
sentence_inputs = [
    SentenceInput(
        text=r["text"],
        block_id=r["source_block"],
        ambiguous=r["analysis"].get("ambiguous"),
        verifiable=r["analysis"].get("verifiable")
    )
    for r in raw_claimify_results if r["type"] == "sentence"
]

# Persist to Neo4j
with Neo4jGraphManager(config) as graph:
    graph.create_claims(claim_inputs)
    graph.create_sentences(sentence_inputs)
    print("Persisted Claimify batch to Neo4j.")
```

## Part 4: Schema Reference

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

## Troubleshooting

### Common Issues

**Connection Errors:**

```python
try:
    with Neo4jGraphManager(config) as graph:
        graph.setup_schema()
except Exception as e:
    print(f"Connection failed: {e}")
    print("Check NEO4J_HOST, NEO4J_USER, NEO4J_PASSWORD environment variables")
```

**Authentication Errors:**

-   Verify `NEO4J_USER` and `NEO4J_PASSWORD` are correct.
-   Check that the user has write permissions.

### Best Practices

1.  **Always use context managers** for database connections.
2.  **Set up the schema** before creating nodes.
3.  **Use batch operations** for efficiency.
4.  **Handle null scores gracefully** for failed agents.

## Next Steps

Now that you understand the Neo4j graph system:

1.  Integrate it into your Claimify pipeline.
2.  Set up evaluation agents to populate scores.
3.  Monitor graph growth and performance.
4.  Consider adding more sophisticated queries and analytics.

For more details on the underlying implementation, see the [Graph System Components](../components/graph_system.md) documentation.