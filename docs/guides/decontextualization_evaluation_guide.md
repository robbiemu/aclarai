# Guide: Decontextualization Evaluation Workflow

This guide explains how to use the components of the decontextualization evaluation workflow. This system is responsible for assessing whether a factual claim can be understood on its own, without its original context.

## 🎯 Purpose

The goal of this workflow is to generate a `decontextualization_score` for each claim. This score is crucial for ensuring that only high-quality, portable claims are promoted into the knowledge base.

The workflow involves three key components:
1.  **`DecontextualizationAgent`**: The LLM-powered agent that performs the evaluation.
2.  **`ClaimEvaluationGraphService`**: The service that persists the score to the Neo4j graph.
3.  **`MarkdownUpdaterService`**: The service that writes the score back to the vault's Markdown files.

## ⚙️ Core Workflow

The following sections provide a deep dive into each component, with usage examples.

### High-Level Workflow Example

Here is a conceptual overview of how these services work together to evaluate a single claim:

The end-to-end process for evaluating a single claim looks like this:

```python
# Assume 'agent', 'graph_service', and 'markdown_service' are already initialized.
# Assume 'claim_data' is a dictionary with claim_id, claim_text, block_id, and source_text.

# 1. Agent evaluates the claim to get a score.
score, message = await agent.evaluate_claim_decontextualization(
    claim_id=claim_data['claim_id'],
    claim_text=claim_data['claim_text'],
    source_id=claim_data['block_id'],
    source_text=claim_data['source_text']
)

if score is None:
    logger.error(f"Evaluation failed for claim {claim_data['claim_id']}: {message}")
    return

# 2. Persist the score to the Neo4j graph.
graph_service.update_decontextualization_score(
    claim_id=claim_data['claim_id'],
    block_id=claim_data['block_id'],
    score=score
)

# 3. Persist the score to the Markdown file.
markdown_service.add_or_update_decontextualization_score(
    filepath_str=claim_data['file_path'],
    block_id=claim_data['block_id'],
    score=score
)
```

## Component Deep Dive & Usage Examples

### 1. The `DecontextualizationAgent`

This agent uses a configured LLM and a set of tools to assess a claim's semantic autonomy.

### Key Behaviors

*   **Tool-Driven:** The agent is provided its tools by the `ToolFactory` based on its role name in the system configuration. For this workflow, it uses a configured `VectorSearchTool` to check if a claim's text appears in diverse contexts across the vault. Highly diverse results suggest the claim is ambiguous and needs more context.
*   **Configurable:** The agent's LLM and retry behavior are configured in `settings/aclarai.config.yaml`.
*   **Prompt-Managed:** The agent's reasoning is guided by the `decontextualization_evaluation.yaml` prompt template, which can be customized by users.
*   **Resilient:** It includes retry logic for transient LLM errors and returns `None` for the score if evaluation fails permanently.

### 2. The `ClaimEvaluationGraphService`

This service is the dedicated interface for writing evaluation scores to the knowledge graph.

#### Initialization
To use the service, you need an active Neo4j driver instance:
```python
from neo4j import GraphDatabase
from aclarai_core.graph import ClaimEvaluationGraphService
from aclarai_shared.config import load_config

# Load config to get Neo4j connection details
config = load_config()
driver = GraphDatabase.driver(
    config.neo4j.get_neo4j_bolt_url(),
    auth=(config.neo4j.user, config.neo4j.password)
)

graph_service = ClaimEvaluationGraphService(neo4j_driver=driver, config=config)
```

#### Preparing Test Data in Neo4j
Before updating scores, you need corresponding `:Claim` and `:Block` nodes linked by an `:ORIGINATES_FROM` relationship. You can create test data with this Cypher query:
```cypher
MERGE (c:Claim {id: "claim_test_123"})
MERGE (b:Block {id: "block_test_456"})
MERGE (c)-[:ORIGINATES_FROM]->(b)
```

#### Updating a Single Score
```python
# Update the score for a single claim
success = graph_service.update_decontextualization_score(
    claim_id="claim_test_123",
    block_id="block_test_456",
    score=0.95
)
print(f"Single update successful: {success}")

# Update with a null score if evaluation failed
success_null = graph_service.update_decontextualization_score(
    claim_id="claim_test_123",
    block_id="block_test_456",
    score=None
)
print(f"Single update with null successful: {success_null}")
```

#### Batch Updating Scores
For efficiency, you can update multiple scores in a single transaction.
```python
# Prepare more test data
# MERGE (c2:Claim {id: "claim_batch_1"}) MERGE (b2:Block {id: "block_batch_1"}) MERGE (c2)-[:ORIGINATES_FROM]->(b2)
# MERGE (c3:Claim {id: "claim_batch_2"}) MERGE (b3:Block {id: "block_batch_2"}) MERGE (c3)-[:ORIGINATES_FROM]->(b3)

scores_to_batch = [
    {"claim_id": "claim_batch_1", "block_id": "block_batch_1", "score": 0.88},
    {"claim_id": "claim_batch_2", "block_id": "block_batch_2", "score": None},
    {"claim_id": "claim_non_existent", "block_id": "block_non_existent", "score": 0.5}, # This will be logged as a warning
]

success_batch = graph_service.batch_update_decontextualization_scores(scores_to_batch)
print(f"Batch update attempt completed: {success_batch}")
```

### Key Behaviors

*   **Targeted Updates:** It specifically targets the `:ORIGINATES_FROM` relationship between a `:Claim` and its source `:Block`.
*   **Property Management:** It sets the `decontextualization_score` property on this relationship. The value can be a `float` or `null`.
*   **Batch Operations:** It includes a `batch_update_decontextualization_scores` method for efficiently updating multiple claims in a single transaction.

### 3. The `MarkdownUpdaterService`

This service handles all modifications to Markdown files in the vault, ensuring data integrity.

### Key Behaviors

*   **Atomic Writes:** All file operations use a safe `write-temp -> fsync -> rename` pattern to prevent file corruption.
*   **Block-Level Targeting:** It finds the correct block in a file using its `aclarai:id`.
*   **Versioning:** It automatically increments the `ver=N` number of any block it modifies, signaling to the sync system that the block has changed.
*   **Metadata Injection:** It adds the score as a new `<!-- aclarai:decontextualization_score=... -->` comment, or updates an existing one.

## 📊 Score Interpretation

The `decontextualization_score` is a float between `0.0` and `1.0`.

Claims with a `null` score or a score below the system's quality threshold are typically excluded from being promoted to summaries or linked to concepts.

## 📚 Related Documentation

-   **Component Overview:** [`docs/components/decontextualization_evaluation_agent.md`](../components/decontextualization_evaluation_agent.md)
-   **Architectural Principles:** [`docs/arch/on-evaluation_agents.md`](../arch/on-evaluation_agents.md)