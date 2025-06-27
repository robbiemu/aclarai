# Guide: Decontextualization Evaluation Workflow

This guide explains how to use the components of the decontextualization evaluation workflow. This system is responsible for assessing whether a factual claim can be understood on its own, without its original context.

## 🎯 Purpose

The goal of this workflow is to generate a `decontextualization_score` for each claim. This score is crucial for ensuring that only high-quality, portable claims are promoted into the knowledge base.

The workflow involves three key components:
1.  **`DecontextualizationAgent`**: The LLM-powered agent that performs the evaluation.
2.  **`ClaimEvaluationGraphService`**: The service that persists the score to the Neo4j graph.
3.  **`MarkdownUpdaterService`**: The service that writes the score back to the vault's Markdown files.

## ⚙️ Core Workflow

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

## 1. The `DecontextualizationAgent`

This agent uses a configured LLM and a set of tools to assess a claim's semantic autonomy.

### Key Behaviors

-   **Tool-Driven:** The agent uses a `VectorSearchTool` (provided by the `ToolFactory`) to check if a claim's text appears in diverse contexts across the vault. Highly diverse results suggest the claim is ambiguous and needs more context.
-   **Configurable:** The agent's LLM and retry behavior are configured in `settings/aclarai.config.yaml`.
-   **Prompt-Managed:** The agent's reasoning is guided by the `decontextualization_evaluation.yaml` prompt template, which can be customized by users.
-   **Resilient:** It includes retry logic for transient LLM errors and returns `None` for the score if evaluation fails permanently.

## 2. The `ClaimEvaluationGraphService`

This service is the dedicated interface for writing evaluation scores to the knowledge graph.

### Key Behaviors

-   **Targeted Updates:** It specifically targets the `:ORIGINATES_FROM` relationship between a `:Claim` and its source `:Block`.
-   **Property Management:** It sets the `decontextualization_score` property on this relationship. The value can be a `float` or `null`.
-   **Batch Operations:** It includes a `batch_update_decontextualization_scores` method for efficiently updating multiple claims in a single transaction.

## 3. The `MarkdownUpdaterService`

This service handles all modifications to Markdown files in the vault, ensuring data integrity.

### Key Behaviors

-   **Atomic Writes:** All file operations use a safe `write-temp -> fsync -> rename` pattern to prevent file corruption.
-   **Block-Level Targeting:** It finds the correct block in a file using its `aclarai:id`.
-   **Versioning:** It automatically increments the `ver=N` number of any block it modifies, signaling to the sync system that the block has changed.
-   **Metadata Injection:** It adds the score as a new `<!-- aclarai:decontextualization_score=... -->` comment, or updates an existing one.

## 📊 Score Interpretation

The `decontextualization_score` is a float between `0.0` and `1.0`.

Claims with a `null` score or a score below the system's quality threshold are typically excluded from being promoted to summaries or linked to concepts.

## 📚 Related Documentation

-   **Component Overview:** [`docs/components/decontextualization_evaluation_agent.md`](../components/decontextualization_evaluation_agent.md)
-   **Architectural Principles:** [`docs/arch/on-evaluation_agents.md`](../arch/on-evaluation_agents.md)
