# Entailment Evaluation Agent

The Entailment Evaluation Agent is responsible for assessing whether a given source text logically entails a claim derived from it. This is a crucial step in the claim processing pipeline to ensure the quality and factual grounding of claims.

## Purpose

This agent determines the degree to which a `source` text (e.g., a block of conversation, a document sentence) supports a given `claim` text. The output is an `entailed_score`, a float between 0.0 and 1.0.

- **1.0**: The source perfectly entails the claim.
- **0.5**: The source partially supports or is related to the claim but does not fully entail it.
- **0.0**: The source does not entail the claim at all, or contradicts it.
- **null**: The evaluation failed after retries (e.g., LLM error, timeout).

## Agent Implementation

- **Class:** `services.aclarai-core.aclarai_core.agents.entailment_agent.EntailmentAgent`
- **Framework:** Uses LlamaIndex's `CodeActAgent` to allow for multi-step reasoning and the use of tools for contextual understanding if needed.
- **Configuration:**
    - The agent's role for tool mapping is `"entailment_agent"` (configured in `tools.agent_tool_mappings`).
    - The LLM model used by the agent can be configured under `model.claimify.entailment` in `aclarai.config.default.yaml` (or user overrides).

## Prompt Structure

The agent uses a dedicated prompt template: `shared/aclarai_shared/prompts/entailment_evaluation.yaml`.

- **Inputs to Prompt:**
    - `source_text`: The premise or source content.
    - `claim_text`: The hypothesis or claim to be evaluated.
- **Instructions to LLM:** The LLM is instructed to analyze the premise and hypothesis, use available tools (`neo4j_query_tool`, `vector_search_utterances`) if necessary for more context, and output a single float score representing the entailment level.

Refer to the YAML file directly for the detailed system prompt and user template.

## Process Flow

1.  **Trigger:** The entailment evaluation process is typically triggered within the `DirtyBlockConsumer` after a source block in a Markdown file is identified as new or modified and has been synchronized with the Neo4j graph.
2.  **Claim Retrieval:** The system fetches claims that originate from this source block and require entailment evaluation (e.g., `entailed_score` is `null` on the `[:ORIGINATES_FROM]` relationship, or the claim is marked for reprocessing).
3.  **Agent Invocation:** For each such claim, the `EntailmentAgent.evaluate_entailment()` method is called with:
    - `claim_id`
    - `claim_text`
    - `source_id` (the `aclarai:id` of the source block)
    - `source_text` (the semantic content of the source block)
4.  **Evaluation:** The agent processes the input using its configured LLM and prompt. It may use tools like `Neo4jQueryTool` or `VectorSearchTool` to gather more context (e.g., surrounding conversation, related utterances).
5.  **Score Return:** The agent returns the `entailed_score` (or `None` on failure).

## Output and Storage

### 1. Neo4j Graph
- The `entailed_score` (or `null` if evaluation failed) is stored as a property on the `[:ORIGINATES_FROM]` relationship connecting the `(:Claim {id: claim_id})` node to its source `(:Block {id: source_id})` node.
- This update is performed by the `_update_entailment_score_in_neo4j` method in `DirtyBlockConsumer`.

### 2. Tier 1 Markdown File
- If the evaluation is successful (score is not `None`), the Markdown file containing the source block is updated:
    - An HTML comment `<!-- aclarai:entailed_score=X.XX -->` (e.g., `<!-- aclarai:entailed_score=0.91 -->`) is added or updated. This comment is typically placed on the line immediately following the main `<!-- aclarai:id=source_id ... -->` comment of the source block.
    - The version number in the source block's main comment (e.g., `ver=N`) is incremented by 1 (e.g., to `ver=N+1`).
- These updates are performed by the `_update_markdown_with_entailment_score` method in `DirtyBlockConsumer`, using atomic file writes.
- Claims with `null` entailment scores do not result in Markdown updates for this score.

## Failure Handling
- The `EntailmentAgent` includes a retry mechanism for LLM calls.
- If evaluation fails after retries, the `entailed_score` will be `None`.
- Downstream processes should handle `null` scores appropriately (e.g., by excluding such claims from further processing or promotion, as per `docs/arch/on-evaluation_agents.md`).
