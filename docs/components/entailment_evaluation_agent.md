# Entailment Evaluation Agent

The Entailment Evaluation Agent is responsible for assessing whether a given source text logically entails a claim derived from it. This is a crucial step in the claim processing pipeline to ensure the quality and factual grounding of claims.

## 🎯 Purpose

This agent determines the degree to which a `source` text (e.g., a block of conversation, a document sentence) supports a given `claim` text. The output is an `entailed_score`, a float between 0.0 and 1.0.

-   **1.0**: The source perfectly entails the claim.
-   **0.5**: The source partially supports or is related to the claim but does not fully entail it.
-   **0.0**: The source does not entail the claim at all, or contradicts it.
-   **null**: The evaluation failed after retries (e.g., LLM error, timeout).

## ⚙️ How it Works

### Agent Implementation

-   **Class:** `services.aclarai-core.aclarai_core.agents.entailment_agent.EntailmentAgent`
-   **Framework:** Uses LlamaIndex's `CodeActAgent` to allow for multi-step reasoning and the use of tools for contextual understanding if needed.
-   **Configuration:**
    -   The agent's role for tool mapping is `"entailment_agent"` (configured in `tools.agent_tool_mappings`).
    -   The LLM model used by the agent can be configured under `model.claimify.entailment` in `aclarai.config.default.yaml` (or user overrides).

### Prompt Structure

The agent uses a dedicated prompt template: `shared/aclarai_shared/prompts/entailment_evaluation.yaml`.

-   **Inputs to Prompt:**
    -   `source_text`: The premise or source content.
    -   `claim_text`: The hypothesis or claim to be evaluated.
-   **Instructions to LLM:** The LLM is instructed to analyze the premise and hypothesis, use available tools (`neo4j_query_tool`, `vector_search_utterances`) if necessary for more context, and output a single float score representing the entailment level.

Refer to the YAML file directly for the detailed system prompt and user template.

## 🔄 Process Flow

The Entailment Agent is invoked as part of the reactive sync loop managed by the `DirtyBlockConsumer`.

1.  **Trigger:** The `DirtyBlockConsumer` receives a message about a new or modified block.
2.  **Claim Retrieval:** The consumer fetches any claims that originate from this block and require evaluation.
3.  **Agent Invocation:** For each claim, the `DirtyBlockConsumer` calls the `EntailmentAgent.evaluate_entailment()` method with the `claim_id`, `claim_text`, `source_id`, and `source_text`.
4.  **Evaluation:** The agent processes the input using its configured LLM and prompt. It may use tools from the `ToolFactory` to gather more context to make an accurate assessment.
5.  **Score Return:** The agent returns the `entailed_score` (or `None` on failure) to the `DirtyBlockConsumer`.
6.  **Persistence:** The `DirtyBlockConsumer` then uses dedicated services to persist the score.

## 💾 Output and Storage

The agent's score is persisted to both the knowledge graph and the source Markdown file.

### 1. Neo4j Graph
-   The `entailed_score` (or `null` if evaluation failed) is stored as a property on the `[:ORIGINATES_FROM]` relationship connecting the `(:Claim {id: claim_id})` node to its source `(:Block {id: source_id})` node.
-   This update is performed by the **`ClaimEvaluationGraphService`**.

### 2. Tier 1 Markdown File
-   If the evaluation is successful (score is not `None`), the **`MarkdownUpdaterService`** updates the source file:
    -   An HTML comment `<!-- aclarai:entailed_score=X.XX -->` is added or updated.
    -   The version number in the source block's main comment (e.g., `ver=N`) is **incremented by 1**.
-   Claims with `null` entailment scores do not result in a new score comment being added to the Markdown file.

## ❌ Failure Handling

-   The `EntailmentAgent` includes a retry mechanism for LLM calls.
-   If evaluation fails after all retries, the `entailed_score` will be `None`.
-   Downstream processes, orchestrated by the `DirtyBlockConsumer`, handle `null` scores appropriately (e.g., by excluding such claims from further processing or promotion, as per `docs/arch/on-evaluation_agents.md`).