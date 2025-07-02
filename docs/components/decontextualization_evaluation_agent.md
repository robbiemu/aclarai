# Decontextualization Evaluation Agent

The Decontextualization Evaluation Agent is a specialized component within the aclarai system responsible for assessing the semantic autonomy of a given factual claim. Its primary goal is to determine if a claim can be fully understood and verified on its own, without needing access to the original source text or conversational context from which it was extracted.

## 🎯 Purpose

This agent evaluates a `claim` against its `source` to produce a `decontextualization_score`, a float between 0.0 and 1.0. This score is crucial for ensuring that claims promoted to the knowledge base are portable, reusable, and unambiguous.

-   **1.0**: claim is perfectly self-contained and understandable in isolation.
-   **0.0**: claim is completely dependent on its source and cannot be understood alone
-   **null**: The evaluation failed after retries.

## ⚙️ How it Works

### Agent Implementation

-   **Class:** `services.aclarai-core.aclarai_core.agents.decontextualization_agent.DecontextualizationAgent`
-   **Framework:** Uses LlamaIndex's `ReActAgent` to allow for multi-step reasoning and the use of tools.
-   **Configuration:**
    -   The agent's role for tool mapping is `"decontextualization_agent"` (configured in `tools.agent_tool_mappings`).
    -   The LLM model used is configured under `model.claimify.decontextualization` in `aclarai.config.yaml`.

### Prompt and Tool Usage

The agent's reasoning is guided by the `decontextualization_evaluation.yaml` prompt template. This prompt instructs the LLM to analyze the claim for ambiguities (e.g., unresolved pronouns, vague references) and to optionally use the `vector_search_utterances` tool.

By searching for the claim's text in the vector store, the agent can determine if similar phrases appear in many different contexts. If so, it's a strong signal that the claim is too generic and lacks the specific context needed to be truly standalone.

## 🔄 Process Flow

The Decontextualization Agent is invoked as part of the reactive sync loop managed by the `DirtyBlockConsumer`.

1.  **Trigger:** The `DirtyBlockConsumer` receives a message about a new or modified block.
2.  **Claim Retrieval:** The consumer fetches any claims that originate from this block and require evaluation.
3.  **Agent Invocation:** For each claim, the `DirtyBlockConsumer` calls the `DecontextualizationAgent.evaluate_claim_decontextualization()` method with the `claim_id`, `claim_text`, `source_id`, and `source_text`.
4.  **Evaluation:** The agent processes the input using its configured LLM, prompt, and tools to generate the `decontextualization_score`.
5.  **Score Return:** The agent returns the score (or `None` on failure) to the `DirtyBlockConsumer`.
6.  **Persistence:** The `DirtyBlockConsumer` then uses dedicated services to persist the score.

## 💾 Output and Storage

The agent's score is persisted to both the knowledge graph and the source Markdown file.

### 1. Neo4j Graph

-   The `decontextualization_score` (or `null`) is stored as a property on the `[:ORIGINATES_FROM]` relationship between the `:Claim` and its source `:Block`.
-   This update is performed by the **`ClaimEvaluationGraphService`**.

### 2. Tier 1 Markdown File

-   If the evaluation is successful (score is not `None`), the **`MarkdownUpdaterService`** updates the source file:
    -   An HTML comment `<!-- aclarai:decontextualization_score=X.XX -->` is added or updated.
    -   The version number in the source block's main comment (e.g., `ver=N`) is **incremented by 1**.
-   Claims with `null` scores do not result in a new score comment being added to the Markdown file.

## ❌ Failure Handling

-   The agent incorporates a retry mechanism with exponential backoff for LLM API calls.
-   If evaluation fails after all retries, the agent returns `None` for the score.
-   Claims with `null` scores are excluded from promotion and strong concept linking by downstream processes.