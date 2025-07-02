# Coverage Evaluation Agent

The Coverage Evaluation Agent is a key component of aclarai's claim quality assessment pipeline. Its primary function is to determine how completely a factual claim captures all the verifiable information from its original source text.

This agent ensures that claims are not just logically supported, but are also comprehensive, preventing the loss of critical details when summarizing information.

## 🎯 Purpose

This agent analyzes a `source` text and a `claim` derived from it to produce two key outputs:

1.  A `coverage_score` (float from 0.0 to 1.0) quantifying the claim's completeness.
2.  A list of `omitted_elements` (verifiable facts from the source that are missing from the claim).

These outputs are crucial for filtering claims and enriching the knowledge graph with information about what was left out.

## ⚙️ How it Works

### Agent Implementation

-   **Class:** `services.aclarai-core.aclarai_core.agents.coverage_agent.CoverageAgent`
-   **Framework:** Uses LlamaIndex's `CodeActAgent` to allow for multi-step reasoning and the use of tools.
-   **Configuration:**
    -   The agent's role for tool mapping is `"coverage_agent"` (configured in `tools.agent_tool_mappings`).
    -   The LLM model used is configured under `model.claimify.coverage` in `aclarai.config.yaml`.
    -   Score Interpretation
        * **1.0**: Claim captures all important verifiable elements from Source
        * **0.8-0.9**: Claim captures most elements, minor omissions that don't significantly impact fact-checking
        * **0.6-0.7**: Claim captures main elements but omits some important details
        * **0.4-0.5**: Claim captures some elements but has significant omissions  
        * **0.0-0.3**: Claim omits many important verifiable elements from Source
        *   **null**: The evaluation failed. Claims with a `null` score are excluded from promotion and linking.
   
### Prompt Structure

The agent uses a dedicated prompt template: `shared/aclarai_shared/prompts/coverage_evaluation.yaml`.

-   **Inputs to Prompt:**
    -   `source_text`: The original source content.
    -   `claim_text`: The claim to be evaluated.
-   **Instructions to LLM:** The LLM is instructed to compare the claim against the source, identify any important verifiable facts that were omitted, and produce a JSON object containing the `coverage_score` and a list of `omitted_elements`.

## 🔄 Process Flow

The Coverage Agent is invoked as part of the reactive sync loop managed by the `DirtyBlockConsumer`.

1.  **Trigger:** The `DirtyBlockConsumer` receives a message about a new or modified block.
2.  **Claim Retrieval:** The consumer fetches any claims that originate from this block and require evaluation.
3.  **Agent Invocation:** For each claim, the `DirtyBlockConsumer` calls the `CoverageAgent.evaluate_coverage()` method with the `claim_id`, `claim_text`, `source_id`, and `source_text`.
4.  **Evaluation:** The agent processes the input using its configured LLM and prompt. It may use tools from the `ToolFactory` (like `WebSearchTool`) to assess the significance of entities to determine if their omission is critical.
5.  **Output Return:** The agent returns the `coverage_score` and `omitted_elements` (or `None` on failure) to the `DirtyBlockConsumer`.
6.  **Persistence:** The `DirtyBlockConsumer` then uses dedicated services to persist the results.

## 💾 Output and Storage

The agent's outputs are persisted to both the knowledge graph and the source Markdown file.

### 1. Neo4j Graph

-   The `coverage_score` (or `null`) is stored as a property on the `[:ORIGINATES_FROM]` relationship between the `:Claim` and its source `:Block`.
-   Each `omitted_element` is created as a new `(:Element)` node in the graph.
-   An `[:OMITS]` relationship is created from the `:Claim` to each of its corresponding `(:Element)` nodes.
-   These updates are performed by the **`ClaimEvaluationGraphService`**.

### 2. Tier 1 Markdown File

-   If the evaluation is successful (score is not `None`), the **`MarkdownUpdaterService`** updates the source file:
    -   An HTML comment `<!-- aclarai:coverage_score=X.XX -->` is added or updated.
    -   The version number in the source block's main comment (e.g., `ver=N`) is **incremented by 1**.
-   Claims with `null` coverage scores do not result in a new score comment being added to the Markdown file.

## ❌ Failure Handling

-   The `CoverageAgent` includes a retry mechanism for LLM calls.
-   If evaluation fails after all retries, the `coverage_score` and `omitted_elements` will be `None`.
-   Downstream processes, orchestrated by the `DirtyBlockConsumer`, handle `null` scores by excluding the claim from promotion or strong concept linking.