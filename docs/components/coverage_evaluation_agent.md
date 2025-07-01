# Coverage Evaluation Agent

## 🎯 Purpose

The Coverage Evaluation Agent is a key component of aclarai's claim quality assessment pipeline. Its primary function is to determine how completely a factual claim captures all the verifiable information from its original source text.

This agent ensures that claims are not just logically supported, but are also comprehensive, preventing the loss of critical details when summarizing information.

## ⚙️ How it Works

### Core Logic

1.  **Input**: The agent receives a `claim_text` and its original `source_text`.
2.  **LLM-Powered Analysis**: It uses a configured Large Language Model (LLM) to compare the claim against the source, guided by the `coverage_evaluation.yaml` prompt.
3.  **Tool-Assisted Evaluation**: The agent requests its toolset from the `ToolFactory` using the role name `coverage_agent`. It can leverage tools like `VectorSearchTool` or `WebSearchTool` (if configured) to assess the significance of entities mentioned in the source to determine if their omission in the claim is critical.
4.  **Dual Output**: The agent produces two key outputs:
    *   A `coverage_score` (float from 0.0 to 1.0) quantifying the claim's completeness.
    *   A list of `omitted_verifiable_elements` (e.g., specific names, dates, or numbers) that were present in the source but missing from the claim.

### Outputs and Storage

The agent's findings are persisted in two places:

*   **Neo4j Graph**:
    *   The `coverage_score` is stored as a property on the `[:ORIGINATES_FROM]` relationship between the `:Claim` and its source `:Block`.
    *   Each `omitted_element` is created as a new `(:Element)` node.
    *   An `[:OMITS]` relationship is created from the `:Claim` to each of its corresponding `(:Element)` nodes, explicitly modeling the missing information.
*   **Markdown File**:
    *   The `coverage_score` is written as a metadata comment (`<!-- aclarai:coverage_score=... -->`) into the source Markdown file.

### Agent Workflow

```mermaid
graph TD
    A[Input: Claim + Source] --> B(CoverageAgent);
    B --> C[Output: coverage_score];
    B --> D[Output: List of Omitted Elements];
    C --> E[Store on [:ORIGINATES_FROM] edge];
    C --> F[Write to Markdown comment];
    D --> G[Create (:Element) nodes & [:OMITS] edges];
```

## 🧩 Role in the Pipeline

The `coverage_score` is a critical factor in the overall quality assessment of a claim. It is combined with the `entailed_score` and `decontextualization_score` (typically via a geometric mean) to produce a final quality score. This score is then used to filter claims, ensuring that only those that are complete, accurate, and self-contained are promoted to summaries or linked to concepts in the knowledge graph.

The `(:Element)` nodes also provide valuable data for future claim enrichment or identifying information gaps in the knowledge base.

## 📚 Related Documentation

*   **Developer Guide:** [`docs/guides/coverage_evaluation_guide.md`](../guides/coverage_evaluation_guide.md)
*   **Architectural Principles:** [`docs/arch/on-evaluation_agents.md`](../arch/on-evaluation_agents.md)
```

---
### New File: `docs/guides/coverage_evaluation_guide.md`

```markdown
# Guide: Coverage Evaluation Workflow

This guide provides a technical overview of the `CoverageAgent` and explains how to use its components to evaluate claim completeness.

## 🎯 Purpose

The goal of this workflow is to generate a `coverage_score` for each claim and to identify any verifiable information from the source that was omitted. This ensures claims are not only accurate but also comprehensive.

## ⚙️ Core Workflow

The end-to-end process for evaluating a single claim involves the `CoverageAgent`, the `ClaimEvaluationGraphService` for Neo4j updates, and the `MarkdownUpdaterService` for file modifications.

### Agent Usage Example

```python
# Assume 'agent' is an initialized CoverageAgent instance.
# Assume 'claim_data' contains claim_id, claim_text, source_id, and source_text.

# 1. Agent evaluates the claim to get a score and omitted elements.
score, elements, status = agent.evaluate_coverage(
    claim_id=claim_data['claim_id'],
    claim_text=claim_data['claim_text'],
    source_id=claim_data['source_id'],
    source_text=claim_data['source_text']
)

if score is None:
    logger.error(f"Evaluation failed for claim {claim_data['claim_id']}: {status}")
    return

# 2. Persist the score to the Neo4j graph.
graph_service.update_coverage_score(
    claim_id=claim_data['claim_id'],
    block_id=claim_data['source_id'],
    score=score
)

# 3. Persist the omitted elements to the graph.
if elements:
    graph_service.create_element_nodes_and_omits_relationships(
        claim_id=claim_data['claim_id'],
        omitted_elements=elements
    )

# 4. Persist the score to the Markdown file.
markdown_service.add_or_update_score(
    filepath_str=claim_data['file_path'],
    block_id=claim_data['source_id'],
    score_name="coverage_score",
    score=score
)
```

## Component Deep Dive

### 1. The `CoverageAgent`

The agent analyzes a claim's completeness relative to its source.

#### Key Behaviors

*   **Tool-Driven:** Uses the `ToolFactory` to acquire tools (like `WebSearchTool`) based on its configured role (`coverage_agent`) to assess the significance of omitted information.
*   **Structured Output:** Returns a tuple containing the `coverage_score`, a list of `omitted_elements`, and a status message.
*   **Resilient:** Includes retry logic for LLM calls and gracefully handles failures by returning a `null` score.

### 2. Handling Omitted Elements

A unique feature of this agent is its ability to identify and structure information that was left out of a claim.

*   **Structure:** The agent returns a list of dictionaries: `[{"text": "omitted fact", "significance": "why it matters"}]`.
*   **Graph Persistence:** The `ClaimEvaluationGraphService` provides a dedicated method to persist this information:
    *   `create_element_nodes_and_omits_relationships()`: This method takes the claim ID and the list of omitted elements. It creates a new `(:Element)` node for each omission and links it to the source `(:Claim)` node with an `[:OMITS]` relationship.
*   **Example Cypher (Conceptual):**
    ```cypher
    // For a claim that omitted "European Commission"
    MATCH (c:Claim {id: "claim_123"})
    CREATE (e:Element {text: "European Commission", significance: "Key organization responsible"})
    CREATE (c)-[:OMITS]->(e)
    ```

### 3. Configuration

*   **LLM Model:** Configure the agent's model in `settings/aclarai.config.yaml` under `model.claimify.coverage`.
*   **Tools:** Configure the agent's tools (e.g., enabling web search) in the `tools.agent_tool_mappings.coverage_agent` section.
*   **Prompt:** The agent's behavior is guided by `shared/aclarai_shared/prompts/coverage_evaluation.yaml`. You can customize this file to refine the evaluation criteria and JSON output format.

### 4. Score Interpretation

* **1.0**: Claim captures all important verifiable elements from Source
* **0.8-0.9**: Claim captures most elements, minor omissions that don't significantly impact fact-checking
* **0.6-0.7**: Claim captures main elements but omits some important details
* **0.4-0.5**: Claim captures some elements but has significant omissions  
* **0.0-0.3**: Claim omits many important verifiable elements from Source
*   **null**: The evaluation failed. Claims with a `null` score are excluded from promotion and linking.

## 📚 Related Documentation

*   **Component Overview:** [`docs/components/coverage_evaluation_agent.md`](../components/coverage_evaluation_agent.md)
*   **Architectural Principles:** [`docs/arch/on-evaluation_agents.md`](../arch/on-evaluation_agents.md)
```

---
### Modified File: `docs/components/markdown_updater_system.md`

```diff
--- a/docs/components/markdown_updater_system.md
+++ b/docs/components/markdown_updater_system.md
@@ -21,19 +21,27 @@
 
 ## 🐍 Usage Example
 
-Any service needing to update a Markdown file can use the `MarkdownUpdaterService`. Here is how the `DecontextualizationAgent` would use it to persist a score:
+Any service, such as an evaluation agent, can use the `MarkdownUpdaterService` to persist scores. The service provides a generic `add_or_update_score` method as well as specific helpers for each score type.
 
-*(Note: The following example illustrates usage of a dedicated `MarkdownUpdaterService`. While this service provides the ideal abstraction, similar update logic might also be directly implemented within calling services like `DirtyBlockConsumer` while adhering to the same principles of atomic writes, version incrementing, and metadata comment standards described here. The `EntailmentAgent`'s score, for example, is also persisted using this pattern.)*
+Here are examples of how different evaluation agents would use the service:
 
 ```python
 from aclarai_core.markdown import MarkdownUpdaterService
 
-# Assume 'score' and other variables are defined
-# score = 0.88
-# block_id_to_update = "blk_abc123" # The aclarai:id of the block whose metadata is updated
-# file_path_of_block = "/path/to/vault/tier1/conversation.md"
-
 # Instantiate the service
 updater = MarkdownUpdaterService()
 
-# Call the specific update method
-success = updater.add_or_update_decontextualization_score(
-    filepath_str=file_path_of_block, # Corrected variable name
-    block_id=block_id_to_update,
-    score=score
+# Example 1: The DecontextualizationAgent persists its score.
+decon_success = updater.add_or_update_decontextualization_score(
+    filepath_str="/path/to/vault/tier1/conversation.md",
+    block_id="blk_abc123",
+    score=0.88
+)
+
+# Example 2: The CoverageAgent persists its score using the generic method.
+coverage_success = updater.add_or_update_score(
+    filepath_str="/path/to/vault/tier1/conversation.md",
+    block_id="blk_abc123",
+    score_name="coverage_score",
+    score=0.77
 )
 
 # Similarly, for an entailment score:
@@ -43,10 +51,10 @@
 #     score=entailment_score
 # )
 
-if success:
-    print(f"Successfully updated block {block_id_to_update} in {file_path_of_block}.")
+if decon_success and coverage_success:
+    print("Successfully updated scores for block blk_abc123.")
 else:
-    print(f"Failed to update block {block_id_to_update}.")
+    print("One or more score updates failed.")
 
 ```