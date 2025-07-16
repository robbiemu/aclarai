# 🧠 LLM Interaction Strategy

This document outlines aclarai's conventions and best practices for interacting with Large Language Models (LLMs) across its various agents. As LLMs are a core intelligence component, this strategy aims to ensure consistent integration, optimize performance and cost, and maintain the quality and reliability of agent outputs.

---

## 🎯 Purpose

To provide clear guidelines for prompt management, response handling, and resource optimization when agents interact with LLMs, leading to predictable and robust AI-driven operations within aclarai.

---

## I. LLM Configuration and Selection

aclarai's agents are designed to dynamically select which LLM to use based on centralized configuration, decoupling model choice from agent code.

*   **How it works:** Each agent specifies a model "role" (e.g., `claimify.selection`, `concept_linker`, `trending_concepts_agent`). The `settings/aclarai.config.yaml` maps these roles to specific LLM providers and models (e.g., `gpt-4`, `claude-3-opus`, `openrouter:gemma-2b`, `ollama:mistral`). Agents then instantiate their assigned LLM using LlamaIndex's `llm` abstractions.
*   **Benefit:** This approach provides flexibility to tune processing quality and cost globally without modifying individual agent code.

---

## II. Externalized Prompt Templates

All LLM prompts, including system instructions, few-shot examples, and output formatting guidelines, are managed as external configuration templates rather than being hardcoded.

*   **How it works:** Agents load prompt templates from external files (e.g., YAML or dedicated template files) at runtime. Default templates are stored in the shared library. Users can override specific keys by creating a corresponding file in `./settings/prompts`, which will be deep-merged over the defaults at runtime. This allows users to directly inspect and modify prompts without duplicating entire template files. The templates define the LLM's role, instructions, and placeholder variables for injecting dynamic context.
*   **Benefit:** This separates prompt engineering from core development, allowing iterative refinement of agent behavior without code changes and simplifying version control of prompts.

---

## III. Agentic Workflows for Structured Data Generation

aclarai's agents are designed as intelligent orchestrators that can reason, gather information using tools, and then provide structured data as their final output. This section details how agents achieve this, focusing on the `EntailmentAgent` as a concrete example.

**EntailmentAgent Overview:** The EntailmentAgent analyzes the logical relationship between claims (`(:Claim)` nodes) and their sources (`(:Block)` nodes), producing an `entailed_score` (float between 0.0 and 1.0) that indicates the degree of logical support the source provides for the claim. This score is stored both as a property on the `[:ORIGINATES_FROM]` relationship in Neo4j and as metadata in the Markdown Tier 1 files. The agent implements robust retry mechanisms and error handling, setting scores to `null` when evaluation fails after all retries.

### A. Developing Tools for Agents

To reason effectively, agents need access to aclarai's knowledge base. We develop specific LlamaIndex tools that expose this data.

**Tool Location in Monorepo:** These common tools (e.g., `Neo4jQueryTool`, `VectorSearchTool`, `WebSearchTool`) are shared across multiple agents. To maximize reusability and maintainability, their implementations reside in the `shared/aclarai_tools/` directory within the monorepo. This ensures a single source of truth for core functionalities.

1.  **`Neo4jQueryTool`:**
    *   **Purpose:** Allows an agent to execute Cypher queries on the Neo4j knowledge graph.
    *   **Use Case for `EntailmentAgent`:** The `EntailmentAgent` might use this tool to:
        *   Retrieve the full text of the conversation (`:Conversation` node and its linked `:Block` sequence) to understand the `source_text` in its original dialogue flow.
        *   Fetch definitions of `(:Concept)` nodes linked to the `claim_text` or `source_text` to clarify domain-specific terms.
        *   Look for other `(:Claim)` nodes semantically similar or potentially contradictory to the current claim.

2.  **`VectorSearchTool`:**
    *   **Purpose:** Enables an agent to perform semantic similarity searches on aclarai's vector stores (`utterances`, `concept_candidates`, `concepts`).
    *   **Use Case for `EntailmentAgent`:** The `EntailmentAgent` might use this tool to:
        *   Find other ways a specific phrase from the `source_text` is used across the vault, helping to disambiguate or confirm its meaning.
        *   Discover implicit connections or common knowledge related to the claim that might support or refute its entailment.

### B. Agent's Reasoning and Tool Use

When an agent (e.g., `EntailmentAgent` implemented as a `CodeActAgent`) is invoked:

1.  **Initial Prompting:** The agent's externalized prompt guides its internal LLM, instructing it on its role, the task (entailment evaluation), and providing access to the available tools.

    *   **Example Prompt Snippet (for `CodeActAgent`):**
        ```
        # ... (System prompt, task description) ...

        # Available Tools:
        # - neo4j_query_tool: Executes Cypher queries on the knowledge graph.
        # - vector_search_tool: Performs semantic searches on vector stores.

        # Your Goal: Determine the entailment score (0.0-1.0) and reasoning.
        # Use the provided tools if you need more context.
        # When you have your final answer, output a JSON object using the format:
        # print(json.dumps({"entailed_score": <float>, "reasoning": "<string>"}))
        ```

2.  **LLM's Internal Deliberation:** The `CodeActAgent`'s internal LLM receives the initial inputs (`claim_text`, `source_text`). It then engages in an iterative reasoning process. If it determines it needs more information to confidently assess entailment, it will generate Python code to call one of the provided tools (e.g., `print(neo4j_query_tool.execute_cypher_query(...))`).

3.  **Tool Execution:** The `CodeActAgent`'s executor runs this generated code, captures the tool's output, and feeds it back to the LLM as part of the ongoing conversation history. The LLM then continues its reasoning, potentially calling more tools.

### C. Producing Final Structured Data (JSON)

This is the critical step where the agent transitions from *acting* (using tools) to *reporting its final structured result*.

1.  **Agent Signals Completion:** After the `CodeActAgent`'s internal LLM has completed its reasoning and gathered all necessary information via tools, it will generate a final Python code block that explicitly outputs the desired JSON data.

    *   **Mechanism:** The prompt instructs the LLM to use `print(json.dumps(...))` for its final answer.
    *   **Example LLM-generated final output code:**
        ```python
        # Agent has reasoned and determined the score
        print(json.dumps({
            "entailed_score": 0.92,
            "reasoning": "The premise directly states the hypothesis, confirming logical entailment based on the full conversation context."
        }))
        ```
    *   The `CodeActAgent`'s executor captures this `print` statement's output. This captured string is the agent's "final answer" for this invocation. It is treated as the agent's final data output, not a tool call, because it is captured from the execution environment's stdout, not interpreted from a tool-calling syntax.

2.  **Post-Processing and Validation:** aclarai's agent (the Python class) receives this captured JSON string from the `CodeActAgent`'s execution. It then performs:
    *   **Parsing:** `json.loads()` to convert the string to a Python dictionary.
    *   **Schema Validation:** Verification that the parsed output adheres to the expected schema (e.g., `entailed_score` is a float between 0.0 and 1.0).
    *   **Error Handling:** If parsing or validation fails, the agent logs the invalid response and triggers specific failure handling (e.g., retries, marking as `null`), as detailed in `docs/arch/on-error_handling_and_resilience.md`.
    *   **Transformation:** Validated output is transformed into aclarai's internal data models for persistence (e.g., Neo4j properties, Markdown metadata).

*   **Benefit:** This approach allows agents to leverage LlamaIndex's powerful reasoning and tool-use capabilities to make informed decisions, while reliably producing structured JSON data for programmatic consumption, ensuring data integrity.

---

## IV. Context Management for Agents

Agents manage LLM context by constructing relevant and concise inputs, building upon LlamaIndex's default capabilities.

### A. Precise Windowing for Sequential Data

For agents processing sequential data, such as the Claimify pipeline's steps (Selection, Disambiguation, Decomposition), aclarai employs a configurable "sliding window" to provide local context.

*   **How it works:** When processing a focal sentence (`sentence_i`), the agent dynamically extracts `p` previous sentences and `f` following sentences from the ordered sequence of a Tier 1 document. This precisely constructed `window_context_string` is then injected into the LLM prompt template.
*   **Benefit:** This custom context construction, defined by `window.claimify.p` and `window.claimify.f` in `settings/aclarai.config.yaml`, ensures that LLM agents receive the optimal semantic window for their specific analytical tasks, leading to higher quality claim extraction and evaluation. It's a deliberate choice to enhance LLM performance for conversational analysis beyond generic document chunking.

### B. Semantic Context from Vector Database

For Retrieval-Augmented Generation (RAG) tasks, agents leverage aclarai's vector stores to build highly relevant LLM contexts.

*   **How it works:** Agents perform semantic similarity searches on the `utterances` vector store (`docs/arch/on-vector_stores.md`) to retrieve relevant conversational chunks, and on the `concepts` vector store (`docs/arch/on-vector_stores.md`) to find semantically related concepts. The retrieved content is then integrated into the LLM's prompt.
*   **Benefit:** This ensures the LLM's context is not diluted by irrelevant data and remains optimally sized, improving accuracy and consistency in RAG-driven tasks like summary generation.

### C. Structured Context from Graph Store

Agents utilize aclarai's Neo4j knowledge graph to provide structured, explicit relationships as context to LLMs.

*   **How it works:** Agents execute targeted Cypher queries (`docs/arch/on-neo4j_interaction.md`) to retrieve specific nodes (e.g., related claims, concept definitions, speaker information) and relationships (e.g., `ORIGINATES_FROM`, `SUPPORTS_CONCEPT`). This structured information is then formatted and injected into the LLM's prompt.
*   **Benefit:** Providing explicit graph relationships and attributes as context allows the LLM to reason over a structured understanding of the knowledge, leading to more accurate and grounded outputs compared to relying solely on unstructured text.
