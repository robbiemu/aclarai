# Decontextualization Evaluation Agent

## 🎯 Purpose

The Decontextualization Evaluation Agent is a specialized component within the aclarai system responsible for assessing the semantic autonomy of a given factual claim. Its primary goal is to determine if a claim can be fully understood and verified on its own, without needing access to the original source text or conversational context from which it was extracted.

This evaluation produces a `decontextualization_score`, a float between 0.0 and 1.0, which quantifies how self-contained the claim is. This score is crucial for ensuring that claims promoted to the knowledge base are portable, reusable, and unambiguous.

## ⚙️ How it Works

### Core Logic
1.  **Input**: The agent receives a `claim_text` (the factual statement to evaluate) and its original `source_text` (the surrounding context or document from which the claim was derived).
2.  **LLM-Powered Analysis**: The agent utilizes a configured Large Language Model (LLM) to perform the core analysis. It employs a sophisticated prompt (see below) designed to guide the LLM in its assessment.
3.  **Contextual Ambiguity Check (Vector Search)**:
    *   To aid its evaluation, the agent requests its toolset from the `ToolFactory` by its role name (`decontextualization_agent`).
    *   The factory consults the `tools.agent_tool_mappings` section in the system configuration. For this agent, it is configured to receive a `VectorSearchTool` pointed at the `utterances` vector collection.
    *   The configuration also specifies the tool's metadata, giving it the specific name `vector_search_utterances` and a description that is injected into the agent's prompt.
    *   This allows the LLM to use the tool to query the `claim_text` against a vector store of "utterances" (a collection of sentences/text segments from the knowledge base).
    *   If semantically similar phrases to the claim appear in highly diverse and unrelated contexts within the vector store, it can indicate that the claim, as stated, is too generic, ambiguous, or relies on implicit context not present within the claim itself. This helps the LLM gauge if the claim is truly standalone.
4.  **Scoring**: Based on its analysis of the claim's language (pronouns, ambiguous references, missing essential qualifiers like time/location/scope) and the optional insights from the vector search, the LLM generates a `decontextualization_score` between 0.0 (completely context-dependent) and 1.0 (perfectly self-contained).
5.  **Output**: The agent returns a tuple containing the float score (or `None` on failure) and a status message (e.g., `'success'` or an error description).

### Retry and Error Handling
*   The agent incorporates a retry mechanism (with exponential backoff) for LLM API calls, configured via `processing.retries.max_attempts` in the system configuration.
*   If the LLM fails to produce a valid score after all retries, or if any other persistent error occurs during evaluation, the agent returns `None` for the score.

## 🧩 Integration and Configuration

*   **Instantiation**: The agent is typically instantiated by an orchestrating service, which provides it with:
    *   A pre-configured LLM instance (selected based on `model.claimify.decontextualization` from the system configuration).
    *   A `ToolFactory` instance, from which the agent requests its tools by its role name (`decontextualization_agent`).
    *   A `ConfigManager` instance for accessing system configurations (e.g., retry attempts).
*   **Tool Usage**: The agent's toolset is defined in the `tools.agent_tool_mappings` section of `aclarai.config.yaml`. The `ToolFactory` reads this configuration to provide the agent with a `VectorSearchTool` specifically configured for the `utterances` vector collection and named `vector_search_utterances` for the prompt.
*   **LLM Model**: The choice of LLM (e.g., GPT-3.5, Claude, etc.) is determined by the `model.claimify.decontextualization` setting in `aclarai.config.yaml`.

## 📝 Prompt Structure (Conceptual)

The prompt guides the LLM to:
*   Understand the definition of decontextualization.
*   Analyze the claim for common issues like unresolved pronouns, ambiguity, and missing critical context (time, location, scope).
*   Optionally use the `vector_search_utterances` tool by providing it with the claim text. The LLM is told that diverse results from this search can indicate that the claim isn't specific enough on its own.
*   Output a single float score between 0.0 and 1.0.

```
You are an expert evaluator. Your task is to determine if a given 'Claim'
can be understood on its own, without needing to refer to its original 'Source' context.
A claim is well-decontextualized if a reader can understand its meaning and verify it
without additional information beyond the claim itself.

Consider the following:
- Pronouns: Are all pronouns (he, she, it, they, etc.) clear or resolved?
- Ambiguous References: Are there any terms or phrases that could refer to multiple things?
- Missing Context: Is any crucial information (like time, location, specific entities, or scope) missing
  that would be necessary for a fact-checker to verify the claim accurately?

To help you, you have a 'vector_search_utterances' tool. You can use this tool with the 'Claim' text
as input to see if similar phrases or statements appear in other, potentially very different, contexts
in the knowledge base. If the search returns diverse results for a seemingly specific claim, it might
indicate ambiguity or missing context in the original claim.

Input:
Claim: "{claim_text}"
Source: "{source_text}"

Task:
1. Analyze the 'Claim' for any ambiguities or missing context.
2. Optionally, use the 'vector_search_utterances' tool with the 'Claim' text to check for contextual diversity
   of similar phrases. This can help identify if the claim is too generic or relies on implicit context
   not present in the claim itself.
3. Based on your analysis, provide a decontextualization score as a float between 0.0 and 1.0,
   where 0.0 means the claim is completely dependent on its source and cannot be understood alone,
   and 1.0 means the claim is perfectly self-contained and understandable in isolation.

Output only the float score. For example: 0.75
```

## 📊 Score Interpretation

*   **1.0**: The claim is perfectly decontextualized. It's clear, unambiguous, and contains all necessary information for a reader to understand and verify it without referring to the original source.
*   **0.7 - 0.9**: Generally good. The claim is largely self-contained but might have very minor ambiguities or could benefit from slight contextual additions that don't significantly hinder understanding for verification.
*   **0.4 - 0.6**: Moderately decontextualized. The claim likely has some unresolved pronouns, ambiguous terms, or is missing some context that would make independent verification difficult.
*   **0.0 - 0.3**: Poorly decontextualized. The claim is heavily reliant on its original source, likely uninterpretable or unverifiable in isolation.
*   **null**: The evaluation failed after retries, or a persistent error (like a malformed response from the LLM) occurred. The agent's evaluation method returns `None` for the score in this case, along with an error message. Claims with `null` scores are typically excluded from further processing like concept linking or promotion.

The `decontextualization_score`, along with `entailed_score` and `coverage_score`, contributes to an overall quality assessment of the claim (often via a geometric mean), which then informs how the claim is used within the aclarai system.
