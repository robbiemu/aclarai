# GitHub Copilot - Project Guidelines and Instructions

This document provides details about the aclarai project, its structure, and the guidelines for implementing features and fixes. If you are asked to work on a something other than a sprint task, these guidelines may still be of interest. If you are asked to work on a sprint task, please follow the workflow outlined below.

## 🎯 Objective

Your primary role is to act as an expert Python developer working on the aclarai project. When asked to process a sprint task document (formatted as `docs/project/epic_M/sprint_N-*.md`), you will analyze all relevant documentation, adhere to the project's coding standards and architecture, and produce a complete and correct implementation.

---

## 🏗️ Repository Structure

This is a monorepo containing multiple services and shared libraries.

-   **`docs/`**: All project documentation, including architecture, design, and sprint plans. This is your primary source for requirements.
-   **`services/`**: Contains the individual applications.
    -   `aclarai-core/`: The main processing engine.
    -   `vault-watcher/`: File system monitoring service.
    -   `scheduler/`: For running periodic jobs.
    -   `aclarai-ui/`: The Gradio-based user interface.
-   **`shared/`**: Reusable Python modules, such as data models and common LlamaIndex tools, to be used across different services.
-   **`.github/workflows/`**: CI/CD pipeline configurations.
-   **`settings/aclarai.config.yaml`**: The central configuration file for the entire application. All configurable parameters must be read from here.
-   **`.env`**: For storing secrets and environment-specific variables (e.g., API keys, database URLs).

---

## 🛠️ Development & Tooling

This project uses `uv` for package and virtual environment management.

-   **Setup:** `uv venv` to create the virtual environment.
-   **Install Dependencies:** `uv pip install -r requirements.txt`
-   **Linting:** `ruff check .`
-   **Formatting:** `ruff format .`
-   **Testing:** `pytest`
-   **Pre-Commit CI Check:** Before any commit (and definitely do this after all other code changes), run formatting and linting. The CI pipeline will enforce these checks.

---

## 🐍 Python & Project Guidelines

1.  **Type Hinting:** All new code **must** include full type hints using the `typing` module.
2.  **Structured Logging:** Adhere strictly to `docs/arch/idea-logging.md`. Logs must go to `stdout`/`stderr` and include `service`, `filename.function_name`, and relevant contextual IDs (`aclarai_id`, `job_id`).
3.  **Configuration Management: Never hardcode values.** All configurable parameters are loaded at runtime by automatically merging the user's `settings/aclarai.config.yaml` over the base `shared/aclarai_shared/aclarai.config.default.yaml`. This provides a complete set of defaults that are safely overridden by user customizations, ensuring no values are ever hardcoded in the application.
4.  **Error Handling & Resilience:** Follow the patterns in `docs/arch/on-error-handling-and-resilience.md`. Use retries with exponential backoff for transient network errors (APIs, DBs), implement atomic file writes (`write-temp -> rename`) for all vault modifications, and handle failures gracefully (e.g., returning `null` scores from evaluation agents).
5.  **LlamaIndex First:** Prefer LlamaIndex abstractions (`VectorStoreIndex`, `Neo4jGraphStore`, `ServiceContext`, agentic tools) for interacting with data stores and LLMs.
6.  **Reusable Code:** Place any logic, data model, or tool that could be used by more than one service in the `shared/` directory to avoid code duplication.
7. **Documentation and comments:** When adding new documentation or comments to code, never reference the documents in the doc/project/epic_M/ folders.
8. More on new **Documentation style:**  If something is explained in docs/project and docs/arch directories, it's fine to include a repeat of that detail in implementation documentation. BUT if something is explained in OTHER implementation documents (like plugins documentation, tutorials, etc.) that could be considered authoritative, then reference those instead of rewriting the content.
9. **No Example code:** Rather than writing example code scripts, write tutorial-blog style documents for inclusion in the docs/tutorials directory. If the code indicates how to add on or work with tools, a guide like the docs/guides/plugin_system_guide.md makes sense. If the task you are implementing is a fairly self-contained component of the projectm a document, instead of writing a README.md in the same directory as the code, use docs/components. To facilitate good style, please see an example in any of these directories before adding your own.

---

##  workflow: Task Analysis & Implementation

When you receive a sprint task file, follow this precise workflow to plan and execute your work.

### 1. **Identify the Core Task**
-   Parse the filename to identify the **Sprint Number (N)** and the **Task Name**.
-   Read the "Descrição" (Description) and "Escopo" (Scope) sections of the task document to understand the primary goal and boundaries of the work item.

### 2. **Establish Situational Awareness**
-   **Consult the Sprint Plan:** Open `docs/project/epic_1/sprint_plan.md`. Locate the entry for the current sprint and task to understand its context within the larger sprint goal.
-   **Consult High-Level Overviews:** Refer to `docs/project/product_definition.md` and `docs/project/technical_overview.md` to understand how this task contributes to the broader product features and technical architecture.

### 3. **Gather Architectural Context**
-   Systematically identify all relevant documents by meticulously scanning the sprint task for **explicit paths** and **implicit keywords**.
-   **Keywords to Documents Mapping:**
    -   "evaluation scores," "entailment," `decontextualization_score` → `docs/arch/on-evaluation_agents.md`
    -   "configuration," "thresholds," "UI panel" → `docs/arch/design_config_panel.md`
    -   "atomic writes," "file conflicts" → `docs/arch/on-filehandle_conflicts.md`
    -   "Neo4j," "Cypher," "graph nodes" → `docs/arch/on-neo4j_interaction.md`
    -   `aclarai:id`, `ver=`, "sync loop" → `docs/arch/on-graph_vault_synchronization.md`
    -   "Top Concepts," "Subject Summaries," agent-written files → `docs/arch/on-writing_vault_documents.md`
    -   "concepts," "vector store," `hnswlib` → `docs/arch/on-concepts.md` & `docs/arch/on-vector_stores.md`
-   **Internalize these requirements** to ensure your implementation is consistent with the established architecture before writing any code.

Note: This project has substantial documentation, it may be helpful to read the docs/README.md guide to orient yourself to the project documentation structure.

### 4. **Handle Dependencies & Blockers**
-   **Identify Sibling Tasks:** Scan the file list for all other tasks in the same sprint (`sprint_N-*.md`).
-   **Analyze Prerequisites:** Read their descriptions. Determine if any must be completed *before* the current task can be fully implemented. (e.g., "Apply evaluation thresholds" depends on the agents that *produce* the scores).
-   **Formulate Action Plan:**
    -   **If NO blockers are found:** Proceed with a full implementation plan.
    -   **If a blocker IS found:**
        -   Clearly state which task(s) are blockers.
        -   Implement **only the unblocked, independent parts** of the current task.
        -   Prepare a pull request message using this template:
            ```markdown
            ### Completed Work
            - Implemented [describe the unblocked part of the task].
            - [List any other independent work completed].

            ---

            ### ⚠️ BLOCKED
            **This PR is partial and a blocker is present.**

            - **Blocked by:** [List the full name(s) of the prerequisite task document(s), e.g., `sprint_7-Implement_entailment_evaluation.md`]
            - **Impact:** The core logic for [describe the blocked functionality] cannot be completed until the prerequisite task is done, as it depends on [describe the specific output/component from the other task].
            ```