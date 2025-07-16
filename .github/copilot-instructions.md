# GitHub Copilot - Project Guidelines and Instructions

This document provides details about the aclarai project, its structure, and guidelines for implementing features and fixes. Whether you're working on specific sprint tasks or general development, these guidelines will help you understand the project's architecture and development standards.

## 🎯 Objective

Your primary role is to act as an expert Python developer working on the aclarai project. When implementing features or fixes, analyze all relevant documentation, adhere to the project's coding standards and architecture, and produce complete and correct implementations. For sprint tasks formatted as `sprint_N-*.md`, follow the specific workflow outlined in the Sprint Task Implementation section.

## 📚 Project Documentation

**Primary Documentation Hub:** [ClarifAI Project Wiki](https://github.com/robbiemu/aclarai/wiki)

The project documentation has been migrated to the GitHub wiki following the completion of Epic 1. This serves as the authoritative source for architecture decisions, technical specifications, and implementation guidance.

Key architectural documents available on the wiki include:
- [Technical Overview](https://github.com/robbiemu/aclarai/wiki/Technical-Overview) - High-level system architecture and workflow
- [Architecture](https://github.com/robbiemu/aclarai/wiki/Architecture) - Docker Compose deployment architecture
- [Design Config Panel](https://github.com/robbiemu/aclarai/wiki/Design-Config-Panel) - UI configuration system design
- [On Graph Vault Synchronization](https://github.com/robbiemu/aclarai/wiki/On-Graph-Vault-Synchronization) - Graph-vault sync strategy
- [On Pluggable Formats](https://github.com/robbiemu/aclarai/wiki/On-Pluggable-Formats) - Format conversion system
- [On Claim Generation](https://github.com/robbiemu/aclarai/wiki/On-Claim-Generation) - Claim extraction process
- [On Concepts](https://github.com/robbiemu/aclarai/wiki/On-Concepts) - Concept creation and drift handling
- [On Evaluation Agents](https://github.com/robbiemu/aclarai/wiki/On-Evaluation-Agents) - Evaluation scoring system
- [On Writing Vault Documents](https://github.com/robbiemu/aclarai/wiki/On-Writing-Vault-Documents) - Document generation standards

## 🏗️ Repository Structure

This is a monorepo containing multiple services and shared libraries.

-   **`docs/`**: Local documentation and guides (architectural docs migrated to wiki)
-   **`services/`**: Contains the individual applications
    -   `aclarai-core/`: The main processing engine
    -   `vault-watcher/`: File system monitoring service
    -   `scheduler/`: For running periodic jobs
    -   `aclarai-ui/`: The Gradio-based user interface
-   **`shared/`**: Reusable Python modules, data models, and common LlamaIndex tools
-   **`.github/workflows/`**: CI/CD pipeline configurations
-   **`settings/aclarai.config.yaml`**: Central configuration file for the entire application
-   **`.env`**: Secrets and environment-specific variables (API keys, database URLs)

## 🛠️ Development & Tooling

This project uses `uv` for package and virtual environment management.

-   **Setup:** `uv venv` to create the virtual environment
-   **Install Dependencies:** `uv pip install -r requirements.txt`
-   **Linting:** `ruff check .`
-   **Formatting:** `ruff format .`
-   **Testing:** `pytest`
-   **Pre-Commit CI Check:** Before any commit, run formatting and linting. The CI pipeline will enforce these checks.

## 🐍 Python & Project Guidelines

### Core Development Standards

1.  **Type Hinting:** All new code **must** include full type hints using the `typing` module
2.  **Structured Logging:** Use structured logging with `service`, `filename.function_name`, and contextual IDs (`aclarai_id`, `job_id`). Logs must go to `stdout`/`stderr`
3.  **Configuration Management:** Never hardcode values. All parameters are loaded from `settings/aclarai.config.yaml` merged over `shared/aclarai_shared/aclarai.config.default.yaml`
4.  **Error Handling & Resilience:** Use retries with exponential backoff for transient errors, implement atomic file writes (`write-temp -> rename`) for vault modifications, and handle failures gracefully
5.  **LlamaIndex First:** Prefer LlamaIndex abstractions (`VectorStoreIndex`, `Neo4jGraphStore`, `ServiceContext`, agentic tools) for data stores and LLMs
6.  **Reusable Code:** Place shared logic, data models, or tools in the `shared/` directory to avoid code duplication

### Documentation Standards

7. **Documentation References:** Never reference documents in `docs/project/epic_M/` folders in new documentation or comments
8. **Documentation Style:** 
   - Reference authoritative implementation documents (plugins, tutorials, guides) rather than rewriting content
   - Repeat details from `docs/project` and `docs/arch` if needed in implementation documentation
   - Use `docs/tutorials/` for tutorial-blog style documents instead of example code scripts
   - Use `docs/guides/` for reference materials (e.g., `docs/guides/plugin_system_guide.md`)
   - Use `docs/components/` for self-contained component documentation instead of local README.md files
9. **No Example Code:** Write comprehensive tutorial, guide, or compoennt documents rather than example scripts.

## 🔄 Sprint Task Implementation Workflow

When working on a sprint task file, follow this workflow:

### 1. **Identify the Core Task**
-   Parse the filename to identify the **Sprint Number (N)** and **Task Name**
-   Read the "Descrição" (Description) and "Escopo" (Scope) sections to understand the primary goal and boundaries

### 2. **Establish Situational Awareness**
-   **Consult Sprint History:** Review [Project History & Epic 1 Planning Archive](https://github.com/robbiemu/aclarai/wiki/Project-History-&-Epic-1-Planning-Archive) for context
-   **Consult High-Level Overviews:** There are several documents of interest in the wiki. For example, refer to [Technical Overview](https://github.com/robbiemu/aclarai/wiki/Technical-Overview) to understand how the task fits into the broader architecture
-   **Consult architectural documentation** various documents in the wiki may also be kept locally for convenience in `docs/arch`, which can shed light on the design principles and implementation strategies of the project.

### 3. **Gather Architectural Context**
-   **Systematically identify relevant documents** by scanning the task for explicit paths and implicit keywords
-   **Keywords to Wiki Documentation Mapping:**
    -   "evaluation scores," "entailment," `decontextualization_score` → [On Evaluation Agents](https://github.com/robbiemu/aclarai/wiki/On-Evaluation-Agents)
    -   "configuration," "thresholds," "UI panel" → [Design Config Panel](https://github.com/robbiemu/aclarai/wiki/Design-Config-Panel)
    -   "Neo4j," "Cypher," "graph nodes" → [Idea neo4J-interaction](https://github.com/robbiemu/aclarai/wiki/Idea-neo4J-ineteraction)
    -   `aclarai:id`, `ver=`, "sync loop" → [On Graph Vault Synchronization](https://github.com/robbiemu/aclarai/wiki/On-Graph-Vault-Synchronization)
    -   "Top Concepts," "Subject Summaries," agent-written files → [On Writing Vault Documents](https://github.com/robbiemu/aclarai/wiki/On-Writing-Vault-Documents)
    -   "concepts," "vector store," `hnswlib` → [On Concepts](https://github.com/robbiemu/aclarai/wiki/On-Concepts)
    -   "claims," "claimify," claim generation → [On Claim Generation](https://github.com/robbiemu/aclarai/wiki/On-Claim-Generation)
    -   "pluggable formats," format conversion → [On Pluggable Formats](https://github.com/robbiemu/aclarai/wiki/On-Pluggable-Formats)
    -   "docker," "deployment," "services" → [Architecture](https://github.com/robbiemu/aclarai/wiki/Architecture)
-   **Internalize requirements** to ensure implementation consistency before writing code

### 4. **Handle Dependencies & Blockers**
-   **Identify Sibling Tasks:** Scan for other tasks in the same sprint (`sprint_N-*.md`)
-   **Analyze Prerequisites:** Determine if any must be completed before the current task
-   **Formulate Action Plan:**
    -   **If NO blockers:** Proceed with full implementation
    -   **If blockers exist:**
        -   State which tasks are blockers
        -   Implement only the unblocked, independent parts
        -   Prepare a pull request with clear blocker documentation:
            ```markdown
            ### Completed Work
            - Implemented [describe unblocked work]
            - [List other independent work completed]

            ---

            ### ⚠️ BLOCKED
            **This PR is partial and a blocker is present.**

            - **Blocked by:** [List prerequisite task document names]
            - **Impact:** [Describe blocked functionality and dependencies]
            ```

## 🔗 Additional Resources

For comprehensive technical documentation, architectural decisions, and implementation guides, visit the [ClarifAI Wiki](https://github.com/robbiemu/aclarai/wiki).
