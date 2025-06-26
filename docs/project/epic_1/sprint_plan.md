# 🧭 Sprint Plan for POC

## 🟢 **Sprint 1: Foundational Infrastructure**

**Goal:** Fully containerized backend with file watching, vault sync, and service scaffolding.

**Tasks:**

* [ ] Start initial repos for the services in the project.
* [ ] Scaffold aclarai frontend repo
  - Create aclarai-ui/ with Dockerfile (Gradio/Python)
  - Set up /import page as the default route
  - Add simple file picker and log area (unattached, can simulate plugin output)
  - No integration work needed: Connect to backend later via REST or file drop

* [ ] Build Docker Compose stack (Neo4j, Postgres w/pgvector, `aclarai-core`, `vault-watcher`, `scheduler`)
* [ ] Implement `.env` injection and `host.docker.internal` fallback for external DBs
* [ ] Set up github and github actions for CI before merges into main.


## 🟡 **Sprint 2: Ingestion and Block Generation**

**Goal:** Ingest ChatGPT JSON → generate Tier 1 Markdown with block anchors + IDs

**Tasks:**

* [ ] Implement default plugin for pluggable format conversion system
* [ ] Embed utterance chunks and store vectors in Postgres via LlamaIndex
  * Use LlamaIndex's TextSplitter (e.g. SentenceSplitter or TokenTextSplitter) to segment each Tier 1 Markdown block into coherent chunks suitable for embedding
  * Embed each chunk using the configured embedding model (e.g. OpenAI, local model)
  * Store resulting vectors in Postgres using LlamaIndex’s pgvector integration, including metadata:
    * `aclarai:id` (parent block)
    * `chunk_index`
    * original text

* [ ] Create Tier 1 Markdown files during import:
  * Compute and check file-level hash to skip duplicate imports
  * Use plugin outputs to emit Markdown files to the vault
  * Annotate each utterance with `aclarai:id` and `^anchor`
  * Embed file-level metadata at top (participants, timestamps, plugin)

## 🔵 **Sprint 3: Claimify MVP**

**Goal:** Extract and embed high-quality claims using the first 3 Claimify roles

**Tasks:**

* [ ] Implement core Claimify pipeline components:
  * Selection → Disambiguation → Decomposition
  * Process one sentence at a time using configured models
  * Structure output for later quality evaluation and graph ingestion

* [ ] Create `(:Claim)` and `(:Sentence)` nodes in Neo4j
* [ ] Create the agent and integration to generate Tier 2 summaries with links back to Tier 1
  * Atomic write logic for Markdown files:
    ✅ **What the task entails**

    #### **1. Implement safe overwrite via temp file**

    Instead of writing directly to `file.md`, you:

    1. Write to `file.md.tmp` using `fs.open()` + `fs.write()` + `fs.fsync()` (or equivalent in your language).
    2. Atomically `rename(file.md.tmp, file.md)` — this is atomic on POSIX filesystems.

    Python example:

    ```python
    import os
    with open("note.md.tmp", "w", encoding="utf-8") as tmp:
        tmp.write(new_content)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace("note.md.tmp", "note.md")  # atomic
    ```
  * Concept linking will be added in Sprint 4 after concepts exist.

* [ ] Bootstrap scheduler container + vault sync job
  * Add a new `aclarai-scheduler` service to the monorepo
  * Use `APScheduler` to run jobs on a cron schedule (e.g. nightly at 2am)
  * Implement `sync_vault_to_graph()` job:
    * Read all Tier 1 Markdown files
    * For each block with a `aclarai:id`, hash the text
    * Compare against stored hash in Neo4j
    * If changed, update node text/hash and mark dirty
  * Log job execution start, finish, and result
  * No external queue or UI controls yet — runs unconditionally

## **Sprint 4:**

**Goal:** Establish reactive vault sync infrastructure for detecting and syncing changed blocks.

* [ ] Vault file watcher with dirty block detection (based on `aclarai:id` comments)
* [ ] Block ID hashing + sync loop: update graph nodes if content changes
* [ ] Create noun phrase extractor on claims + summaries
  * Fetch `(:Claim)` and `(:Summary)` nodes from the graph
  * Extract noun phrases using spaCy from each node’s text
  * Normalize each noun phrase (lowercase, lemmatize, strip punctuation)
  * Embed and store each phrase in the `concept_candidates` vector index with metadata
  * Mark each entry as "pending" for future deduplication and promotion

* [ ] Use hnswlib for embedding-based concept detection
  * Initialize the HNSW index using the `concept_candidates` vector store.
  * Use it to detect duplicates:
    * For each candidate, query the index for similar items (e.g., cosine_sim ≥ 0.9)
    * If a match exists, link it to the existing concept or skip promotion.
  * Store results: Mark each as "merged" or "promoted" depending on match.

## 🟣 **Sprint 5: Concept Linking and Tier 3 Generation**

**Goal:** Create and link concepts to claims and summaries.

**Tasks:**

* [ ] Enhance Tier 2 summaries to include linked concepts
  * Scan each summary block for SUPPORTS_CONCEPT and MENTIONS_CONCEPT relationships
  * Embed [[Concept]] links inline where relevant in Tier 2 Markdown

* [ ] Create/update Tier 3 Markdown files (`[[Concept]]`) and `(:Concept)` nodes
* [ ] Link claims to concepts with `SUPPORTS_CONCEPT`, `MENTIONS_CONCEPT`, etc.
* [ ] Refresh embeddings from concept files nightly

## 🔶 **Sprint 6: Config Panel & Automation Control**

**Goal:** Support runtime customization of models, windows, and thresholds.

**Tasks:**


* [ ] Implement aclarai's core configuration system, providing **both** a UI panel and a persistent YAML config file, for:
  * LLM and embedding model selections (e.g., for Claimify stages).
  * Claim and concept processing thresholds (e.g., similarity, quality).
  * Claimify context window parameters (p, f).

* [ ] Add configuration controls for scheduled jobs
  * Support `enabled`, `manual_only`, and UI toggles for jobs like concept hygiene, sync, summary agents.

* [ ] Add “pause automation” feature (via file flags or UI switch)

## 🧪 **Sprint 7: Evaluation Agents & Filtering**

**Goal:** Support the full 9-role Claimify flow including evaluation agents and claim filtering. Use `entailed_score`, `coverage_score`, and `decontextualization_score` as defined in `on-evaluation_agents.md`.

**Tasks:**

*   [ ] **Implement Shared Tool Factory for Agents**
    *   Create a centralized `ToolFactory` in `shared/aclarai_shared/tools/` that provides agents with a standard set of tools based on `settings/aclarai.config.yaml`.
    *   Implement `Neo4jQueryTool` and `VectorSearchTool`.
    *   Implement a pluggable `WebSearchTool` that is **only enabled if a provider and API key are configured**.
    *   Define the `tools:` configuration schema in `aclarai.config.default.yaml`.
    *   Ensure agents can be initialized with the tools provided by the factory.

*   [ ] Implement entailment evaluation agent
    *   **Implement as a LlamaIndex `CodeActAgent`** that uses tools from the **Shared Tool Factory**.
    *   Uses source + claim to produce `entailed_score`.
    *   Writes score to graph edges and Markdown metadata.
    *   Sets `null` on failure with retries.

*   [ ] Implement coverage evaluation agent
    *   **Implement as a LlamaIndex `CodeActAgent`** that uses tools from the **Shared Tool Factory**, including the conditional `WebSearchTool`.
    *   Computes `coverage_score` from claim + source.
    *   Extracts omitted verifiable elements.
    *   Adds `(:Element)` nodes and `[:OMITS]` edges to graph.
    *   Writes score to Markdown and graph; handles null.

*   [ ] Implement decontextualization evaluation agent
    *   **Implement as a LlamaIndex `CodeActAgent`** that uses tools from the **Shared Tool Factory**.
    *   Determines `decontextualization_score` from claim + source.
    *   Writes to graph + Markdown.
    *   Follows same retry/null logic.

*   [ ] Apply evaluation thresholds to linking and filtering
    *   Computes geomean from all scores.
    *   Skips concept linking, promotion, or summary export for claims below threshold or with nulls.
    *   Integrates into existing claim-to-concept and vault update logic.
  
## 🧪 **Sprint 8: Import UX & Evaluation Display**

**Goal:** Surface evaluation results in the vault and support import via Gradio. 
Build on the pluggable import system introduced in Sprint 2 by adding a coordinated plugin orchestrator and user-facing import workflow.

**Tasks:**

* [ ] Create plugin manager and import orchestrator
  * Scans all known plugins and runs the first one where `can_accept()` returns true
  * Passes file to plugin and records import status
  * Supports fallback plugin if none match

* [ ] Build Gradio Import Panel
  * File picker for uploading raw conversation files
  * Selects appropriate format plugin automatically via `can_accept()`
  * Invokes plugin orchestrator to emit Tier 1 Markdown
  * Displays import status (success, skipped, error)
  * Supports fallback plugin if no plugin accepts the file

* [ ] Display evaluation scores in Markdown metadata
  * Append `<!-- aclarai:... -->` blocks to Tier 1 Markdown after evaluation
  * One line per score: entailment, coverage, decontextualization
  * Use consistent formatting and null handling

## 🟣 Sprint 9: Concept Highlight Pages

**🎯 Goal:** Generate `Top Concepts.md`, `Trending Topics.md`, and `[[Concept]]` pages using agentic processes and the existing aclarai graph and vector stores.


### Tasks

* [ ] Implement Top Concepts Job
  * Run Neo4j PageRank on `(:Concept)` nodes
  * Select top N by score
  * Write `Top Concepts.md` listing concept names, rank, and backlinks

* [ ] Implement Trending Topics Job
  * Track `SUPPORTS_CONCEPT` and `MENTIONS_CONCEPT` edge creation timestamps
  * Compute per-concept mention deltas over the last 7 days
  * Write `Trending Topics - <date>.md` with top changed concepts

* [ ] Implement Concept Summary Agent
  * For each canonical `(:Concept)`, generate a `[[Concept]]` Markdown file
  * Pull supporting claims, summaries, utterances, and related concepts from local sources
  * Include:
    * `## Concept: <name>`
    * Bullet-point examples with `^aclarai:id`
    * See Also section
  * Skip if insufficient claim links

* [ ] Design Config Panel UI for Concept Highlight Jobs
  * Add inputs for top_concepts and trending_topics under a new “Highlight & Summary” section
  * Support metric, count, percent, window_days, and target_file fields
  * Include toggle inputs for min_mentions and preview field for output filenames
  * Display selected agent model (model.trending_concepts_agent) as read-only or dropdown

* [ ] Schedule Highlight & Concept Jobs
  * Add `concept_highlight_refresh` and `concept_summary_refresh` to the scheduler
  * Ensure output uses atomic write and supports vault sync

## 🟡 **Sprint 10: Subject-Level Clustering & Summary**

**🎯 Goal:** Cluster related concepts and generate `[[Subject:XYZ]]` pages via an external-research-capable agent. Establish bi-directional links to member concepts.

### Tasks

* [ ] Implement Concept Clustering Job
  * Use concept embedding HNSW index to form thematic clusters
  * Filter by minimum cluster size and similarity threshold
  * Cache group assignments

* [ ] Implement Subject Summary Agent
  * For each concept cluster, generate a `[[Subject:XYZ]]` Markdown file
  * Pull shared claims, common summaries, and top concept names
  * **May use web search or open-source references** to provide background context
  * Include:
    * `## Subject: <name>`
    * Member `[[Concept]]`s with backlinks
    * Sectioned summary of key themes or issues

* [ ] Design Config Panel UI for Subject Summary & Concept Summary Agents
  * For subject_summaries: sliders or inputs for similarity_threshold, min_concepts, max_concepts
  * Toggle switches for allow_web_search and skip_if_incoherent
  * Dropdown or read-only field for model.subject_summary
  * For concept_summaries: input for max_examples, toggles for skip_if_no_claims, include_see_also
  * Collapsible section for agent-specific settings grouped under “Highlight & Summary”

* [ ] Link Concepts to Subjects
  * Add footer links in each `[[Concept]]` to its subject page
  * Optionally update the graph with `(:Concept)-[:PART_OF]->(:Subject)` edges

* [ ] Schedule Subject Jobs
  * Add `subject_group_refresh` to the scheduler with configurable frequency
  * Ensure safe overwrite and optional pause toggle
