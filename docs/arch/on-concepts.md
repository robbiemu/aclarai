# 🧠 Concept Creation and Drift Handling
aclarai extracts, organizes, and updates concepts in a lightweight but effective way that suits individual users and small teams. This approach builds on the existing block-level processing and nightly maintenance workflows.

---

## 1. 📌 How Concepts Are Created

Concepts are born automatically during sentence/claim extraction batches.

| Step                                 | What Happens                                                                                                                                                     |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1.1. Extract surface candidates**  | After running Claimify on a batch, scan resulting sentences and summaries using lightweight noun phrase extraction (e.g., spaCy or regex chunking). _see concept_candidates Vector Table_             |
| **1.2. Normalize candidates**        | Lowercase, trim whitespace, strip punctuation, singularize nouns. E.g., `["dashboards", "Dashboard", "Dash board"] → "dashboard"`.                               |
| **1.3. Check for existing concepts** | Use `hnswlib` to run a cosine similarity search on all current concept embeddings. If a match ≥ `0.9` is found, reuse that concept. Otherwise, create a new one. |
| **1.4. Create or update the graph**  | \n- New concept → create `(:Concept {id, name, embedding, aliases})`\n- Existing concept → update `last_seen` and expand `aliases` if necessary                  |
| **1.5. Link to claims**              | For each extracted claim, create `(:Claim)-[:SUPPORTS_CONCEPT {strength}]→(:Concept)` or `CONTRADICTS_CONCEPT`.                                                  |
---

### 🧩 Properties of `(:Concept)` nodes

Each canonical concept is represented by a `(:Concept)` node in the graph. These nodes contain both identifying information and operational metadata used during syncing, search, and linking.

| Property         | Type        | Description                                                                 |
|------------------|-------------|-----------------------------------------------------------------------------|
| `name`           | `String`    | The canonical name of the concept (also used in the Markdown filename)     |
| `embedding_hash` | `String`    | SHA256 hash of the Markdown text used to produce the current embedding     |
| `last_updated`   | `Datetime`  | Timestamp of last embedding refresh (usually from nightly concept sync)    |
| `version`        | `Int`       | Parsed from the concept file’s `aclarai:id` block to track vault edits    |
| `status`         | `String`    | Optional: may be `"active"`, `"merged"`, or `"deprecated"` for hygiene ops |


---

### 📦 `concept_candidates` Vector Table

Extracted noun phrases are embedded and stored in a **persistent vector table**, called `concept_candidates`. This vector space:

* **Spans the entire vault**, not just the current file
* Accumulates candidates from every processed claim and summary
* Enables **fast nearest-neighbor search** to detect existing concepts before creating new ones

Each row includes:

* `text`: the noun phrase (e.g. `"slice object"`)
* `embedding`: vector representation
* `source_claim_id` and `aclarai_id` for traceability
* `status`: e.g., `"pending"`, `"merged"`, `"promoted"`

This table supports:

* Concept deduplication across time and documents
* Canonical concept creation via similarity threshold (≥ 0.9)
* Concept linking without introducing noise into the final graph

> 🔁 Like `utterances`, this is a cumulative, appendable vector index—but used specifically for short phrases, not full utterances.

---

Yes — perfect approach. Let’s just **add a concise new section** after the existing description of `:Concept` nodes in `on-concepts.md`.

The document already refers to `:Concept` nodes in the graph, but it does **not yet explain** that a vector store is used for **semantic matching** when linking claims or resolving new candidates.

---

### ✅ Suggested Addition (Minimal Insertion)

Add after the explanation of concept promotion or `:Concept` node creation:

---

### 🧠 Concept Vector Index (Canonical Concepts)

Once concepts are promoted to canonical `:Concept` nodes, they are also embedded and stored in a separate **concepts vector table**. This allows:

* **Fast nearest-neighbor lookup** when linking new claims or summaries
* **Semantic similarity** grouping across promoted concepts (e.g., `"GPU crash"` ≈ `"CUDA failure"`)
* **Deduplication enforcement** when additional noun phrases are proposed

This vector index is used primarily during:

* **Linking:** `(:Claim)-[:MENTIONS_CONCEPT]->(:Concept)`
* **UI refinement** (e.g. concept grouping or merging suggestions)

> 🔍 Tier 3 Markdown files are generated via graph traversal, not vector similarity — the concept vector DB supports semantic alignment only.

---

## 2. 🔄 Nightly Concept Refresh Job

Once per day, the same maintenance job that syncs Markdown and Neo4j also performs basic concept hygiene.

| Step                                    | What It Does                                                                                                                                                                                                                                                                              |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **2.1. Refresh embeddings**             | For each concept, embed the first paragraph of its Tier 3 Markdown page (usually the definition or summary). Store new vector.                                                                                                                                                            |
| **2.2. Detect duplicates**              | Run pairwise similarity via `hnswlib`. If two concepts are ≥ `0.95` similar, mark one as a duplicate.                                                                                                                                                                                     |
| **2.3. Auto-merge low-risk duplicates** | If:\n- Both concepts have low total claim count (e.g., < 10), and\n- One is clearly newer (`created_at`), then:\n  - Move claims to older concept\n  - Merge aliases\n  - Delete the duplicate node\n  - Add a `redirect` note in the Tier 3 file (e.g., “Merged from \[\[Old Concept]]”) |

> 📝 No user review queues, moderation UIs, or merge confirmation dialogs are required.

---

## 3. ✂️ Optional Concept Split Detection (Later)

Not included in the MVP, but the groundwork is in place:

* Over time, if a concept’s linked claims split into two unrelated semantic clusters, we can detect that via **intra-concept claim embedding divergence**.
* When needed, this can trigger a split into multiple new concepts with redistributed claims and backlink notes.

---

### 4. 🔁 How This Fits The Graph-Vault Sync Loop

| Pipeline phase                    | Concept actions                                                                           |
| --------------------------------- | ----------------------------------------------------------------------------------------- |
| **During sentence batch**         | Extract new concepts and link claims immediately after Claimify.                          |
| **Nightly job**                   | Refresh embeddings, detect & merge concept duplicates.                                    |
| **Manual edits to concept files** | Cause the associated `(:Concept)` node to be marked dirty and reprocessed the next night. |

---

### ✅ Summary

* Uses `hnswlib` for fast, lightweight concept matching
* Merges similar concepts automatically based on similarity threshold and size
* Updates embeddings from Tier 3 page definitions every night
* No UI review flow required—MVP stays simple and autonomous
* Fully integrated into the existing nightly sync and block-based claim extraction pipeline

This plan keeps aclarai usable for individuals from day one while still laying the groundwork for smarter concept maintenance over time.
