## 📦 Vector Store Summary (Approved)

This document summarizes the vector tables we’ve committed to implementing, their purpose, and the sprints/tasks where they are used.

---

### ✅ 1. `utterances`
**Description**: Embeddings of Tier 1 block-level utterance chunks.

**Used In:**
- 🟡 Sprint 2 — *"Embed utterances and save vectors to Postgres"*
- 🔵 Sprint 3 — Used by summary agent to retrieve relevant utterances during Tier 2 summary generation
- 🟣 Sprint 4 — RAG context source for concept generation

**Purpose:**
- Vault-wide similarity search
- Supports example mining, duplicate detection, RAG summaries

---

### ✅ 2. `concept_candidates`
**Description**: Noun phrases extracted from claims and summaries, embedded and staged for deduplication.

**Used In:**
- 🟣 Sprint 4 — *"Create noun phrase extractor"*, *"Use HNSWlib for concept detection"*

**Purpose:**
- Support nearest-neighbor grouping
- Canonical concept promotion pipeline
- Staging area before inserting into `:Concept` graph

---

### ✅ 3. `concepts`
**Description**: Canonical `(:Concept)` terms, embedded to support semantic search and linking.

**Used In:**
- 🟣 Sprint 4 — *"Link claims to concepts"*
- 🔵 Sprint 5+ — Similarity-based exploration and disambiguation
- 🟡 Sprint 10 — *"Implement Concept clustering job"*

**Purpose:**
- Detect semantic duplicates
- Enable concept-to-concept similarity
- Assist in onboarding new claims or summaries
- Support efficient bulk retrieval for clustering and analytics

#### **Access Patterns**

The `concepts` vector store supports two primary access patterns:

-   **Similarity Search (`similarity_search`):** Used for discovery, such as finding a few concepts semantically similar to a new claim or another concept. This is a "one-to-few" operation.
-   **Bulk Embedding Retrieval (`get_embeddings_for_concepts`):** Used for data-intensive tasks like clustering, where the embeddings for a large, known set of concepts are required. This is a "many-to-many" operation performed in a single, efficient database query. **This pattern should be used by the Concept Clustering Job to avoid N+1 query performance issues.**

---

### ✅ 4. 🔁 Used in RAG (but no new vector DBs)
- Claims: retrieved from graph, not vectorized
- Summaries: retrieved from graph, not vectorized

**Covered in:**
- 📄 *Tier3 Rag Workflow* design doc

---

### ❌ Declined Vector Stores
| Candidate            | Reason Rejected                                  |
|----------------------|--------------------------------------------------|
| `claim_vectors`      | Redundant; not used in any sprint                |
| `summary_vectors`    | Covered by RAG over utterances/claims/summaries |
| `rag_passages`       | Future-use; redundant with Tier 3 RAG design     |

---

This summary reflects vector table requirements through Sprint 4 and supports future RAG capabilities without unnecessary expansion.
