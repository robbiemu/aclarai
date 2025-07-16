## 🧠 Tier 3 Concept RAG Workflow Design

### Goal
Enable structured Markdown generation for `[[Concept]]` pages using a hybrid RAG (Retrieval-Augmented Generation) strategy, leveraging both the graph and existing vector stores.

---

### 🎯 Input: `(:Concept)` node (e.g. `slice`)

### 🔁 Retrieval Sources

| Source              | Type      | Method                | Purpose                                 |
|---------------------|-----------|------------------------|-----------------------------------------|
| Claims              | Graph     | Cypher traversal       | Pull direct mentions                    |
| Summaries           | Graph     | Summary links          | Pull summarizing context                |
| Utterances          | Vector    | `utterances`    | Pull vault-wide natural language usage  |
| Concept candidates  | Vector    | `concept_candidates`   | Pull alternate phrasings                |
| Concepts            | Vector    | `concepts`             | Pull semantically related concepts      |

No new vector tables are introduced.

---

### 🔧 Retrieval Process

```python
def retrieve_rag_context(concept_text: str) -> dict:
    claims = query_graph_mentions(concept_text)  # Cypher
    summaries = query_summary_links(concept_text)  # Cypher
    utterances = find_similar_utterances(concept_text)
    candidate_phrases = find_similar_candidates(concept_text)
    related_concepts = find_related_concepts(concept_text)

    return {
        "claims": claims,
        "summaries": summaries,
        "utterances": utterances,
        "candidate_phrases": candidate_phrases,
        "related_concepts": related_concepts,
    }
```

---

### 🪄 Agent Prompt Template

```text
Concept: slice

---
Relevant Claims:
- [The user] received a type error from Pylance. ^blk_abc123
- A slice cannot be assigned to an int in __setitem__. ^blk_def456

---
Real-world Usage:
- "slice[None, None, None] cannot be assigned" ^blk_xyz111

---
Alternate Phrases:
- slice object
- Python slice

---
Similar Concepts:
- indexing
- __setitem__

---
Generate a Markdown page that defines this concept, explains its usage, and links to relevant aclarai IDs.
```

---

### 📦 Output: Tier 3 Markdown File (`[[Concept]]`)
- `## Concept: <term>`
- Claims and utterance links with `^aclarai:id`
- Optional `See also` section

---

### ✅ Summary
- No new vector DBs are needed
- Graph + 3 vector stores support rich, document-quality generation
- Fully agent-compatible


