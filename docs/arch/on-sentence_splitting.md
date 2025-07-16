# ✅ Sentence Splitting Strategy for Claimify Input

> 🎯 **Goal:** Break Tier 1 blocks into semantically coherent, complete-enough chunks for the Claimify pipeline, with every chunk going through Selection.


## 🎯 Sentence Splitting Strategy

This strategy splits Tier 1 blocks into coherent, agent-ready **sentence units** for the Claimify pipeline. The goal is to:

* Merge incomplete lead-ins (`...I get:`) with the next line.
* Keep quoted diagnostics or phrases as standalone if meaningful.
* Avoid over-splitting around code or newline boundaries.
* Be language-aware, but not rigid, prioritizing utility over strict grammar.
* Ensure output is **ready for Selection → Disambiguation → Decomposition**.

---

## ✅ Top Sentence Splitting Options

| Option                            | Summary                                                      | Fit for Your Use Case                   |
| --------------------------------- | ------------------------------------------------------------ | --------------------------------------- |
| **spaCy `doc.sents`**             | Rule-based sentence boundary detection (for known languages) | 🟡 Decent, needs patching for fragments |
| **NLTK `PunktSentenceTokenizer`** | Unsupervised, trained on language samples                    | 🟡 Over-splits code, too formal         |
| **LlamaIndex `SentenceSplitter`** | Chunking utility w/ token count + overlap + line breaks      | 🟢 Strong candidate, customizable       |
| **Langchain `TextSplitter`**      | Similar to LlamaIndex, token-aware + overlap support         | 🟡 Needs manual alignment with IDs      |
| **Custom rule-based splitter**    | Build your own with regex, indents, punctuation, colons      | 🟢 Best control, most effort            |
| **Stanza / syntactic tools**      | Tree-aware, language-specific                                | 🔴 Overkill for this preprocessing task |

---
Absolutely — you’re right to emphasize this: **we are not discarding content.** Every chunk should still go through the Claimify pipeline and either:

* Produce one or more `(:Claim)` nodes
  **or**
* Get stored as a `(:Sentence)` node (with `claimified: false`)

So let’s revise the approach with that in mind.

---

## 🥇 Recommended: **Hybrid Sentence Splitter Strategy**

### 🔧 Base Layer:

* Use **`LlamaIndex.SentenceSplitter`** with:

  * `chunk_size=300`
  * `chunk_overlap=30`
  * `keep_separator=True`

This gives:

* Token-aware splitting
* Language-agnostic sentence boundaries
* Works well on Markdown blocks, even informal ones

---

### 🧠 Post-processing Rules (Optional Enhancements)

These **improve semantic coherence**, especially for agentic processing:

1. **Merge colon-ended lead-ins**

   * If a sentence ends in `:` and the next one starts with lowercase or quote/code → merge
   * e.g. `in the else block I get:` + `O argumento do tipo...`

2. **Short prefix merger**

   * If a sentence is < 5 tokens and is followed by something more complete → merge forward
   * e.g. `Example:` + `"The model failed"`

3. **No discards**

   * Code fragments, single symbols, or diagnostics **are retained**
   * They are passed to Claimify like any other sentence
   * If no claim is found → they are stored as `(:Sentence)` nodes only

4. **Linebreak preservation**

   * Keep newline structure intact to help with aligning back to original Markdown if needed

---

### ✅ Result

Each Tier 1 block produces a list of **chunked sentence inputs**, where:

* Each chunk is grammatically plausible (enough)
* Each chunk is guaranteed to:

  * Get a `aclarai:id`
  * Be sent to the Claimify pipeline
  * Be recorded as either a `(:Claim)` or `(:Sentence)`

---

## 🧾 Output Format

Each output chunk includes:

* `text`: the chunked input for Claimify
* `aclarai_block_id`: ID of the parent Tier 1 block
* `chunk_index`: ordinal within block
* (optional) `offset_start`, `offset_end` if you track line or char spans

---

## 📌 Summary

| Stage             | Tool                          | Behavior                                                              |
| ----------------- | ----------------------------- | --------------------------------------------------------------------- |
| Base splitting    | `LlamaIndex.SentenceSplitter` | Token-aware, chunked output                                           |
| Postprocess merge | Custom wrapper                | Handles colons, short prefixes, etc.                                  |
| Result per chunk  | Claimify input                | One sentence (or merged) → Selection → Disambiguation → Decomposition |
| If no claim       | `(:Sentence)`                 | Preserves traceability and context                                    |
