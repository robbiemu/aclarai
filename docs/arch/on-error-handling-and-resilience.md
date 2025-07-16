# 🛡️ Error Handling and Resilience Strategy

This document defines the error handling and resilience strategy used across distributed components in the aclarai system. It outlines guiding principles and shared mechanisms that help the system remain robust, fault-tolerant, and transparent when facing unexpected failures.

---

## 🎯 Purpose

Establish a consistent approach for anticipating, detecting, responding to, and recovering from errors—minimizing data loss, maximizing uptime, and providing actionable diagnostics.

## 🔧 Toolkit at a Glance

The following table summarizes the core resilience techniques used across the system:

| Mechanism                     | When to Use                       | Key Components                     |
|------------------------------|-----------------------------------|------------------------------------|
| Structured Logging            | Always                            | All components                     |
| Retry + Backoff              | Transient API/db errors           | LLM agents, Neo4j, Postgres        |
| Atomic File Writes           | Config and Markdown updates       | `aclarai-core`, Obsidian outputs  |
| Graceful Degradation         | Low-quality or invalid data       | Plugins, Evaluation Agents         |
| Circuit Breakers | Persistent service failures       | External APIs (LLMs, databases)    |
| Idempotency & Deduplication  | Reprocessing, sync jobs           | `vault-to-graph`, scheduler        |

## I. Guiding Principles

1. **Fail Gracefully, Not Silently:**
   Errors should be explicitly logged and diagnosed. Non-critical failures should degrade functionality without crashing the system.

2. **Protect Data Integrity:**
   All data writes (e.g., vault, Neo4j) must be atomic and avoid corruption.

3. **Separate Transient and Permanent Errors:**
   Use retries only for transient issues (e.g., timeouts). Fail fast on permanent issues (e.g., invalid configs).

4. **Isolate Failures:**
   Errors in one part should not affect unrelated components.

5. **Enable Observability:**
   Errors must be visible through logs or status panels to allow rapid debugging.

6. **Use Idempotency Where Possible:**
   Repeated operations (e.g., retries) should not create duplicates or side effects.

---

## II. Common Error Types

1. **Transient Errors**
   Temporary issues that usually resolve (e.g., network failures, rate limits).

2. **Permanent Operational Errors**
   Persistent issues due to misconfiguration or infrastructure (e.g., invalid API keys, disk full).

3. **Data Errors**
   Input-related problems (e.g., malformed JSON, unrecognized formats).

4. **Logical Errors**
   Bugs in the application logic (e.g., invalid state transitions).

---

## III. Resilience Toolkit

### 1. **Structured Logging**

* **Use:** All services (`aclarai-core`, `vault-watcher`, `scheduler`, `aclarai-ui`).
* **How:** Use Python’s `logging` with structured output.
* **Why:** Enables centralized log collection, traceability using IDs (e.g., `claim_id`, `block_id`).

---

### 2. **Retries with Exponential Backoff**

* **Use:** Transient errors from LLM APIs, Neo4j, Postgres, file I/O.
* **How:** Max 3 retries with exponential backoff.
* **Why:** Improves resilience without manual intervention.
* **Examples:** LLM evaluation agents, entailment evaluations.

---

### 3. **Atomic File Writes**

* **Use:** Writing Markdown, config files.
* **How:** `write-temp → fsync → rename` pattern.
* **Why:** Ensures complete, consistent writes.
* **Examples:** Tier 1–3 creation, config updates.

---

### 4. **Graceful Degradation / Filtering**

* **Use:** Data errors or failed quality checks.
* **How:** Log, filter out, and optionally annotate with `null` scores or error flags.
* **Why:** Keeps low-quality data from polluting the graph.
* **Examples:** Evaluation agents, duplicate detection, fallback formats.

---

### 5. **Circuit Breakers)*

* **Use:** Repeated failures from external services.
* **How:** Trip after failure threshold, test periodically for recovery.
* **Why:** Prevents cascading failures.
* **Note:** Considered for post-MVP.

---

### 6. **Idempotency & Deduplication**

* **Use:** Synchronization and reprocessing.
* **How:** Unique `aclarai:id`, `ver=`, and hash checks.
* **Why:** Safe retries and parallelism.
* **Examples:** `vault-to-graph`, scheduler jobs.

---

## IV. Error Handling by Component

### `aclarai-core`

* Focus: Data integrity and orchestration.
* Patterns: Atomic writes, `try/except`, error filtering based on evaluation results.

### LLM Agents (Claimify, Evaluation, Summary)

* Focus: API interactions.
* Patterns: API retries, null scores on failure, detailed logging.

### Plugins (Format Conversion)

* Focus: Parsing diverse inputs.
* Patterns: Return `None` on failure, log parsing errors, inform UI.

### `vault-watcher`

* Focus: File system monitoring.
* Patterns: I/O resilience, batch/throttle events, fallback via full sync.

### Neo4j & Postgres

* Focus: Persistent storage.
* Patterns: Retry with backoff, transaction wrapping, specific exception handling.

### `aclarai-scheduler`

* Focus: Job orchestration.
* Patterns: Job-level retries, isolation, lifecycle logging, respect global pause flag.

---

## V. Observability

System health and error states are surfaced through:

* **Structured Logs:** Available from containers and CLI.
* **Gradio Status Panels:**

  * *Review Panel:* Job status, pause state, and error summaries.
  * *Import Panel:* Real-time ingestion progress and summaries.
* **Markdown Metadata:** Key evaluation results and error flags embedded directly in files.
