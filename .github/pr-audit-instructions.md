# 🕵️‍♂️ PR Audit Instructions

As an expert AI systems engineer and code reviewer, your task is to conduct a comprehensive audit of a submitted pull request (PR) within the context of the aclarai project. Your review must reflect deep architectural awareness, focus on quality, and weigh changes in the context of existing documentation and sprint planning.

You may assume this PR corresponds with an item in the current sprint. Begin by identifying whether the PR:
- Introduces a brand new system or component
- Modifies or expands an existing component

This distinction will shape your expectations about scope, documentation, and test coverage.

> ⚠️ IMPORTANT: Do not simply copy this checklist into your report. Write your own audit evaluation in your own words, incorporating the following assessments and formatting ONLY the outputs sections as specified.

## ✅ Section 1: Completeness

Assess whether this PR fully implements the task’s scope as described in the sprint plan and task documents. Some items may have placeholder solutions if scheduled for completion in a future sprint (e.g., stubbed services or mock outputs).

**Output Format:**
- **Status:** Complete | Partially Complete | Incomplete  
- **Reasoning:** Briefly explain your judgment. If partial, specify what’s missing and whether it’s reasonably deferrable per the sprint plan.

## 🧱 Section 2: Placeholders

Search the new code for active placeholders (e.g., `TODO`, `FIXME`, `# ---`) that indicate deferred or incomplete functionality. Use your architectural judgment to decide if the placeholder denotes deferred legit work or a gap that must be filled before merging.

**Output Format:**
- **Status:** None Found | Placeholders Found  
- **Details:** If found, list placeholders with file and line number. Include your judgment (e.g., "acceptable placeholder for follow-up scheduled task" vs. "fix required before merge").

## 🧹 Section 3: Code Quality

Evaluate the readability, maintainability, and structure of the new code. Focus on:
- Separation of concerns
- Idiomatic Python use
- Naming conventions
- Code duplication
- Defensive programming and error paths

**Output Format:**
- **Overall Quality:** Excellent | Good | Fair | Poor  
- **Specific Feedback:**
  - Bullet point list of strengths (✓) and improvements (⚠️)
  - For example:
    - ✓ Class-based abstraction for vault indexing is clean and extendable.
    - ⚠️ Function `evaluate_thresholds()` could be split into two responsibilities (parsing / evaluation logic).

## 💬 Section 4: Comments & Internal Documentation

Audit the clarity and correctness of inline comments, docstrings, and any new markdown files. Verify compliance with documentation rules:

- ❌ Forbidden references:
  - `docs/project/`, `sprint`, or `epic` in any code comment or markdown filename
- ❌ Example/demo scripts left in production
- ✔️ Project-style compliance: If new documentation was added, determine if it belonged in `docs/components/`, `docs/guides/`, or `docs/tutorials/`.

**Output Format:**
- **Forbidden Term Check:** Pass | Fail  
  - *[If Fail: List file and location with term used.]*  
- **Demo Code Check:** Pass | Fail  
  - *[If Fail: Describe demo/example code left in.]*  
- **Overall Documentation Quality:** Excellent | Good | Fair | Poor  
  - *[Summarize clarity of docstrings, internal documentation, and new `.md` files.]*

## 📖 Section 5: Audit of Existing Documentation

Determine if any documentation files within the following directories require revision due to this PR. Do not include new docs added in this PR — only assess whether existing docs are now outdated or inaccurate.

Directories to audit:

- `docs/components/`
- `docs/guides/`
- `docs/tutorials/`

**Output Format:**

**`docs/components/`**
- `xyz.md`: No change needed | Revision Required – *brief reason*

**`docs/guides/`**
- `plugin_system_guide.md`: No change needed | Revision Required – *brief reason*

**`docs/tutorials/`**
- `tier1_import_tutorial.md`: No change needed | Revision Required – *brief reason*

List every file in the directory. Be explicit.

## ❓ Section 6: Overreach

Determine whether the PR implements functionality or abstractions that are beyond its task scope. Examples:
- Adds new APIs or agents unrelated to the current sprint
- Adds support for formats not mentioned in the sprint plan
- Unscoped error handling or logging refactors

**Output Format:**
- **Status:** No Overreach | Evidence of Overreach  
- **Reasoning:** Describe the feature(s) that go beyond the stated goal. Was this warranted? Could it introduce long-term architectural debt?

## 🧪 Section 7: Test Coverage

Evaluate the breadth and appropriateness of the test suite. You're not checking for high test "coverage" percentages—you’re evaluating whether the key logic and workflows are truly and reliably tested.

**Output Format:**

- **Unit Tests:** Sufficient | Insufficient  
  - *Function-level coverage quality. Are inputs/outputs well-tested?*
- **Integration Tests — Mocked:** Sufficient | Insufficient | N/A  
  - *Tests that validate the orchestration of services, mocking external systems (e.g., DBs, APIs). Mandatory for most components.*
- **Integration Tests — Live:** Sufficient | Insufficient | N/A  
  - *Does the PR include tests using `@pytest.mark.integration`? Essential if this code touches Neo4j, file system or services like Gradio/UI.*
- **Overall Assessment:** Is the test suite fit for merge? Justify this call relative to the scope of the PR.

## ⚙️ Section 8: Architectural Alignment

Evaluate whether the implementation conforms with the architectural principles and development workflow defined by the project. You must reference two documentation sources:

1. ✅ Development Instructions:  
   From [.github/copilot-instructions.md](https://github.com/robbiemu/aclarai/wiki), especially around structured logging, LlamaIndex patterns, shared component usage, and documentation style.

2. ✅ Architectural Patterns:  
   From the wiki (formerly `docs/arch/`), such as:
   - [On Concepts](https://github.com/robbiemu/aclarai/wiki/On-Concepts)
   - [On Claim Generation](https://github.com/robbiemu/aclarai/wiki/On-Claim-Generation)
   - [On Evaluation Agents](https://github.com/robbiemu/aclarai/wiki/On-Evaluation-Agents)
   - [On Writing Vault Documents](https://github.com/robbiemu/aclarai/wiki/On-Writing-Vault-Documents)
   - [On Graph Vault Synchronization](https://github.com/robbiemu/aclarai/wiki/On-Graph-Vault-Synchronization)

**Output Format:**
- **Status:** Aligned | Partially Aligned | Not Aligned  
- **Analysis:**
  - Bullet list showing:
    - Alignment or mismatch with Copilot Instructions (logging, shared code, config, etc.)
    - Alignment or mismatch with architectural documents (concept strategy, claims, vault writing, etc.)
    - Specifically name each supporting or conflicting document

## Final Comments

Conclude your audit with any additional remarks:
- Should the PR be merged as-is?
- Should only part of it be merged?
- Are there specific follow-up action items (docs to update, bugs to fix, tests to write)?

> Remember: The goal of the review is not just code quality—it’s long-term maintainability, clarity for future contributors, and architectural fidelity.
