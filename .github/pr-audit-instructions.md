### **PR Audit Instructions**

As an expert code reviewer and AI software architect, your task is to perform a comprehensive audit of the provided pull request. Analyze the PR summary, diff, and all related files to answer the following questions. Take your time and do a deep dive into these.

You can assume that this is during that sprint in the `sprint_plan.md`. Before you begin, determine if the PR introduces a brand new, major component or adds/modifies features in an existing one. Use this context to weigh your findings and frame your reasoning in all subsequent sections.

#### **Section 1: Completeness**

**Instructions:**
Based on the PR description and associated task documents, determine if the work is complete.

It may be that placeholders or mocks were used where real work was intended. If this sprint task will be followed by a later task in the `sprint_plan.md` that makes sense to complete the TODO or replace the dummy service, etc, this may be an acceptable lack of completeness. Please detail your findings in such a case.

**Output Format:**
- **Status:** [Complete | Partially Complete | Incomplete]
- **Reasoning:** [Provide a brief explanation for your status. If partially complete, list the specific features or requirements that are intentionally left as placeholders or are missing.]

---

#### **Section 2: Placeholders**

**Instructions:**
Scan the new and changed code for any `TODO`, `FIXME`, or other placeholder comments.

**Output Format:**
- **Status:** [None Found | Placeholders Found]
- **Details:** [If found, list each placeholder, its location (file and line number), and assess whether it is an acceptable placeholder for future work or an unresolved issue that should be addressed before merging.]

---

#### **Section 3: Code Quality**

**Instructions:**
Review the code for quality, readability, and adherence to Pythonic principles. Be detail oriented, don't let small naming convention variances, incomplete algorithms, or unrealized methods get in the way of production code.

**Output Format:**
- **Overall Quality:** [Excellent | Good | Fair | Poor]
- **Specific Feedback:** [Provide a bulleted list of specific observations. Mention strong points (e.g., "Good separation of concerns in `agent.py`") and areas for improvement (e.g., "The function `xyz` could be simplified by...").]

---

#### **Section 4: Comments and Internal Documentation**

**Instructions:**
Audit the code comments, docstrings, and any new documentation files for clarity and correctness. Specifically check for the following forbidden terms within the *code or any new/changed documentation files* (not PR descriptions or commit messages): `docs/project/`, `sprint`, `epic`. Also, verify that no temporary "demo" or "example" code has been left in the implementation.

**Output Format:**
- **Forbidden Term Check:** [Pass | Fail] - *[If Fail, list the file and line number where a forbidden term was found.]*
- **Demo Code Check:** [Pass | Fail] - *[If Fail, describe the leftover demo code.]*
- **Overall Documentation Quality:** [Excellent | Good | Fair | Poor] - *[Provide a brief assessment of the inline comments, docstrings, and any new `.md` files.]*

---

#### **Section 5: Audit of Existing Documentation**

**Instructions:**
You will audit the *existing* documentation to identify files that require updates due to the changes in this PR. Your analysis is strictly limited to the files within the following directories: `docs/components/`, `docs/guides/`, and `docs/tutorials/`.

**Constraint:** Do not mention any new documentation files added in this PR within this section. This section is *only* for auditing existing files.

**Output Format:**

**`docs/components/`**
- `file_a.md`: [No change needed | Revision Required] - *[Brief reason if required]*
- `file_b.md`: [No change needed | Revision Required] - *[Brief reason if required]*
- ...and so on for all files in the directory.

**`docs/guides/`**
- `file_c.md`: [No change needed | Revision Required] - *[Brief reason if required]*
- ...

**`docs/tutorials/`**
- `file_d.md`: [No change needed | Revision Required] - *[Brief reason if required]*
- ...

---

#### **Section 6: Overreach**

**Instructions:**
Assess whether the PR includes work that is outside its stated scope or introduces premature features that should be handled in a future task. (Consider this specifically in relation to the `sprint_plan.md`.)

**Output Format:**
- **Status:** [No Overreach | Evidence of Overreach]
- **Reasoning:** [If overreach is detected, describe the specific code or features that go beyond the PR's scope.]

---

#### **Section 7: Test Coverage**

**Instructions:**
Evaluate the test suite's effectiveness. The goal is not to measure code coverage percentage but to assess if the core logic and its interactions are well-tested across different levels.

**Output Format:**
-   **Unit Tests:** [Sufficient | Insufficient]
    -   **Reasoning:** *[Assess if the logic of individual functions and classes is well-tested in isolation. Mention any gaps.]*

-   **Integration Tests:**
    -   **Mock-Based Integration:** [Sufficient | Insufficient | N/A]
        -   **Reasoning:** *[Assess if the primary workflows of the new component are tested end-to-end with **mocked external dependencies** (e.g., databases, APIs). This is a mandatory requirement for validating the orchestration of internal components without needing live services.]*
    -   **Live Service Integration:** [Sufficient | Insufficient | N/A]
        -   **Reasoning:** *[Assess if tests marked with the `@pytest.mark.integration` decorator exist. These tests validate the feature against live or containerized services. Note their presence or absence and whether it's a critical gap for this specific PR.]*

-   **Overall Assessment:** [Provide a brief summary of the test suite's quality, considering all test types, weighing the concerns based on if the PR introduces a brand new component or just adds features to existing components. Conclude if the testing is adequate for merging.]

---

#### **Section 8: Architectural Alignment**

**Instructions:**
Review the implementation against the project's architectural principles and development instructions. Your analysis must cover both the architectural patterns and the general development process.

**Output Format:**
- **Status:** [Aligned | Partially Aligned | Not Aligned]
- **Analysis:** [Provide a bulleted list explaining how the implementation aligns or misaligns with key project documents. You must check against both sources:]
  - **Development Instructions (`.github/copilot-instructions.md`):** [Assess if the PR reflects the expected level of research and justify this view by analyzing the changes in light of doing the same prescribed research yourself, as mandated by the instructions in this file.]
  - **Architectural Patterns (`docs/arch/`, `docs/project/*.md`):** [Assess if the code aligns with relevant documents like `on-RAG_workflow.md`, `on-filehandle_conflicts.md`, etc. Name the specific documents considered.]