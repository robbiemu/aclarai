# Shared Utility Scripts

This directory contains command-line utility scripts for the `aclarai-shared` package. These scripts are intended for manual execution, testing, or administrative tasks related to shared components.

## Available Scripts

### `import_cli.py`
- **Purpose**: A command-line interface for importing conversation files into the vault as Tier 1 Markdown documents.
- **Usage**:
  ```bash
  # Import a single file
  python -m shared.aclarai_shared.scripts.import_cli --file /path/to/your/chat.txt

  # Import an entire directory
  python -m shared.aclarai_shared.scripts.import_cli --directory /path/to/exports
  ```

### `run_concept_summary_agent.py`
- **Purpose**: A command-line interface for manually running the `ConceptSummaryAgent`. This agent generates detailed Markdown pages for each canonical concept in the knowledge graph using a RAG workflow.
- **Usage**:
  ```bash
  # Perform a dry run without writing files
  python -m shared.aclarai_shared.scripts.run_concept_summary_agent --dry-run

  # Run with a custom configuration file
  python -m shared.aclarai_shared.scripts.run_concept_summary_agent --config custom.yaml
  ```

### `manual_test_subject_summary_agent.py`
- **Purpose**: A manual verification script for the `SubjectSummaryAgent`. It runs the agent against mocked data and prints the generated Markdown content to the console, allowing a developer to visually inspect the output quality without needing live services.
- **Usage**:
  ```bash
  # Run the verification script
  python -m shared.aclarai_shared.scripts.manual_test_subject_summary_agent
  ```