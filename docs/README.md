## 📚 ClarifAI Project Documentation: Your Compass

Welcome to the ClarifAI documentation hub! This file serves as your central point of orientation, guiding you through the various architectural decisions, technical specifications, and project plans that define ClarifAI.

Whether you're a new team member, a seasoned contributor, or just exploring the project, this guide will help you find the information you need quickly.

## 📖 Documentation Philosophy: Finding the Right Document

Many of our technical documents have been migrated to the [project wiki](https://github.com/robbiemu/aclarai/wiki). The wiki serves as our primary documentation repository following the completion of Epic 1.

Some documentation, particularly documentation of actually implemented structures in the codebase, are maintained in the codebase. Of note, this particularly includes some overlapping documentating in the `docs/arch` folder.

| Category | Purpose | Answers the Question... | Example |
| :--- | :--- | :--- | :--- |
| **Architecture** | **Why?** (Architectural Decisions) | "Why was it built this way?" | [On Vector Stores](https://github.com/robbiemu/aclarai/wiki/On-Vector-Stores) explains the rationale for our vector database choices. |
| **Components**| **What?** (System Overviews) | "What does this system do?" | Component overviews provide high-level summaries of major systems. |
| **Guides** | **How does it work?** (Reference) | "What are my options for X?" | Reference guides for supported configurations and options. |
| **Tutorials**| **How do I do it?** (Instruction) | "How do I use X to achieve a goal?" | Step-by-step recipes for common tasks. |

## 📝 Documentation Categories

### 🚀 1. Project Vision & Product

These documents articulate the core purpose, user value, and high-level technical direction of ClarifAI.

*   **[ClarifAI: Technical Overview](https://github.com/robbiemu/aclarai/wiki/Technical-Overview)**
*   **[Selected User Stories for ClarifAI](https://github.com/robbiemu/aclarai/wiki/Epic_1)**

### 🧩 2. Core Component Overviews (The "What")

These documents provide high-level summaries of ClarifAI's major systems. They are the best starting point for understanding what each component does.

*   **Block Syncing Loop**
*   **Claimify Pipeline System**
*   **Embedding System**
*   **Graph System (Neo4j)**
*   **Import System**
*   **Scheduler System**
*   **UI System (Gradio)**
*   **Vault Watcher System**

### 🧱 3. Architecture & Design Principles (The "Why")

These documents detail the foundational architectural choices and cross-cutting design principles that govern development.

*   **[ClarifAI Deployment Architecture (Docker Compose Edition)](https://github.com/robbiemu/aclarai/wiki/Architecture)**
*   **[UI Design Docs](https://github.com/robbiemu/aclarai/wiki/Design-Config-Panel)**
*   **Error Handling and Resilience Strategy**
*   **Logging Strategy**
*   **LLM Interaction Strategy**
*   **Obsidian File Handle Conflicts**

### ⚙️ 4. Core Systems & Data Flow

These documents specify the implementation details of core services and how data flows and is stored.

*   **[ClarifAI Graph–Vault Sync Design](https://github.com/robbiemu/aclarai/wiki/On-Graph-Vault-Synchronization)**
*   **[Pluggable Format Conversion System](https://github.com/robbiemu/aclarai/wiki/On-Pluggable-Formats)**
*   **Vector Store Summary**
*   **[Neo4j Interaction Strategy](https://github.com/robbiemu/aclarai/wiki/Idea-neo4J-ineteraction)**
*   **Vault Layout and Document Type Inference**

### 🧠 5. Agent & AI Logic (Deep Dives)

These documents explain the intricate workings of ClarifAI's intelligent agents and their underlying AI logic.

*   **[Claim Generation Walkthrough](https://github.com/robbiemu/aclarai/wiki/On-Claim-Generation)**
*   **[Concept Creation and Drift Handling](https://github.com/robbiemu/aclarai/wiki/On-Concepts)**
*   **[Evaluation Roles (Agents)](https://github.com/robbiemu/aclarai/wiki/On-Evaluation-Agents)**
*   **Linking Claims to Concepts**
*   **Tier 3 Concept RAG Workflow Design**

### 📚 6. Guides & Tutorials (The "How")

These documents provide practical instructions and comprehensive references for using, configuring, and contributing to ClarifAI.

*   **Guides (Reference Manuals):**
    *   **Claimify Pipeline Guide**
    *   **Embedding Models Guide**
    *   **Scheduler Setup Guide**
*   **Tutorials (Step-by-Step Lessons):**
    *   **End-to-End Claimify Tutorial**
    *   **Importing Your First Conversation**
    *   **Working with the Neo4j Graph**

### 🗓️ 7. Development & Process

These documents cover our agile development process and sprint plans.

*   **[Sprint Plan for POC (Epic 1)](https://github.com/robbiemu/aclarai/wiki/Project-History-&-Epic-1-Planning-Archive)**

## 🔗 Additional Resources

For comprehensive project documentation, visit the [ClarifAI Wiki](https://github.com/robbiemu/aclarai/wiki) where you'll find detailed technical specifications, architectural decisions, and implementation guides.

## 🤝 Contributing to Documentation

This documentation is a living asset. If you find errors, omissions, or areas that could be clearer, please feel free to open an issue or submit a pull request. Your contributions help make ClarifAI better for everyone.
