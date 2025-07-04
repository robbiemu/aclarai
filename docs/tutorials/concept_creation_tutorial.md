# Concept Creation Tutorial

This tutorial explains the end-to-end flow from text processing to the creation of `[[Concept]]` Markdown files in aclarai, showing how the concept promotion pipeline automatically generates Tier 3 concept documents.

## Overview

The aclarai concept creation system follows this pipeline:

1. **Text Processing** → Extract noun phrases from claims and summaries
2. **Concept Detection** → Identify similar concepts or promote new ones
3. **Neo4j Integration** → Create (:Concept) nodes for promoted concepts
4. **Tier 3 File Creation** → Generate `[[Concept]]` Markdown files in the vault

## The Complete Workflow

### Step 1: Noun Phrase Extraction

When processing claims or summaries, aclarai extracts noun phrases that could become concepts:

```python
from aclarai_shared.noun_phrase_extraction import NounPhraseExtractor

# Initialize extractor
extractor = NounPhraseExtractor()

# Process a claim or summary
result = extractor.extract_from_text(
    text="Machine learning is revolutionizing data analysis in healthcare.",
    source_node_id="claim_123",
    source_node_type="claim",
    aclarai_id="doc_456"
)

# This extracts candidates like:
# - "machine learning"
# - "data analysis" 
# - "healthcare"
```

### Step 2: Concept Detection and Promotion

The system then analyzes each candidate to determine if it should be merged with existing concepts or promoted to a new concept:

```python
from aclarai_shared.concept_detection import ConceptDetector

detector = ConceptDetector()

# Process a batch of candidates
detection_batch = detector.process_candidates_batch(candidates)

# Results contain actions:
# - MERGED: Similar concept already exists
# - PROMOTED: New concept should be created
```

### Step 3: Concept Node Creation

Promoted concepts become (:Concept) nodes in the Neo4j graph:

```python
from aclarai_shared.graph import Neo4jGraphManager

neo4j_manager = Neo4jGraphManager()

# Create concept nodes for promoted candidates
concepts = neo4j_manager.create_concepts(promoted_concept_inputs)

# Each concept gets:
# - Unique concept_id (e.g., "concept_machine_learning_123")
# - Source traceability to original claim/summary
# - Timestamp and version tracking
```

### Step 4: Automated Concept Summary Generation

The final step of generating rich, detailed Markdown files for each concept is now automated. This process is handled by the `concept_summary_refresh` scheduled job, which runs daily.

The job uses the **Concept Summary Agent** to create comprehensive `[[Concept]]` pages. The agent employs a sophisticated RAG (Retrieval-Augmented Generation) workflow that automatically:

- Retrieves supporting claims and summaries from Neo4j.
- Finds related concepts via vector similarity search.
- Generates intelligent content using an LLM.
- Creates properly formatted Markdown files using atomic writes to prevent data corruption.

This automation ensures that your concept pages are consistently and regularly updated without manual intervention, reflecting the latest state of your knowledge graph.

The Concept Summary Agent uses a sophisticated RAG workflow to create content:

1. **Graph Retrieval**: Fetches claims and summaries related to the concept via Neo4j relationships
2. **Vector Search**: Finds semantically related concepts and utterances using vector similarity
3. **LLM Generation**: Synthesizes retrieved context into coherent concept definitions and examples
4. **Atomic Writing**: Safely writes complete files to prevent corruption during vault monitoring

## Generated File Structure

### Filename Generation

Concept filenames are generated from the concept text using filesystem-safe transformations:

- **Input**: "machine learning"
- **Filename**: `machine_learning.md`
- **Rules**:
  - Convert to lowercase
  - Replace spaces with underscores
  - Remove special characters
  - Ensure uniqueness

### File Content Template

Each generated concept file follows this enhanced structure created by the Concept Summary Agent:

```markdown
## Concept: machine learning

Machine learning is a field of artificial intelligence focused on algorithms that improve automatically through experience, enabling computers to learn patterns from data without explicit programming for each task.

### Examples
- ML algorithms can identify patterns in medical imaging data to assist diagnosis ^claim_ml_medical_123
- Recommendation systems use machine learning to personalize content delivery ^summary_ml_recommendations_456
- Natural language processing applies ML techniques to understand human language ^claim_nlp_ml_789

### See Also
- [[artificial intelligence]]
- [[data science]]
- [[neural networks]]

<!-- aclarai:id=concept_machine_learning_123 ver=1 -->
^concept_machine_learning_123
```

Key improvements over basic concept files:
- **Intelligent Definitions**: LLM-generated explanations based on available context
- **Rich Examples**: Real claims and summaries that support or mention the concept
- **Semantic Relationships**: Vector search finds related concepts automatically
- **Proper Anchoring**: All examples include ^anchor references for Obsidian linking

## Key Components Explained

### aclarai ID and Anchors

Each concept file includes two important markers:

1. **aclarai ID Comment**: `<!-- aclarai:id=concept_machine_learning_123 ver=1 -->`
   - Links the file to the Neo4j (:Concept) node
   - Includes version number for tracking changes
   - Used for synchronization between vault and graph

2. **Obsidian Anchor**: `^concept_machine_learning_123`
   - Allows direct linking to this concept from other notes
   - Enables Obsidian's block reference system
   - Supports concept relationship mapping

### File Location

Concept files are created in the configured concepts directory:

```yaml
# In settings/aclarai.config.yaml
paths:
  concepts: "concepts"  # Creates files in vault/concepts/
```

## Integration with Obsidian

The generated concept files integrate seamlessly with Obsidian:

### Linking to Concepts

Reference concepts in your notes using standard Obsidian syntax:

```markdown
The advances in [[machine learning]] have transformed how we approach data analysis.

You can also link to specific concept anchors:
![[machine_learning#^concept_machine_learning_123]]
```

### Concept Graph Visualization

Obsidian's graph view will show relationships between:
- Source documents (Tier 1 files)
- Generated concepts (Tier 3 files)
- Claims and summaries that reference concepts

## Automated Workflow Example

Here's how the complete pipeline works in practice:

```python
from aclarai_core.concept_processor import ConceptProcessor

# Initialize the processor (handles all steps automatically)
processor = ConceptProcessor()

# Process a block (claim or summary)
result = processor.process_block_for_concepts(
    block={
        "aclarai_id": "claim_456",
        "semantic_text": "Machine learning algorithms are improving medical diagnosis accuracy."
    },
    block_type="claim"
)

# Result includes:
# - Extracted noun phrases: ["machine learning algorithms", "medical diagnosis"]
# - Detection actions: ["promoted", "merged"]
# - Created concept files: ["machine_learning_algorithms.md"]
# - Neo4j concept nodes created
```

## Configuration

Customize concept creation behavior in your `settings/aclarai.config.yaml`:

```yaml
concept_detection:
  similarity_threshold: 0.85  # Higher = more strict concept merging

concept_summaries:
  model: "gpt-4"              # LLM for generating concept content
  max_examples: 5             # Maximum examples per concept page
  skip_if_no_claims: true     # Skip concepts without supporting evidence
  include_see_also: true      # Include related concepts section
  
embedding:
  default_model: "sentence-transformers/all-MiniLM-L6-v2"
  
paths:
  concepts: "concepts"  # Where concept files are created
```

## Error Handling

The system includes robust error handling:

- **File Write Failures**: Logged but don't stop concept node creation
- **Database Errors**: Concept promotion continues without file creation
- **Network Issues**: Embedding failures fallback gracefully
- **Duplicate Prevention**: Existing files are never overwritten

## Monitoring and Logging

Track concept creation through structured logs:

```bash
# View concept creation activity
grep "Created.*Tier 3 Markdown files" aclarai.log

# Monitor concept promotion rates
grep "promoted concepts" aclarai.log
```

## Next Steps

- **Manual Curation**: Review and edit generated concept pages to add domain expertise
- **Concept Linking**: The agent automatically creates [[wiki-style links]] to related concepts
- **Example Enrichment**: As more content is processed, concept pages automatically gain more examples
- **Vector Search Optimization**: Fine-tune similarity thresholds to improve related concept discovery
- **Custom Prompts**: Modify the agent's LLM prompts for domain-specific concept generation

The Concept Summary Agent provides an intelligent foundation for building a comprehensive knowledge graph that grows automatically as you process more content through aclarai, creating rich, interconnected concept pages that serve as the backbone of your knowledge base.