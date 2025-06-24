"""
End-to-end test for Tier 3 concept file creation without external dependencies.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from aclarai_shared.config import PathsConfig, aclaraiConfig
from aclarai_shared.graph.models import Concept
from aclarai_shared.tier3_concept.writer import ConceptFileWriter


def test_end_to_end_concept_creation():
    """Test the complete flow from concept creation to file writing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup configuration
        config = aclaraiConfig(vault_path=temp_dir, paths=PathsConfig(tier3="concepts"))
        # Create the concept file writer
        writer = ConceptFileWriter(config)
        # Create a sample concept (as would be done by ConceptProcessor)
        concept = Concept(
            concept_id="concept_machine_learning_123",
            text="machine learning",
            source_candidate_id="candidate_456",
            source_node_id="claim_789",
            source_node_type="claim",
            aclarai_id="doc_abc123",
            version=1,
            timestamp=datetime.now(timezone.utc),
        )
        # Write the concept file
        success = writer.write_concept_file(concept)
        assert success is True
        # Verify the file was created
        expected_file = Path(temp_dir) / "concepts" / "machine_learning.md"
        assert expected_file.exists()
        # Verify the file content follows the correct format
        content = expected_file.read_text()
        # Check required elements are present
        assert "## Concept: machine learning" in content
        assert "### Examples" in content
        assert "### See Also" in content
        assert "<!-- aclarai:id=concept_machine_learning_123 ver=1 -->" in content
        assert "^concept_machine_learning_123" in content
        # Verify the structure
        lines = content.strip().split("\n")
        assert lines[0] == "## Concept: machine learning"
        assert lines[-2] == "<!-- aclarai:id=concept_machine_learning_123 ver=1 -->"
        assert lines[-1] == "^concept_machine_learning_123"
        print("✅ End-to-end test passed: concept file created successfully")
        print(f"📁 File created at: {expected_file}")
        print(f"📄 Content preview:\n{content[:200]}...")


def test_multiple_concepts_creation():
    """Test creating multiple concept files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = aclaraiConfig(vault_path=temp_dir, paths=PathsConfig(tier3="concepts"))
        writer = ConceptFileWriter(config)
        # Create multiple concepts
        concepts = [
            Concept(
                concept_id="concept_ai_001",
                text="artificial intelligence",
                source_candidate_id="cand_001",
                source_node_id="claim_001",
                source_node_type="claim",
                aclarai_id="doc_001",
                version=1,
                timestamp=datetime.now(timezone.utc),
            ),
            Concept(
                concept_id="concept_dl_002",
                text="deep learning",
                source_candidate_id="cand_002",
                source_node_id="claim_002",
                source_node_type="claim",
                aclarai_id="doc_002",
                version=1,
                timestamp=datetime.now(timezone.utc),
            ),
        ]
        # Write all concept files
        for concept in concepts:
            success = writer.write_concept_file(concept)
            assert success is True
        # Verify all files were created
        concepts_dir = Path(temp_dir) / "concepts"
        created_files = list(concepts_dir.glob("*.md"))
        assert len(created_files) == 2
        expected_files = {"artificial_intelligence.md", "deep_learning.md"}
        actual_files = {f.name for f in created_files}
        assert actual_files == expected_files
        print("✅ Multiple concepts test passed: all files created correctly")
        print(f"📁 Files created: {[f.name for f in created_files]}")


if __name__ == "__main__":
    test_end_to_end_concept_creation()
    test_multiple_concepts_creation()
