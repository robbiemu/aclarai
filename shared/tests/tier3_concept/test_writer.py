"""
Tests for the Tier 3 Concept file writer functionality.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from aclarai_shared.config import PathsConfig, aclaraiConfig
from aclarai_shared.graph.models import Concept
from aclarai_shared.tier3_concept.writer import ConceptFileWriter


class TestConceptFileWriter:
    """Test the ConceptFileWriter class."""

    def test_initialization(self):
        """Test ConceptFileWriter initialization."""
        writer = ConceptFileWriter()
        assert writer.config is not None
        assert writer.concepts_dir is not None

    def test_initialization_with_config(self):
        """Test ConceptFileWriter initialization with custom config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = aclaraiConfig(
                vault_path=temp_dir, paths=PathsConfig(tier3="custom_concepts")
            )
            writer = ConceptFileWriter(config)
            expected_path = Path(temp_dir) / "custom_concepts"
            assert writer.concepts_dir == expected_path

    def test_generate_concept_filename(self):
        """Test concept filename generation."""
        writer = ConceptFileWriter()
        # Test normal text
        filename = writer._generate_concept_filename("machine learning")
        assert filename == "machine_learning.md"
        # Test text with special characters
        filename = writer._generate_concept_filename("AI/ML & data science!")
        assert filename == "AI_ML_data_science.md"
        # Test empty text
        filename = writer._generate_concept_filename("")
        assert filename == "unnamed_concept.md"
        # Test very long text
        long_text = "a" * 250
        filename = writer._generate_concept_filename(long_text)
        assert len(filename) <= 203  # 200 + ".md"
        assert filename.endswith(".md")

    def test_generate_concept_content(self):
        """Test concept content generation."""
        writer = ConceptFileWriter()
        concept = Concept(
            concept_id="concept_test123",
            text="machine learning",
            source_candidate_id="candidate_123",
            source_node_id="claim_456",
            source_node_type="claim",
            aclarai_id="doc_789",
            version=1,
            timestamp=datetime.now(timezone.utc),
        )
        content = writer._generate_concept_content(concept)
        # Check required elements are present
        assert "## Concept: machine learning" in content
        assert "### Examples" in content
        assert "### See Also" in content
        assert "<!-- aclarai:id=concept_test123 ver=1 -->" in content
        assert "^concept_test123" in content
        # Check content structure
        lines = content.split("\n")
        assert lines[0] == "## Concept: machine learning"
        assert lines[-2] == "<!-- aclarai:id=concept_test123 ver=1 -->"
        assert lines[-1] == "^concept_test123"

    def test_write_concept_file_success(self):
        """Test successful concept file writing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = aclaraiConfig(
                vault_path=temp_dir, paths=PathsConfig(tier3="concepts")
            )
            writer = ConceptFileWriter(config)
            concept = Concept(
                concept_id="concept_test123",
                text="machine learning",
                source_candidate_id="candidate_123",
                source_node_id="claim_456",
                source_node_type="claim",
                aclarai_id="doc_789",
                version=1,
                timestamp=datetime.now(timezone.utc),
            )
            # Write the file
            success = writer.write_concept_file(concept)
            assert success is True
            # Check file was created
            expected_file = Path(temp_dir) / "concepts" / "machine_learning.md"
            assert expected_file.exists()
            # Check file content
            content = expected_file.read_text()
            assert "## Concept: machine learning" in content
            assert "<!-- aclarai:id=concept_test123 ver=1 -->" in content
            assert "^concept_test123" in content

    def test_write_concept_file_directory_creation(self):
        """Test that concepts directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = aclaraiConfig(
                vault_path=temp_dir, paths=PathsConfig(tier3="new_concepts")
            )
            writer = ConceptFileWriter(config)
            # Concepts directory shouldn't exist yet
            concepts_dir = Path(temp_dir) / "new_concepts"
            assert not concepts_dir.exists()
            concept = Concept(
                concept_id="concept_test123",
                text="test concept",
                source_candidate_id="candidate_123",
                source_node_id="claim_456",
                source_node_type="claim",
                aclarai_id="doc_789",
                version=1,
                timestamp=datetime.now(timezone.utc),
            )
            # Write the file - should create directory
            success = writer.write_concept_file(concept)
            assert success is True
            # Check directory was created
            assert concepts_dir.exists()
            assert concepts_dir.is_dir()
            # Check file was created
            expected_file = concepts_dir / "test_concept.md"
            assert expected_file.exists()

    def test_atomic_write_safety(self):
        """Test that atomic write is used correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = aclaraiConfig(
                vault_path=temp_dir, paths=PathsConfig(tier3="concepts")
            )
            writer = ConceptFileWriter(config)
            concept = Concept(
                concept_id="concept_atomic_test",
                text="atomic write test",
                source_candidate_id="candidate_123",
                source_node_id="claim_456",
                source_node_type="claim",
                aclarai_id="doc_789",
                version=1,
                timestamp=datetime.now(timezone.utc),
            )
            # Write the file
            success = writer.write_concept_file(concept)
            assert success is True
            # Verify no temporary files are left behind
            concepts_dir = Path(temp_dir) / "concepts"
            temp_files = list(concepts_dir.glob(".*tmp*"))
            assert len(temp_files) == 0, f"Found temporary files: {temp_files}"
            # Verify file content is complete
            expected_file = concepts_dir / "atomic_write_test.md"
            content = expected_file.read_text()
            assert "## Concept: atomic write test" in content
            assert "<!-- aclarai:id=concept_atomic_test ver=1 -->" in content
            assert "^concept_atomic_test" in content
