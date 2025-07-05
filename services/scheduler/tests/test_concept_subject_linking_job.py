"""
Tests for the Concept Subject Linking Job.
These tests validate the logic for linking concepts to their subjects
with footer links and optional Neo4j edges.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from services.scheduler.aclarai_scheduler.concept_subject_linking_job import (
    ConceptSubjectLinkingJob,
)
from shared.aclarai_shared.config import ConceptSubjectLinkingJobConfig


class TestConceptSubjectLinkingJob:
    """Test the ConceptSubjectLinkingJob class."""

    def test_initialization(self):
        """Test job initialization."""
        mock_config = Mock()
        mock_config.vault_path = "/test/vault"
        mock_config.scheduler.jobs.concept_subject_linking = ConceptSubjectLinkingJobConfig(
            enabled=True,
            manual_only=False,
            cron="0 8 * * *",
            description="Test job",
            create_neo4j_edges=False,
            batch_size=50,
            footer_section_title="Part of Subjects",
        )

        job = ConceptSubjectLinkingJob(
            config=mock_config,
            neo4j_manager=Mock(),
            concept_clustering_job=Mock(),
        )

        assert job.config == mock_config
        assert job.vault_path == Path("/test/vault")
        assert job.concepts_dir == Path("/test/vault/concepts")
        assert not job.job_config.create_neo4j_edges
        assert job.job_config.batch_size == 50

    def test_extract_concept_name_from_file(self):
        """Test extracting concept name from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            concepts_dir = Path(temp_dir) / "concepts"
            concepts_dir.mkdir()

            # Create a valid concept file
            concept_file = concepts_dir / "test_concept.md"
            content = """# Test Concept

This is a test concept.

<!-- aclarai:id=concept_test_concept ver=1 -->
^concept_test_concept"""
            with open(concept_file, "w", encoding="utf-8") as f:
                f.write(content)

            # Create a mock job
            mock_config = Mock()
            mock_config.vault_path = temp_dir
            mock_config.scheduler.jobs.concept_subject_linking = ConceptSubjectLinkingJobConfig()
            
            job = ConceptSubjectLinkingJob(
                config=mock_config,
                neo4j_manager=Mock(),
                concept_clustering_job=Mock(),
            )

            concept_name = job._extract_concept_name_from_file(concept_file)
            assert concept_name == "test_concept"

            # Test non-concept file
            non_concept_file = concepts_dir / "not_concept.md"
            with open(non_concept_file, "w", encoding="utf-8") as f:
                f.write("This is not a concept file")

            concept_name = job._extract_concept_name_from_file(non_concept_file)
            assert concept_name is None

    def test_add_subject_footer_links(self):
        """Test adding subject footer links to content."""
        mock_config = Mock()
        mock_config.vault_path = "/test/vault"
        mock_config.scheduler.jobs.concept_subject_linking = ConceptSubjectLinkingJobConfig(
            footer_section_title="Part of Subjects"
        )
        
        job = ConceptSubjectLinkingJob(
            config=mock_config,
            neo4j_manager=Mock(),
            concept_clustering_job=Mock(),
        )

        content = """# Test Concept

This is a test concept.

<!-- aclarai:id=concept_test_concept ver=1 -->
^concept_test_concept"""

        subject_names = ["Machine Learning", "Data Science"]
        updated_content = job._add_subject_footer_links(content, subject_names)

        # Check that footer section was added
        assert "## Part of Subjects" in updated_content
        assert "- [[Subject:Machine Learning]]" in updated_content
        assert "- [[Subject:Data Science]]" in updated_content

        # Check that it was added before metadata
        assert updated_content.index("## Part of Subjects") < updated_content.index("<!-- aclarai:id=")

    def test_add_subject_footer_links_already_exists(self):
        """Test that footer links are not added if they already exist."""
        mock_config = Mock()
        mock_config.vault_path = "/test/vault"
        mock_config.scheduler.jobs.concept_subject_linking = ConceptSubjectLinkingJobConfig(
            footer_section_title="Part of Subjects"
        )
        
        job = ConceptSubjectLinkingJob(
            config=mock_config,
            neo4j_manager=Mock(),
            concept_clustering_job=Mock(),
        )

        content = """# Test Concept

This is a test concept.

## Part of Subjects

- [[Subject:Existing Subject]]

<!-- aclarai:id=concept_test_concept ver=1 -->
^concept_test_concept"""

        subject_names = ["Machine Learning"]
        updated_content = job._add_subject_footer_links(content, subject_names)

        # Content should remain unchanged
        assert updated_content == content

    def test_increment_version(self):
        """Test incrementing version in content."""
        mock_config = Mock()
        mock_config.vault_path = "/test/vault"
        mock_config.scheduler.jobs.concept_subject_linking = ConceptSubjectLinkingJobConfig()
        
        job = ConceptSubjectLinkingJob(
            config=mock_config,
            neo4j_manager=Mock(),
            concept_clustering_job=Mock(),
        )

        content = """# Test Concept

This is a test concept.

<!-- aclarai:id=concept_test_concept ver=1 -->
^concept_test_concept"""

        updated_content = job._increment_version(content)

        assert "ver=2" in updated_content
        assert "ver=1" not in updated_content

    def test_get_subject_name_by_cluster_id(self):
        """Test getting subject name by cluster ID."""
        mock_neo4j = Mock()
        mock_neo4j.execute_query.return_value = [{"name": "Machine Learning"}]

        mock_config = Mock()
        mock_config.vault_path = "/test/vault"
        mock_config.scheduler.jobs.concept_subject_linking = ConceptSubjectLinkingJobConfig()
        
        job = ConceptSubjectLinkingJob(
            config=mock_config,
            neo4j_manager=mock_neo4j,
            concept_clustering_job=Mock(),
        )

        subject_name = job._get_subject_name_by_cluster_id(1)
        assert subject_name == "Machine Learning"

        # Test not found
        mock_neo4j.execute_query.return_value = []
        subject_name = job._get_subject_name_by_cluster_id(999)
        assert subject_name is None

    def test_update_concept_file(self):
        """Test updating a concept file with subject links."""
        with tempfile.TemporaryDirectory() as temp_dir:
            concepts_dir = Path(temp_dir) / "concepts"
            concepts_dir.mkdir()

            # Create a concept file
            concept_file = concepts_dir / "test_concept.md"
            content = """# Test Concept

This is a test concept.

<!-- aclarai:id=concept_test_concept ver=1 -->
^concept_test_concept"""
            with open(concept_file, "w", encoding="utf-8") as f:
                f.write(content)

            mock_config = Mock()
            mock_config.vault_path = temp_dir
            mock_config.scheduler.jobs.concept_subject_linking = ConceptSubjectLinkingJobConfig(
                footer_section_title="Part of Subjects"
            )
            
            job = ConceptSubjectLinkingJob(
                config=mock_config,
                neo4j_manager=Mock(),
                concept_clustering_job=Mock(),
            )

            # Update file
            subject_names = ["Machine Learning"]
            result = job._update_concept_file(concept_file, "test_concept", subject_names)

            assert result is True

            # Check file was updated
            with open(concept_file, "r", encoding="utf-8") as f:
                updated_content = f.read()

            assert "## Part of Subjects" in updated_content
            assert "- [[Subject:Machine Learning]]" in updated_content
            assert "ver=2" in updated_content

    def test_update_concept_file_no_changes(self):
        """Test updating a concept file when no changes are needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            concepts_dir = Path(temp_dir) / "concepts"
            concepts_dir.mkdir()

            # Create a concept file that already has the footer
            concept_file = concepts_dir / "test_concept.md"
            content = """# Test Concept

This is a test concept.

## Part of Subjects

- [[Subject:Machine Learning]]

<!-- aclarai:id=concept_test_concept ver=1 -->
^concept_test_concept"""
            with open(concept_file, "w", encoding="utf-8") as f:
                f.write(content)

            mock_config = Mock()
            mock_config.vault_path = temp_dir
            mock_config.scheduler.jobs.concept_subject_linking = ConceptSubjectLinkingJobConfig(
                footer_section_title="Part of Subjects"
            )
            
            job = ConceptSubjectLinkingJob(
                config=mock_config,
                neo4j_manager=Mock(),
                concept_clustering_job=Mock(),
            )

            # Try to update file
            subject_names = ["Machine Learning"]
            result = job._update_concept_file(concept_file, "test_concept", subject_names)

            assert result is False  # No changes made

    def test_create_neo4j_edge(self):
        """Test creating Neo4j edge between concept and subject."""
        mock_neo4j = Mock()
        mock_neo4j.execute_query.return_value = [{"c": {"name": "test_concept"}, "s": {"name": "Machine Learning"}}]

        mock_config = Mock()
        mock_config.vault_path = "/test/vault"
        mock_config.scheduler.jobs.concept_subject_linking = ConceptSubjectLinkingJobConfig()
        
        job = ConceptSubjectLinkingJob(
            config=mock_config,
            neo4j_manager=mock_neo4j,
            concept_clustering_job=Mock(),
        )

        result = job._create_neo4j_edge("test_concept", "Machine Learning")
        assert result is True

        # Verify the query was called
        mock_neo4j.execute_query.assert_called_once()
        args, kwargs = mock_neo4j.execute_query.call_args
        assert "MERGE (c)-[:PART_OF]->(s)" in args[0]
        assert args[1]["concept_name"] == "test_concept"
        assert args[1]["subject_name"] == "Machine Learning"

    @patch('services.scheduler.aclarai_scheduler.concept_subject_linking_job.write_file_atomically')
    def test_run_job_success(self, mock_write_atomic):
        """Test successful job execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            concepts_dir = Path(temp_dir) / "concepts"
            concepts_dir.mkdir()

            # Create concept files
            concept_file = concepts_dir / "test_concept.md"
            content = """# Test Concept

This is a test concept.

<!-- aclarai:id=concept_test_concept ver=1 -->
^concept_test_concept"""
            with open(concept_file, "w", encoding="utf-8") as f:
                f.write(content)

            # Mock dependencies
            mock_clustering_job = Mock()
            mock_clustering_job.get_cluster_assignments.return_value = {"test_concept": 1}

            mock_neo4j = Mock()
            mock_neo4j.execute_query.return_value = [{"name": "Machine Learning"}]

            mock_config = Mock()
            mock_config.vault_path = temp_dir
            mock_config.scheduler.jobs.concept_subject_linking = ConceptSubjectLinkingJobConfig(
                footer_section_title="Part of Subjects"
            )

            job = ConceptSubjectLinkingJob(
                config=mock_config,
                neo4j_manager=mock_neo4j,
                concept_clustering_job=mock_clustering_job,
            )

            # Run job
            stats = job.run_job()

            assert stats["success"] is True
            assert stats["concepts_processed"] == 1
            assert stats["concepts_linked"] == 1
            assert stats["files_updated"] == 1
            assert stats["concepts_skipped"] == 0
            assert len(stats["error_details"]) == 0

    def test_run_job_no_cluster_assignments(self):
        """Test job execution when no cluster assignments are available."""
        mock_clustering_job = Mock()
        mock_clustering_job.get_cluster_assignments.return_value = None

        mock_config = Mock()
        mock_config.vault_path = "/tmp"
        mock_config.scheduler.jobs.concept_subject_linking = ConceptSubjectLinkingJobConfig()

        job = ConceptSubjectLinkingJob(
            config=mock_config,
            neo4j_manager=Mock(),
            concept_clustering_job=mock_clustering_job,
        )

        stats = job.run_job()

        assert stats["success"] is False
        assert "No cluster assignments available" in stats["error_details"]

    def test_run_job_no_concept_files(self):
        """Test job execution when no concept files are found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty concepts directory
            concepts_dir = Path(temp_dir) / "concepts"
            concepts_dir.mkdir()

            mock_clustering_job = Mock()
            mock_clustering_job.get_cluster_assignments.return_value = {"concept1": 1}

            mock_config = Mock()
            mock_config.vault_path = temp_dir
            mock_config.scheduler.jobs.concept_subject_linking = ConceptSubjectLinkingJobConfig()

            job = ConceptSubjectLinkingJob(
                config=mock_config,
                neo4j_manager=Mock(),
                concept_clustering_job=mock_clustering_job,
            )

            stats = job.run_job()

            assert stats["success"] is False
            assert "No concept files found" in stats["error_details"]