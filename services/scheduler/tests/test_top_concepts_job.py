"""
Unit tests for Top Concepts Job functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from aclarai_scheduler.top_concepts_job import TopConceptsJob
from aclarai_shared import load_config


class TestTopConceptsJob:
    """Test TopConceptsJob functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = load_config(validate=False)
        # Ensure top concepts job is enabled for testing
        self.config.scheduler.jobs.top_concepts.enabled = True
        self.config.scheduler.jobs.top_concepts.manual_only = False

    def test_top_concepts_job_initialization(self):
        """Test that TopConceptsJob initializes correctly."""
        with patch("aclarai_scheduler.top_concepts_job.Neo4jGraphManager"):
            job = TopConceptsJob(self.config)

            assert job.config == self.config
            assert job.job_config == self.config.scheduler.jobs.top_concepts
            assert job.vault_path == Path(self.config.vault_path)
            assert job.target_file_path == job.vault_path / job.job_config.target_file

    def test_config_loading(self):
        """Test that top concepts job configuration is loaded correctly."""
        # Test top concepts job config exists
        top_concepts = self.config.scheduler.jobs.top_concepts
        assert hasattr(top_concepts, "enabled")
        assert hasattr(top_concepts, "manual_only")
        assert hasattr(top_concepts, "cron")
        assert hasattr(top_concepts, "description")
        assert hasattr(top_concepts, "metric")
        assert hasattr(top_concepts, "count")
        assert hasattr(top_concepts, "percent")
        assert hasattr(top_concepts, "target_file")

        # Test default values
        assert top_concepts.metric == "pagerank"
        assert top_concepts.count == 25
        assert top_concepts.percent is None
        assert top_concepts.target_file == "Top Concepts.md"

    @patch("aclarai_scheduler.top_concepts_job.Neo4jGraphManager")
    def test_project_concept_graph_success(self, mock_neo4j_manager):
        """Test successful graph projection."""
        # Mock Neo4j manager
        mock_manager_instance = MagicMock()
        mock_neo4j_manager.return_value = mock_manager_instance

        # Mock query results
        mock_manager_instance.query.side_effect = [
            [],  # Drop graph query (no existing graph)
            [
                {"graphName": "concept_graph", "nodeCount": 10, "relationshipCount": 15}
            ],  # Project query
        ]

        job = TopConceptsJob(self.config)
        job.neo4j_manager = mock_manager_instance

        result = job._project_concept_graph()

        assert result is True
        assert mock_manager_instance.query.call_count == 2

    @patch("aclarai_scheduler.top_concepts_job.Neo4jGraphManager")
    def test_execute_pagerank_success(self, mock_neo4j_manager):
        """Test successful PageRank execution."""
        # Mock Neo4j manager
        mock_manager_instance = MagicMock()
        mock_neo4j_manager.return_value = mock_manager_instance

        # Mock PageRank result
        mock_manager_instance.query.return_value = [
            {"nodePropertiesWritten": 10, "ranIterations": 8}
        ]

        job = TopConceptsJob(self.config)
        job.neo4j_manager = mock_manager_instance

        result = job._execute_pagerank()

        assert result is True
        mock_manager_instance.query.assert_called_once()

    @patch("aclarai_scheduler.top_concepts_job.Neo4jGraphManager")
    def test_get_top_concepts_with_count(self, mock_neo4j_manager):
        """Test getting top concepts with count configuration."""
        # Mock Neo4j manager
        mock_manager_instance = MagicMock()
        mock_neo4j_manager.return_value = mock_manager_instance

        # Mock concept results
        mock_manager_instance.query.return_value = [
            {"name": "Concept A", "score": 0.95},
            {"name": "Concept B", "score": 0.87},
            {"name": "Concept C", "score": 0.72},
        ]

        job = TopConceptsJob(self.config)
        job.neo4j_manager = mock_manager_instance
        job.job_config.count = 3
        job.job_config.percent = None

        concepts = job._get_top_concepts()

        assert len(concepts) == 3
        assert concepts[0] == ("Concept A", 0.95)
        assert concepts[1] == ("Concept B", 0.87)
        assert concepts[2] == ("Concept C", 0.72)

    @patch("aclarai_scheduler.top_concepts_job.Neo4jGraphManager")
    def test_get_top_concepts_with_percent(self, mock_neo4j_manager):
        """Test getting top concepts with percent configuration."""
        # Mock Neo4j manager
        mock_manager_instance = MagicMock()
        mock_neo4j_manager.return_value = mock_manager_instance

        # Mock queries: first count, then results
        mock_manager_instance.query.side_effect = [
            [{"total_count": 20}],  # Count query
            [
                {"name": "Concept A", "score": 0.95},
                {"name": "Concept B", "score": 0.87},
            ],  # Results query with calculated limit
        ]

        job = TopConceptsJob(self.config)
        job.neo4j_manager = mock_manager_instance
        job.job_config.count = None
        job.job_config.percent = 10.0  # 10% of 20 = 2 concepts

        concepts = job._get_top_concepts()

        assert len(concepts) == 2
        assert mock_manager_instance.query.call_count == 2

    def test_generate_markdown_content(self):
        """Test markdown content generation."""
        with patch("aclarai_scheduler.top_concepts_job.Neo4jGraphManager"):
            job = TopConceptsJob(self.config)

            concepts = [
                ("Machine Learning", 0.95),
                ("Neural Networks", 0.87),
                ("Deep Learning", 0.72),
            ]

            content = job._generate_markdown_content(concepts)

            assert "## Top Concepts" in content
            assert "[[Machine Learning]]" in content
            assert "[[Neural Networks]]" in content
            assert "[[Deep Learning]]" in content
            assert "Ranked #1 with PageRank score 0.9500" in content
            assert "Ranked #2 with PageRank score 0.8700" in content
            assert "Ranked #3 with PageRank score 0.7200" in content
            assert "<!-- aclarai:id=" in content
            assert " ver=1 -->" in content

    def test_generate_markdown_content_empty(self):
        """Test markdown content generation with no concepts."""
        with patch("aclarai_scheduler.top_concepts_job.Neo4jGraphManager"):
            job = TopConceptsJob(self.config)

            content = job._generate_markdown_content([])

            assert "## Top Concepts" in content
            assert "*No concepts found with PageRank scores.*" in content
            assert "<!-- aclarai:id=" in content

    def test_write_file_atomically(self):
        """Test atomic file writing."""
        with patch("aclarai_scheduler.top_concepts_job.Neo4jGraphManager"):
            job = TopConceptsJob(self.config)

            # Use a temporary directory for testing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = Path(temp_dir) / "test_top_concepts.md"
                job.target_file_path = temp_file

                content = "# Test Content\nSome test markdown content."

                result = job._write_file_atomically(content)

                assert result is True
                assert temp_file.exists()

                # Verify content was written correctly
                with open(temp_file, "r", encoding="utf-8") as f:
                    written_content = f.read()

                assert written_content == content

    @patch("aclarai_scheduler.top_concepts_job.Neo4jGraphManager")
    def test_run_job_success_flow(self, mock_neo4j_manager):
        """Test complete job execution flow."""
        # Mock Neo4j manager
        mock_manager_instance = MagicMock()
        mock_neo4j_manager.return_value = mock_manager_instance

        # Mock all Neo4j queries for successful flow
        mock_manager_instance.query.side_effect = [
            [],  # Drop graph
            [
                {"graphName": "concept_graph", "nodeCount": 5, "relationshipCount": 8}
            ],  # Project graph
            [{"nodePropertiesWritten": 5, "ranIterations": 6}],  # PageRank
            [  # Get top concepts
                {"name": "Test Concept", "score": 0.85},
            ],
            [],  # Cleanup graph
        ]

        # Use temporary directory for file output
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config = self.config
            temp_config.vault_path = temp_dir

            job = TopConceptsJob(temp_config)
            job.neo4j_manager = mock_manager_instance

            stats = job.run_job()

            assert stats["success"] is True
            assert stats["pagerank_executed"] is True
            assert stats["file_written"] is True
            assert stats["concepts_analyzed"] == 1
            assert stats["top_concepts_selected"] == 1
            assert len(stats["error_details"]) == 0

            # Verify file was created
            target_file = Path(temp_dir) / job.job_config.target_file
            assert target_file.exists()
