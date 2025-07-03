# Integration tests for Top Concepts Job functionality.


import re
import sys
import tempfile
from pathlib import Path

import pytest

from services.scheduler.aclarai_scheduler.top_concepts_job import TopConceptsJob
from shared.aclarai_shared import load_config
from shared.aclarai_shared.graph.neo4j_manager import Neo4jGraphManager

# Add project root (monorepo root) to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.integration
class TestTopConceptsIntegration:
    """Integration tests for TopConceptsJob."""

    def setup_method(self):
        """Set up test fixtures and live Neo4j connection."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_vault_path = Path(self.temp_dir.name)

        self.config = load_config(validate=False)
        self.config.vault_path = str(self.temp_vault_path)
        self.config.scheduler.jobs.top_concepts.enabled = True
        self.config.scheduler.jobs.top_concepts.manual_only = False
        self.config.scheduler.jobs.top_concepts.count = 3  # Limit to top 3 for testing

        self.neo4j_manager = Neo4jGraphManager(self.config)

        # Clear Neo4j before each test to ensure a clean state
        self.neo4j_manager.execute_query("MATCH (n) DETACH DELETE n")

        # Create sample data for the success test case
        self._create_sample_neo4j_data()

    def teardown_method(self):
        """Clean up temporary directory and clear Neo4j."""
        self.temp_dir.cleanup()
        # Clear Neo4j after each test
        self.neo4j_manager.execute_query("MATCH (n) DETACH DELETE n")
        self.neo4j_manager.close()

    def _create_sample_neo4j_data(self):
        """Create sample Concept nodes and relationships for PageRank to analyze."""
        # Create concepts WITHOUT the pagerank_score property.
        # The relationships are what PageRank will use to calculate centrality.
        self.neo4j_manager.execute_query("""
            CREATE (ml:Concept {name: 'Machine Learning'})
            CREATE (nn:Concept {name: 'Neural Networks'})
            CREATE (dl:Concept {name: 'Deep Learning'})
            CREATE (ai:Concept {name: 'Artificial Intelligence'})
            CREATE (cv:Concept {name: 'Computer Vision'})
            CREATE (nlp:Concept {name: 'Natural Language Processing'})

            // Create relationships that will influence the PageRank score.
            // A highly connected node like 'Machine Learning' will naturally get a higher score.
            CREATE (ml)-[:RELATED_TO]->(nn)
            CREATE (ml)-[:RELATED_TO]->(dl)
            CREATE (ml)-[:RELATED_TO]->(ai)
            CREATE (dl)-[:RELATED_TO]->(nn)
            CREATE (dl)-[:RELATED_TO]->(cv)
            CREATE (dl)-[:RELATED_TO]->(nlp)
            CREATE (ai)-[:RELATED_TO]->(dl)
            CREATE (nn)-[:RELATED_TO]->(nlp)
        """)

    def test_top_concepts_job_e2e_success(self):
        """Test end-to-end execution of TopConceptsJob, verifying relative rank order."""
        # The data is set up in setup_method

        # Initialize and run the job
        job = TopConceptsJob(self.config, neo4j_manager=self.neo4j_manager)
        stats = job.run_job()

        # Assertions on job statistics
        assert stats["success"] is True
        assert stats["pagerank_executed"] is True
        assert stats["file_written"] is True
        assert (
            stats["top_concepts_selected"] == 3
        )  # We configured the job to get the top 3
        assert len(stats["error_details"]) == 0

        # Verify the output file content and, most importantly, the order of concepts
        output_file_path = (
            self.temp_vault_path / self.config.scheduler.jobs.top_concepts.target_file
        )
        assert output_file_path.exists()

        content = output_file_path.read_text()
        print(f"Generated content:\n{content}")

        # 1. Extract the concept names from the markdown list in the order they appear.
        # This is more robust than checking for hardcoded ranks or scores.
        # The regex finds list items (-), optional whitespace (\s*), and captures the content within [[...]].
        extracted_concepts = re.findall(r"-\s*\[\[(.*?)\]\]", content)

        # 2. Define the expected order of the top 3 concepts based on the graph's structure.
        # In our sample data, 'Machine Learning' is the most central node, followed by the
        # other highly interconnected nodes 'Deep Learning' and 'Neural Networks'.
        expected_order = [
            "Machine Learning",
            "Deep Learning",
            "Neural Networks",
        ]

        # 3. Assert that the extracted list of concepts matches the expected order.
        # This confirms the ranking logic is working correctly.
        assert extracted_concepts == expected_order, (
            f"Expected order {expected_order}, but got {extracted_concepts}"
        )

        # 4. Verify that lower-ranked concepts were correctly excluded from the file.
        assert "[[Artificial Intelligence]]" not in content
        assert "[[Computer Vision]]" not in content
        assert "[[Natural Language Processing]]" not in content

        # 5. Verify that the required metadata is still present in the file.
        assert "<!-- aclarai:id=file_top_concepts_" in content
        assert " ver=1 -->" in content
        assert "^file_top_concepts_" in content

        # 6. Verify that the temporary 'pagerank_score' property was cleaned up from the graph.
        result = self.neo4j_manager.execute_query(
            "MATCH (c:Concept) WHERE c.pagerank_score IS NOT NULL RETURN c"
        )
        assert len(result) == 0, (
            "pagerank_score property should be removed from all nodes after job completion"
        )

    def test_top_concepts_job_no_concepts(self):
        """Test end-to-end execution when no concepts are found in Neo4j."""
        # Clear all data to ensure no concepts are present for this test
        self.neo4j_manager.execute_query("MATCH (n) DETACH DELETE n")

        # Initialize and run the job
        job = TopConceptsJob(self.config, neo4j_manager=self.neo4j_manager)
        stats = job.run_job()

        # Assertions
        assert (
            stats["success"] is True
        )  # Job should still succeed, just with empty output
        assert stats["pagerank_executed"] is True
        assert stats["file_written"] is True
        assert stats["concepts_analyzed"] == 0
        assert stats["top_concepts_selected"] == 0
        assert len(stats["error_details"]) == 0

        # Verify the output file content
        output_file_path = (
            self.temp_vault_path / self.config.scheduler.jobs.top_concepts.target_file
        )
        assert output_file_path.exists()

        content = output_file_path.read_text()
        print(f"Generated content (no concepts):\n{content}")

        assert "## Top Concepts" in content
        assert "*No concepts found with PageRank scores.*" in content
        assert "<!-- aclarai:id=file_top_concepts_" in content
        assert " ver=1 -->" in content
        assert "^file_top_concepts_" in content

        # Verify that no pagerank_score property exists
        result = self.neo4j_manager.execute_query(
            "MATCH (c:Concept) WHERE c.pagerank_score IS NOT NULL RETURN c.pagerank_score AS score"
        )
        assert len(result) == 0
