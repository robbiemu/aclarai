import re
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from services.scheduler.aclarai_scheduler.concept_highlight_refresh import (
    ConceptHighlightRefreshJob,
)
from shared.aclarai_shared import load_config
from shared.aclarai_shared.graph.neo4j_manager import Neo4jGraphManager

# Add project root (monorepo root) to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.integration
class TestConceptHighlightRefreshIntegration:
    """Integration tests for ConceptHighlightRefreshJob."""

    def setup_method(self):
        """Set up test fixtures and live Neo4j connection."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_vault_path = Path(self.temp_dir.name)

        self.config = load_config(validate=False)
        self.config.vault_path = str(self.temp_vault_path)
        self.config.scheduler.jobs.top_concepts.count = 3
        self.config.scheduler.jobs.trending_topics.window_days = 7
        self.config.scheduler.jobs.trending_topics.min_mentions = 1

        self.neo4j_manager = Neo4jGraphManager(self.config)

        # Clear Neo4j before each test to ensure a clean state
        self.neo4j_manager.execute_query("MATCH (n) DETACH DELETE n", allow_dangerous_operations=True)

        # Create sample data for the success test case
        self._create_sample_neo4j_data()

    def teardown_method(self):
        """Clean up temporary directory and clear Neo4j."""
        self.temp_dir.cleanup()
        # Clear Neo4j after each test
        self.neo4j_manager.execute_query("MATCH (n) DETACH DELETE n", allow_dangerous_operations=True)
        self.neo4j_manager.close()

    def _create_sample_neo4j_data(self):
        """Create sample Concept nodes and relationships for PageRank and Trending Topics to analyze."""
        now = datetime.utcnow()
        recent_timestamp = (now - timedelta(days=2)).isoformat()
        old_timestamp = (now - timedelta(days=10)).isoformat()

        self.neo4j_manager.execute_query(f"""
            CREATE (ml:Concept {{name: 'Machine Learning'}})
            CREATE (nn:Concept {{name: 'Neural Networks'}})
            CREATE (dl:Concept {{name: 'Deep Learning'}})
            CREATE (ai:Concept {{name: 'Artificial Intelligence'}})
            CREATE (cv:Concept {{name: 'Computer Vision'}})
            CREATE (nlp:Concept {{name: 'Natural Language Processing'}})

            CREATE (claim1:Claim {{text: 'claim 1'}})
            CREATE (claim2:Claim {{text: 'claim 2'}})
            CREATE (claim3:Claim {{text: 'claim 3'}})
            CREATE (claim4:Claim {{text: 'claim 4'}})

            CREATE (ml)-[:RELATED_TO]->(nn)
            CREATE (ml)-[:RELATED_TO]->(dl)
            CREATE (ml)-[:RELATED_TO]->(ai)
            CREATE (dl)-[:RELATED_TO]->(nn)
            CREATE (dl)-[:RELATED_TO]->(cv)
            CREATE (dl)-[:RELATED_TO]->(nlp)
            CREATE (ai)-[:RELATED_TO]->(dl)
            CREATE (nn)-[:RELATED_TO]->(nlp)

            CREATE (claim1)-[:MENTIONS_CONCEPT {{created_at: '{recent_timestamp}'}}]->(nlp)
            CREATE (claim2)-[:MENTIONS_CONCEPT {{created_at: '{recent_timestamp}'}}]->(nlp)
            CREATE (claim3)-[:MENTIONS_CONCEPT {{created_at: '{old_timestamp}'}}]->(cv)
            CREATE (claim4)-[:MENTIONS_CONCEPT {{created_at: '{recent_timestamp}'}}]->(cv)
        """)

    def test_concept_highlight_refresh_job_e2e_success(self):
        """Test end-to-end execution of ConceptHighlightRefreshJob."""
        job = ConceptHighlightRefreshJob(self.config)
        stats = job.run_job()

        assert stats["success"] is True
        assert stats["top_concepts_stats"]["success"] is True
        assert stats["trending_topics_stats"]["success"] is True
        assert stats["top_concepts_stats"]["file_written"] is True
        assert stats["trending_topics_stats"]["file_written"] is True
        assert len(stats["error_details"]) == 0

        # Verify Top Concepts file
        top_concepts_file = (
            self.temp_vault_path / self.config.scheduler.jobs.top_concepts.target_file
        )
        assert top_concepts_file.exists()
        top_concepts_content = top_concepts_file.read_text()
        extracted_top_concepts = re.findall(r"-\s*\[\[(.*?)\]\]", top_concepts_content)
        expected_top_concepts = ["Machine Learning", "Deep Learning", "Neural Networks"]
        assert extracted_top_concepts == expected_top_concepts

        # Verify Trending Topics file
        today = datetime.utcnow().strftime("%Y-%m-%d")
        trending_topics_file_name = (
            self.config.scheduler.jobs.trending_topics.target_file.format(date=today)
        )
        trending_topics_file = self.temp_vault_path / trending_topics_file_name
        assert trending_topics_file.exists()
        trending_topics_content = trending_topics_file.read_text()
        extracted_trending_topics = re.findall(
            r"-\s*\[\[(.*?)\]\]", trending_topics_content
        )
        # NLP has 2 recent mentions, CV has 1 recent and 1 old. So NLP should be higher.
        assert "Natural Language Processing" in extracted_trending_topics
        assert "Computer Vision" in extracted_trending_topics
