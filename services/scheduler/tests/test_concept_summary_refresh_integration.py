import sys
import tempfile
from pathlib import Path

import pytest

from services.scheduler.aclarai_scheduler.concept_summary_refresh import (
    ConceptSummaryRefreshJob,
)
from shared.aclarai_shared import load_config
from shared.aclarai_shared.graph.neo4j_manager import Neo4jGraphManager

# Add project root (monorepo root) to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.integration
class TestConceptSummaryRefreshIntegration:
    """Integration tests for ConceptSummaryRefreshJob."""

    def setup_method(self):
        """Set up test fixtures and live Neo4j connection."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_vault_path = Path(self.temp_dir.name)

        self.config = load_config(validate=False)
        self.config.vault_path = str(self.temp_vault_path)
        (self.temp_vault_path / self.config.paths.concepts).mkdir(exist_ok=True)

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
        """Create sample Concept nodes and relationships for the job to process."""
        self.neo4j_manager.execute_query("""
            CREATE (c1:Concept {name: 'Test Concept 1', is_canonical: true})
            CREATE (c2:Concept {name: 'Test Concept 2', is_canonical: true})
            CREATE (c3:Concept {name: 'Non-Canonical Concept', is_canonical: false})
            CREATE (claim1:Claim {text: 'claim 1'})
            CREATE (claim1)-[:SUPPORTS_CONCEPT]->(c1)
        """)

    def test_concept_summary_refresh_job_e2e_success(self):
        """Test end-to-end execution of ConceptSummaryRefreshJob."""
        job = ConceptSummaryRefreshJob(self.config)
        stats = job.run_job()

        assert stats["success"] is True
        assert stats["concepts_processed"] == 2
        assert stats["concepts_updated"] > 0  # Depending on LLM, may not update all
        assert stats["concepts_skipped"] == 0
        assert stats["errors"] == 0

        # Verify that files were created for canonical concepts
        concept1_file = (
            self.temp_vault_path / self.config.paths.concepts / "test_concept_1.md"
        )
        concept2_file = (
            self.temp_vault_path / self.config.paths.concepts / "test_concept_2.md"
        )
        non_canonical_file = (
            self.temp_vault_path
            / self.config.paths.concepts
            / "non_canonical_concept.md"
        )

        assert concept1_file.exists()
        assert concept2_file.exists()
        assert not non_canonical_file.exists()

        # Verify content of a generated file
        content = concept1_file.read_text()
        assert "## Concept: Test Concept 1" in content
        assert "<!-- aclarai:id=concept_test_concept_1" in content
