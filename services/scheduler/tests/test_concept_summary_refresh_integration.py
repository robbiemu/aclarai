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
        # Ensure the concept summary agent uses the temporary directory
        self.config.paths.vault = str(self.temp_vault_path)
        (self.temp_vault_path / self.config.paths.tier3).mkdir(
            exist_ok=True, parents=True
        )

        self.neo4j_manager = Neo4jGraphManager(self.config)

        # Clear Neo4j before each test to ensure a clean state
        self.neo4j_manager.execute_query(
            "MATCH (n) DETACH DELETE n", allow_dangerous_operations=True
        )

        # Create sample data for the success test case
        self._create_sample_neo4j_data()

    def teardown_method(self):
        """Clean up temporary directory and clear Neo4j."""
        self.temp_dir.cleanup()
        # Clear Neo4j after each test
        self.neo4j_manager.execute_query(
            "MATCH (n) DETACH DELETE n", allow_dangerous_operations=True
        )
        self.neo4j_manager.close()

    def _create_sample_neo4j_data(self):
        """Create sample Concept nodes and relationships for the job to process."""
        self.neo4j_manager.execute_query("""
            CREATE (c1:Concept {name: 'Test Concept 1', is_canonical: true})
            CREATE (c2:Concept {name: 'Test Concept 2', is_canonical: true})
            CREATE (c3:Concept {name: 'Non-Canonical Concept', is_canonical: false})
            CREATE (claim1:Claim {text: 'claim 1'})
            CREATE (claim2:Claim {text: 'claim 2'})
            CREATE (claim1)-[:SUPPORTS_CONCEPT]->(c1)
            CREATE (claim2)-[:SUPPORTS_CONCEPT]->(c2)
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
            self.temp_vault_path / self.config.paths.tier3 / "test_concept_1.md"
        )
        concept2_file = (
            self.temp_vault_path / self.config.paths.tier3 / "test_concept_2.md"
        )
        non_canonical_file = (
            self.temp_vault_path / self.config.paths.tier3 / "non_canonical_concept.md"
        )

        assert concept1_file.exists()
        assert concept2_file.exists()
        assert not non_canonical_file.exists()

        # Verify content of a generated file
        content = concept1_file.read_text()
        assert "## Concept: Test Concept 1" in content
        assert "<!-- aclarai:id=concept_test_concept_1" in content
        # Verify version metadata is properly set (should default to 1 when no version in Neo4j)
        assert "ver=1" in content
        assert "ver=None" not in content

    def test_concept_summary_version_metadata_handling(self):
        """Test that version metadata is correctly handled in generated concept files."""
        # Create concepts with different version scenarios
        self.neo4j_manager.execute_query(
            """
            MATCH (n) DETACH DELETE n
        """,
            allow_dangerous_operations=True,
        )

        # Create proper concept and claim structure following the architecture
        self.neo4j_manager.execute_query("""
            CREATE (c1:Concept {name: 'Concept With Version', is_canonical: true, version: 5, id: 'c1', aclarai_id: 'concept_with_version'})
            CREATE (c2:Concept {name: 'Concept With Zero Version', is_canonical: true, version: 0, id: 'c2', aclarai_id: 'concept_with_zero_version'})
            CREATE (c3:Concept {name: 'Concept No Version', is_canonical: true, id: 'c3', aclarai_id: 'concept_no_version'})
            CREATE (claim1:Claim {text: 'This is a supporting claim for concept 1', aclarai_id: 'claim_1'})
            CREATE (claim2:Claim {text: 'This is a supporting claim for concept 2', aclarai_id: 'claim_2'})
            CREATE (claim3:Claim {text: 'This is a supporting claim for concept 3', aclarai_id: 'claim_3'})
            CREATE (claim1)-[:SUPPORTS_CONCEPT]->(c1)
            CREATE (claim2)-[:SUPPORTS_CONCEPT]->(c2)
            CREATE (claim3)-[:SUPPORTS_CONCEPT]->(c3)
        """)

        # Disable skip_if_no_claims for this test to focus on version handling
        # This is acceptable for testing purposes while maintaining architectural alignment
        original_skip_setting = getattr(
            self.config.concept_summaries, "skip_if_no_claims", True
        )
        self.config.concept_summaries.skip_if_no_claims = False

        try:
            job = ConceptSummaryRefreshJob(self.config)
            stats = job.run_job()

            print(f"\nDebug: Job stats: {stats}")

            assert stats["success"] is True
            assert stats["concepts_processed"] == 3
            print(
                f"Debug: concepts_processed={stats['concepts_processed']}, concepts_skipped={stats['concepts_skipped']}"
            )

            # Debug: check what files were actually created
            concepts_dir = self.temp_vault_path / self.config.paths.tier3
            actual_files = (
                list(concepts_dir.glob("*.md")) if concepts_dir.exists() else []
            )
            print(f"\nDebug: Files created in {concepts_dir}:")
            for f in actual_files:
                print(f"  - {f.name}")

            # Check concept with explicit version 5
            concept_v5_file = (
                self.temp_vault_path
                / self.config.paths.tier3
                / "Concept_With_Version.md"
            )
            assert concept_v5_file.exists()
            content_v5 = concept_v5_file.read_text()
            assert "ver=5" in content_v5
            assert "ver=None" not in content_v5

            # Check concept with version 0 (should preserve 0, not default to 1)
            concept_v0_file = (
                self.temp_vault_path
                / self.config.paths.tier3
                / "Concept_With_Zero_Version.md"
            )
            assert concept_v0_file.exists()
            content_v0 = concept_v0_file.read_text()
            assert "ver=0" in content_v0
            assert "ver=1" not in content_v0
            assert "ver=None" not in content_v0

            # Check concept with no version (should default to 1)
            concept_no_v_file = (
                self.temp_vault_path / self.config.paths.tier3 / "Concept_No_Version.md"
            )
            assert concept_no_v_file.exists()
            content_no_v = concept_no_v_file.read_text()
            assert "ver=1" in content_no_v
            assert "ver=None" not in content_no_v

        finally:
            # Restore original setting
            self.config.concept_summaries.skip_if_no_claims = original_skip_setting
