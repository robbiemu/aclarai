"""
Integration tests for the scheduler service and VaultSyncJob execution.
Tests the complete scheduler functionality against live Neo4j database.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from aclarai_scheduler.vault_sync import VaultSyncJob
from aclarai_shared import load_config
from aclarai_shared.graph.neo4j_manager import Neo4jGraphManager


@pytest.mark.integration
class TestSchedulerIntegration:
    """Integration tests for scheduler service requiring live Neo4j."""

    @pytest.fixture(scope="class")
    def integration_neo4j_manager(self):
        """Fixture to set up a connection to a real Neo4j database for testing."""
        if not os.getenv("NEO4J_PASSWORD"):
            pytest.skip("NEO4J_PASSWORD not set for integration tests.")

        config = load_config(validate=True)
        manager = Neo4jGraphManager(config=config)
        manager.setup_schema()
        # Clean up any existing test data
        with manager.session() as session:
            session.run(
                "MATCH (n) WHERE n.id STARTS WITH 'test_scheduler_' DETACH DELETE n"
            )
        yield manager
        # Clean up after tests
        with manager.session() as session:
            session.run(
                "MATCH (n) WHERE n.id STARTS WITH 'test_scheduler_' DETACH DELETE n"
            )
        manager.close()

    @pytest.fixture
    def temp_vault_path(self):
        """Create a temporary vault directory with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_path = Path(temp_dir)

            # Create tier1 directory structure
            tier1_path = vault_path / "tier1"
            tier1_path.mkdir(parents=True)

            # Create a test markdown file
            test_file = tier1_path / "test_document.md"
            test_content = """# Test Document

## Introduction
aclarai:id=test_scheduler_block_001 ver=1 hash=abc123

This is a test block for scheduler integration testing.
The system should process this content and create graph nodes.

## Conclusion
aclarai:id=test_scheduler_block_002 ver=1 hash=def456

This is another test block to verify batch processing.
"""
            test_file.write_text(test_content)

            yield vault_path

    @pytest.mark.integration
    def test_scheduler_runs_vault_sync_job_e2e(
        self, temp_vault_path, integration_neo4j_manager
    ):
        """
        Tests that the scheduler correctly executes the vault sync job against a live DB.
        1. Set up a temp vault with a markdown file and a known block.
        2. Pre-populate Neo4j with a "stale" version of that block.
        3. Manually trigger the VaultSyncJob.
        4. Query Neo4j to verify the block's node was updated (version incremented, etc.).
        """
        # Pre-populate Neo4j with stale version of the block
        with integration_neo4j_manager.session() as session:
            session.run("""
                CREATE (:Block {
                    id: 'test_scheduler_block_001',
                    content: 'Old content that should be updated',
                    version: 0,
                    hash: 'old_hash_123',
                    file_path: 'tier1/test_document.md',
                    aclarai_id: 'test_scheduler_block_001'
                })
            """)

            # Verify the stale block exists
            result = session.run(
                "MATCH (b:Block {id: 'test_scheduler_block_001'}) RETURN b.version as version"
            )
            record = result.single()
            assert record is not None
            assert record["version"] == 0

        # Create VaultSyncJob with temp vault path
        config = load_config(validate=True)
        # Override vault path in config for this test
        with patch.object(config, "vault_path", str(temp_vault_path)):
            vault_sync_job = VaultSyncJob(config=config)

            # Execute the sync job
            stats = vault_sync_job.run_sync()

            # Verify job completed successfully
            assert stats["files_processed"] >= 1
            assert stats["blocks_processed"] >= 2  # We have 2 blocks in the test file
            assert (
                stats["blocks_updated"] >= 1
            )  # At least the stale block should be updated
            assert stats["errors"] == 0

        # Query Neo4j to verify the block was updated
        with integration_neo4j_manager.session() as session:
            result = session.run("""
                MATCH (b:Block {id: 'test_scheduler_block_001'})
                RETURN b.version as version, b.hash as hash, b.content as content
            """)
            record = result.single()
            assert record is not None
            assert record["version"] > 0  # Version should have been incremented
            assert record["hash"] == "abc123"  # Hash should match the new content
            assert (
                "This is a test block for scheduler integration testing"
                in record["content"]
            )

        # Verify the second block was also processed
        with integration_neo4j_manager.session() as session:
            result = session.run("""
                MATCH (b:Block {id: 'test_scheduler_block_002'})
                RETURN b.version as version, b.hash as hash
            """)
            record = result.single()
            assert record is not None
            assert record["version"] >= 1
            assert record["hash"] == "def456"

    @pytest.mark.integration
    def test_vault_sync_job_handles_new_blocks(
        self, temp_vault_path, integration_neo4j_manager
    ):
        """
        Test that VaultSyncJob correctly creates new blocks that don't exist in Neo4j.
        """
        # Ensure no blocks exist initially
        with integration_neo4j_manager.session() as session:
            session.run(
                "MATCH (b:Block) WHERE b.id STARTS WITH 'test_scheduler_' DELETE b"
            )

        # Create VaultSyncJob and run it
        config = load_config(validate=True)
        with patch.object(config, "vault_path", str(temp_vault_path)):
            vault_sync_job = VaultSyncJob(config=config)
            stats = vault_sync_job.run_sync()

            # Verify job stats
            assert stats["files_processed"] >= 1
            assert stats["blocks_processed"] >= 2
            assert stats["blocks_new"] >= 2  # Both blocks should be new
            assert stats["errors"] == 0

        # Verify both blocks were created
        with integration_neo4j_manager.session() as session:
            result = session.run("""
                MATCH (b:Block)
                WHERE b.id IN ['test_scheduler_block_001', 'test_scheduler_block_002']
                RETURN count(b) as block_count
            """)
            record = result.single()
            assert record["block_count"] == 2

    @pytest.mark.integration
    def test_vault_sync_job_handles_unchanged_blocks(
        self, temp_vault_path, integration_neo4j_manager
    ):
        """
        Test that VaultSyncJob correctly identifies unchanged blocks and doesn't update them.
        """
        # Pre-populate Neo4j with up-to-date blocks
        with integration_neo4j_manager.session() as session:
            session.run("""
                CREATE (:Block {
                    id: 'test_scheduler_block_001',
                    content: 'This is a test block for scheduler integration testing.\\nThe system should process this content and create graph nodes.',
                    version: 1,
                    hash: 'abc123',
                    file_path: 'tier1/test_document.md',
                    aclarai_id: 'test_scheduler_block_001'
                })
            """)
            session.run("""
                CREATE (:Block {
                    id: 'test_scheduler_block_002',
                    content: 'This is another test block to verify batch processing.',
                    version: 1,
                    hash: 'def456',
                    file_path: 'tier1/test_document.md',
                    aclarai_id: 'test_scheduler_block_002'
                })
            """)

        # Run sync job
        config = load_config(validate=True)
        with patch.object(config, "vault_path", str(temp_vault_path)):
            vault_sync_job = VaultSyncJob(config=config)
            stats = vault_sync_job.run_sync()

            # Verify that blocks were identified as unchanged
            assert stats["files_processed"] >= 1
            assert stats["blocks_processed"] >= 2
            assert stats["blocks_unchanged"] >= 2  # Both blocks should be unchanged
            assert stats["blocks_updated"] == 0
            assert stats["blocks_new"] == 0
            assert stats["errors"] == 0

        # Verify blocks remain at version 1
        with integration_neo4j_manager.session() as session:
            result = session.run("""
                MATCH (b:Block)
                WHERE b.id IN ['test_scheduler_block_001', 'test_scheduler_block_002']
                RETURN b.id as id, b.version as version
                ORDER BY b.id
            """)
            records = list(result)
            assert len(records) == 2
            for record in records:
                assert record["version"] == 1  # Version should remain unchanged
