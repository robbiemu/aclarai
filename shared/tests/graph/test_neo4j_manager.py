import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from aclarai_shared.config import load_config
from aclarai_shared.graph.models import ClaimInput, SentenceInput
from aclarai_shared.graph.neo4j_manager import Neo4jGraphManager


class TestNeo4jGraphManagerUnit:
    """Unit tests for Neo4jGraphManager using a mocked driver."""

    @pytest.fixture
    def mock_manager(self):
        """Fixture to create a Neo4jGraphManager with a mocked driver and session."""
        # Patch the driver to prevent real connections
        with patch("aclarai_shared.graph.neo4j_manager.GraphDatabase.driver"):
            config = Mock()
            config.neo4j.get_neo4j_bolt_url.return_value = "bolt://mock:7687"
            config.neo4j.user = "mock"
            config.neo4j.password = "mock"
            config.processing = {"retries": {"max_attempts": 1}}

            manager = Neo4jGraphManager(config)

            mock_session = MagicMock()
            patcher = patch.object(Neo4jGraphManager, "session", new_callable=MagicMock)
            mock_session_method = patcher.start()
            mock_session_method.return_value.__enter__.return_value = mock_session

            # Store the session mock on the manager instance for easy access in tests
            manager.mock_session = mock_session

            yield manager

            # Stop the patcher after the test is done
            patcher.stop()
            manager.close()

    def test_create_claims_unit(self, mock_manager):
        """Test that create_claims generates the correct Cypher query."""
        claim_inputs = [ClaimInput(text="test claim", block_id="b1")]
        mock_manager.create_claims(claim_inputs)

        mock_manager.mock_session.run.assert_called_once()

        args, kwargs = mock_manager.mock_session.run.call_args
        query = args[0]
        params = kwargs["claims_data"]

        assert "UNWIND $claims_data AS data" in query
        assert "MERGE (c:Claim {id: data.id})" in query
        assert "MERGE (c)-[:ORIGINATES_FROM]->(b)" in query
        assert len(params) == 1
        assert params[0]["text"] == "test claim"

    def test_create_sentences_unit(self, mock_manager):
        """Test that create_sentences generates the correct Cypher query."""
        sentence_inputs = [SentenceInput(text="test sentence", block_id="b1")]
        mock_manager.create_sentences(sentence_inputs)

        mock_manager.mock_session.run.assert_called_once()

        args, kwargs = mock_manager.mock_session.run.call_args
        query = args[0]
        params = kwargs["sentences_data"]

        assert "UNWIND $sentences_data AS data" in query
        assert "MERGE (s:Sentence {id: data.id})" in query
        assert "MERGE (s)-[:ORIGINATES_FROM]->(b)" in query
        assert len(params) == 1
        assert params[0]["text"] == "test sentence"


@pytest.mark.integration
class TestNeo4jManagerIntegration:
    """Integration tests for Neo4jGraphManager requiring a live database."""

    @pytest.fixture(scope="class")
    def integration_manager(self):
        """Fixture to set up a connection to a real Neo4j database for testing."""
        if not os.getenv("NEO4J_PASSWORD"):
            pytest.skip("NEO4J_PASSWORD not set for integration tests.")

        config = load_config(validate=True)
        manager = Neo4jGraphManager(config=config)
        with manager.session() as session:
            session.run("MATCH (n) DETACH DELETE n", allow_dangerous_operations=True)
        yield manager
        manager.close()

    def test_connection_and_schema(self, integration_manager):
        """Test that the manager connects and sets up the schema."""
        integration_manager.setup_schema()
        with integration_manager.session() as session:
            result = session.run("SHOW CONSTRAINTS YIELD name")
            constraint_names = {record["name"] for record in result}
            assert "claim_id_unique" in constraint_names
            assert "sentence_id_unique" in constraint_names

    def test_create_and_get_claims(self, integration_manager):
        """Test creating claims and retrieving them."""
        with integration_manager.session() as session:
            session.run("CREATE (:Block {id: 'block-for-claim'})")
        claim_inputs = [
            ClaimInput(text="integration test claim", block_id="block-for-claim")
        ]
        created_claims = integration_manager.create_claims(claim_inputs)
        retrieved_claim = integration_manager.get_claim_by_id(
            created_claims[0].claim_id
        )
        assert retrieved_claim is not None
        assert retrieved_claim["text"] == "integration test claim"

    def test_create_and_get_sentences(self, integration_manager):
        """Test creating sentences and retrieving them."""
        with integration_manager.session() as session:
            session.run("CREATE (:Block {id: 'block-for-sentence'})")
        sentence_inputs = [
            SentenceInput(
                text="integration test sentence", block_id="block-for-sentence"
            )
        ]
        created_sentences = integration_manager.create_sentences(sentence_inputs)
        retrieved_sentence = integration_manager.get_sentence_by_id(
            created_sentences[0].sentence_id
        )
        assert retrieved_sentence is not None
        assert retrieved_sentence["text"] == "integration test sentence"

    def test_node_count(self, integration_manager):
        """Test node counting functionality."""
        with integration_manager.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            session.run("CREATE (:Block {id: 'b1'}), (:Block {id: 'b2'})")
        integration_manager.create_claims([ClaimInput(text="c1", block_id="b1")])
        integration_manager.create_sentences(
            [
                SentenceInput(text="s1", block_id="b1"),
                SentenceInput(text="s2", block_id="b2"),
            ]
        )
        counts = integration_manager.count_nodes()
        assert counts["claims"] == 1
        assert counts["sentences"] == 2
        assert counts["blocks"] == 2
