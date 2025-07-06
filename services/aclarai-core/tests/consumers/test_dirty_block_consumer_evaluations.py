import json
import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pika
import pytest
from aclarai_core.dirty_block_consumer import DirtyBlockConsumer
from aclarai_shared.config import aclaraiConfig
from aclarai_shared.graph.neo4j_manager import Neo4jGraphManager
from aclarai_shared.tools.factory import ToolFactory
from llama_index.core.llms.llm import LLM as LlamaLLM


@pytest.fixture
def mock_consumer_config():
    config = MagicMock(spec=aclaraiConfig)
    config.vault_path = "/fake/vault"
    config.rabbitmq_host = "fake_host"
    config.dict = MagicMock(
        return_value={"tools": {"agent_tool_mappings": {"entailment_agent": []}}}
    )
    config.processing = MagicMock()
    config.processing.retries = {"max_attempts": 3}
    config.model = MagicMock()
    config.model.claimify = {"entailment": "mock_llm", "default": "mock_llm"}
    config.neo4j = MagicMock()
    config.neo4j.host = "neo4j_mock_host"
    config.neo4j.port = 7687
    config.neo4j.user = "user"
    config.neo4j.password = "pass"
    config.neo4j.get_neo4j_bolt_url.return_value = "bolt://neo4j_mock_host:7687"
    config.embedding = MagicMock()
    config.embedding.pgvector = {"collection_name": "utterances", "embed_dim": 384}
    config.embedding.models = {"default": "mock_model"}
    config.embedding.device = "cpu"
    config.embedding.batch_size = 1
    config.embedding.chunking = {}
    config.paths = MagicMock()
    return config


@pytest.fixture
def mock_graph_manager_fixture():
    # This remains useful for tests that need to control its methods directly.
    return MagicMock(spec=Neo4jGraphManager)


@pytest.fixture
def mock_llm_fixture():
    return MagicMock(spec=LlamaLLM)


@pytest.fixture
def mock_tool_factory_fixture():
    return MagicMock(spec=ToolFactory)


@pytest.fixture
# THIS IS THE CRUCIAL FIX: Patch Neo4jGraphManager where it's imported by the consumer.
@patch("aclarai_core.dirty_block_consumer.Neo4jGraphManager")
@patch("aclarai_core.dirty_block_consumer.MarkdownUpdaterService")
@patch("aclarai_core.dirty_block_consumer.ClaimEvaluationGraphService")
@patch("aclarai_core.dirty_block_consumer.RabbitMQManager")
@patch("aclarai_core.dirty_block_consumer.BlockParser")
@patch("aclarai_core.dirty_block_consumer.ConceptProcessor")
@patch("aclarai_core.dirty_block_consumer.DecontextualizationAgent")
@patch("aclarai_core.dirty_block_consumer.CoverageAgent")
@patch("aclarai_core.dirty_block_consumer.EntailmentAgent")
@patch("aclarai_shared.tools.vector_store_manager.aclaraiVectorStoreManager")
def consumer_instance(
    MockVectorStoreManager,
    MockEntailmentAgent,
    MockCoverageAgent,
    MockDecontextualizationAgent,
    MockConceptProcessor,
    MockBlockParser,
    MockRabbitMQManager,
    MockGraphService,
    MockMarkdownService,
    _MockNeo4jManager,  # The new mock from the added patch
    mock_consumer_config,
    mock_graph_manager_fixture,
    mock_llm_fixture,
    mock_tool_factory_fixture,
):
    """
    Fixture for creating a DirtyBlockConsumer instance with all dependencies mocked.
    The patch for Neo4jGraphManager prevents the real class from being instantiated
    in the consumer's __init__, thus avoiding the network connectivity check.
    """
    with patch.object(
        DirtyBlockConsumer, "_initialize_llm", return_value=mock_llm_fixture
    ):
        consumer = DirtyBlockConsumer(config=mock_consumer_config)

        # The consumer's self.graph_manager is already a mock thanks to the patch.
        # We can overwrite it again if we need to use a specific mock instance like
        # mock_graph_manager_fixture, which is good practice for clarity.
        consumer.graph_manager = mock_graph_manager_fixture

        # Assign other mocked dependencies
        consumer.entailment_agent = MockEntailmentAgent()
        consumer.coverage_agent = MockCoverageAgent()
        consumer.decontextualization_agent = MockDecontextualizationAgent()
        MockBlockParser.return_value.extract_aclarai_blocks.return_value = []
        consumer.rabbitmq_manager = MockRabbitMQManager()
        consumer.block_parser = MockBlockParser()
        consumer.concept_processor = MockConceptProcessor()
        consumer.vector_store_manager = MockVectorStoreManager()
        consumer.tool_factory = mock_tool_factory_fixture

        # The patched services are automatically instantiated as mocks.
        # We can assign them explicitly to the consumer instance for clarity and control.
        consumer.graph_service = MockGraphService()
        consumer.markdown_service = MockMarkdownService()

        return consumer


class TestDirtyBlockConsumerEvaluationMethods:
    def test_get_claims_success(self, consumer_instance, mock_graph_manager_fixture):
        source_block_id = "s1"
        expected_claims = [
            {"id": "c1", "text": "Claim 1"},
            {"id": "c2", "text": "Claim 2"},
        ]
        mock_graph_manager_fixture.execute_query.return_value = expected_claims

        claims = consumer_instance._get_claims_for_evaluation(source_block_id)

        assert claims == expected_claims
        expected_query = """
        MATCH (claim:Claim)-[r:ORIGINATES_FROM]->(block:Block {id: $source_block_id})
        WHERE r.entailed_score IS NULL OR claim.needs_reprocessing = true
        RETURN claim.id AS id, claim.text AS text
        """.strip()
        mock_graph_manager_fixture.execute_query.assert_called_once_with(
            expected_query, parameters={"source_block_id": source_block_id}
        )

    def test_get_claims_none_found(self, consumer_instance, mock_graph_manager_fixture):
        source_block_id = "s1"
        mock_graph_manager_fixture.execute_query.return_value = []

        claims = consumer_instance._get_claims_for_evaluation(source_block_id)
        assert claims == []

    @patch("aclarai_core.dirty_block_consumer.logger")
    def test_get_claims_neo4j_exception(
        self, mock_logger, consumer_instance, mock_graph_manager_fixture
    ):
        source_block_id = "s1"
        mock_graph_manager_fixture.execute_query.side_effect = Exception(
            "DB connection error"
        )

        claims = consumer_instance._get_claims_for_evaluation(source_block_id)

        assert claims == []
        assert mock_logger.error.called
        assert (
            "Failed to fetch claims for evaluation" in mock_logger.error.call_args[0][0]
        )


class TestDirtyBlockConsumerProcessBlockIntegration:
    @patch.object(DirtyBlockConsumer, "_read_block_from_file")
    @patch.object(DirtyBlockConsumer, "_sync_block_with_graph")
    @patch.object(DirtyBlockConsumer, "_get_claims_for_evaluation")
    def test_process_block_success_flow_all_agents(
        self,
        mock_get_claims,
        mock_sync_graph,
        mock_read_block,
        consumer_instance,
    ):
        message = {
            "aclarai_id": "s1",
            "file_path": "test_file.md",
            "change_type": "modified",
        }
        mock_block_data = {
            "aclarai_id": "s1",
            "semantic_text": "source text",
            "version": 1,
        }
        mock_claim_data = [{"id": "c1", "text": "claim text"}]

        mock_read_block.return_value = mock_block_data
        mock_sync_graph.return_value = True
        mock_get_claims.return_value = mock_claim_data

        # Mock all three evaluation agents
        consumer_instance.entailment_agent.evaluate_entailment.return_value = (
            0.9,
            "success",
        )
        consumer_instance.coverage_agent.evaluate_coverage.return_value = (
            0.8,
            [],  # omitted_elements
            "success",
        )
        consumer_instance.decontextualization_agent.evaluate_claim_decontextualization.return_value = (
            0.7,
            "success",
        )

        result = consumer_instance._process_dirty_block(message)

        assert result is True
        mock_read_block.assert_called_once_with(Path("test_file.md"), "s1")
        mock_sync_graph.assert_called_once_with(mock_block_data, Path("test_file.md"))
        mock_get_claims.assert_called_once_with("s1")

        # Verify all three agents are called
        consumer_instance.entailment_agent.evaluate_entailment.assert_called_once_with(
            claim_id="c1",
            claim_text="claim text",
            source_id="s1",
            source_text="source text",
        )
        consumer_instance.coverage_agent.evaluate_coverage.assert_called_once_with(
            claim_id="c1",
            claim_text="claim text",
            source_id="s1",
            source_text="source text",
        )
        consumer_instance.decontextualization_agent.evaluate_claim_decontextualization.assert_called_once_with(
            claim_id="c1",
            claim_text="claim text",
            source_id="s1",
            source_text="source text",
        )

        # Verify all three scores are updated in graph
        assert consumer_instance.graph_service.update_relationship_score.call_count == 3
        consumer_instance.graph_service.update_relationship_score.assert_any_call(
            "c1", "s1", "entailed_score", 0.9
        )
        consumer_instance.graph_service.update_relationship_score.assert_any_call(
            "c1", "s1", "coverage_score", 0.8
        )
        consumer_instance.graph_service.update_relationship_score.assert_any_call(
            "c1", "s1", "decontextualization_score", 0.7
        )

        # Verify all three scores are updated in markdown
        expected_filepath = str(Path(consumer_instance.vault_path) / "test_file.md")
        assert consumer_instance.markdown_service.add_or_update_score.call_count == 3
        consumer_instance.markdown_service.add_or_update_score.assert_any_call(
            expected_filepath, "s1", "entailed_score", 0.9
        )
        consumer_instance.markdown_service.add_or_update_score.assert_any_call(
            expected_filepath, "s1", "coverage_score", 0.8
        )
        consumer_instance.markdown_service.add_or_update_score.assert_any_call(
            expected_filepath, "s1", "decontextualization_score", 0.7
        )

    @patch.object(DirtyBlockConsumer, "_read_block_from_file")
    @patch.object(DirtyBlockConsumer, "_sync_block_with_graph")
    @patch.object(DirtyBlockConsumer, "_get_claims_for_evaluation")
    def test_process_block_null_scores_not_written_to_markdown(
        self,
        mock_get_claims,
        mock_sync_graph,
        mock_read_block,
        consumer_instance,
    ):
        """Test that null scores are stored in graph but not written to markdown."""
        message = {
            "aclarai_id": "s1",
            "file_path": "test_file.md",
            "change_type": "modified",
        }
        mock_block_data = {
            "aclarai_id": "s1",
            "semantic_text": "source text",
            "version": 1,
        }
        mock_claim_data = [{"id": "c1", "text": "claim text"}]

        mock_read_block.return_value = mock_block_data
        mock_sync_graph.return_value = True
        mock_get_claims.return_value = mock_claim_data

        # All agents return null scores
        consumer_instance.entailment_agent.evaluate_entailment.return_value = (
            None,
            "LLM error",
        )
        consumer_instance.coverage_agent.evaluate_coverage.return_value = (
            None,
            None,  # omitted_elements
            "LLM timeout",
        )
        consumer_instance.decontextualization_agent.evaluate_claim_decontextualization.return_value = (
            None,
            "Parse error",
        )

        result = consumer_instance._process_dirty_block(message)

        assert result is True

        # Verify all scores are still updated in graph (even null ones)
        assert consumer_instance.graph_service.update_relationship_score.call_count == 3
        consumer_instance.graph_service.update_relationship_score.assert_any_call(
            "c1", "s1", "entailed_score", None
        )
        consumer_instance.graph_service.update_relationship_score.assert_any_call(
            "c1", "s1", "coverage_score", None
        )
        consumer_instance.graph_service.update_relationship_score.assert_any_call(
            "c1", "s1", "decontextualization_score", None
        )

        # Verify NO scores are written to markdown (per architecture requirements)
        consumer_instance.markdown_service.add_or_update_score.assert_not_called()

    @patch.object(DirtyBlockConsumer, "_read_block_from_file")
    @patch.object(DirtyBlockConsumer, "_sync_block_with_graph")
    @patch.object(DirtyBlockConsumer, "_get_claims_for_evaluation")
    def test_process_block_entailment_agent_fails(
        self,
        mock_get_claims,
        mock_sync_graph,
        mock_read_block,
        consumer_instance,
    ):
        """Test backward compatibility - old test behavior still works."""
        message = {
            "aclarai_id": "s1",
            "file_path": "test_file.md",
            "change_type": "modified",
        }
        mock_block_data = {
            "aclarai_id": "s1",
            "semantic_text": "source text",
            "version": 1,
        }
        mock_claim_data = [{"id": "c1", "text": "claim text"}]

        mock_read_block.return_value = mock_block_data
        mock_sync_graph.return_value = True
        mock_get_claims.return_value = mock_claim_data

        # Entailment fails, but other agents succeed
        consumer_instance.entailment_agent.evaluate_entailment.return_value = (
            None,
            "LLM error",
        )
        consumer_instance.coverage_agent.evaluate_coverage.return_value = (
            0.8,
            [],
            "success",
        )
        consumer_instance.decontextualization_agent.evaluate_claim_decontextualization.return_value = (
            0.7,
            "success",
        )

        result = consumer_instance._process_dirty_block(message)
        assert result is True

        # Only the non-null scores should be written to markdown
        assert consumer_instance.markdown_service.add_or_update_score.call_count == 2
        expected_filepath = str(Path(consumer_instance.vault_path) / "test_file.md")
        consumer_instance.markdown_service.add_or_update_score.assert_any_call(
            expected_filepath, "s1", "coverage_score", 0.8
        )
        consumer_instance.markdown_service.add_or_update_score.assert_any_call(
            expected_filepath, "s1", "decontextualization_score", 0.7
        )

    @patch.object(DirtyBlockConsumer, "_read_block_from_file")
    @patch.object(DirtyBlockConsumer, "_sync_block_with_graph")
    @patch.object(DirtyBlockConsumer, "_get_claims_for_evaluation")
    def test_process_block_no_claims_for_evaluation(
        self,
        mock_get_claims,
        mock_sync_graph,
        mock_read_block,
        consumer_instance,
    ):
        message = {
            "aclarai_id": "s1",
            "file_path": "test_file.md",
            "change_type": "modified",
        }
        mock_block_data = {
            "aclarai_id": "s1",
            "semantic_text": "source text",
            "version": 1,
        }

        mock_read_block.return_value = mock_block_data
        mock_sync_graph.return_value = True
        mock_get_claims.return_value = []

        result = consumer_instance._process_dirty_block(message)
        assert result is True
        consumer_instance.entailment_agent.evaluate_entailment.assert_not_called()
        consumer_instance.graph_service.update_relationship_score.assert_not_called()
        consumer_instance.markdown_service.add_or_update_score.assert_not_called()

    def test_process_block_deleted_message(self, consumer_instance):
        message = {
            "aclarai_id": "s1",
            "file_path": "test_file.md",
            "change_type": "deleted",
        }

        with (
            patch.object(consumer_instance, "_read_block_from_file") as mock_read,
            patch.object(consumer_instance, "_sync_block_with_graph") as mock_sync,
        ):
            result = consumer_instance._process_dirty_block(message)
            assert result is True
            mock_read.assert_not_called()
            mock_sync.assert_not_called()

    @patch.object(DirtyBlockConsumer, "_read_block_from_file")
    def test_process_block_read_file_fails(self, mock_read_block, consumer_instance):
        message = {
            "aclarai_id": "s1",
            "file_path": "test_file.md",
            "change_type": "modified",
        }
        mock_read_block.return_value = None

        result = consumer_instance._process_dirty_block(message)
        assert result is False

    @patch.object(DirtyBlockConsumer, "_read_block_from_file")
    @patch.object(DirtyBlockConsumer, "_sync_block_with_graph")
    def test_process_block_sync_graph_fails(
        self, mock_sync_graph, mock_read_block, consumer_instance
    ):
        message = {
            "aclarai_id": "s1",
            "file_path": "test_file.md",
            "change_type": "modified",
        }
        mock_block_data = {
            "aclarai_id": "s1",
            "semantic_text": "source text",
            "version": 1,
        }

        mock_read_block.return_value = mock_block_data
        mock_sync_graph.return_value = False

        result = consumer_instance._process_dirty_block(message)
        assert result is False

    @patch.object(DirtyBlockConsumer, "_read_block_from_file")
    @patch.object(DirtyBlockConsumer, "_sync_block_with_graph")
    @patch("aclarai_core.dirty_block_consumer.logger")
    def test_process_block_no_llm_skips_evaluation(
        self,
        mock_logger,
        mock_sync_graph,
        mock_read_block,
        consumer_instance,
    ):
        # Save original agents
        original_entailment_agent = consumer_instance.entailment_agent
        original_coverage_agent = consumer_instance.coverage_agent
        original_decontextualization_agent = consumer_instance.decontextualization_agent

        # Set all agents to None to simulate missing LLM
        consumer_instance.entailment_agent = None
        consumer_instance.coverage_agent = None
        consumer_instance.decontextualization_agent = None

        message = {
            "aclarai_id": "s1",
            "file_path": "test_file.md",
            "change_type": "modified",
        }
        mock_block_data = {
            "aclarai_id": "s1",
            "semantic_text": "source text",
            "version": 1,
        }
        mock_read_block.return_value = mock_block_data
        mock_sync_graph.return_value = True

        result = consumer_instance._process_dirty_block(message)

        assert result is True
        mock_logger.error.assert_called_with(
            # THIS IS THE ONLY LINE THAT CHANGES
            "Evaluation agents not initialized (EntailmentAgent, CoverageAgent, DecontextualizationAgent), skipping evaluation and concept processing for block s1.",
            extra={"aclarai_id": "s1"},
        )
        # Verify no agent methods were called
        original_entailment_agent.evaluate_entailment.assert_not_called()
        original_coverage_agent.evaluate_coverage.assert_not_called()
        original_decontextualization_agent.evaluate_claim_decontextualization.assert_not_called()


@pytest.mark.integration
class TestConsumerEvaluationIntegration:
    """
    End-to-end integration tests for the consumer's evaluation workflow.
    Requires live RabbitMQ and Neo4j services.
    """

    @pytest.fixture(scope="class")
    def integration_neo4j_manager(self):
        """Fixture to set up a connection to a real Neo4j database for testing."""
        if not os.getenv("NEO4J_PASSWORD"):
            pytest.skip("NEO4J_PASSWORD not set for integration tests.")

        from aclarai_shared import load_config

        config = load_config(validate=True)
        manager = Neo4jGraphManager(config=config)
        manager.setup_schema()
        # Clean up any existing test data
        with manager.session() as session:
            session.run(
                "MATCH (n) WHERE n.id STARTS WITH 'eval_integ_test_' DETACH DELETE n",
                allow_dangerous_operations=True,
            )
        yield manager
        # Clean up after tests
        with manager.session() as session:
            session.run(
                "MATCH (n) WHERE n.id STARTS WITH 'eval_integ_test_' DETACH DELETE n",
                allow_dangerous_operations=True,
            )
        manager.close()

    @pytest.fixture
    def rabbitmq_connection(self):
        """Fixture for RabbitMQ connection."""
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host="localhost", port=5672)
            )
            channel = connection.channel()
            queue_name = "test_aclarai_dirty_blocks_evaluation"
            channel.queue_declare(queue=queue_name, durable=True)
            yield connection, channel, queue_name
            # Cleanup
            channel.queue_purge(queue=queue_name)
            connection.close()
        except pika.exceptions.AMQPConnectionError:
            pytest.skip("RabbitMQ not available for integration testing")

    @pytest.fixture
    def temp_markdown_file(self):
        """Creates a temporary markdown file with a versioned block."""
        content = """# Evaluation Integration Test
A claim to be evaluated. <!-- aclarai:id=eval_integ_test_block_1 ver=1 -->
^eval_integ_test_block_1
"""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, encoding="utf-8", suffix=".md"
        ) as tmp_file:
            tmp_file.write(content)
            filepath = tmp_file.name
        yield Path(filepath)
        os.unlink(filepath)

    def test_consumer_triggers_evaluations_and_persists_scores(
        self, integration_neo4j_manager, rabbitmq_connection, temp_markdown_file
    ):
        """
        Tests the full reactive flow: RabbitMQ message -> Consumer -> Agent evaluation -> Neo4j & Markdown updates.
        """
        # --- ARRANGE ---
        connection, channel, queue_name = rabbitmq_connection
        block_id = "eval_integ_test_block_1"
        claim_id = "eval_integ_test_claim_1"

        # 1. Pre-populate Neo4j with a Claim and Block needing evaluation
        with integration_neo4j_manager.session() as session:
            session.run(
                """
                MERGE (c:Claim {id: $claim_id, text: 'A claim to be evaluated.'})
                MERGE (b:Block {id: $block_id, text: 'A claim to be evaluated.', hash: 'initial_hash', version: 1})
                MERGE (c)-[r:ORIGINATES_FROM]->(b)
                """,
                claim_id=claim_id,
                block_id=block_id,
            )

        # 2. Publish a "dirty block" message to RabbitMQ
        message = {
            "aclarai_id": block_id,
            "file_path": str(temp_markdown_file),
            "change_type": "modified",
        }
        channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=2),
        )

        # 3. Instantiate the consumer and mock the LLM-based agents
        with patch("aclarai_core.dirty_block_consumer.load_config") as mock_load_config:
            mock_config = MagicMock()
            mock_config.vault_path = temp_markdown_file.parent
            mock_config.rabbitmq_host = "localhost"
            mock_config.vault_watcher.queue_name = queue_name
            # Mock other necessary config attributes
            mock_config.dict.return_value = {"tools": {}}
            mock_config.processing.retries = {"max_attempts": 1}
            mock_load_config.return_value = mock_config

            consumer = DirtyBlockConsumer(config=mock_config)

            # Mock the agent evaluations to return predictable scores
            consumer.entailment_agent.evaluate_entailment.return_value = (
                0.91,
                "success",
            )
            consumer.coverage_agent.evaluate_coverage.return_value = (
                0.77,
                [],
                "success",
            )
            consumer.decontextualization_agent.evaluate_claim_decontextualization.return_value = (
                0.88,
                "success",
            )

            # 4. Run the consumer in a thread to process the message
            processed = threading.Event()
            original_process_message = consumer._process_dirty_block

            def mock_process_message(msg):
                original_process_message(msg)
                processed.set()

            consumer._process_dirty_block = mock_process_message

            consumer_thread = threading.Thread(target=consumer.start_consuming)
            consumer_thread.daemon = True
            consumer_thread.start()

            # --- ACT ---
            # Wait for the message to be processed
            if not processed.wait(timeout=15):
                pytest.fail(
                    "Message was not processed by the consumer within the timeout."
                )

            # Stop the consumer
            consumer.stop_consuming()
            consumer_thread.join(timeout=5)

        # --- ASSERT ---
        # 1. Verify Neo4j was updated correctly
        with integration_neo4j_manager.session() as session:
            result = session.run(
                """
                MATCH (:Claim {id: $claim_id})-[r:ORIGINATES_FROM]->(:Block {id: $block_id})
                RETURN r.entailed_score AS entailed, r.coverage_score AS coverage, r.decontextualization_score AS decontext
                """,
                claim_id=claim_id,
                block_id=block_id,
            )
            record = result.single()
            assert record is not None, "Relationship not found in Neo4j"
            assert record["entailed"] == 0.91
            assert record["coverage"] == 0.77
            assert record["decontext"] == 0.88

        # 2. Verify the Markdown file was updated correctly
        updated_content = temp_markdown_file.read_text(encoding="utf-8")
        assert "<!-- aclarai:entailed_score=0.91 -->" in updated_content
        assert "<!-- aclarai:coverage_score=0.77 -->" in updated_content
        assert "<!-- aclarai:decontextualization_score=0.88 -->" in updated_content

        # Version should be incremented for each score update
        assert "ver=4" in updated_content  # Original ver=1, +1 for each of 3 scores
