from dataclasses import asdict
import json
import os
import tempfile
import time
import threading
from pathlib import Path
import pika
import pytest
from unittest.mock import MagicMock, patch


from aclarai_shared import load_config
from aclarai_core.dirty_block_consumer import DirtyBlockConsumer
from aclarai_shared.config import (
    aclaraiConfig,
    ModelConfig,
    ClaimifyModelConfig,
    ProcessingConfig,
    EmbeddingConfig,
    VaultWatcherConfig,
    DatabaseConfig,
    LLMConfig,
    PathsConfig,
    SchedulerConfig,
    ThresholdConfig,
    ConceptsConfig,
    NounPhraseExtractionConfig,
    ConceptSummariesConfig,
    SubjectSummariesConfig,
    WindowConfig,
)
from aclarai_shared.graph.neo4j_manager import Neo4jGraphManager
from aclarai_shared.tools.factory import ToolFactory
from llama_index.core.llms.llm import LLM as LlamaLLM
import yaml


@pytest.fixture
def mock_consumer_config():
    # Alternative approach: Create a real dataclass instance and selectively mock parts
    try:
        # Try to create a real config instance with default values
        config = aclaraiConfig()

        # Now mock just the parts we need to control
        config.model = MagicMock()
        config.model.claimify = MagicMock()
        config.model.claimify.entailment = "mock_llm"
        config.model.claimify.default = "mock_llm"

        config.processing = MagicMock()
        config.processing.temperature = 0.1
        config.processing.max_tokens = 1000

        config.vault_watcher = MagicMock()
        config.vault_watcher.queue_name = "mock_dirty_blocks_queue"

        config.neo4j = MagicMock()
        config.neo4j.get_neo4j_bolt_url.return_value = "bolt://neo4j_mock_host:7687"

        config.vault_path = "/fake/vault"
        config.rabbitmq_host = "fake_host"
        config.rabbitmq_port = 5672
        config.rabbitmq_user = "user"
        config.rabbitmq_password = "password"

        # Mock the nested configs that we need to control
        config.llm = MagicMock()
        config.embedding = MagicMock()
        config.paths = MagicMock()
        config.scheduler = MagicMock()
        config.threshold = MagicMock()
        config.concepts = MagicMock()
        config.noun_phrase_extraction = MagicMock()
        config.concept_summaries = MagicMock()
        config.subject_summaries = MagicMock()
        config.window = {"claimify": MagicMock()}

    except Exception:
        # Fallback: create a completely mocked config if real instantiation fails
        config = MagicMock()

        # Set up all the attributes we need
        config.model = MagicMock()
        config.model.claimify = MagicMock()
        config.model.claimify.entailment = "mock_llm"
        config.model.claimify.default = "mock_llm"

        config.processing = MagicMock()
        config.processing.temperature = 0.1
        config.processing.max_tokens = 1000

        config.vault_watcher = MagicMock()
        config.vault_watcher.queue_name = "mock_dirty_blocks_queue"

        config.neo4j = MagicMock()
        config.neo4j.get_neo4j_bolt_url.return_value = "bolt://neo4j_mock_host:7687"

        config.vault_path = "/fake/vault"
        config.rabbitmq_host = "fake_host"
        config.rabbitmq_port = 5672
        config.rabbitmq_user = "user"
        config.rabbitmq_password = "password"

        # Add other necessary nested mocks
        config.llm = MagicMock()
        config.embedding = MagicMock()
        config.paths = MagicMock()
        config.scheduler = MagicMock()
        config.threshold = MagicMock()
        config.concepts = MagicMock()
        config.noun_phrase_extraction = MagicMock()
        config.concept_summaries = MagicMock()
        config.subject_summaries = MagicMock()
        config.window = {"claimify": MagicMock()}

    def mock_asdict(obj):
        if isinstance(obj, MagicMock):
            return {"tools": {"agent_tool_mappings": {"entailment_agent": []}}}
        from dataclasses import asdict as real_asdict

        return real_asdict(obj)

    with patch("aclarai_core.dirty_block_consumer.asdict", side_effect=mock_asdict):
        yield config


@pytest.fixture
def mock_graph_manager_fixture():
    return MagicMock(spec=Neo4jGraphManager)


@pytest.fixture
def mock_llm_fixture():
    return MagicMock(spec=LlamaLLM)


@pytest.fixture
def mock_tool_factory_fixture():
    return MagicMock(spec=ToolFactory)


@pytest.fixture
@patch("aclarai_core.dirty_block_consumer.Neo4jGraphManager")  # 1. _MockNeo4jManager
@patch(
    "aclarai_core.dirty_block_consumer.MarkdownUpdaterService"
)  # 2.  MockMarkdownService
@patch(
    "aclarai_core.dirty_block_consumer.ClaimEvaluationGraphService"
)  # 3.  MockGraphService
@patch("aclarai_core.dirty_block_consumer.RabbitMQManager")  # 4.  MockRabbitMQManager
@patch("aclarai_core.dirty_block_consumer.BlockParser")  # 5.  MockBlockParser
@patch("aclarai_core.dirty_block_consumer.ConceptProcessor")  # 6.  MockConceptProcessor
@patch(
    "aclarai_core.dirty_block_consumer.DecontextualizationAgent"
)  # 7.  MockDecontextualizationAgent
@patch("aclarai_core.dirty_block_consumer.CoverageAgent")  # 8.  MockCoverageAgent
@patch("aclarai_core.dirty_block_consumer.EntailmentAgent")  # 9.  MockEntailmentAgent
@patch(
    "aclarai_shared.tools.vector_store_manager.aclaraiVectorStoreManager"
)  # 10.  MockVectorStoreManager
def consumer_instance(
    MockVectorStoreManager,  # 10. (last patch applied = first parameter)
    MockEntailmentAgent,  # 9.
    MockCoverageAgent,  # 8.
    MockDecontextualizationAgent,  # 7.
    MockConceptProcessor,  # 6.
    MockBlockParser,  # 5.
    MockRabbitMQManager,  # 4.
    MockGraphService,  # 3.
    MockMarkdownService,  # 2.
    _MockNeo4jManager,  # 1. (first patch applied = last parameter)
    # Fixtures come after all the patched arguments
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

        consumer.graph_manager = mock_graph_manager_fixture
        consumer.entailment_agent = MockEntailmentAgent()
        consumer.coverage_agent = MockCoverageAgent()
        consumer.decontextualization_agent = MockDecontextualizationAgent()
        MockBlockParser.return_value.extract_aclarai_blocks.return_value = []
        consumer.rabbitmq_manager = MockRabbitMQManager()
        consumer.block_parser = MockBlockParser()
        consumer.concept_processor = MockConceptProcessor()
        consumer.vector_store_manager = MockVectorStoreManager()
        consumer.tool_factory = mock_tool_factory_fixture
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
            "content_hash": "new_hash",
        }
        mock_claim_data = [{"id": "c1", "text": "claim text"}]

        mock_read_block.return_value = mock_block_data
        mock_sync_graph.return_value = True
        mock_get_claims.return_value = mock_claim_data

        consumer_instance.entailment_agent.evaluate_entailment.return_value = (
            0.9,
            "success",
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
        mock_read_block.assert_called_once_with(Path("test_file.md"), "s1")
        mock_sync_graph.assert_called_once_with(mock_block_data, Path("test_file.md"))
        mock_get_claims.assert_called_once_with("s1")

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
        message = {
            "aclarai_id": "s1",
            "file_path": "test_file.md",
            "change_type": "modified",
        }
        mock_block_data = {
            "aclarai_id": "s1",
            "semantic_text": "source text",
            "version": 1,
            "content_hash": "new_hash",
        }
        mock_claim_data = [{"id": "c1", "text": "claim text"}]

        mock_read_block.return_value = mock_block_data
        mock_sync_graph.return_value = True
        mock_get_claims.return_value = mock_claim_data

        consumer_instance.entailment_agent.evaluate_entailment.return_value = (
            None,
            "LLM error",
        )
        consumer_instance.coverage_agent.evaluate_coverage.return_value = (
            None,
            None,
            "LLM timeout",
        )
        consumer_instance.decontextualization_agent.evaluate_claim_decontextualization.return_value = (
            None,
            "Parse error",
        )

        result = consumer_instance._process_dirty_block(message)

        assert result is True

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
        message = {
            "aclarai_id": "s1",
            "file_path": "test_file.md",
            "change_type": "modified",
        }
        mock_block_data = {
            "aclarai_id": "s1",
            "semantic_text": "source text",
            "version": 1,
            "content_hash": "new_hash",
        }
        mock_claim_data = [{"id": "c1", "text": "claim text"}]

        mock_read_block.return_value = mock_block_data
        mock_sync_graph.return_value = True
        mock_get_claims.return_value = mock_claim_data

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
            "content_hash": "new_hash",
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
            "content_hash": "new_hash",
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
            "content_hash": "new_hash",
        }
        mock_read_block.return_value = mock_block_data
        mock_sync_graph.return_value = True

        result = consumer_instance._process_dirty_block(message)

        assert result is True
        mock_logger.error.assert_called_with(
            "Evaluation agents not initialized (EntailmentAgent, CoverageAgent, DecontextualizationAgent), skipping evaluation and concept processing for block s1.",
            extra={"aclarai_id": "s1"},
        )
        # Verify no agent methods were called
        assert not original_entailment_agent.evaluate_entailment.called
        assert not original_coverage_agent.evaluate_coverage.called
        assert not original_decontextualization_agent.evaluate_claim_decontextualization.called


@pytest.mark.integration
class TestConsumerEvaluationIntegration:
    """
    End-to-end integration tests for the consumer's evaluation workflow.
    Requires live RabbitMQ and Neo4j services.
    """

    @pytest.fixture(scope="class")
    def integration_neo4j_manager(self):
        """Fixture to set up a connection to a real Neo4j database using app config."""
        # Load the configuration as the single source of truth.
        config = load_config(validate=True)

        if False:
            config_dict = asdict(config)

            # 2. Dump that dictionary to YAML (or JSON, or whatever you want)
            print("\n--- Dumping dictionary to YAML ---")
            final_settings_yaml = yaml.dump(
                config_dict, sort_keys=False, default_flow_style=False
            )

            # 3. Print the result
            print("\n--- Final Merged Configuration ---")
            print(final_settings_yaml)

        # Check for the necessary configuration from the config object.
        try:
            if not config.neo4j.password:
                pytest.skip("Neo4j password not set in the application configuration.")
        except AttributeError:
            pytest.skip("Neo4j configuration section is missing or incomplete.")

        # Proceed with the real manager instance.
        manager = Neo4jGraphManager(config=config)
        manager.setup_schema()
        # Clean up any existing test data
        with manager.session() as session:
            session.run(
                "MATCH (n) WHERE n.id STARTS WITH 'eval_integ_test_' DETACH DELETE n"
            )
        yield manager
        # Clean up after tests
        with manager.session() as session:
            session.run(
                "MATCH (n) WHERE n.id STARTS WITH 'eval_integ_test_' DETACH DELETE n"
            )
        manager.close()

    @pytest.fixture
    def rabbitmq_setup(self):
        """
        Fixture to set up and tear down the RabbitMQ test queue.
        It connects, ensures the queue exists and is empty, then disconnects.
        It yields only the queue name for the test to use.
        """
        try:
            config = load_config(validate=True)
            connection_params = pika.ConnectionParameters(
                host=config.rabbitmq_host,
                port=config.rabbitmq_port,
                credentials=pika.PlainCredentials(
                    config.rabbitmq_user, config.rabbitmq_password
                ),
            )
            connection = pika.BlockingConnection(connection_params)
            channel = connection.channel()
            queue_name = "test_aclarai_dirty_blocks_evaluation"
            channel.queue_declare(queue=queue_name, durable=True)
            channel.queue_purge(queue=queue_name)  # Ensure queue is empty before test
            connection.close()
            yield queue_name
        except pika.exceptions.AMQPConnectionError as e:
            pytest.skip(f"RabbitMQ not available for integration testing: {e}")

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
        self, integration_neo4j_manager, rabbitmq_setup, temp_markdown_file
    ):
        """
        Tests the full reactive flow: RabbitMQ message -> Consumer -> Agent evaluation -> Neo4j & Markdown updates.
        """
        # --- ARRANGE ---
        # The fixture now provides just the queue name
        queue_name = rabbitmq_setup
        block_id = "eval_integ_test_block_1"
        claim_id = "eval_integ_test_claim_1"

        # 1. Pre-populate the REAL Neo4j database (no change here).
        with integration_neo4j_manager.session() as session:
            session.run(
                """
                MERGE (c:Claim {id: $claim_id, text: 'A claim to be evaluated.'})
                MERGE (b:Block {id: $block_id, text: 'A claim to be evaluated.', hash: 'initial_hash', version: 1})
                MERGE (c)-[r:ORIGINATES_FROM]->(b)
                SET r.entailed_score = NULL
                """,
                claim_id=claim_id,
                block_id=block_id,
            )

        # 2. Instantiate the real consumer (no change here).
        config = load_config(validate=True)
        config.vault_path = str(temp_markdown_file.parent)
        config.vault_watcher.queue_name = queue_name  # Use the test queue name
        consumer = DirtyBlockConsumer(config=config)

        # 3. Mock external dependencies (no change here).
        consumer.entailment_agent = MagicMock(
            evaluate_entailment=MagicMock(return_value=(0.91, "success"))
        )
        consumer.coverage_agent = MagicMock(
            evaluate_coverage=MagicMock(return_value=(0.77, [], "success"))
        )
        consumer.decontextualization_agent = MagicMock(
            evaluate_claim_decontextualization=MagicMock(return_value=(0.88, "success"))
        )
        consumer.concept_processor = MagicMock()

        # 4. Run the consumer in a thread, using an Event to signal completion.
        processed = threading.Event()
        original_process_message = consumer._process_dirty_block

        def mock_process_message_wrapper(msg):
            # This wrapper calls the real method and then sets the event
            original_process_message(msg)
            processed.set()

        consumer._process_dirty_block = mock_process_message_wrapper

        consumer_thread = threading.Thread(target=consumer.start_consuming)
        consumer_thread.daemon = True
        consumer_thread.start()
        time.sleep(1)  # Give 1 second for consumer  to connect and start listening

        # 5. Publish the message using a short-lived, self-contained connection.
        # This completely avoids the fixture lifecycle race condition.
        publisher_connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=config.rabbitmq_host,
                port=config.rabbitmq_port,
                credentials=pika.PlainCredentials(
                    config.rabbitmq_user, config.rabbitmq_password
                ),
            )
        )
        publisher_channel = publisher_connection.channel()
        message = {
            "aclarai_id": block_id,
            "file_path": str(temp_markdown_file),
            "change_type": "modified",
        }
        publisher_channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=2),  # Durable message
        )
        publisher_connection.close()  # Connection can be closed immediately.

        # --- ACT ---
        # Wait for the message to be processed or timeout.
        if not processed.wait(timeout=15):
            # If it fails, stop the consumer gracefully before failing the test.
            consumer.stop_consuming()
            consumer_thread.join(timeout=5)
            pytest.fail("Message was not processed by the consumer within the timeout.")

        # --- SHUTDOWN ---
        # Use the new graceful shutdown mechanism.
        consumer.stop_consuming()
        consumer_thread.join(timeout=5)

        assert not consumer_thread.is_alive(), (
            "Consumer thread did not shut down correctly."
        )

        # --- ASSERT ---
        # (No changes to the assertion block)
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

        updated_content = temp_markdown_file.read_text(encoding="utf-8")
        assert "<!-- aclarai:entailed_score=0.91 -->" in updated_content
        assert "<!-- aclarai:coverage_score=0.77 -->" in updated_content
        assert "<!-- aclarai:decontextualization_score=0.88 -->" in updated_content
        assert "ver=4" in updated_content
