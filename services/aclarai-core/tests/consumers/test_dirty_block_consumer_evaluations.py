from pathlib import Path
from unittest.mock import MagicMock, patch

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
@patch("aclarai_core.dirty_block_consumer.EntailmentAgent")
@patch("aclarai_shared.tools.vector_store_manager.aclaraiVectorStoreManager")
def consumer_instance(
    MockVectorStoreManager,
    MockEntailmentAgent,
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
    def test_process_block_success_flow(
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
        consumer_instance.entailment_agent.evaluate_entailment.return_value = (
            0.9,
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
        consumer_instance.graph_service.update_relationship_score.assert_called_once_with(
            "c1", "s1", "entailed_score", 0.9
        )
        expected_filepath = str(Path(consumer_instance.vault_path) / "test_file.md")
        consumer_instance.markdown_service.add_or_update_score.assert_called_once_with(
            expected_filepath, "s1", "entailed_score", 0.9
        )

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
        }
        mock_claim_data = [{"id": "c1", "text": "claim text"}]

        mock_read_block.return_value = mock_block_data
        mock_sync_graph.return_value = True
        mock_get_claims.return_value = mock_claim_data
        consumer_instance.entailment_agent.evaluate_entailment.return_value = (
            None,
            "LLM error",
        )

        result = consumer_instance._process_dirty_block(message)
        assert result is True
        consumer_instance.graph_service.update_relationship_score.assert_called_once_with(
            "c1", "s1", "entailed_score", None
        )
        consumer_instance.markdown_service.add_or_update_score.assert_not_called()

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
        original_agent = consumer_instance.entailment_agent
        consumer_instance.entailment_agent = None

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
            "EntailmentAgent not initialized, skipping evaluation for block s1.",
            extra={"aclarai_id": "s1"},
        )
        original_agent.evaluate_entailment.assert_not_called()
