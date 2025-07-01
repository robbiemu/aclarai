from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

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
    # Provide a dictionary for the ToolFactory via the mocked .dict() method
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
    return MagicMock(spec=Neo4jGraphManager)


@pytest.fixture
def mock_llm_fixture():
    return MagicMock(spec=LlamaLLM)


@pytest.fixture
def mock_tool_factory_fixture():
    return MagicMock(spec=ToolFactory)


@pytest.fixture
@patch("aclarai_core.dirty_block_consumer.RabbitMQManager")
@patch("aclarai_core.dirty_block_consumer.BlockParser")
@patch("aclarai_core.dirty_block_consumer.ConceptProcessor")
@patch("aclarai_core.dirty_block_consumer.EntailmentAgent")
@patch(
    "aclarai_shared.tools.vector_store_manager.aclaraiVectorStoreManager"
)  # Patch the concrete manager
def consumer_instance(
    MockVectorStoreManager,
    MockEntailmentAgent,
    MockConceptProcessor,
    MockBlockParser,
    MockRabbitMQManager,
    mock_consumer_config,
    mock_graph_manager_fixture,
    mock_llm_fixture,
    mock_tool_factory_fixture,
):
    with patch.object(
        DirtyBlockConsumer, "_initialize_llm", return_value=mock_llm_fixture
    ):
        consumer = DirtyBlockConsumer(config=mock_consumer_config)
        consumer.graph_manager = mock_graph_manager_fixture
        consumer.entailment_agent = MockEntailmentAgent()
        # Fix for TypeError: make extract_aclarai_blocks return an iterable
        MockBlockParser.return_value.extract_aclarai_blocks.return_value = []
        consumer.rabbitmq_manager = MockRabbitMQManager()
        consumer.block_parser = MockBlockParser()
        consumer.concept_processor = MockConceptProcessor()
        consumer.vector_store_manager = MockVectorStoreManager()
        consumer.tool_factory = mock_tool_factory_fixture
        return consumer


class TestDirtyBlockConsumerEvaluationMethods:
    def test_update_neo4j_success_with_score(
        self, consumer_instance, mock_graph_manager_fixture
    ):
        claim_id, source_id, score = "c1", "s1", 0.95
        mock_graph_manager_fixture.execute_query.return_value = [
            {"updatedScore": score}
        ]

        consumer_instance._update_entailment_score_in_neo4j(claim_id, source_id, score)

        expected_query = """
        MATCH (claim:Claim {id: $claim_id})-[rel:ORIGINATES_FROM]->(block:Block {id: $source_id})
        SET rel.entailed_score = $entailed_score
        RETURN claim.id AS claimId, rel.entailed_score AS updatedScore
        """.strip()
        params = {"claim_id": claim_id, "source_id": source_id, "entailed_score": score}
        mock_graph_manager_fixture.execute_query.assert_called_once_with(
            expected_query, parameters=params
        )

    def test_update_neo4j_score_is_none(
        self, consumer_instance, mock_graph_manager_fixture
    ):
        claim_id, source_id = "c1", "s1"
        mock_graph_manager_fixture.execute_query.return_value = [{"updatedScore": None}]

        consumer_instance._update_entailment_score_in_neo4j(claim_id, source_id, None)

        params = {"claim_id": claim_id, "source_id": source_id, "entailed_score": None}
        mock_graph_manager_fixture.execute_query.assert_called_with(
            ANY, parameters=params
        )

    @patch("aclarai_core.dirty_block_consumer.logger")
    def test_update_neo4j_no_results(
        self, mock_logger, consumer_instance, mock_graph_manager_fixture
    ):
        claim_id, source_id, score = "c1", "s1", 0.8
        mock_graph_manager_fixture.execute_query.return_value = []

        consumer_instance._update_entailment_score_in_neo4j(claim_id, source_id, score)

        assert mock_logger.warning.called
        assert "did not return results" in mock_logger.warning.call_args[0][0]

    @patch("aclarai_core.dirty_block_consumer.logger")
    def test_update_neo4j_exception(
        self, mock_logger, consumer_instance, mock_graph_manager_fixture
    ):
        claim_id, source_id, score = "c1", "s1", 0.8
        mock_graph_manager_fixture.execute_query.side_effect = Exception("DB error")

        consumer_instance._update_entailment_score_in_neo4j(claim_id, source_id, score)

        assert mock_logger.error.called
        assert (
            "Failed to update entailment score in Neo4j"
            in mock_logger.error.call_args[0][0]
        )

    @patch("aclarai_core.dirty_block_consumer.write_file_atomically")
    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    def test_update_markdown_new_score(
        self,
        mock_exists,
        mock_read_text,
        mock_write_atomic,
        consumer_instance,
    ):
        source_filepath, source_block_id, score = "/fake/vault/test.md", "b1", 0.75
        mock_exists.return_value = True
        initial_content = f"Line 1\n<!-- aclarai:id={source_block_id} ver=1 -->\nLine 3"
        mock_read_text.return_value = initial_content

        consumer_instance._update_markdown_with_entailment_score(
            source_filepath, source_block_id, score
        )

        expected_content = f"Line 1\n<!-- aclarai:id={source_block_id} ver=2 -->\n<!-- aclarai:entailed_score={score:.2f} -->\nLine 3"
        mock_write_atomic.assert_called_once_with(
            Path(source_filepath), expected_content
        )

    @patch("aclarai_core.dirty_block_consumer.write_file_atomically")
    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    def test_update_markdown_update_existing_score(
        self,
        mock_exists,
        mock_read_text,
        mock_write_atomic,
        consumer_instance,
    ):
        source_filepath, source_block_id, score = "/fake/vault/test.md", "b1", 0.88
        mock_exists.return_value = True
        initial_content = f"Line 1\n<!-- aclarai:id={source_block_id} ver=2 -->\n<!-- aclarai:entailed_score=0.50 -->\nLine 4"
        mock_read_text.return_value = initial_content

        consumer_instance._update_markdown_with_entailment_score(
            source_filepath, source_block_id, score
        )

        expected_content = f"Line 1\n<!-- aclarai:id={source_block_id} ver=3 -->\n<!-- aclarai:entailed_score={score:.2f} -->\nLine 4"
        mock_write_atomic.assert_called_once_with(
            Path(source_filepath), expected_content
        )

    @patch("aclarai_core.dirty_block_consumer.write_file_atomically")
    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("aclarai_core.dirty_block_consumer.logger")
    def test_update_markdown_block_id_not_found(
        self,
        mock_logger,
        mock_exists,
        mock_read_text,
        mock_write_atomic,
        consumer_instance,
    ):
        source_filepath, source_block_id, score = (
            "/fake/vault/test.md",
            "b_unknown",
            0.75,
        )
        mock_exists.return_value = True
        mock_read_text.return_value = "Some content without the target block ID."

        consumer_instance._update_markdown_with_entailment_score(
            source_filepath, source_block_id, score
        )

        mock_write_atomic.assert_not_called()
        assert mock_logger.warning.called
        assert (
            "Could not find aclarai:id comment" in mock_logger.warning.call_args[0][0]
        )

    @patch("aclarai_core.dirty_block_consumer.write_file_atomically")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("aclarai_core.dirty_block_consumer.logger")
    def test_update_markdown_file_not_found(
        self, mock_logger, mock_exists, mock_write_atomic, consumer_instance
    ):
        source_filepath, source_block_id, score = (
            "/fake/vault/nonexistent.md",
            "b1",
            0.75,
        )
        mock_exists.return_value = False

        consumer_instance._update_markdown_with_entailment_score(
            source_filepath, source_block_id, score
        )

        mock_write_atomic.assert_not_called()
        assert mock_logger.error.called
        assert "Markdown file not found" in mock_logger.error.call_args[0][0]

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
    @patch.object(DirtyBlockConsumer, "_update_entailment_score_in_neo4j")
    @patch.object(DirtyBlockConsumer, "_update_markdown_with_entailment_score")
    def test_process_block_success_flow(
        self,
        mock_update_markdown,
        mock_update_neo4j,
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
        mock_update_neo4j.assert_called_once_with("c1", "s1", 0.9)
        expected_filepath = str(Path(consumer_instance.vault_path) / "test_file.md")
        mock_update_markdown.assert_called_once_with(expected_filepath, "s1", 0.9)

    @patch.object(DirtyBlockConsumer, "_read_block_from_file")
    @patch.object(DirtyBlockConsumer, "_sync_block_with_graph")
    @patch.object(DirtyBlockConsumer, "_get_claims_for_evaluation")
    @patch.object(DirtyBlockConsumer, "_update_entailment_score_in_neo4j")
    @patch.object(DirtyBlockConsumer, "_update_markdown_with_entailment_score")
    def test_process_block_entailment_agent_fails(
        self,
        mock_update_markdown,
        mock_update_neo4j,
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
        mock_update_neo4j.assert_called_once_with("c1", "s1", None)
        mock_update_markdown.assert_not_called()

    @patch.object(DirtyBlockConsumer, "_read_block_from_file")
    @patch.object(DirtyBlockConsumer, "_sync_block_with_graph")
    @patch.object(DirtyBlockConsumer, "_get_claims_for_evaluation")
    @patch.object(DirtyBlockConsumer, "_update_entailment_score_in_neo4j")
    @patch.object(DirtyBlockConsumer, "_update_markdown_with_entailment_score")
    def test_process_block_no_claims_for_evaluation(
        self,
        mock_update_markdown,
        mock_update_neo4j,
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
        mock_update_neo4j.assert_not_called()
        mock_update_markdown.assert_not_called()

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
