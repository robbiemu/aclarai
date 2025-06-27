"""
Tests for the decontextualization evaluation workflow, including the agent,
Neo4j storage, and Markdown updates.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

# Modules to test
from aclarai_core.agents.decontextualization_agent import DecontextualizationAgent
from aclarai_core.graph.claim_evaluation_graph_service import (
    ClaimEvaluationGraphService,
)
from aclarai_core.markdown.markdown_updater_service import MarkdownUpdaterService
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.tools import BaseTool, ToolMetadata
from neo4j import Driver


# Mock objects for dependencies
class MockLLM:
    """Mocks the LlamaIndex LLM for predictable responses."""

    def __init__(
        self, response_text: str = "0.85", throw_exception: Optional[Exception] = None
    ):
        self.response_text = response_text
        self.throw_exception = throw_exception
        self.chat_messages = []  # To inspect prompts

    async def achat(self, messages, **kwargs):
        self.chat_messages.append(messages)
        if self.throw_exception:
            raise self.throw_exception
        # ReActAgent calls chat, then the response object's .response attribute
        # The actual response content is often in message.content or similar
        # For this mock, we'll assume the ReActAgent interaction results in this being the final string
        return ChatResponse(
            message=MagicMock(content=self.response_text),
            raw={"response": self.response_text},
        )

    def chat(
        self, messages, **kwargs
    ):  # Synchronous version often used by ReActAgent internals
        self.chat_messages.append(messages)
        if self.throw_exception:
            raise self.throw_exception
        return ChatResponse(
            message=MagicMock(content=self.response_text),
            raw={"response": self.response_text},
        )


class MockTool(BaseTool):
    """Mocks a LlamaIndex BaseTool."""

    def __init__(
        self,
        name: str,
        description: str,
        call_return_value: Any = "Tool search results.",
    ):
        self._name = name
        self._description = description
        self._metadata = ToolMetadata(name=name, description=description)
        self.call_return_value = call_return_value
        self.last_call_args = None
        self.last_call_kwargs = None

    @property
    def metadata(self):
        return self._metadata

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.last_call_args = args
        self.last_call_kwargs = kwargs
        return self.call_return_value


class MockToolFactory:
    """Mocks the ToolFactory."""

    def __init__(self, vector_search_tool: Optional[MockTool] = None):
        self.vector_search_tool = vector_search_tool or MockTool(
            name="vector_search_utterances", description="Searches utterances"
        )

    def get_tool(
        self,
        tool_name: str,
        collection_name: Optional[str] = None,
        metadata: Optional[ToolMetadata] = None,
    ) -> Optional[BaseTool]:
        if tool_name == "VectorSearchTool" and collection_name == "utterances":
            # Update mock tool's metadata if provided, useful for verifying agent setup
            if metadata:
                self.vector_search_tool._name = metadata.name
                self.vector_search_tool._description = metadata.description
                self.vector_search_tool._metadata = metadata
            return self.vector_search_tool
        return None


class MockConfigManager:
    """Mocks the ConfigManager."""

    def __init__(self, configs: Optional[Dict[str, Any]] = None):
        self.configs = configs or {}
        self.configs.setdefault("processing.retries.max_attempts", "3")
        self.configs.setdefault("model.claimify.decontextualization", "mock_llm")

    def get(self, key: str, default: Any = None) -> Any:
        return self.configs.get(key, default)


@pytest.fixture
def mock_config_manager():
    return MockConfigManager()


@pytest.fixture
def mock_vector_search_tool():
    return MockTool(name="vector_search_utterances", description="Searches utterances")


@pytest.fixture
def mock_tool_factory(mock_vector_search_tool):
    return MockToolFactory(vector_search_tool=mock_vector_search_tool)


# --- DecontextualizationAgent Tests ---
@pytest.mark.asyncio
async def test_decontextualization_agent_success(
    mock_tool_factory, mock_config_manager
):
    """Test successful evaluation by the agent."""
    mock_llm = MockLLM(response_text="0.92")
    agent = DecontextualizationAgent(
        llm=mock_llm, tool_factory=mock_tool_factory, config_manager=mock_config_manager
    )  # type: ignore

    score, message = await agent.evaluate_claim_decontextualization(
        claim_id="c1",
        claim_text="Claim text",
        source_id="s1",
        source_text="Source text",
    )

    assert score == 0.92
    assert message == "success"
    # Check if the prompt was formatted as expected (basic check)
    assert 'Claim: "Claim text"' in mock_llm.chat_messages[-1]
    assert 'Source: "Source text"' in mock_llm.chat_messages[-1]


@pytest.mark.asyncio
async def test_decontextualization_agent_llm_parse_error(
    mock_tool_factory, mock_config_manager
):
    """Test agent handling of LLM returning non-float score."""
    mock_llm = MockLLM(response_text="not a float")
    agent = DecontextualizationAgent(
        llm=mock_llm, tool_factory=mock_tool_factory, config_manager=mock_config_manager
    )  # type: ignore

    score, message = await agent.evaluate_claim_decontextualization(
        claim_id="c1", claim_text="Claim", source_id="s1", source_text="Source"
    )

    assert score is None
    assert "invalid score format" in message.lower()


@pytest.mark.asyncio
async def test_decontextualization_agent_llm_score_out_of_range(
    mock_tool_factory, mock_config_manager
):
    """Test agent handling of LLM returning score out of 0.0-1.0 range."""
    mock_llm = MockLLM(response_text="1.5")
    agent = DecontextualizationAgent(
        llm=mock_llm, tool_factory=mock_tool_factory, config_manager=mock_config_manager
    )  # type: ignore

    score, message = await agent.evaluate_claim_decontextualization(
        claim_id="c1", claim_text="Claim", source_id="s1", source_text="Source"
    )

    assert score is None
    assert "out-of-range value" in message.lower()


@pytest.mark.asyncio
@patch(
    "tenacity.AsyncRetrying.iter", return_value=iter([1, 2, 3])
)  # Mock retries to run 3 times then fail
async def test_decontextualization_agent_llm_exception_after_retries(
    mock_tool_factory, # mock_retry_iter_unused removed
): # mock_config_manager removed, mock_retry_iter renamed
    """Test agent handling of LLM consistently raising exceptions."""
    mock_llm = MockLLM(throw_exception=RuntimeError("LLM API Error"))
    # Configure retries to be 3, matching the patch, directly in a new mock config
    mock_config_manager_for_this_test = MockConfigManager(
        {"processing.retries.max_attempts": "3"}
    )
    agent = DecontextualizationAgent(
        llm=mock_llm,
        tool_factory=mock_tool_factory,
        config_manager=mock_config_manager_for_this_test, # Use specific mock config
    )  # type: ignore

    score, message = await agent.evaluate_claim_decontextualization(
        claim_id="c1", claim_text="Claim", source_id="s1", source_text="Source"
    )

    assert score is None
    assert "error during decontextualization evaluation" in message.lower()
    assert "llm api error" in message.lower()
    assert (
        "after 3 retries" in message.lower()
    )  # Check if max_retries from config is mentioned
    assert mock_llm.chat_messages  # Ensure LLM was called
    # Tenacity's call count is tricky to get directly from mock_retry_iter here
    # but we expect chat_messages to have been populated multiple times if retries happened.
    # The number of actual calls to LLM depends on how ReActAgent handles it internally with tenacity.
    # For a direct tenacity call, len(mock_llm.chat_messages) would be 3.


@pytest.mark.asyncio
async def test_decontextualization_agent_tool_usage_in_prompt(
    mock_vector_search_tool, mock_config_manager
):
    """Verify that the prompt correctly instructs the LLM on tool usage."""
    mock_llm = MockLLM(response_text="0.7")
    # Need a tool factory that provides the specific mock_vector_search_tool
    tool_factory_with_specific_tool = MockToolFactory(
        vector_search_tool=mock_vector_search_tool
    )
    agent = DecontextualizationAgent(
        llm=mock_llm,
        tool_factory=tool_factory_with_specific_tool,
        config_manager=mock_config_manager,
    )  # type: ignore

    await agent.evaluate_claim_decontextualization(
        claim_id="c1",
        claim_text="A generic claim",
        source_id="s1",
        source_text="Source",
    )

    # Check the prompt sent to the LLM
    prompt_text = mock_llm.chat_messages[-1]
    assert "To help you, you have a 'vector_search_utterances' tool." in prompt_text
    assert (
        "Optionally, use the 'vector_search_utterances' tool with the 'Claim' text"
        in prompt_text
    )
    # This doesn't test if the ReActAgent *actually* calls the tool, as that depends on LLM output.
    # Testing the ReActAgent's tool execution loop is more complex and LLM-dependent.


# --- ClaimEvaluationGraphService Tests ---
@pytest.fixture
def mock_neo4j_driver():
    driver = MagicMock(spec=Driver)
    # Mock the session context manager
    mock_session = MagicMock()
    driver.session.return_value.__enter__.return_value = mock_session
    return driver, mock_session


def test_graph_service_update_score_success(mock_neo4j_driver, mock_config_manager):
    driver, mock_session = mock_neo4j_driver
    mock_result = MagicMock()
    mock_result.single.return_value = {"updated_count": 1}
    mock_session.run.return_value = mock_result

    service = ClaimEvaluationGraphService(
        neo4j_driver=driver, config_manager=mock_config_manager
    )
    success = service.update_decontextualization_score("c1", "b1", 0.75)

    assert success is True
    mock_session.run.assert_called_once()
    call_args = mock_session.run.call_args[0]
    assert "SET r.decontextualization_score = $score" in call_args[0]  # Query
    assert call_args[1] == {"claim_id": "c1", "block_id": "b1", "score": 0.75}  # Params


def test_graph_service_update_score_null_success(
    mock_neo4j_driver, mock_config_manager
):
    driver, mock_session = mock_neo4j_driver
    mock_result = MagicMock()
    mock_result.single.return_value = {"updated_count": 1}
    mock_session.run.return_value = mock_result

    service = ClaimEvaluationGraphService(
        neo4j_driver=driver, config_manager=mock_config_manager
    )
    success = service.update_decontextualization_score("c1", "b1", None)

    assert success is True
    call_args = mock_session.run.call_args[0]
    assert call_args[1]["score"] is None


def test_graph_service_update_score_no_relationship(
    mock_neo4j_driver, mock_config_manager, caplog
):
    driver, mock_session = mock_neo4j_driver
    mock_result = MagicMock()
    mock_result.single.return_value = {
        "updated_count": 0
    }  # Simulate no relationship found/updated
    mock_session.run.return_value = mock_result

    service = ClaimEvaluationGraphService(
        neo4j_driver=driver, config_manager=mock_config_manager
    )
    with caplog.at_level(logging.WARNING):
        success = service.update_decontextualization_score(
            "c_nonexistent", "b_nonexistent", 0.5
        )

    assert success is False
    assert (
        "failed to update decontextualization_score: no relationship found"
        in caplog.text.lower()
    )


def test_graph_service_update_score_db_error(
    mock_neo4j_driver, mock_config_manager, caplog
):
    driver, mock_session = mock_neo4j_driver
    mock_session.run.side_effect = Exception("Neo4j connection error")

    service = ClaimEvaluationGraphService(
        neo4j_driver=driver, config_manager=mock_config_manager
    )
    with caplog.at_level(logging.ERROR):
        success = service.update_decontextualization_score("c1", "b1", 0.5)

    assert success is False
    assert "error updating decontextualization_score" in caplog.text.lower()
    assert "neo4j connection error" in caplog.text.lower()


def test_graph_service_batch_update_scores(
    mock_neo4j_driver, mock_config_manager, caplog
):
    driver, mock_session = mock_neo4j_driver

    # Simulate Neo4j returning results for processed items
    mock_db_results = [
        {"processed_claim_id": "c_batch_1", "updated_count": 1},
        {"processed_claim_id": "c_batch_2", "updated_count": 1},
        # c_batch_3 will simulate a miss (no relationship found)
    ]
    mock_session.run.return_value = iter(mock_db_results)

    service = ClaimEvaluationGraphService(
        neo4j_driver=driver, config_manager=mock_config_manager
    )
    scores_data = [
        {"claim_id": "c_batch_1", "block_id": "b_batch_1", "score": 0.88},
        {"claim_id": "c_batch_2", "block_id": "b_batch_2", "score": None},
        {
            "claim_id": "c_batch_3",
            "block_id": "b_batch_3",
            "score": 0.7,
        },  # This one will be "missed"
    ]

    with caplog.at_level(logging.INFO):
        success = service.batch_update_decontextualization_scores(scores_data)

    assert success is True
    mock_session.run.assert_called_once()
    call_args = mock_session.run.call_args[0]
    assert "UNWIND $scores_batch AS score_entry" in call_args[0]
    assert call_args[1] == {"scores_batch": scores_data}

    assert (
        "successfully batch-updated decontextualization_score for claim c_batch_1"
        in caplog.text.lower()
    )
    assert (
        "successfully batch-updated decontextualization_score for claim c_batch_2"
        in caplog.text.lower()
    )
    assert (
        "failed to batch-update decontextualization_score: no relationship found for claim c_batch_3"
        in caplog.text.lower()
    )


# --- MarkdownUpdaterService Tests ---
@pytest.fixture
def markdown_service():
    return MarkdownUpdaterService()


@pytest.fixture
def temp_markdown_file():
    content = """Line 0
This is block one. <!-- aclarai:id=blk_abc123 ver=1 -->
^blk_abc123
Some other text.
<!-- aclarai:decontextualization_score=0.50 -->
Line for block two. <!-- aclarai:id=blk_xyz789 ver=3 -->
^blk_xyz789
This is block three. <!-- aclarai:id=blk_multi ver=1 -->
This is the second line of block three.
<!-- aclarai:id=blk_another ver=5 -->
"""
    # Use NamedTemporaryFile to ensure it's cleaned up, but get its path for MarkdownUpdaterService
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, encoding="utf-8", suffix=".md"
    ) as tmp_file:
        tmp_file.write(content)
        filepath = tmp_file.name

    yield Path(filepath)  # Provide the Path object to the test

    os.unlink(filepath)  # Explicit cleanup


def test_markdown_updater_add_score_new_block(markdown_service, temp_markdown_file):
    success = markdown_service.add_or_update_decontextualization_score(
        str(temp_markdown_file), "blk_abc123", 0.77
    )
    assert success is True
    content = temp_markdown_file.read_text(encoding="utf-8")
    assert "<!-- aclarai:decontextualization_score=0.77 -->" in content
    assert (
        "This is block one. <!-- aclarai:id=blk_abc123 ver=2 -->" in content
    )  # Version incremented


def test_markdown_updater_update_existing_score_logic(
    markdown_service, temp_markdown_file, caplog
):
    # This test targets the logic within _find_block_and_update_score
    # The current _find_block_and_update_score places the new score *before* the ID line
    # and tries to remove an old score if it was on the line *before* that.
    # The initial content for blk_xyz789 has the score comment *before* the ID line.

    with caplog.at_level(logging.DEBUG):  # To see "Removing old score comment"
        success = markdown_service.add_or_update_decontextualization_score(
            str(temp_markdown_file),
            "blk_xyz789",
            None,  # New score is null
        )
    assert success is True
    content = temp_markdown_file.read_text(encoding="utf-8")

    assert "<!-- aclarai:decontextualization_score=null -->" in content
    assert (
        "Line for block two. <!-- aclarai:id=blk_xyz789 ver=4 -->" in content
    )  # Version incremented
    assert (
        "<!-- aclarai:decontextualization_score=0.50 -->" not in content
    )  # Old score removed

    # Verify the new score is correctly positioned relative to its block
    # (i.e., before the line with blk_xyz789 ver=4)
    lines = content.splitlines()
    found_new_score_line = -1
    found_block_id_line = -1
    for i, line_text in enumerate(lines):
        if (
            "<!-- aclarai:decontextualization_score=null -->" in line_text
            and "blk_xyz789" not in line_text # Ensure it's the new one for the correct block
            and i + 1 < len(lines)
            and "blk_xyz789 ver=4" in lines[i + 1]
        ):
            found_new_score_line = i
        if "Line for block two. <!-- aclarai:id=blk_xyz789 ver=4 -->" in line_text:
            found_block_id_line = i

    assert found_new_score_line != -1, "New score comment for blk_xyz789 not found or not positioned correctly"
    assert found_block_id_line != -1, "Block ID line for blk_xyz789 not found"
    assert found_new_score_line == found_block_id_line - 1, (
        "New score comment not immediately before its block ID line"
    )

    # Check logging for old score removal
    assert (
        "removing old score comment: <!-- aclarai:decontextualization_score=0.50 -->"
        in caplog.text.lower()
    )


def test_markdown_updater_block_not_found(markdown_service, temp_markdown_file, caplog):
    with caplog.at_level(logging.WARNING):
        success = markdown_service.add_or_update_decontextualization_score(
            str(temp_markdown_file), "blk_nonexistent", 0.5
        )
    assert success is False
    assert "block id 'blk_nonexistent' not found" in caplog.text.lower()


def test_markdown_updater_file_not_found(markdown_service, caplog):
    with caplog.at_level(logging.ERROR):
        success = markdown_service.add_or_update_decontextualization_score(
            "non_existent_file.md", "blk_abc123", 0.5
        )
    assert success is False
    assert "markdown file not found" in caplog.text.lower()


@patch("os.replace")
@patch("os.fsync")
@patch("os.fdopen")
def test_markdown_updater_atomic_write_failure(
    mock_os_replace,    # mock_fdopen_unused and mock_fsync_unused removed
    markdown_service,
    temp_markdown_file,
    caplog,
):
    mock_os_replace.side_effect = OSError("Disk full")  # Simulate failure during rename

    with caplog.at_level(logging.ERROR):
        success = markdown_service.add_or_update_decontextualization_score(
            str(temp_markdown_file), "blk_abc123", 0.6
        )

    assert success is False
    assert "failed to atomically write" in caplog.text.lower()
    assert "disk full" in caplog.text.lower()
    # Ensure temp file (if created by mkstemp mock path) would be attempted to be cleaned up
    # This part is harder to test without deeper os patching or passing temp_file_path from _atomic_write


def test_markdown_updater_preserves_other_blocks(markdown_service, temp_markdown_file):
    """Test that updating one block doesn't corrupt others."""
    markdown_service.add_or_update_decontextualization_score(
        str(temp_markdown_file), "blk_abc123", 0.1
    )
    content_after_first_update = temp_markdown_file.read_text(encoding="utf-8")

    # Check original content of other blocks still exists
    assert (
        "<!-- aclarai:decontextualization_score=0.50 -->" in content_after_first_update
    )  # For blk_xyz789
    assert (
        "Line for block two. <!-- aclarai:id=blk_xyz789 ver=3 -->"
        in content_after_first_update
    )
    assert (
        "This is block three. <!-- aclarai:id=blk_multi ver=1 -->"
        in content_after_first_update
    )
    assert "<!-- aclarai:id=blk_another ver=5 -->" in content_after_first_update

    markdown_service.add_or_update_decontextualization_score(
        str(temp_markdown_file), "blk_multi", 0.2
    )
    content_after_second_update = temp_markdown_file.read_text(encoding="utf-8")

    # Check first updated block is still correct
    assert (
        "<!-- aclarai:decontextualization_score=0.1 -->" in content_after_second_update
    )
    assert (
        "This is block one. <!-- aclarai:id=blk_abc123 ver=2 -->"
        in content_after_second_update
    )
    # Check second updated block
    assert (
        "<!-- aclarai:decontextualization_score=0.2 -->" in content_after_second_update
    )
    assert (
        "This is block three. <!-- aclarai:id=blk_multi ver=2 -->"
        in content_after_second_update
    )
    # Check other blocks are still fine
    assert (
        "<!-- aclarai:decontextualization_score=0.50 -->" in content_after_second_update
    )  # For blk_xyz789 (still original score)
    assert (
        "Line for block two. <!-- aclarai:id=blk_xyz789 ver=3 -->"
        in content_after_second_update
    )
    assert "<!-- aclarai:id=blk_another ver=5 -->" in content_after_second_update


# Integration Style Test (Conceptual)
# This would require more setup, like a running Neo4j instance and actual LLM calls or more sophisticated mocks.
# @pytest.mark.asyncio
# async def test_full_decontextualization_workflow(mock_neo4j_driver, temp_markdown_file, mock_config_manager):
#     # 1. Setup: Mock LLM, real ToolFactory (or one that returns a functional mock tool),
#     #    real GraphService with mocked driver, real MarkdownUpdaterService.
#
#     # Mock LLM to return a specific score
#     mock_llm = MockLLM(response_text="0.85")
#     mock_tool_factory_inst = MockToolFactory() # Using the basic mock tool
#
#     agent = DecontextualizationAgent(llm=mock_llm, tool_factory=mock_tool_factory_inst, config_manager=mock_config_manager)
#
#     driver, mock_session_neo4j = mock_neo4j_driver
#     mock_neo4j_result = MagicMock()
#     mock_neo4j_result.single.return_value = {"updated_count": 1}
#     mock_session_neo4j.run.return_value = mock_neo4j_result
#     graph_service = ClaimEvaluationGraphService(neo4j_driver=driver, config_manager=mock_config_manager)
#
#     markdown_service_inst = MarkdownUpdaterService()
#
#     # Test data
#     test_claim_id = "blk_abc123" # This ID is in temp_markdown_file
#     test_block_id = "blk_abc123" # In this case, claim_id and block_id are the same for simplicity
#     test_claim_text = "This is block one."
#     test_source_text = "Some source for block one."
#
#     # 2. Execute: Agent evaluates
#     score, eval_message = await agent.evaluate_claim_decontextualization(
#         claim_id=test_claim_id, claim_text=test_claim_text,
#         source_id=test_block_id, source_text=test_source_text
#     )
#     assert score == 0.85
#     assert eval_message == "success"
#
#     # 3. Persist: Store score in Neo4j
#     neo4j_success = graph_service.update_decontextualization_score(
#         claim_id=test_claim_id, block_id=test_block_id, score=score
#     )
#     assert neo4j_success is True
#     mock_session_neo4j.run.assert_called_with(
#         Any, # Query string
#         {'claim_id': test_claim_id, 'block_id': test_block_id, 'score': 0.85}
#     )
#
#     # 4. Persist: Store score in Markdown
#     md_success = markdown_service_inst.add_or_update_decontextualization_score(
#         str(temp_markdown_file), test_block_id, score
#     )
#     assert md_success is True
#
#     # 5. Verify: Check Markdown content
#     content = temp_markdown_file.read_text(encoding="utf-8")
#     assert "<!-- aclarai:decontextualization_score=0.85 -->" in content
#     assert f"{test_claim_text} <!-- aclarai:id={test_block_id} ver=2 -->" in content # Version incremented
#
#     # Add more assertions as needed for a full integration test.
