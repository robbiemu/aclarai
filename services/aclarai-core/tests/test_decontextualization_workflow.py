"""
Tests for the decontextualization evaluation workflow, including the agent,
Neo4j storage, and Markdown updates.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Modules to test
from aclarai_core.agents.decontextualization_agent import DecontextualizationAgent
from aclarai_core.graph.claim_evaluation_graph_service import (
    ClaimEvaluationGraphService,
)
from aclarai_core.markdown.markdown_updater_service import MarkdownUpdaterService
from aclarai_shared.config import (
    LLMConfig,
    ProcessingConfig,
    aclaraiConfig,
)
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.base.response.schema import Response
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
        self.metadata = MagicMock()
        self.metadata.context_window = 4096  # Default context window
        self.callback_manager = MagicMock()

    async def achat(self, messages, **kwargs):  # noqa: ARG002
        # ReActAgent can send multiple messages (e.g., system, user).
        # We join them to test the full context sent to the LLM.
        full_prompt_content = "\n".join(m.content for m in messages)
        self.chat_messages.append(full_prompt_content)

        if self.throw_exception:
            raise self.throw_exception
        return ChatResponse(
            message=ChatMessage(content=self.response_text),
            response=self.response_text,
        )

    def chat(
        self,
        messages,
        **kwargs,  # noqa: ARG002
    ):  # Synchronous version often used by ReActAgent internals
        # ReActAgent can send multiple messages (e.g., system, user).
        # We join them to test the full context sent to the LLM.
        full_prompt_content = "\n".join(m.content for m in messages)
        self.chat_messages.append(full_prompt_content)

        if self.throw_exception:
            raise self.throw_exception
        return ChatResponse(
            message=ChatMessage(content=self.response_text),
            response=self.response_text,
        )


class MockReActAgent:
    """Mocks the ReActAgent for testing."""

    def __init__(
        self, response_text: str = "0.85", throw_exception: Optional[Exception] = None
    ):
        self.response_text = response_text
        self.throw_exception = throw_exception
        self.last_input = None
        self.call_count = 0

    def run(self, input: str) -> Response:
        self.last_input = input
        self.call_count += 1

        if self.throw_exception:
            raise self.throw_exception

        # Return a Response object like the real ReActAgent
        response = Response(response=self.response_text)
        return response


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

    def __init__(self, tools: Optional[List[BaseTool]] = None):
        self._tools = tools or [
            MockTool(name="vector_search", description="Searches utterances")
        ]
        self._agent_tool_cache = {}

    def get_tools_for_agent(
        self, agent_name: str
    ) -> List[BaseTool | Callable[..., Any]]:
        # Simulate caching behavior
        if agent_name not in self._agent_tool_cache:
            self._agent_tool_cache[agent_name] = self._tools
        return self._agent_tool_cache[agent_name]


@pytest.fixture
def mock_aclarai_config() -> MagicMock:
    """Provides a mock AclaraiConfig object for tests."""
    mock_config = MagicMock(spec=aclaraiConfig)
    mock_config.processing = MagicMock(spec=ProcessingConfig)
    mock_config.processing.retries = {"max_attempts": 3}
    mock_config.llm = MagicMock(spec=LLMConfig)
    mock_config.llm.model_params = {"claimify": {"decontextualization": "mock_llm"}}
    return mock_config


@pytest.fixture
def mock_vector_search_tool():
    return MockTool(name="vector_search_utterances", description="Searches utterances")


@pytest.fixture
def mock_graph_service_config():
    """Provide a mock configuration for ClaimEvaluationGraphService."""
    return {
        "neo4j": {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
        }
    }


@pytest.fixture
def mock_tool_factory():
    # Provide the correctly named tool to align with the prompt's instructions
    return MockToolFactory(
        tools=[
            MockTool(name="vector_search_utterances", description="Searches utterances")
        ]
    )


# --- DecontextualizationAgent Tests ---
# In file: test_decontextualization_workflow.py
def test_decontextualization_agent_success(mock_tool_factory, mock_aclarai_config):
    """Test successful evaluation by the agent."""
    mock_llm = MockLLM(response_text="0.92")

    # Mock the ReActAgent
    mock_react_agent = MockReActAgent(response_text="0.92")

    with (
        patch(
            "aclarai_core.agents.decontextualization_agent.load_prompt_template"
        ) as mock_load_prompt,
        patch(
            "aclarai_core.agents.decontextualization_agent.ReActAgent",
            return_value=mock_react_agent,
        ),
    ):
        # Define test data and the prompt template
        claim_text_val = "Claim text"
        source_text_val = "Source text"
        prompt_template = """Input:
Claim: "{claim_text}"
Source: "{source_text}"

Task:
1. Analyze the 'Claim' for any ambiguities or missing context.
2. Optionally, use the 'vector_search_utterances' tool with the 'Claim' text to check for contextual diversity.
3. Based on your analysis, provide a decontextualization score as a float between 0.0 and 1.0.
Output only the float score. For example: 0.75"""

        # FIX: Format the template before setting it as the mock's return value
        mock_load_prompt.return_value = prompt_template.format(
            claim_text=claim_text_val, source_text=source_text_val
        )

        agent = DecontextualizationAgent(
            llm=mock_llm, tool_factory=mock_tool_factory, config=mock_aclarai_config
        )

        score, message = agent.evaluate_claim_decontextualization(
            claim_id="c1",
            claim_text=claim_text_val,
            source_id="s1",
            source_text=source_text_val,
        )

    assert score == 0.92
    assert message == "success"

    # Verify the ReActAgent was called correctly
    assert mock_react_agent.call_count == 1
    assert 'Claim: "Claim text"' in mock_react_agent.last_input
    assert 'Source: "Source text"' in mock_react_agent.last_input


def test_decontextualization_agent_llm_parse_error(
    mock_tool_factory, mock_aclarai_config
):
    """Test agent handling of LLM returning non-float score."""
    mock_llm = MockLLM(response_text="not a float")
    mock_react_agent = MockReActAgent(response_text="not a float")

    with (
        patch(
            "aclarai_core.agents.decontextualization_agent.load_prompt_template"
        ) as mock_load_prompt,
        patch(
            "aclarai_core.agents.decontextualization_agent.ReActAgent",
            return_value=mock_react_agent,
        ),
    ):
        mock_load_prompt.return_value = "Mock prompt"

        agent = DecontextualizationAgent(
            llm=mock_llm, tool_factory=mock_tool_factory, config=mock_aclarai_config
        )

        score, message = agent.evaluate_claim_decontextualization(
            claim_id="c1", claim_text="Claim", source_id="s1", source_text="Source"
        )

    assert score is None
    assert "invalid score format" in message.lower()


def test_decontextualization_agent_llm_score_out_of_range(
    mock_tool_factory, mock_aclarai_config
):
    """Test agent handling of LLM returning score out of 0.0-1.0 range."""
    mock_llm = MockLLM(response_text="1.5")
    mock_react_agent = MockReActAgent(response_text="1.5")

    with (
        patch(
            "aclarai_core.agents.decontextualization_agent.load_prompt_template"
        ) as mock_load_prompt,
        patch(
            "aclarai_core.agents.decontextualization_agent.ReActAgent",
            return_value=mock_react_agent,
        ),
    ):
        mock_load_prompt.return_value = "Mock prompt"

        agent = DecontextualizationAgent(
            llm=mock_llm, tool_factory=mock_tool_factory, config=mock_aclarai_config
        )

        score, message = agent.evaluate_claim_decontextualization(
            claim_id="c1", claim_text="Claim", source_id="s1", source_text="Source"
        )

    assert score is None
    assert "out-of-range value" in message.lower()


def test_decontextualization_agent_llm_exception_after_retries(
    mock_tool_factory, mock_aclarai_config
):
    """Test agent handling of LLM consistently raising exceptions."""
    original_exception = RuntimeError("LLM API Error")
    mock_llm = MockLLM(throw_exception=original_exception)
    mock_react_agent = MockReActAgent(throw_exception=original_exception)
    mock_aclarai_config.processing.retries = {"max_attempts": 3}

    with (
        patch(
            "aclarai_core.agents.decontextualization_agent.load_prompt_template",
            return_value="A valid mock prompt string",
        ) as mock_load_prompt,
        patch(
            "aclarai_core.agents.decontextualization_agent.ReActAgent",
            return_value=mock_react_agent,
        ),
    ):
        agent = DecontextualizationAgent(
            llm=mock_llm,
            tool_factory=mock_tool_factory,
            config=mock_aclarai_config,
        )

        score, message = agent.evaluate_claim_decontextualization(
            claim_id="c1", claim_text="Claim", source_id="s1", source_text="Source"
        )

    assert score is None
    expected_error_snippet = f"Root cause: {repr(original_exception)}"
    assert expected_error_snippet in message

    mock_load_prompt.assert_called_once_with(
        "decontextualization_evaluation",
        claim_text="Claim",
        source_text="Source",
    )


def test_decontextualization_agent_tool_usage_in_prompt(mock_aclarai_config):
    """Verify that the prompt correctly instructs the LLM on tool usage."""
    mock_llm = MockLLM(response_text="0.7")
    mock_react_agent = MockReActAgent(response_text="0.7")

    # Need a tool factory that provides the specific mock_vector_search_tool
    tool_factory_with_specific_tool = MockToolFactory(
        tools=[
            MockTool(name="vector_search_utterances", description="Searches utterances")
        ]
    )

    with (
        patch(
            "aclarai_core.agents.decontextualization_agent.load_prompt_template"
        ) as mock_load_prompt,
        patch(
            "aclarai_core.agents.decontextualization_agent.ReActAgent",
            return_value=mock_react_agent,
        ) as mock_react_from_tools,
    ):
        mock_load_prompt.return_value = "use the 'vector_search_utterances' tool"

        agent = DecontextualizationAgent(
            llm=mock_llm,
            tool_factory=tool_factory_with_specific_tool,
            config=mock_aclarai_config,
        )

        agent.evaluate_claim_decontextualization(
            claim_id="c1",
            claim_text="A generic claim",
            source_id="s1",
            source_text="Source",
        )

    # Verify ReActAgent was created with the correct tools
    mock_react_from_tools.assert_called_once()
    call_args = mock_react_from_tools.call_args
    tools_passed = call_args[1]["tools"]  # keyword argument 'tools'

    # Check that the tool with the correct name was passed
    tool_names = [tool.metadata.name for tool in tools_passed]
    assert "vector_search_utterances" in tool_names

    # Check that the prompt contains the tool usage instruction
    assert "use the 'vector_search_utterances' tool" in mock_react_agent.last_input


# --- ClaimEvaluationGraphService Tests ---
@pytest.fixture
def mock_neo4j_driver():
    driver = MagicMock(spec=Driver)
    # Mock the session context manager
    mock_session = MagicMock()
    driver.session.return_value.__enter__.return_value = mock_session
    return driver, mock_session


def test_graph_service_update_score_success(
    mock_neo4j_driver, mock_graph_service_config
):
    driver, mock_session = mock_neo4j_driver
    mock_result = MagicMock()
    mock_result.single.return_value = {"updated_count": 1}
    mock_session.run.return_value = mock_result

    service = ClaimEvaluationGraphService(
        neo4j_driver=driver, config=mock_graph_service_config
    )
    success = service.update_decontextualization_score("c1", "b1", 0.75)

    assert success is True
    mock_session.run.assert_called_once()
    call_args = mock_session.run.call_args[0]
    assert "SET r.decontextualization_score = $score" in call_args[0]  # Query
    assert call_args[1] == {"claim_id": "c1", "block_id": "b1", "score": 0.75}  # Params


def test_graph_service_update_score_null_success(
    mock_neo4j_driver, mock_graph_service_config
):
    driver, mock_session = mock_neo4j_driver
    mock_result = MagicMock()
    mock_result.single.return_value = {"updated_count": 1}
    mock_session.run.return_value = mock_result

    service = ClaimEvaluationGraphService(
        neo4j_driver=driver, config=mock_graph_service_config
    )
    success = service.update_decontextualization_score("c1", "b1", None)

    assert success is True
    call_args = mock_session.run.call_args[0]
    assert call_args[1]["score"] is None


def test_graph_service_update_score_no_relationship(
    mock_neo4j_driver, mock_graph_service_config, caplog
):
    driver, mock_session = mock_neo4j_driver
    mock_result = MagicMock()
    mock_result.single.return_value = {
        "updated_count": 0
    }  # Simulate no relationship found/updated
    mock_session.run.return_value = mock_result

    service = ClaimEvaluationGraphService(
        neo4j_driver=driver, config=mock_graph_service_config
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
    mock_neo4j_driver, mock_graph_service_config, caplog
):
    driver, mock_session = mock_neo4j_driver
    mock_session.run.side_effect = Exception("Neo4j connection error")

    service = ClaimEvaluationGraphService(
        neo4j_driver=driver, config=mock_graph_service_config
    )
    with caplog.at_level(logging.ERROR):
        success = service.update_decontextualization_score("c1", "b1", 0.5)

    assert success is False
    assert "error updating decontextualization_score" in caplog.text.lower()
    assert "neo4j connection error" in caplog.text.lower()


def test_graph_service_batch_update_scores(
    mock_neo4j_driver, mock_graph_service_config, caplog
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
        neo4j_driver=driver, config=mock_graph_service_config
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
            and "blk_xyz789"
            not in line_text  # Ensure it's the new one for the correct block
            and i + 1 < len(lines)
            and "blk_xyz789 ver=4" in lines[i + 1]
        ):
            found_new_score_line = i
        if "Line for block two. <!-- aclarai:id=blk_xyz789 ver=4 -->" in line_text:
            found_block_id_line = i

    assert found_new_score_line != -1, (
        "New score comment for blk_xyz789 not found or not positioned correctly"
    )
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


@patch(
    "aclarai_core.markdown.markdown_updater_service.MarkdownUpdaterService._atomic_write"
)
def test_markdown_updater_atomic_write_failure(
    mock_atomic_write,
    markdown_service,
    temp_markdown_file,
    caplog,
):
    mock_atomic_write.return_value = False  # Simulate failure during atomic write

    with caplog.at_level(logging.ERROR):
        success = markdown_service.add_or_update_decontextualization_score(
            str(temp_markdown_file), "blk_abc123", 0.6
        )

    assert success is False
    assert "atomic write failed" in caplog.text.lower()


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


def test_graph_service_generic_update_relationship_score(
    mock_neo4j_driver, mock_graph_service_config
):
    """Test the generic update_relationship_score method."""
    driver, mock_session = mock_neo4j_driver
    mock_result = MagicMock()
    mock_result.single.return_value = {"updated_count": 1}
    mock_session.run.return_value = mock_result

    service = ClaimEvaluationGraphService(
        neo4j_driver=driver, config=mock_graph_service_config
    )
    
    # Test valid score names
    valid_scores = ["entailed_score", "decontextualization_score", "coverage_score"]
    for score_name in valid_scores:
        success = service.update_relationship_score(
            "claim_test", "block_test", score_name, 0.85
        )
        assert success is True  # Mock always returns success

    # Test invalid score name
    with pytest.raises(ValueError, match="Invalid score_name"):
        service.update_relationship_score(
            "claim_test", "block_test", "invalid_score", 0.85
        )


def test_markdown_service_generic_add_or_update_score(markdown_service, temp_markdown_file):
    """Test the generic add_or_update_score method with different score types."""
    # Test entailed_score
    success = markdown_service.add_or_update_score(
        str(temp_markdown_file), "blk_abc123", "entailed_score", 0.78
    )
    assert success is True

    content = temp_markdown_file.read_text(encoding="utf-8")
    assert "<!-- aclarai:entailed_score=0.78 -->" in content
    assert "<!-- aclarai:id=blk_abc123 ver=2 -->" in content

    # Test coverage_score on same block
    success = markdown_service.add_or_update_score(
        str(temp_markdown_file), "blk_abc123", "coverage_score", 0.92
    )
    assert success is True

    content = temp_markdown_file.read_text(encoding="utf-8")
    assert "<!-- aclarai:coverage_score=0.92 -->" in content
    assert "<!-- aclarai:id=blk_abc123 ver=3 -->" in content

    # Test null score
    success = markdown_service.add_or_update_score(
        str(temp_markdown_file), "blk_abc123", "decontextualization_score", None
    )
    assert success is True

    content = temp_markdown_file.read_text(encoding="utf-8")
    assert "<!-- aclarai:decontextualization_score=null -->" in content
    assert "<!-- aclarai:id=blk_abc123 ver=4 -->" in content
