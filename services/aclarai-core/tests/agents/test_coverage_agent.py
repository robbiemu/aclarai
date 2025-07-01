import json
from unittest.mock import MagicMock, patch

import pytest
from aclarai_core.agents.coverage_agent import CoverageAgent
from aclarai_shared.config import aclaraiConfig
from aclarai_shared.tools.factory import ToolFactory
from llama_index.core.base.response.schema import Response as LlamaResponse
from llama_index.core.llms.llm import LLM as LlamaLLM


# Minimal config for testing
@pytest.fixture
def mock_config():
    config = MagicMock(spec=aclaraiConfig)
    config.processing = MagicMock()
    config.processing.retries = {"max_attempts": 3}
    return config


@pytest.fixture
def mock_llm():
    return MagicMock(spec=LlamaLLM)


@pytest.fixture
def mock_tool_factory():
    factory = MagicMock(spec=ToolFactory)
    factory.get_tools_for_agent.return_value = []
    return factory


@pytest.fixture
def coverage_agent_instance(mock_llm, mock_tool_factory, mock_config):
    with patch(
        "aclarai_core.agents.coverage_agent.CodeActAgent"
    ) as MockCodeActAgent:
        mock_agent_internal_instance = MockCodeActAgent.from_tools.return_value
        agent_under_test = CoverageAgent(
            llm=mock_llm, tool_factory=mock_tool_factory, config=mock_config
        )
        agent_under_test.agent = mock_agent_internal_instance
        return agent_under_test


class TestCoverageAgent:
    @patch("aclarai_core.agents.coverage_agent.load_prompt_template")
    def test_evaluate_coverage_success_with_omitted_elements(
        self, mock_load_prompt, coverage_agent_instance
    ):
        """Test successful coverage evaluation with omitted elements identified."""
        mock_load_prompt.return_value = {
            "template": "user_prompt",
            "system_prompt": "system_prompt",
        }
        
        # Mock response with coverage score and omitted elements
        response_json = {
            "coverage_score": 0.7,
            "omitted_elements": [
                {"text": "European Commission", "significance": "Key organization responsible for policy"},
                {"text": "2023", "significance": "Specific timeframe for the action"}
            ],
            "reasoning": "Claim captures main point but omits key details"
        }
        
        mock_chat_response = MagicMock(spec=LlamaResponse)
        mock_chat_response.response = json.dumps(response_json)
        coverage_agent_instance.agent.chat.return_value = mock_chat_response

        score, elements, msg = coverage_agent_instance.evaluate_coverage(
            "claim1", "EU approved funding", "src1", "In 2023, the European Commission approved funding"
        )

        assert score == 0.7
        assert len(elements) == 2
        assert elements[0]["text"] == "European Commission"
        assert elements[0]["significance"] == "Key organization responsible for policy"
        assert elements[1]["text"] == "2023"
        assert msg == "success"
        coverage_agent_instance.agent.chat.assert_called_once_with("user_prompt")
        coverage_agent_instance.agent.update_prompts.assert_called_once_with(
            {"agent_worker:system_prompt": "system_prompt"}
        )

    @patch("aclarai_core.agents.coverage_agent.load_prompt_template")
    def test_evaluate_coverage_success_no_omitted_elements(
        self, mock_load_prompt, coverage_agent_instance
    ):
        """Test successful coverage evaluation with no omitted elements."""
        mock_load_prompt.return_value = {
            "template": "user_prompt",
            "system_prompt": "system_prompt",
        }
        
        response_json = {
            "coverage_score": 1.0,
            "omitted_elements": [],
            "reasoning": "Claim captures all verifiable elements from source"
        }
        
        mock_chat_response = MagicMock(spec=LlamaResponse)
        mock_chat_response.response = json.dumps(response_json)
        coverage_agent_instance.agent.chat.return_value = mock_chat_response

        score, elements, msg = coverage_agent_instance.evaluate_coverage(
            "claim1", "claim_text", "src1", "src_text"
        )

        assert score == 1.0
        assert elements == []
        assert msg == "success"

    @patch("aclarai_core.agents.coverage_agent.load_prompt_template")
    def test_evaluate_coverage_llm_error_with_retries(
        self, mock_load_prompt, coverage_agent_instance
    ):
        """Test coverage evaluation with LLM errors and retries."""
        mock_load_prompt.return_value = {"template": "user_prompt"}
        coverage_agent_instance.agent.chat.side_effect = ValueError("LLM API error")

        max_retries = coverage_agent_instance.max_retries
        score, elements, msg = coverage_agent_instance.evaluate_coverage(
            "claim1", "claim_text", "src1", "src_text"
        )

        assert score is None
        assert elements is None
        assert "LLM API error" in msg
        assert f"after {max_retries} retries" in msg
        assert coverage_agent_instance.agent.chat.call_count == max_retries

    @patch("aclarai_core.agents.coverage_agent.load_prompt_template")
    def test_evaluate_coverage_invalid_json_response(
        self, mock_load_prompt, coverage_agent_instance
    ):
        """Test handling of invalid JSON response from LLM."""
        mock_load_prompt.return_value = {"template": "user_prompt"}
        mock_chat_response = MagicMock(spec=LlamaResponse)
        mock_chat_response.response = "not valid json"
        coverage_agent_instance.agent.chat.return_value = mock_chat_response

        score, elements, msg = coverage_agent_instance.evaluate_coverage(
            "claim1", "claim_text", "src1", "src_text"
        )

        assert score is None
        assert elements is None
        assert "invalid JSON response" in msg

    @patch("aclarai_core.agents.coverage_agent.load_prompt_template")
    def test_evaluate_coverage_missing_coverage_score(
        self, mock_load_prompt, coverage_agent_instance
    ):
        """Test handling of response missing coverage_score."""
        mock_load_prompt.return_value = {"template": "user_prompt"}
        
        response_json = {
            "omitted_elements": [],
            "reasoning": "Some reasoning"
        }
        
        mock_chat_response = MagicMock(spec=LlamaResponse)
        mock_chat_response.response = json.dumps(response_json)
        coverage_agent_instance.agent.chat.return_value = mock_chat_response

        score, elements, msg = coverage_agent_instance.evaluate_coverage(
            "claim1", "claim_text", "src1", "src_text"
        )

        assert score is None
        assert elements is None
        assert "Missing or invalid coverage_score" in msg

    @patch("aclarai_core.agents.coverage_agent.load_prompt_template")
    def test_evaluate_coverage_score_out_of_range_high(
        self, mock_load_prompt, coverage_agent_instance
    ):
        """Test handling of coverage score above 1.0."""
        mock_load_prompt.return_value = {"template": "user_prompt"}
        
        response_json = {
            "coverage_score": 1.5,
            "omitted_elements": [],
            "reasoning": "Invalid high score"
        }
        
        mock_chat_response = MagicMock(spec=LlamaResponse)
        mock_chat_response.response = json.dumps(response_json)
        coverage_agent_instance.agent.chat.return_value = mock_chat_response

        score, elements, msg = coverage_agent_instance.evaluate_coverage(
            "claim1", "claim_text", "src1", "src_text"
        )

        assert score is None
        assert elements is None
        assert "out-of-range coverage score" in msg
        assert "Score: 1.5" in msg

    @patch("aclarai_core.agents.coverage_agent.load_prompt_template")
    def test_evaluate_coverage_score_out_of_range_low(
        self, mock_load_prompt, coverage_agent_instance
    ):
        """Test handling of coverage score below 0.0."""
        mock_load_prompt.return_value = {"template": "user_prompt"}
        
        response_json = {
            "coverage_score": -0.3,
            "omitted_elements": [],
            "reasoning": "Invalid low score"
        }
        
        mock_chat_response = MagicMock(spec=LlamaResponse)
        mock_chat_response.response = json.dumps(response_json)
        coverage_agent_instance.agent.chat.return_value = mock_chat_response

        score, elements, msg = coverage_agent_instance.evaluate_coverage(
            "claim1", "claim_text", "src1", "src_text"
        )

        assert score is None
        assert elements is None
        assert "out-of-range coverage score" in msg
        assert "Score: -0.3" in msg

    @patch("aclarai_core.agents.coverage_agent.load_prompt_template")
    def test_evaluate_coverage_invalid_omitted_elements_format(
        self, mock_load_prompt, coverage_agent_instance
    ):
        """Test handling of invalid omitted_elements format."""
        mock_load_prompt.return_value = {"template": "user_prompt"}
        
        response_json = {
            "coverage_score": 0.8,
            "omitted_elements": [
                {"text": "Valid element"},
                "invalid element format",  # This should be a dict
                {"missing_text_field": "value"}  # Missing required 'text' field
            ],
            "reasoning": "Mixed valid and invalid elements"
        }
        
        mock_chat_response = MagicMock(spec=LlamaResponse)
        mock_chat_response.response = json.dumps(response_json)
        coverage_agent_instance.agent.chat.return_value = mock_chat_response

        score, elements, msg = coverage_agent_instance.evaluate_coverage(
            "claim1", "claim_text", "src1", "src_text"
        )

        assert score == 0.8
        assert len(elements) == 1  # Only valid element should be included
        assert elements[0]["text"] == "Valid element"
        assert elements[0]["significance"] == ""  # Default when missing
        assert msg == "success"

    @patch("aclarai_core.agents.coverage_agent.load_prompt_template")
    def test_evaluate_coverage_prompt_load_failure(
        self, mock_load_prompt, coverage_agent_instance
    ):
        """Test handling of prompt template loading failure."""
        mock_load_prompt.side_effect = FileNotFoundError("Prompt file missing")

        score, elements, msg = coverage_agent_instance.evaluate_coverage(
            "claim1", "claim_text", "src1", "src_text"
        )

        assert score is None
        assert elements is None
        assert "Failed to load coverage prompt template" in msg
        assert "Prompt file missing" in msg
        coverage_agent_instance.agent.chat.assert_not_called()

    @patch("aclarai_core.agents.coverage_agent.load_prompt_template")
    def test_malformed_prompt_template_data(
        self, mock_load_prompt, coverage_agent_instance
    ):
        """Test handling of malformed prompt template data."""
        mock_load_prompt.return_value = "just a string"

        score, elements, msg = coverage_agent_instance.evaluate_coverage(
            "claim1", "claim_text", "src1", "src_text"
        )

        assert score is None
        assert elements is None
        assert "Failed to load coverage prompt template" in msg
        assert "not a dictionary or is missing 'template' key" in msg
        coverage_agent_instance.agent.chat.assert_not_called()

    @patch("aclarai_core.agents.coverage_agent.load_prompt_template")
    @patch("aclarai_core.agents.coverage_agent.logger")
    def test_system_prompt_update_logic(
        self, mock_logger, mock_load_prompt, coverage_agent_instance
    ):
        """Test system prompt update logic."""
        # Case 1: System prompt is present
        mock_load_prompt.return_value = {
            "template": "user_prompt_content",
            "system_prompt": "custom_system_prompt",
        }
        
        response_json = {"coverage_score": 0.9, "omitted_elements": []}
        mock_chat_response = MagicMock(spec=LlamaResponse)
        mock_chat_response.response = json.dumps(response_json)
        coverage_agent_instance.agent.chat.return_value = mock_chat_response

        coverage_agent_instance.evaluate_coverage("c1", "ct", "s1", "st")
        coverage_agent_instance.agent.update_prompts.assert_called_with(
            {"agent_worker:system_prompt": "custom_system_prompt"}
        )
        mock_logger.warning.assert_not_called()

        # Reset mocks
        coverage_agent_instance.agent.update_prompts.reset_mock()
        coverage_agent_instance.agent.chat.reset_mock()
        mock_load_prompt.reset_mock()
        mock_logger.reset_mock()

        # Case 2: System prompt is not present (is None)
        mock_load_prompt.return_value = {
            "template": "user_prompt_content_2",
            "system_prompt": None,
        }
        mock_chat_response_2 = MagicMock(spec=LlamaResponse)
        mock_chat_response_2.response = json.dumps(response_json)
        coverage_agent_instance.agent.chat.return_value = mock_chat_response_2

        coverage_agent_instance.evaluate_coverage("c2", "ct2", "s2", "st2")

        # Check if system prompt update was called (it shouldn't be)
        called_update_prompts_for_system = False
        for call_args in coverage_agent_instance.agent.update_prompts.call_args_list:
            if "agent_worker:system_prompt" in call_args[0][0]:
                called_update_prompts_for_system = True
                break
        assert not called_update_prompts_for_system

        mock_logger.warning.assert_called_once()
        assert "No system_prompt found" in mock_logger.warning.call_args[0][0]
        coverage_agent_instance.agent.chat.assert_called_once_with("user_prompt_content_2")