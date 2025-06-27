from unittest.mock import MagicMock, patch  # Added call for retry check

import pytest
from aclarai_core.agents.entailment_agent import EntailmentAgent  # Corrected
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
def entailment_agent_instance(mock_llm, mock_tool_factory, mock_config):
    with patch(
        "aclarai_core.agents.entailment_agent.CodeActAgent"
    ) as MockCodeActAgent:  # Corrected
        mock_agent_internal_instance = MockCodeActAgent.from_tools.return_value
        agent_under_test = EntailmentAgent(
            llm=mock_llm, tool_factory=mock_tool_factory, config=mock_config
        )
        agent_under_test.agent = mock_agent_internal_instance
        return agent_under_test


class TestEntailmentAgent:
    @patch("aclarai_core.agents.entailment_agent.load_prompt_template")  # Corrected
    def test_evaluate_entailment_success(
        self, mock_load_prompt, entailment_agent_instance
    ):
        mock_load_prompt.return_value = {
            "template": "user_prompt",
            "system_prompt": "system_prompt",
        }
        mock_chat_response = MagicMock(spec=LlamaResponse)
        mock_chat_response.response = "0.85"
        entailment_agent_instance.agent.chat.return_value = mock_chat_response

        score, msg = entailment_agent_instance.evaluate_entailment(
            "claim1", "claim_text", "src1", "src_text"
        )

        assert score == 0.85
        assert msg == "success"
        entailment_agent_instance.agent.chat.assert_called_once_with("user_prompt")
        entailment_agent_instance.agent.update_prompts.assert_called_once_with(
            {"agent_worker:system_prompt": "system_prompt"}
        )

    @patch("aclarai_core.agents.entailment_agent.load_prompt_template")  # Corrected
    def test_evaluate_entailment_llm_error_with_retries(
        self, mock_load_prompt, entailment_agent_instance
    ):
        mock_load_prompt.return_value = {"template": "user_prompt"}
        entailment_agent_instance.agent.chat.side_effect = ValueError("LLM API error")

        max_retries = entailment_agent_instance.max_retries
        score, msg = entailment_agent_instance.evaluate_entailment(
            "claim1", "claim_text", "src1", "src_text"
        )

        assert score is None
        assert "LLM API error" in msg
        assert f"after {max_retries} retries" in msg
        assert entailment_agent_instance.agent.chat.call_count == max_retries

    @patch("aclarai_core.agents.entailment_agent.load_prompt_template")  # Corrected
    def test_evaluate_entailment_invalid_score_format(
        self, mock_load_prompt, entailment_agent_instance
    ):
        mock_load_prompt.return_value = {"template": "user_prompt"}
        mock_chat_response = MagicMock(spec=LlamaResponse)
        mock_chat_response.response = "not a float"
        entailment_agent_instance.agent.chat.return_value = mock_chat_response

        score, msg = entailment_agent_instance.evaluate_entailment(
            "claim1", "claim_text", "src1", "src_text"
        )

        assert score is None
        assert "invalid score format" in msg
        assert "Expected a float, got: 'not a float'" in msg

    @patch("aclarai_core.agents.entailment_agent.load_prompt_template")  # Corrected
    def test_evaluate_entailment_score_out_of_range_high(
        self, mock_load_prompt, entailment_agent_instance
    ):
        mock_load_prompt.return_value = {"template": "user_prompt"}
        mock_chat_response = MagicMock(spec=LlamaResponse)
        mock_chat_response.response = "1.5"
        entailment_agent_instance.agent.chat.return_value = mock_chat_response

        score, msg = entailment_agent_instance.evaluate_entailment(
            "claim1", "claim_text", "src1", "src_text"
        )

        assert score is None
        assert "out-of-range score" in msg
        assert "Score: 1.5" in msg

    @patch("aclarai_core.agents.entailment_agent.load_prompt_template")  # Corrected
    def test_evaluate_entailment_score_out_of_range_low(
        self, mock_load_prompt, entailment_agent_instance
    ):
        mock_load_prompt.return_value = {"template": "user_prompt"}
        mock_chat_response = MagicMock(spec=LlamaResponse)
        mock_chat_response.response = "-0.5"
        entailment_agent_instance.agent.chat.return_value = mock_chat_response

        score, msg = entailment_agent_instance.evaluate_entailment(
            "claim1", "claim_text", "src1", "src_text"
        )

        assert score is None
        assert "out-of-range score" in msg
        assert "Score: -0.5" in msg

    @patch("aclarai_core.agents.entailment_agent.load_prompt_template")  # Corrected
    def test_evaluate_entailment_prompt_load_failure(
        self, mock_load_prompt, entailment_agent_instance
    ):
        mock_load_prompt.side_effect = FileNotFoundError("Prompt file missing")

        score, msg = entailment_agent_instance.evaluate_entailment(
            "claim1", "claim_text", "src1", "src_text"
        )

        assert score is None
        assert "Failed to load entailment prompt template" in msg
        assert "Prompt file missing" in msg
        entailment_agent_instance.agent.chat.assert_not_called()

    @patch("aclarai_core.agents.entailment_agent.load_prompt_template")  # Corrected
    @patch("aclarai_core.agents.entailment_agent.logger")  # Corrected
    def test_system_prompt_update_logic(
        self, mock_logger, mock_load_prompt, entailment_agent_instance
    ):
        # Case 1: System prompt is present
        mock_load_prompt.return_value = {
            "template": "user_prompt_content",
            "system_prompt": "custom_system_prompt",
        }
        mock_chat_response = MagicMock(spec=LlamaResponse)
        mock_chat_response.response = "0.7"
        entailment_agent_instance.agent.chat.return_value = mock_chat_response

        entailment_agent_instance.evaluate_entailment("c1", "ct", "s1", "st")
        entailment_agent_instance.agent.update_prompts.assert_called_with(
            {"agent_worker:system_prompt": "custom_system_prompt"}
        )
        mock_logger.warning.assert_not_called()

        entailment_agent_instance.agent.update_prompts.reset_mock()
        entailment_agent_instance.agent.chat.reset_mock()
        mock_load_prompt.reset_mock()
        mock_logger.reset_mock()

        # Case 2: System prompt is not present (is None)
        mock_load_prompt.return_value = {
            "template": "user_prompt_content_2",
            "system_prompt": None,
        }
        mock_chat_response_2 = MagicMock(spec=LlamaResponse)
        mock_chat_response_2.response = "0.6"
        entailment_agent_instance.agent.chat.return_value = mock_chat_response_2

        entailment_agent_instance.evaluate_entailment("c2", "ct2", "s2", "st2")

        called_update_prompts_for_system = False
        for call_args in entailment_agent_instance.agent.update_prompts.call_args_list:
            if "agent_worker:system_prompt" in call_args[0][0]:
                called_update_prompts_for_system = True
                break
        assert not called_update_prompts_for_system

        mock_logger.warning.assert_called_once()
        assert "No system_prompt found" in mock_logger.warning.call_args[0][0]
        entailment_agent_instance.agent.chat.assert_called_once_with(
            "user_prompt_content_2"
        )

    @patch("aclarai_core.agents.entailment_agent.load_prompt_template")  # Corrected
    def test_malformed_prompt_template_data(
        self, mock_load_prompt, entailment_agent_instance
    ):
        mock_load_prompt.return_value = "just a string"

        score, msg = entailment_agent_instance.evaluate_entailment(
            "claim1", "claim_text", "src1", "src_text"
        )

        assert score is None
        assert "Failed to load entailment prompt template" in msg
        assert "not a dictionary or is missing 'template' key" in msg
        entailment_agent_instance.agent.chat.assert_not_called()

        mock_load_prompt.reset_mock()
        entailment_agent_instance.agent.chat.reset_mock()

        mock_load_prompt.return_value = {"user_facing_prompt": "user_prompt"}
        score, msg = entailment_agent_instance.evaluate_entailment(
            "claim1", "claim_text", "src1", "src_text"
        )
        assert score is None
        assert "Failed to load entailment prompt template" in msg
        assert "not a dictionary or is missing 'template' key" in msg
        entailment_agent_instance.agent.chat.assert_not_called()
