"""
Integration test for the coverage evaluation agent to verify end-to-end functionality.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from aclarai_core.agents.coverage_agent import CoverageAgent
from aclarai_core.graph.claim_evaluation_graph_service import ClaimEvaluationGraphService
from aclarai_core.markdown.markdown_updater_service import MarkdownUpdaterService
from aclarai_shared.config import aclaraiConfig
from aclarai_shared.tools.factory import ToolFactory
from llama_index.core.base.response.schema import Response as LlamaResponse
from llama_index.core.llms.llm import LLM as LlamaLLM


class TestCoverageAgentIntegration:
    """Integration tests for the coverage evaluation agent."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock(spec=aclaraiConfig)
        config.processing = MagicMock()
        config.processing.retries = {"max_attempts": 3}
        return config

    @pytest.fixture
    def mock_llm(self):
        return MagicMock(spec=LlamaLLM)

    @pytest.fixture
    def mock_tool_factory(self):
        factory = MagicMock(spec=ToolFactory)
        factory.get_tools_for_agent.return_value = []
        return factory

    @pytest.fixture
    def mock_neo4j_driver(self):
        return MagicMock()

    @pytest.fixture
    def coverage_agent(self, mock_llm, mock_tool_factory, mock_config):
        with patch("aclarai_core.agents.coverage_agent.CodeActAgent") as MockCodeActAgent:
            mock_agent_internal_instance = MockCodeActAgent.from_tools.return_value
            agent = CoverageAgent(
                llm=mock_llm, tool_factory=mock_tool_factory, config=mock_config
            )
            agent.agent = mock_agent_internal_instance
            return agent

    @pytest.fixture
    def graph_service(self, mock_neo4j_driver, mock_config):
        return ClaimEvaluationGraphService(mock_neo4j_driver, mock_config)

    @pytest.fixture
    def markdown_service(self):
        return MarkdownUpdaterService()

    @patch("aclarai_core.agents.coverage_agent.load_prompt_template")
    def test_coverage_evaluation_end_to_end_workflow(
        self, 
        mock_load_prompt,
        coverage_agent,
        graph_service,
        markdown_service,
        mock_neo4j_driver
    ):
        """Test the complete coverage evaluation workflow from agent to storage."""
        
        # Setup: Mock prompt loading
        mock_load_prompt.return_value = {
            "template": "Evaluate coverage for claim: {claim_text} from source: {source_text}",
            "system_prompt": "You are a coverage evaluator."
        }
        
        # Setup: Mock LLM response with coverage results
        response_json = {
            "coverage_score": 0.75,
            "omitted_elements": [
                {"text": "European Commission", "significance": "Key organization"},
                {"text": "€7.5 billion", "significance": "Specific amount"}
            ],
            "reasoning": "Claim captures main idea but omits key details"
        }
        
        mock_chat_response = MagicMock(spec=LlamaResponse)
        mock_chat_response.response = json.dumps(response_json)
        coverage_agent.agent.chat.return_value = mock_chat_response
        
        # Step 1: Evaluate coverage with the agent
        claim_id = "claim_123"
        claim_text = "EU approved climate funding in 2023"
        source_id = "block_456"
        source_text = "In 2023, the European Commission approved €7.5 billion in climate funding for renewable energy projects"
        
        score, elements, status = coverage_agent.evaluate_coverage(
            claim_id, claim_text, source_id, source_text
        )
        
        # Verify agent results
        assert score == 0.75
        assert len(elements) == 2
        assert elements[0]["text"] == "European Commission"
        assert elements[1]["text"] == "€7.5 billion"
        assert status == "success"
        
        # Step 2: Store coverage score in Neo4j
        with patch.object(graph_service, 'update_relationship_score') as mock_update_score:
            mock_update_score.return_value = True
            
            score_stored = graph_service.update_coverage_score(claim_id, source_id, score)
            assert score_stored is True
            mock_update_score.assert_called_once_with(claim_id, source_id, "coverage_score", 0.75)
        
        # Step 3: Create Element nodes and OMITS relationships
        mock_session = MagicMock()
        mock_neo4j_driver.session.return_value.__enter__.return_value = mock_session
        mock_neo4j_driver.session.return_value.__exit__.return_value = None
        
        mock_result = MagicMock()
        mock_record = MagicMock()
        mock_record.__getitem__.return_value = 2  # created_elements count
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        
        elements_created = graph_service.create_element_nodes_and_omits_relationships(
            claim_id, elements
        )
        assert elements_created is True
        
        # Verify the Cypher query was called correctly
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        parameters = call_args[0][1]
        
        assert "CREATE (e:Element {text: element.text, significance: element.significance})" in query
        assert "CREATE (c)-[:OMITS]->(e)" in query
        assert parameters["claim_id"] == claim_id
        assert parameters["elements"] == elements
        
        # Step 4: Update Markdown file with coverage score
        with patch.object(markdown_service, '_atomic_write') as mock_atomic_write, \
             patch('builtins.open') as mock_open, \
             patch('pathlib.Path.exists') as mock_exists:
            
            mock_exists.return_value = True
            mock_atomic_write.return_value = True
            
            # Mock file content with aclarai:id
            mock_file_content = """
# Sample Document

This is a test claim. <!-- aclarai:id=claim_123 ver=1 -->

Some other content.
"""
            mock_open.return_value.__enter__.return_value.read.return_value = mock_file_content
            
            markdown_updated = markdown_service.add_or_update_score(
                "/path/to/file.md", claim_id, "coverage_score", score
            )
            
            assert markdown_updated is True
            mock_atomic_write.assert_called_once()
            
            # Verify the updated content includes the coverage score
            updated_content = mock_atomic_write.call_args[0][1]
            assert "<!-- aclarai:coverage_score=0.75 -->" in updated_content
            assert "<!-- aclarai:id=claim_123 ver=2 -->" in updated_content

    @patch("aclarai_core.agents.coverage_agent.load_prompt_template")
    def test_coverage_evaluation_handles_null_score(
        self,
        mock_load_prompt,
        coverage_agent,
        graph_service,
        markdown_service
    ):
        """Test handling of null coverage scores throughout the workflow."""
        
        # Setup: Mock prompt loading
        mock_load_prompt.return_value = {
            "template": "user_prompt",
            "system_prompt": "system_prompt"
        }
        
        # Setup: Mock LLM failure (agent returns null)
        coverage_agent.agent.chat.side_effect = Exception("LLM error")
        
        # Step 1: Agent evaluation fails and returns null
        score, elements, status = coverage_agent.evaluate_coverage(
            "claim_123", "claim_text", "source_456", "source_text"
        )
        
        assert score is None
        assert elements is None
        assert "error" in status.lower()
        
        # Step 2: Store null score in Neo4j
        with patch.object(graph_service, 'update_relationship_score') as mock_update_score:
            mock_update_score.return_value = True
            
            score_stored = graph_service.update_coverage_score("claim_123", "source_456", None)
            assert score_stored is True
            mock_update_score.assert_called_once_with("claim_123", "source_456", "coverage_score", None)
        
        # Step 3: No Element nodes should be created for null evaluation
        elements_created = graph_service.create_element_nodes_and_omits_relationships(
            "claim_123", elements or []
        )
        assert elements_created is True  # Empty list succeeds
        
        # Step 4: Markdown should still be updated with null score
        with patch.object(markdown_service, '_atomic_write') as mock_atomic_write, \
             patch('builtins.open') as mock_open, \
             patch('pathlib.Path.exists') as mock_exists:
            
            mock_exists.return_value = True
            mock_atomic_write.return_value = True
            
            mock_file_content = """
# Sample Document

This is a test claim. <!-- aclarai:id=claim_123 ver=1 -->
"""
            mock_open.return_value.__enter__.return_value.read.return_value = mock_file_content
            
            markdown_updated = markdown_service.add_or_update_score(
                "/path/to/file.md", "claim_123", "coverage_score", None
            )
            
            assert markdown_updated is True
            
            # Verify null score is properly formatted in markdown
            updated_content = mock_atomic_write.call_args[0][1]
            assert "<!-- aclarai:coverage_score=null -->" in updated_content
            assert "<!-- aclarai:id=claim_123 ver=2 -->" in updated_content

    def test_coverage_agent_tool_integration(self, mock_tool_factory, mock_llm, mock_config):
        """Test that the coverage agent properly integrates with the ToolFactory."""
        
        # Mock tools from factory with proper BaseTool structure
        from llama_index.core.tools import BaseTool
        
        mock_neo4j_tool = MagicMock(spec=BaseTool)
        mock_neo4j_tool.metadata = MagicMock()
        mock_neo4j_tool.metadata.name = "neo4j_query_tool"
        
        mock_vector_tool = MagicMock(spec=BaseTool)
        mock_vector_tool.metadata = MagicMock()
        mock_vector_tool.metadata.name = "vector_search_utterances"
        
        mock_web_tool = MagicMock(spec=BaseTool)
        mock_web_tool.metadata = MagicMock()
        mock_web_tool.metadata.name = "web_search"
        
        mock_tool_factory.get_tools_for_agent.return_value = [
            mock_neo4j_tool, mock_vector_tool, mock_web_tool
        ]
        
        with patch("aclarai_core.agents.coverage_agent.CodeActAgent") as MockCodeActAgent:
            agent = CoverageAgent(
                llm=mock_llm, tool_factory=mock_tool_factory, config=mock_config
            )
            
            # Verify tools were requested for coverage_agent
            mock_tool_factory.get_tools_for_agent.assert_called_once_with("coverage_agent")
            
            # Verify agent was initialized with the tools
            MockCodeActAgent.assert_called_once()
            call_kwargs = MockCodeActAgent.call_args[1]
            assert call_kwargs["tools"] == [mock_neo4j_tool, mock_vector_tool, mock_web_tool]
            assert call_kwargs["llm"] == mock_llm
            
            # Verify tool names are stored for logging
            assert "neo4j_query_tool" in agent.tool_names
            assert "vector_search_utterances" in agent.tool_names
            assert "web_search" in agent.tool_names