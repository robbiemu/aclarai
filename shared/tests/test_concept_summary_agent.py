import unittest
from unittest.mock import MagicMock, patch

from aclarai_shared.concept_summary_agent.agent import ConceptSummaryAgent
from llama_index.core.base.response.schema import Response


class TestConceptSummaryAgent(unittest.TestCase):
    """
    Tests for the ConceptSummaryAgent, focusing on the integration of sub-agents
    and other critical functionalities.
    """

    def setUp(self):
        """Set up a reusable agent for testing."""
        self.mock_config = MagicMock()
        self.mock_config.llm.provider = "openai"
        self.mock_config.llm.model = "test-model"
        self.mock_config.llm.api_key = "test-key"
        self.mock_config.paths.vault = "/tmp/vault"
        self.mock_config.paths.tier3 = "concepts"
        self.mock_config.concept_summaries.max_examples = 5
        self.mock_config.concept_summaries.skip_if_no_claims = True
        self.mock_config.concept_summaries.include_see_also = True

        # Patch dependencies
        self.patcher_create_agent = patch(
            "aclarai_shared.concept_summary_agent.agent.create_retrieval_agent"
        )
        self.patcher_tool_factory = patch(
            "aclarai_shared.concept_summary_agent.agent.ToolFactory"
        )
        self.patcher_neo4j = patch(
            "aclarai_shared.concept_summary_agent.agent.Neo4jGraphManager"
        )
        self.patcher_vector_store = patch(
            "aclarai_shared.concept_summary_agent.agent.aclaraiVectorStoreManager"
        )
        self.patcher_openai = patch("aclarai_shared.concept_summary_agent.agent.OpenAI")
        self.patcher_write_atomic = patch(
            "aclarai_shared.concept_summary_agent.agent.write_file_atomically"
        )

        self.mock_create_retrieval_agent = self.patcher_create_agent.start()
        self.mock_tool_factory = self.patcher_tool_factory.start()
        self.mock_neo4j_manager = self.patcher_neo4j.start()
        self.mock_vector_store_manager = self.patcher_vector_store.start()
        self.mock_openai = self.patcher_openai.start()
        self.mock_write_atomic = self.patcher_write_atomic.start()

        # Mock the sub-agents
        self.mock_claim_agent = MagicMock()
        self.mock_claim_agent.chat.return_value = Response(
            response=["claim 1 ^id1", "claim 2 ^id2"]
        )
        self.mock_related_concepts_agent = MagicMock()
        self.mock_related_concepts_agent.chat.return_value = Response(
            response=["related concept 1", "related concept 2"]
        )
        self.mock_create_retrieval_agent.side_effect = [
            self.mock_claim_agent,
            self.mock_related_concepts_agent,
        ]

        self.agent = ConceptSummaryAgent(config=self.mock_config)
        self.agent.llm = MagicMock()

    def tearDown(self):
        self.patcher_create_agent.stop()
        self.patcher_tool_factory.stop()
        self.patcher_neo4j.stop()
        self.patcher_vector_store.stop()
        self.patcher_openai.stop()
        self.patcher_write_atomic.stop()

    def test_generate_concept_slug_and_filename(self):
        """Test the slug and filename generation."""
        self.assertEqual(
            self.agent.generate_concept_slug("Test/Concept"), "test_concept"
        )
        self.assertEqual(
            self.agent.generate_concept_slug("API/REST endpoints"), "api_rest_endpoints"
        )

        self.assertEqual(
            self.agent.generate_concept_filename("Machine Learning"),
            "Machine_Learning.md",
        )

        self.assertEqual(
            self.agent.generate_concept_filename("Test/Concept"), "Test_Concept.md"
        )

    def test_generate_concept_page_with_sub_agents(self):
        """Test that generate_concept_page correctly calls the sub-agents."""
        # Arrange
        self.mock_claim_agent.chat.return_value = Response(
            response=["claim 1 ^id1", "claim 2 ^id2"]
        )
        self.mock_related_concepts_agent.chat.return_value = Response(
            response=["related concept 1", "related concept 2"]
        )
        self.agent.llm.complete.return_value = Response(response="Generated content")
        concept = {"id": "1", "text": "Test Concept"}

        # Act
        self.agent.generate_concept_page(concept)

        # Assert
        self.mock_claim_agent.chat.assert_called_once_with(
            'Find all claims related to the concept: "Test Concept"'
        )
        self.mock_related_concepts_agent.chat.assert_called_once_with(
            'Find concepts semantically similar to: "Test Concept"'
        )
        self.agent.llm.complete.assert_called_once()
        self.mock_write_atomic.assert_called_once()

    def test_template_fallback_on_llm_failure(self):
        """Test that the agent falls back to template generation if the LLM fails."""
        # Arrange
        self.mock_claim_agent.chat.return_value = Response(response=["claim 1 ^id1"])
        self.mock_related_concepts_agent.chat.return_value = Response(
            response=["related concept 1"]
        )
        self.agent.llm.complete.side_effect = Exception("LLM is down")
        concept = {"id": "1", "text": "Test Concept"}

        # Act
        self.agent.generate_concept_page(concept)

        # Assert
        self.mock_write_atomic.assert_called_once()
        written_content = self.mock_write_atomic.call_args[0][1]
        self.assertIn("## Concept: Test Concept", written_content)
        self.assertIn("- claim 1 ^id1", written_content)
        self.assertIn("- [[related concept 1]]", written_content)

    def test_skip_if_no_claims(self):
        """Test that a concept is skipped if skip_if_no_claims is True and no claims are found."""
        # Arrange
        self.agent.skip_if_no_claims = True
        self.mock_claim_agent.chat.return_value = Response(response=[])
        self.mock_related_concepts_agent.chat.return_value = Response(
            response=["related concept 1"]
        )
        concept = {"id": "1", "text": "Test Concept"}

        # Act
        result = self.agent.generate_concept_page(concept)

        # Assert
        self.assertFalse(result)
        self.agent.llm.complete.assert_not_called()
        self.mock_write_atomic.assert_not_called()

    def test_run_agent_workflow(self):
        """Test the end-to-end run_agent workflow."""
        # Arrange
        concepts = [{"id": "1", "text": "Concept 1"}, {"id": "2", "text": "Concept 2"}]
        self.agent.get_canonical_concepts = MagicMock(return_value=concepts)
        self.mock_claim_agent.chat.return_value = Response(response=["claim 1 ^id1"])
        self.mock_related_concepts_agent.chat.return_value = Response(
            response=["related 1"]
        )
        self.agent.llm.complete.return_value = Response(response="LLM content")

        # Act
        result = self.agent.run_agent()

        # Assert
        self.assertEqual(result["concepts_processed"], 2)
        self.assertEqual(result["concepts_generated"], 2)
        self.assertEqual(result["concepts_skipped"], 0)
        self.assertEqual(self.agent.llm.complete.call_count, 2)
        self.assertEqual(self.mock_write_atomic.call_count, 2)


if __name__ == "__main__":
    unittest.main()
