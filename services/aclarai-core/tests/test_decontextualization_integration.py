"""
Tests for the decontextualization evaluation workflow, including the agent,
Neo4j storage, and Markdown updates.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest
from aclarai_core.agents.decontextualization_agent import DecontextualizationAgent
from aclarai_core.graph.claim_evaluation_graph_service import (
    ClaimEvaluationGraphService,
)
from aclarai_core.markdown.markdown_updater_service import MarkdownUpdaterService
from aclarai_shared.config import aclaraiConfig
from aclarai_shared.tools.factory import ToolFactory
from aclarai_shared.tools.vector_store_manager import VectorStore, VectorStoreManager
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from neo4j import GraphDatabase


# --- Mock Objects for Dependencies ---
class MockLLM:
    """Mocks the LlamaIndex LLM for predictable responses."""

    def __init__(self, response_text: str = "0.85"):
        self.response_text = response_text
        self.chat_messages = []

    async def achat(self, messages, **kwargs):  # noqa: ARG002
        self.chat_messages.append(messages)
        return ChatResponse(
            message=MagicMock(content=self.response_text),
            raw={"response": self.response_text},
        )  # type: ignore

    def chat(self, messages, **kwargs):  # noqa: ARG002
        self.chat_messages.append(messages)
        return ChatResponse(
            message=MagicMock(content=self.response_text),
            raw={"response": self.response_text},
        )  # type: ignore


class MockVectorStoreManager(VectorStoreManager):
    """Mocks the VectorStoreManager to avoid a real Postgres dependency."""

    def get_store(self, name: str) -> Optional[VectorStore]:
        if name == "utterances":
            mock_store = MagicMock(spec=VectorStore)
            mock_store.query.return_value = VectorStoreQueryResult(
                nodes=[], similarities=[]
            )
            return mock_store
        return None


# --- Pytest Fixtures ---
@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for our test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="class")
@pytest.mark.integration
def integration_neo4j_driver():
    """Fixture to set up a connection to a real Neo4j database for testing."""
    if not os.getenv("NEO4J_PASSWORD"):
        pytest.skip("NEO4J_PASSWORD not set for integration tests.")

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()

    # Clean up before tests
    with driver.session() as session:
        session.run("MATCH (n) WHERE n.id STARTS WITH 'integ_test_' DETACH DELETE n")
        session.run(
            "MATCH (n) WHERE n.claim_id STARTS WITH 'integ_test_' DETACH DELETE n"
        )

    yield driver

    # Clean up after tests
    with driver.session() as session:
        session.run("MATCH (n) WHERE n.id STARTS WITH 'integ_test_' DETACH DELETE n")
        session.run(
            "MATCH (n) WHERE n.claim_id STARTS WITH 'integ_test_' DETACH DELETE n"
        )
    driver.close()


@pytest.fixture
def temp_markdown_file_with_content():
    """Creates a temporary markdown file with a versioned block."""
    content = """
# Integration Test Document

This is the first block of text.

Here is the claim we will evaluate. <!-- aclarai:id=integ_test_block_1 ver=1 -->
^integ_test_block_1

Some text after the block.
"""
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, encoding="utf-8", suffix=".md"
    ) as tmp_file:
        tmp_file.write(content)
        filepath = tmp_file.name

    yield Path(filepath)

    os.unlink(filepath)


@pytest.mark.integration
class TestDecontextualizationWorkflowIntegration:
    """End-to-end tests for the decontextualization workflow."""

    @pytest.mark.asyncio
    async def test_full_decontextualization_workflow(
        self, integration_neo4j_driver, temp_markdown_file_with_content
    ):
        """
        Verifies the full workflow:
        1. Agent evaluates a claim and returns a score.
        2. Graph service persists the score to the correct relationship in Neo4j.
        3. Markdown service updates the .md file with the score and increments the version.
        """
        # --- ARRANGE ---
        # 1. Mock dependencies that are not under test (LLM, Config, VectorStore)
        mock_llm = MockLLM(response_text="0.85")
        mock_config = aclaraiConfig
        mock_vsm = MockVectorStoreManager()
        mock_tool_factory = ToolFactory(config={}, vector_store_manager=mock_vsm)

        # 2. Instantiate the services with a mix of real and mock components
        agent = DecontextualizationAgent(
            llm=mock_llm, tool_factory=mock_tool_factory, config=mock_config
        )
        graph_service = ClaimEvaluationGraphService(
            neo4j_driver=integration_neo4j_driver, config=mock_config
        )
        markdown_service = MarkdownUpdaterService()

        # 3. Prepare test data and pre-populate the real Neo4j DB
        claim_id = "integ_test_claim_1"
        block_id = "integ_test_block_1"
        claim_text = "Here is the claim we will evaluate."
        source_text = "This is the source context for the claim."

        with integration_neo4j_driver.session() as session:
            session.run(
                """
                MERGE (c:Claim {id: $claim_id, text: $claim_text})
                MERGE (b:Block {id: $block_id})
                MERGE (c)-[:ORIGINATES_FROM]->(b)
            """,
                claim_id=claim_id,
                block_id=block_id,
                claim_text=claim_text,
            )

        # --- ACT ---
        # 1. Run the agent evaluation
        score, message = await agent.evaluate_claim_decontextualization(
            claim_id=claim_id,
            claim_text=claim_text,
            source_id=block_id,
            source_text=source_text,
        )

        # 2. Persist the score to the graph
        graph_success = graph_service.update_decontextualization_score(
            claim_id=claim_id, block_id=block_id, score=score
        )

        # 3. Persist the score to the Markdown file
        md_success = markdown_service.add_or_update_decontextualization_score(
            filepath_str=str(temp_markdown_file_with_content),
            block_id=block_id,
            score=score,
        )

        # --- ASSERT ---
        # 1. Agent assertions
        assert score == 0.85
        assert message == "success"

        # 2. Graph service assertions
        assert graph_success is True
        with integration_neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (:Claim {id: $claim_id})-[r:ORIGINATES_FROM]->(:Block {id: $block_id})
                RETURN r.decontextualization_score AS score
            """,
                claim_id=claim_id,
                block_id=block_id,
            )
            record = result.single()
            assert record is not None
            assert record["score"] == 0.85

        # 3. Markdown service assertions
        assert md_success is True
        updated_content = temp_markdown_file_with_content.read_text(encoding="utf-8")

        # Verify the score comment was added
        assert "<!-- aclarai:decontextualization_score=0.85 -->" in updated_content

        # Verify the version number was incremented from 1 to 2
        assert "<!-- aclarai:id=integ_test_block_1 ver=2 -->" in updated_content
        assert "ver=1" not in updated_content  # Ensure old version is gone
