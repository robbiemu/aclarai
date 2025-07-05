#!/usr/bin/env python3
"""
Script to test the Subject Summary Agent.
This script demonstrates the usage of the SubjectSummaryAgent.
"""

from typing import cast
from unittest.mock import MagicMock

from aclarai_shared.graph.neo4j_manager import Neo4jGraphManager
from aclarai_shared.subject_summary_agent import SubjectSummaryAgent


def main():
    """Demonstrate the Subject Summary Agent functionality."""
    print("Testing Subject Summary Agent...")

    # Create mock configuration
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_config.llm.provider = "openai"
    mock_config.llm.api_key = None  # Will fall back to template
    mock_config.subject_summaries.model = "gpt-4"
    mock_config.subject_summaries.allow_web_search = False
    mock_config.subject_summaries.min_concepts = 2
    mock_config.subject_summaries.max_concepts = 10
    mock_config.subject_summaries.skip_if_incoherent = False

    # Create mock clustering job with sample data
    class MockClusteringJob:
        def get_cluster_assignments(self):
            return {
                "Machine Learning": 0,
                "Deep Learning": 0,
                "Neural Networks": 0,
                "Data Science": 1,
                "Statistics": 1,
                "Analytics": 1,
            }

    # Create mock Neo4j manager
    class MockNeo4jManager:
        def execute_query(self, query):
            # Return sample shared claims
            if "claim" in query.lower():
                return [
                    {
                        "text": "Machine learning is a subset of artificial intelligence",
                        "aclarai_id": "claim_123",
                        "related_concepts": ["Machine Learning", "Deep Learning"],
                    }
                ]
            # Return sample summaries
            return [
                {
                    "text": "Data science combines statistics and machine learning",
                    "aclarai_id": "summary_456",
                    "related_concepts": ["Data Science", "Statistics"],
                }
            ]

    # Initialize the agent
    agent = SubjectSummaryAgent(
        config=mock_config,
        neo4j_manager=cast(Neo4jGraphManager, MockNeo4jManager()),
        clustering_job=MockClusteringJob(),
        vector_store_manager=None,
    )

    print("✓ Agent initialized successfully")

    # Test slug generation
    slug = agent.generate_subject_slug("Machine Learning & AI")
    print(f"✓ Generated slug: {slug}")

    # Test filename generation
    filename = agent.generate_subject_filename("Machine Learning & AI")
    print(f"✓ Generated filename: {filename}")

    # Test subject name generation
    concepts = ["Machine Learning", "Deep Learning", "Neural Networks"]
    subject_name = agent.generate_subject_name(concepts)
    print(f"✓ Generated subject name: {subject_name}")

    # Test content generation (template mode since no real LLM)
    context = {
        "shared_claims": [{"text": "ML is a subset of AI", "aclarai_id": "claim_123"}],
        "common_summaries": [],
    }
    content = agent._generate_template_content(subject_name, concepts, context)
    print("✓ Generated template content:")
    print("=" * 50)
    print(content)
    print("=" * 50)

    # Test cluster concept extraction
    cluster_assignments = {
        "Machine Learning": 0,
        "Deep Learning": 0,
        "Neural Networks": 0,
        "Data Science": 1,
    }
    cluster_0_concepts = agent.get_cluster_concepts(cluster_assignments, 0)
    print(f"✓ Cluster 0 concepts: {cluster_0_concepts}")

    # Test run agent (will use mock data)
    result = agent.run_agent()
    print(f"✓ Agent run result: {result}")

    print("\n✨ All tests completed successfully!")


if __name__ == "__main__":
    main()
