"""
Tests for the Subject Summary Agent.
This module tests the functionality of the SubjectSummaryAgent that generates
[[Subject:XYZ]] Markdown pages for concept clusters.
"""

from unittest.mock import MagicMock, Mock

from aclarai_shared.subject_summary_agent import SubjectSummaryAgent


class MockNeo4jManager:
    """Mock Neo4j manager for testing."""

    def __init__(self):
        self.execute_query = Mock()

    def close(self):
        pass


class MockClusteringJob:
    """Mock clustering job for testing."""

    def __init__(self):
        self.cluster_assignments = None

    def get_cluster_assignments(self):
        return self.cluster_assignments

    def set_cluster_assignments(self, assignments):
        self.cluster_assignments = assignments


def test_subject_summary_agent_initialization():
    """Test SubjectSummaryAgent initialization with default config."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_config.llm.provider = "openai"
    mock_config.llm.model = "gpt-4"
    mock_config.llm.api_key = "test-key"
    mock_config.subject_summaries.model = "gpt-4"
    mock_config.subject_summaries.allow_web_search = True
    mock_config.subject_summaries.min_concepts = 3
    mock_config.subject_summaries.max_concepts = 15
    mock_config.subject_summaries.skip_if_incoherent = False
    mock_config.model_dump.return_value = {}

    mock_neo4j = MockNeo4jManager()
    mock_clustering_job = MockClusteringJob()

    agent = SubjectSummaryAgent(
        config=mock_config,
        neo4j_manager=mock_neo4j,
        clustering_job=mock_clustering_job,
    )

    assert agent.config == mock_config
    assert agent.neo4j_manager == mock_neo4j
    assert agent.clustering_job == mock_clustering_job
    assert agent.model_name == "gpt-4"


def test_generate_subject_slug():
    """Test subject slug generation."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_config.llm.provider = "openai"
    mock_config.subject_summaries.model = "gpt-4"
    mock_config.subject_summaries.allow_web_search = False
    mock_config.subject_summaries.min_concepts = 3
    mock_config.subject_summaries.max_concepts = 15
    mock_config.subject_summaries.skip_if_incoherent = False

    mock_neo4j = MockNeo4jManager()
    mock_clustering_job = MockClusteringJob()

    agent = SubjectSummaryAgent(
        config=mock_config,
        neo4j_manager=mock_neo4j,
        clustering_job=mock_clustering_job,
        vector_store_manager=None,
    )

    # Test normal subject name
    slug = agent.generate_subject_slug("Machine Learning & AI")
    assert slug == "machine_learning_ai"

    # Test subject name with special characters
    slug = agent.generate_subject_slug("GPU/CPU Performance")
    assert slug == "gpu_cpu_performance"

    # Test empty name
    slug = agent.generate_subject_slug("")
    assert slug == "untitled_subject"


def test_generate_subject_filename():
    """Test subject filename generation."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_config.llm.provider = "openai"
    mock_config.subject_summaries.model = "gpt-4"
    mock_config.subject_summaries.allow_web_search = False
    mock_config.subject_summaries.min_concepts = 3
    mock_config.subject_summaries.max_concepts = 15
    mock_config.subject_summaries.skip_if_incoherent = False

    mock_neo4j = MockNeo4jManager()
    mock_clustering_job = MockClusteringJob()

    agent = SubjectSummaryAgent(
        config=mock_config,
        neo4j_manager=mock_neo4j,
        clustering_job=mock_clustering_job,
        vector_store_manager=None,
    )

    # Test normal subject name
    filename = agent.generate_subject_filename("Machine Learning")
    assert filename == "Subject_Machine_Learning.md"

    # Test subject name with special characters
    filename = agent.generate_subject_filename("GPU/CPU & Performance")
    assert filename == "Subject_GPU_CPU_Performance.md"


def test_generate_subject_name():
    """Test subject name generation from concepts."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_config.llm.provider = "openai"
    mock_config.subject_summaries.model = "gpt-4"
    mock_config.subject_summaries.allow_web_search = False
    mock_config.subject_summaries.min_concepts = 3
    mock_config.subject_summaries.max_concepts = 15
    mock_config.subject_summaries.skip_if_incoherent = False

    mock_neo4j = MockNeo4jManager()
    mock_clustering_job = MockClusteringJob()

    agent = SubjectSummaryAgent(
        config=mock_config,
        neo4j_manager=mock_neo4j,
        clustering_job=mock_clustering_job,
        vector_store_manager=None,
    )

    # Test single concept
    name = agent.generate_subject_name(["Machine Learning"])
    assert name == "Machine Learning"

    # Test two concepts
    name = agent.generate_subject_name(["Machine Learning", "AI"])
    assert name == "Machine Learning & AI"

    # Test three concepts
    name = agent.generate_subject_name(["Machine Learning", "AI", "Deep Learning"])
    assert name == "Machine Learning & AI & Deep Learning"

    # Test many concepts
    name = agent.generate_subject_name(["ML", "AI", "DL", "CNN", "RNN"])
    assert name == "ML & Related Topics"

    # Test empty concepts
    name = agent.generate_subject_name([])
    assert name == "Untitled Subject"


def test_get_cluster_concepts():
    """Test getting concepts for a specific cluster."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_config.llm.provider = "openai"
    mock_config.subject_summaries.model = "gpt-4"
    mock_config.subject_summaries.allow_web_search = False
    mock_config.subject_summaries.min_concepts = 3
    mock_config.subject_summaries.max_concepts = 15
    mock_config.subject_summaries.skip_if_incoherent = False

    mock_neo4j = MockNeo4jManager()
    mock_clustering_job = MockClusteringJob()

    agent = SubjectSummaryAgent(
        config=mock_config,
        neo4j_manager=mock_neo4j,
        clustering_job=mock_clustering_job,
        vector_store_manager=None,
    )

    # Test cluster concept extraction
    cluster_assignments = {
        "Machine Learning": 0,
        "Deep Learning": 0,
        "Neural Networks": 0,
        "Data Science": 1,
        "Statistics": 1,
    }

    cluster_0_concepts = agent.get_cluster_concepts(cluster_assignments, 0)
    assert set(cluster_0_concepts) == {"Machine Learning", "Deep Learning", "Neural Networks"}

    cluster_1_concepts = agent.get_cluster_concepts(cluster_assignments, 1)
    assert set(cluster_1_concepts) == {"Data Science", "Statistics"}

    # Test non-existent cluster
    cluster_2_concepts = agent.get_cluster_concepts(cluster_assignments, 2)
    assert cluster_2_concepts == []


def test_should_skip_cluster():
    """Test cluster skipping logic."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_config.llm.provider = "openai"
    mock_config.subject_summaries.model = "gpt-4"
    mock_config.subject_summaries.allow_web_search = False
    mock_config.subject_summaries.min_concepts = 3
    mock_config.subject_summaries.max_concepts = 5
    mock_config.subject_summaries.skip_if_incoherent = True

    mock_neo4j = MockNeo4jManager()
    mock_clustering_job = MockClusteringJob()

    agent = SubjectSummaryAgent(
        config=mock_config,
        neo4j_manager=mock_neo4j,
        clustering_job=mock_clustering_job,
        vector_store_manager=None,
    )

    # Test too few concepts
    concepts = ["Machine Learning", "AI"]
    context = {"shared_claims": [], "common_summaries": []}
    assert agent.should_skip_cluster(concepts, context) is True

    # Test too many concepts
    concepts = ["ML", "AI", "DL", "CNN", "RNN", "LSTM"]
    assert agent.should_skip_cluster(concepts, context) is True

    # Test incoherent cluster (no shared claims or summaries)
    concepts = ["Machine Learning", "AI", "Deep Learning"]
    context = {"shared_claims": [], "common_summaries": []}
    assert agent.should_skip_cluster(concepts, context) is True

    # Test coherent cluster (has shared claims)
    context = {"shared_claims": [{"text": "AI is useful"}], "common_summaries": []}
    assert agent.should_skip_cluster(concepts, context) is False

    # Test coherent cluster (has common summaries)
    context = {"shared_claims": [], "common_summaries": [{"text": "Summary of AI"}]}
    assert agent.should_skip_cluster(concepts, context) is False


def test_get_cluster_assignments_no_clustering_job():
    """Test getting cluster assignments with no clustering job."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_config.llm.provider = "openai"
    mock_config.subject_summaries.model = "gpt-4"
    mock_config.subject_summaries.allow_web_search = False
    mock_config.subject_summaries.min_concepts = 3
    mock_config.subject_summaries.max_concepts = 15
    mock_config.subject_summaries.skip_if_incoherent = False

    mock_neo4j = MockNeo4jManager()

    agent = SubjectSummaryAgent(
        config=mock_config,
        neo4j_manager=mock_neo4j,
        clustering_job=None,
        vector_store_manager=None,
    )

    assignments = agent.get_cluster_assignments()
    assert assignments is None


def test_get_cluster_assignments_with_clustering_job():
    """Test getting cluster assignments with clustering job."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_config.llm.provider = "openai"
    mock_config.subject_summaries.model = "gpt-4"
    mock_config.subject_summaries.allow_web_search = False
    mock_config.subject_summaries.min_concepts = 3
    mock_config.subject_summaries.max_concepts = 15
    mock_config.subject_summaries.skip_if_incoherent = False

    mock_neo4j = MockNeo4jManager()
    mock_clustering_job = MockClusteringJob()

    test_assignments = {"ML": 0, "AI": 0, "Statistics": 1}
    mock_clustering_job.set_cluster_assignments(test_assignments)

    agent = SubjectSummaryAgent(
        config=mock_config,
        neo4j_manager=mock_neo4j,
        clustering_job=mock_clustering_job,
        vector_store_manager=None,
    )

    assignments = agent.get_cluster_assignments()
    assert assignments == test_assignments


def test_retrieve_shared_claims():
    """Test retrieving shared claims for concepts."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_config.llm.provider = "openai"
    mock_config.subject_summaries.model = "gpt-4"
    mock_config.subject_summaries.allow_web_search = False
    mock_config.subject_summaries.min_concepts = 3
    mock_config.subject_summaries.max_concepts = 15
    mock_config.subject_summaries.skip_if_incoherent = False

    mock_neo4j = MockNeo4jManager()
    mock_clustering_job = MockClusteringJob()

    # Mock the Neo4j query result
    mock_claims = [
        {
            "text": "Machine learning is a subset of AI",
            "aclarai_id": "claim_123",
            "related_concepts": ["Machine Learning", "AI"],
        }
    ]
    mock_neo4j.execute_query.return_value = mock_claims

    agent = SubjectSummaryAgent(
        config=mock_config,
        neo4j_manager=mock_neo4j,
        clustering_job=mock_clustering_job,
        vector_store_manager=None,
    )

    concept_names = ["Machine Learning", "AI"]
    claims = agent.retrieve_shared_claims(concept_names)

    assert len(claims) == 1
    assert claims[0]["text"] == "Machine learning is a subset of AI"
    assert claims[0]["aclarai_id"] == "claim_123"
    assert "Machine Learning" in claims[0]["related_concepts"]
    assert "AI" in claims[0]["related_concepts"]


def test_run_agent_no_clusters():
    """Test running the agent with no cluster assignments."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_config.llm.provider = "openai"
    mock_config.subject_summaries.model = "gpt-4"
    mock_config.subject_summaries.allow_web_search = False
    mock_config.subject_summaries.min_concepts = 3
    mock_config.subject_summaries.max_concepts = 15
    mock_config.subject_summaries.skip_if_incoherent = False

    mock_neo4j = MockNeo4jManager()
    mock_clustering_job = MockClusteringJob()
    mock_clustering_job.set_cluster_assignments(None)

    agent = SubjectSummaryAgent(
        config=mock_config,
        neo4j_manager=mock_neo4j,
        clustering_job=mock_clustering_job,
        vector_store_manager=None,
    )

    result = agent.run_agent()

    assert result["success"] is True
    assert result["clusters_processed"] == 0
    assert result["subjects_generated"] == 0
    assert result["subjects_skipped"] == 0
    assert result["errors"] == 0


def test_generate_template_content():
    """Test template-based content generation."""
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_config.llm.provider = "openai"
    mock_config.subject_summaries.model = "gpt-4"
    mock_config.subject_summaries.allow_web_search = False
    mock_config.subject_summaries.min_concepts = 3
    mock_config.subject_summaries.max_concepts = 15
    mock_config.subject_summaries.skip_if_incoherent = False

    mock_neo4j = MockNeo4jManager()
    mock_clustering_job = MockClusteringJob()

    agent = SubjectSummaryAgent(
        config=mock_config,
        neo4j_manager=mock_neo4j,
        clustering_job=mock_clustering_job,
        vector_store_manager=None,
    )

    subject_name = "Machine Learning & AI"
    concepts = ["Machine Learning", "AI", "Deep Learning"]
    context = {
        "shared_claims": [{"text": "AI is useful", "aclarai_id": "claim_123"}],
        "common_summaries": [],
    }

    content = agent._generate_template_content(subject_name, concepts, context)

    assert "## Subject: Machine Learning & AI" in content
    assert "### Included Concepts" in content
    assert "[[Machine Learning]]" in content
    assert "[[AI]]" in content
    assert "[[Deep Learning]]" in content
    assert "### Common Threads" in content
    assert "<!-- aclarai:id=subject_machine_learning_ai ver=1 -->" in content
    assert "^subject_machine_learning_ai" in content


import pytest


@pytest.mark.integration
def test_cypher_query_parameterization_integration():
    """
    Integration test to validate Cypher queries use parameterized queries.
    This test validates the agent's interaction with Neo4j to ensure security.
    """
    mock_config = MagicMock()
    mock_config.paths.vault = "/tmp/test_vault"
    mock_config.paths.tier3 = "concepts"
    mock_config.llm.provider = "openai"
    mock_config.subject_summaries.model = "gpt-4"
    mock_config.subject_summaries.allow_web_search = False
    mock_config.subject_summaries.min_concepts = 3
    mock_config.subject_summaries.max_concepts = 15
    mock_config.subject_summaries.skip_if_incoherent = False

    mock_neo4j = MockNeo4jManager()
    mock_clustering_job = MockClusteringJob()

    agent = SubjectSummaryAgent(
        config=mock_config,
        neo4j_manager=mock_neo4j,
        clustering_job=mock_clustering_job,
        vector_store_manager=None,
    )

    # Test that shared claims query uses parameterized queries
    concepts = ["Machine Learning", "AI"]
    agent.retrieve_shared_claims(concepts)

    # Verify that execute_query was called with parameters
    mock_neo4j.execute_query.assert_called()
    call_args = mock_neo4j.execute_query.call_args
    
    # Check that the query uses parameterized format
    query = call_args[0][0]  # First positional argument (query)
    assert "$concept_names" in query
    assert "'" not in query or query.count("'") <= 2  # No inline string interpolation
    
    # Check that parameters were passed
    kwargs = call_args[1]  # Keyword arguments
    assert "parameters" in kwargs
    assert "concept_names" in kwargs["parameters"]
    assert kwargs["parameters"]["concept_names"] == concepts
    assert kwargs.get("read_only") is True


@pytest.mark.integration  
def test_prompt_template_loading_integration():
    """
    Integration test to validate prompt templates are loaded from YAML files.
    This test ensures the agent properly uses external prompt templates.
    """
    from pathlib import Path
    
    # Check that the required prompt files exist
    prompts_dir = Path(__file__).parent.parent / "shared" / "aclarai_shared" / "prompts"
    
    required_prompts = [
        "subject_summary_definition.yaml",
        "subject_summary_concept_blurb.yaml", 
        "subject_summary_common_threads.yaml"
    ]
    
    for prompt_file in required_prompts:
        prompt_path = prompts_dir / prompt_file
        assert prompt_path.exists(), f"Prompt file {prompt_file} not found at {prompt_path}"
        
        # Verify the file contains required YAML structure
        import yaml
        with open(prompt_path, 'r') as f:
            prompt_data = yaml.safe_load(f)
            
        assert "role" in prompt_data
        assert "description" in prompt_data
        assert "template" in prompt_data
        assert "variables" in prompt_data
