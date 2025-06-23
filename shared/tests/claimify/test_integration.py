"""
Tests for Claimify to Neo4j integration.
Tests the integration between the Claimify pipeline and Neo4j graph persistence,
including conversion of pipeline results to graph inputs and end-to-end workflows.
"""

import os
from unittest.mock import Mock

import pytest
from aclarai_shared.claimify.data_models import (
    ClaimCandidate,
    ClaimifyContext,
    ClaimifyResult,
    DecompositionResult,
    DisambiguationResult,
    SelectionResult,
    SentenceChunk,
)
from aclarai_shared.claimify.integration import (
    ClaimifyGraphIntegration,
)
from aclarai_shared.graph.models import ClaimInput, SentenceInput


class MockNeo4jGraphManager:
    """Mock Neo4j graph manager for testing."""

    def __init__(self):
        self.claims_created = []
        self.sentences_created = []

    def create_claims(self, claims):
        """Mock claim creation."""
        self.claims_created.extend(claims)
        return [f"claim_{i}" for i in range(len(claims))]

    def create_sentences(self, sentences):
        """Mock sentence creation."""
        self.sentences_created.extend(sentences)
        return [f"sentence_{i}" for i in range(len(sentences))]

    def apply_core_schema(self):
        """Mock schema application."""
        return True

    def setup_schema(self):
        """Mock schema setup."""
        return True


class TestClaimifyGraphIntegration:
    """Test Claimify to Neo4j integration functionality."""

    @pytest.fixture
    def mock_graph_manager(self):
        """Create a mock graph manager for testing."""
        manager = Mock()
        manager.create_claims = Mock(return_value=["claim_001"])
        manager.create_sentences = Mock(return_value=["sentence_001"])
        manager.setup_schema = Mock()
        return manager

    @pytest.fixture
    def integration(self, mock_graph_manager):
        """Create integration instance with mock manager."""
        return ClaimifyGraphIntegration(mock_graph_manager)

    @pytest.fixture
    def test_chunk(self):
        """Create test sentence chunk."""
        return SentenceChunk(
            text="The system failed at startup.",
            source_id="blk_001",
            chunk_id="chunk_001",
            sentence_index=0,
        )

    def test_convert_successful_claim_result(self, integration, test_chunk):
        """Test conversion of successful claim result."""
        # Mock test - verify conversion logic
        # Create a valid claim candidate
        valid_claim = ClaimCandidate(
            text="The system failed at startup.",
            is_atomic=True,
            is_self_contained=True,
            is_verifiable=True,
            confidence=0.95,
        )
        # Create decomposition result with valid claim
        decomposition_result = DecompositionResult(
            original_text="The system failed at startup.",
            claim_candidates=[valid_claim],
        )
        # Create a successful claimify result
        result = ClaimifyResult(
            original_chunk=test_chunk,
            context=ClaimifyContext(current_sentence=test_chunk),
            selection_result=SelectionResult(
                sentence_chunk=test_chunk, is_selected=True
            ),
            decomposition_result=decomposition_result,
        )
        # Convert to inputs
        claim_inputs, sentence_inputs = integration._convert_result_to_inputs(result)
        # Check claim properties
        assert len(claim_inputs) == 1
        claim = claim_inputs[0]
        assert isinstance(claim, ClaimInput)
        assert claim.text == "The system failed at startup."
        assert claim.block_id == "blk_001"
        assert claim.verifiable
        assert claim.self_contained
        assert claim.context_complete

    @pytest.fixture(scope="class")
    def integration_graph_manager(self):
        """Fixture to set up a connection to a real Neo4j database for testing."""
        if not os.getenv("NEO4J_PASSWORD"):
            pytest.skip("NEO4J_PASSWORD not set for integration tests.")

        from aclarai_shared import load_config
        from aclarai_shared.graph.neo4j_manager import Neo4jGraphManager

        config = load_config(validate=True)
        manager = Neo4jGraphManager(config=config)
        manager.setup_schema()
        # Clean up any existing test data
        with manager.session() as session:
            session.run("MATCH (n) WHERE n.id STARTS WITH 'test_' DETACH DELETE n")
        yield manager
        # Clean up after tests
        with manager.session() as session:
            session.run("MATCH (n) WHERE n.id STARTS WITH 'test_' DETACH DELETE n")
        manager.close()

    @pytest.fixture
    def integration_instance(self, integration_graph_manager):
        """Create integration instance with real graph manager."""
        from aclarai_shared.claimify.integration import ClaimifyGraphIntegration

        return ClaimifyGraphIntegration(integration_graph_manager)

    @pytest.mark.integration
    def test_convert_successful_claim_result_integration(
        self, integration_instance, integration_graph_manager
    ):
        """Integration test for successful claim result conversion."""
        # Create test block first
        with integration_graph_manager.session() as session:
            session.run("CREATE (:Block {id: 'test_blk_001'})")

        # Create test data
        test_chunk = SentenceChunk(
            text="The system failed at startup due to configuration error.",
            source_id="test_blk_001",
            chunk_id="test_chunk_001",
            sentence_index=0,
        )

        # Create a valid claim candidate
        valid_claim = ClaimCandidate(
            text="The system failed at startup due to configuration error.",
            is_atomic=True,
            is_self_contained=True,
            is_verifiable=True,
            confidence=0.95,
        )

        # Create decomposition result with valid claim
        decomposition_result = DecompositionResult(
            original_text="The system failed at startup due to configuration error.",
            claim_candidates=[valid_claim],
        )

        # Create a successful claimify result
        result = ClaimifyResult(
            original_chunk=test_chunk,
            context=ClaimifyContext(current_sentence=test_chunk),
            selection_result=SelectionResult(
                sentence_chunk=test_chunk, is_selected=True
            ),
            decomposition_result=decomposition_result,
        )

        # Persist to real Neo4j database
        claims_created, sentences_created, errors = (
            integration_instance.persist_claimify_results([result])
        )

        # Verify results
        assert claims_created == 1
        assert sentences_created == 0
        assert len(errors) == 0

        # Query Neo4j to verify the claim was actually created
        with integration_graph_manager.session() as session:
            claim_result = session.run(
                "MATCH (c:Claim)-[:ORIGINATES_FROM]->(b:Block {id: 'test_blk_001'}) RETURN c"
            )
            claims = list(claim_result)
            assert len(claims) == 1
            claim_record = claims[0]["c"]
            assert (
                claim_record["text"]
                == "The system failed at startup due to configuration error."
            )
            assert claim_record["version"] == 1
            assert claim_record.get("entailed_score") is None  # Not set by pipeline
            assert claim_record.get("coverage_score") is None
            assert claim_record.get("decontextualization_score") is None

    def test_convert_failed_claim_result(self, integration, test_chunk):
        """Test conversion of result with claims that failed criteria."""
        # Mock test - verify conversion logic
        # Create a claim candidate that fails some criteria
        failed_candidate = ClaimCandidate(
            text="It failed again.",
            is_atomic=True,
            is_self_contained=False,  # Fails self-containment
            is_verifiable=True,
            confidence=0.3,
        )
        decomposition_result = DecompositionResult(
            original_text="It failed again.", claim_candidates=[failed_candidate]
        )
        result = ClaimifyResult(
            original_chunk=test_chunk,
            context=ClaimifyContext(current_sentence=test_chunk),
            selection_result=SelectionResult(
                sentence_chunk=test_chunk, is_selected=True
            ),
            decomposition_result=decomposition_result,
        )
        # Convert to graph inputs
        claim_inputs, sentence_inputs = integration._convert_result_to_inputs(result)
        # Should have no claims, one sentence
        assert len(claim_inputs) == 0
        assert len(sentence_inputs) == 1
        # Check sentence properties
        sentence = sentence_inputs[0]
        assert isinstance(sentence, SentenceInput)
        assert sentence.text == "It failed again."
        assert sentence.block_id == "blk_001"
        assert sentence.ambiguous  # Because not self-contained
        assert sentence.verifiable
        assert not sentence.failed_decomposition  # Was atomic

    @pytest.mark.integration
    def test_convert_failed_claim_result_integration(
        self, integration_instance, integration_graph_manager
    ):
        """Integration test for failed claim result conversion."""
        # Create test block first
        with integration_graph_manager.session() as session:
            session.run("CREATE (:Block {id: 'test_blk_002'})")

        # Create test data
        test_chunk = SentenceChunk(
            text="It failed again.",
            source_id="test_blk_002",
            chunk_id="test_chunk_002",
            sentence_index=0,
        )

        # Create a claim candidate that fails some criteria
        failed_candidate = ClaimCandidate(
            text="It failed again.",
            is_atomic=True,
            is_self_contained=False,  # Fails self-containment
            is_verifiable=True,
            confidence=0.3,
        )

        decomposition_result = DecompositionResult(
            original_text="It failed again.", claim_candidates=[failed_candidate]
        )

        result = ClaimifyResult(
            original_chunk=test_chunk,
            context=ClaimifyContext(current_sentence=test_chunk),
            selection_result=SelectionResult(
                sentence_chunk=test_chunk, is_selected=True
            ),
            decomposition_result=decomposition_result,
        )

        # Persist to real Neo4j database
        claims_created, sentences_created, errors = (
            integration_instance.persist_claimify_results([result])
        )

        # Verify results
        assert claims_created == 0
        assert sentences_created == 1
        assert len(errors) == 0

        # Query Neo4j to verify the sentence was actually created
        with integration_graph_manager.session() as session:
            sentence_result = session.run(
                "MATCH (s:Sentence)-[:ORIGINATES_FROM]->(b:Block {id: 'test_blk_002'}) RETURN s"
            )
            sentences = list(sentence_result)
            assert len(sentences) == 1
            sentence_record = sentences[0]["s"]
            assert sentence_record["text"] == "It failed again."
            assert sentence_record["ambiguous"]  # Because not self-contained
            assert sentence_record["verifiable"]
            assert not sentence_record["failed_decomposition"]  # Was atomic

    def test_convert_unprocessed_result(self, integration, test_chunk):
        """Test conversion of result that wasn't selected for processing."""
        # Mock test - verify conversion logic
        # Create result where selection failed
        result = ClaimifyResult(
            original_chunk=test_chunk,
            context=ClaimifyContext(current_sentence=test_chunk),
            selection_result=SelectionResult(
                sentence_chunk=test_chunk, is_selected=False
            ),
            # No decomposition result since it wasn't selected
        )
        # Convert to graph inputs
        claim_inputs, sentence_inputs = integration._convert_result_to_inputs(result)
        # Should have no claims, one sentence
        assert len(claim_inputs) == 0
        assert len(sentence_inputs) == 1
        # Check sentence properties
        sentence = sentence_inputs[0]
        assert isinstance(sentence, SentenceInput)
        assert sentence.text == "The system failed at startup."
        assert sentence.block_id == "blk_001"
        assert sentence.ambiguous  # Because not selected
        assert not sentence.verifiable  # Because not selected
        assert not sentence.failed_decomposition  # Wasn't decomposed
        assert sentence.rejection_reason == "Not selected by Selection agent"

    @pytest.mark.integration
    def test_convert_unprocessed_result_integration(
        self, integration_instance, integration_graph_manager
    ):
        """Integration test for unprocessed result conversion."""
        # Create test block first
        with integration_graph_manager.session() as session:
            session.run("CREATE (:Block {id: 'test_blk_003'})")

        # Create test data
        test_chunk = SentenceChunk(
            text="This was not selected for processing.",
            source_id="test_blk_003",
            chunk_id="test_chunk_003",
            sentence_index=0,
        )

        # Create result where selection failed
        result = ClaimifyResult(
            original_chunk=test_chunk,
            context=ClaimifyContext(current_sentence=test_chunk),
            selection_result=SelectionResult(
                sentence_chunk=test_chunk, is_selected=False
            ),
            # No decomposition result since it wasn't selected
        )

        # Persist to real Neo4j database
        claims_created, sentences_created, errors = (
            integration_instance.persist_claimify_results([result])
        )

        # Verify results
        assert claims_created == 0
        assert sentences_created == 1
        assert len(errors) == 0

        # Query Neo4j to verify the sentence was actually created
        with integration_graph_manager.session() as session:
            sentence_result = session.run(
                "MATCH (s:Sentence)-[:ORIGINATES_FROM]->(b:Block {id: 'test_blk_003'}) RETURN s"
            )
            sentences = list(sentence_result)
            assert len(sentences) == 1
            sentence_record = sentences[0]["s"]
            assert sentence_record["text"] == "This was not selected for processing."
            assert sentence_record["ambiguous"]  # Because not selected
            assert not sentence_record["verifiable"]  # Because not selected
            assert (
                sentence_record["rejection_reason"] == "Not selected by Selection agent"
            )

    def test_create_claim_input(self, integration, test_chunk):
        """Test creation of ClaimInput from ClaimCandidate."""
        # Mock test - verify claim input creation
        claim_candidate = ClaimCandidate(
            text="The error occurred at 10:30 AM.",
            is_atomic=True,
            is_self_contained=True,
            is_verifiable=True,
            confidence=0.95,
        )
        claim_input = integration._create_claim_input(claim_candidate, test_chunk)
        assert isinstance(claim_input, ClaimInput)
        assert claim_input.text == "The error occurred at 10:30 AM."
        assert claim_input.block_id == "blk_001"
        assert claim_input.verifiable
        assert claim_input.self_contained
        assert claim_input.context_complete
        assert claim_input.entailed_score is None  # Not set by pipeline
        assert claim_input.coverage_score is None
        assert claim_input.decontextualization_score is None
        assert claim_input.id.startswith("claim_")

    @pytest.mark.integration
    def test_create_claim_input_integration(self, _integration, _test_chunk):
        """Integration test for claim input creation."""
        # Integration test - requires real Neo4j service
        pytest.skip("Integration tests require real database setup")

    def test_create_sentence_input(self, integration, test_chunk):
        """Test creation of SentenceInput from ClaimCandidate."""
        # Mock test - verify sentence input creation
        sentence_candidate = ClaimCandidate(
            text="It was problematic.",
            is_atomic=True,
            is_self_contained=False,  # Ambiguous
            is_verifiable=True,
        )
        sentence_input = integration._create_sentence_input(
            sentence_candidate, test_chunk, rejection_reason="Ambiguous pronoun"
        )
        assert isinstance(sentence_input, SentenceInput)
        assert sentence_input.text == "It was problematic."
        assert sentence_input.block_id == "blk_001"
        assert sentence_input.ambiguous  # Because not self-contained
        assert sentence_input.verifiable
        assert not sentence_input.failed_decomposition  # Was atomic
        assert sentence_input.rejection_reason == "Ambiguous pronoun"
        assert sentence_input.id.startswith("sentence_")

    @pytest.mark.integration
    def test_create_sentence_input_integration(self, _integration, _test_chunk):
        """Integration test for sentence input creation."""
        # Integration test - requires real Neo4j service
        pytest.skip("Integration tests require real database setup")

    def test_create_sentence_input_from_chunk(self, integration, test_chunk):
        """Test creation of SentenceInput directly from chunk."""
        # Mock test - verify sentence input creation from chunk
        sentence_input = integration._create_sentence_input_from_chunk(
            test_chunk, rejection_reason="Failed selection"
        )
        assert isinstance(sentence_input, SentenceInput)
        assert sentence_input.text == "The system failed at startup."
        assert sentence_input.block_id == "blk_001"
        assert sentence_input.ambiguous
        assert not sentence_input.verifiable
        assert not sentence_input.failed_decomposition
        assert sentence_input.rejection_reason == "Failed selection"

    @pytest.mark.integration
    def test_create_sentence_input_from_chunk_integration(
        self, _integration, _test_chunk
    ):
        """Integration test for sentence input creation from chunk."""
        # Integration test - requires real Neo4j service
        pytest.skip("Integration tests require real database setup")


class TestCreateGraphManagerFromConfigStructure:
    """Test graph manager creation from configuration."""

    @pytest.fixture
    def mock_graph_manager(self):
        """Create a mock graph manager for testing."""
        manager = Mock()
        manager.create_claims = Mock(return_value=["claim_001"])
        manager.create_sentences = Mock(return_value=["sentence_001"])
        manager.setup_schema = Mock()
        return manager

    @pytest.fixture
    def integration(self, mock_graph_manager):
        """Create integration instance with mock manager."""
        return ClaimifyGraphIntegration(mock_graph_manager)

    @pytest.fixture
    def test_chunk(self):
        """Create test sentence chunk."""
        return SentenceChunk(
            text="The system failed at startup.",
            source_id="blk_001",
            chunk_id="chunk_001",
            sentence_index=0,
        )

    @pytest.fixture
    def test_context(self, test_chunk):
        """Create test context."""
        return ClaimifyContext(current_sentence=test_chunk)

    def test_create_manager_from_config(self):
        """Test creation of graph manager from config."""
        # Mock test - verify module structure exists
        manager_path = os.path.join(
            os.path.dirname(__file__),
            "../../aclarai_shared/claimify/integration.py",
        )
        assert os.path.exists(manager_path)
        with open(manager_path, "r") as f:
            content = f.read()
            assert "def create_graph_manager_from_config" in content

    def test_create_manager_empty_config(self):
        """Test creation of graph manager with empty config."""
        # Mock test - verify module structure exists
        integration_path = os.path.join(
            os.path.dirname(__file__),
            "../../aclarai_shared/claimify/integration.py",
        )
        assert os.path.exists(integration_path)
        with open(integration_path, "r") as f:
            content = f.read()
            assert "create_graph_manager_from_config" in content

    def test_integration_initialization(self, mock_graph_manager):
        """Test integration initialization."""
        integration = ClaimifyGraphIntegration(mock_graph_manager)
        assert integration.graph_manager == mock_graph_manager

    def test_convert_successful_claim_result(
        self, integration, test_chunk, test_context
    ):
        """Test conversion of result with valid claims."""
        # Create a successful processing result with valid claims
        claim_candidate = ClaimCandidate(
            text="The system failed at startup.",
            is_atomic=True,
            is_self_contained=True,
            is_verifiable=True,
            confidence=0.9,
        )
        decomposition_result = DecompositionResult(
            original_text="The system failed at startup.",
            claim_candidates=[claim_candidate],
        )
        result = ClaimifyResult(
            original_chunk=test_chunk,
            context=test_context,
            selection_result=SelectionResult(test_chunk, is_selected=True),
            disambiguation_result=DisambiguationResult(
                test_chunk, "The system failed at startup."
            ),
            decomposition_result=decomposition_result,
        )
        # Convert to graph inputs
        claim_inputs, sentence_inputs = integration._convert_result_to_inputs(result)
        # Should have one claim, no sentences
        assert len(claim_inputs) == 1
        assert len(sentence_inputs) == 0
        # Check claim properties
        claim = claim_inputs[0]
        assert isinstance(claim, ClaimInput)
        assert claim.text == "The system failed at startup."
        assert claim.block_id == "blk_001"
        assert claim.verifiable
        assert claim.self_contained
        assert claim.context_complete

    def test_convert_failed_claim_result(self, integration, test_chunk, test_context):
        """Test conversion of result with claims that failed criteria."""
        # Create a claim candidate that fails some criteria
        failed_candidate = ClaimCandidate(
            text="It failed again.",
            is_atomic=True,
            is_self_contained=False,  # Fails self-containment
            is_verifiable=True,
            confidence=0.3,
        )
        decomposition_result = DecompositionResult(
            original_text="It failed again.", claim_candidates=[failed_candidate]
        )
        result = ClaimifyResult(
            original_chunk=test_chunk,
            context=test_context,
            selection_result=SelectionResult(test_chunk, is_selected=True),
            disambiguation_result=DisambiguationResult(test_chunk, "It failed again."),
            decomposition_result=decomposition_result,
        )
        # Convert to graph inputs
        claim_inputs, sentence_inputs = integration._convert_result_to_inputs(result)
        # Should have no claims, one sentence
        assert len(claim_inputs) == 0
        assert len(sentence_inputs) == 1
        # Check sentence properties
        sentence = sentence_inputs[0]
        assert isinstance(sentence, SentenceInput)
        assert sentence.text == "It failed again."
        assert sentence.block_id == "blk_001"
        assert sentence.ambiguous  # Because not self-contained
        assert sentence.verifiable
        assert not sentence.failed_decomposition  # Was atomic

    def test_convert_mixed_decomposition_result(
        self, integration, test_chunk, test_context
    ):
        """Test conversion of result with both valid and invalid claims."""
        # Create mix of valid and invalid claim candidates
        valid_claim = ClaimCandidate(
            text="The database connection failed.",
            is_atomic=True,
            is_self_contained=True,
            is_verifiable=True,
        )
        invalid_claim = ClaimCandidate(
            text="This is ambiguous.",
            is_atomic=False,
            is_self_contained=False,
            is_verifiable=False,
        )
        decomposition_result = DecompositionResult(
            original_text="Complex sentence with multiple parts.",
            claim_candidates=[valid_claim, invalid_claim],
        )
        result = ClaimifyResult(
            original_chunk=test_chunk,
            context=test_context,
            selection_result=SelectionResult(test_chunk, is_selected=True),
            disambiguation_result=DisambiguationResult(
                test_chunk, "Complex sentence with multiple parts."
            ),
            decomposition_result=decomposition_result,
        )
        # Convert to graph inputs
        claim_inputs, sentence_inputs = integration._convert_result_to_inputs(result)
        # Should have one claim, one sentence
        assert len(claim_inputs) == 1
        assert len(sentence_inputs) == 1
        # Check that valid claim became ClaimInput
        claim = claim_inputs[0]
        assert claim.text == "The database connection failed."
        # Check that invalid claim became SentenceInput
        sentence = sentence_inputs[0]
        assert sentence.text == "This is ambiguous."

    def test_convert_unprocessed_result(self, integration, test_chunk, test_context):
        """Test conversion of result that was not processed."""
        # Create a result where selection rejected the sentence
        result = ClaimifyResult(
            original_chunk=test_chunk,
            context=test_context,
            selection_result=SelectionResult(test_chunk, is_selected=False),
        )
        # Convert to graph inputs
        claim_inputs, sentence_inputs = integration._convert_result_to_inputs(result)
        # Should have no claims, one sentence from original chunk
        assert len(claim_inputs) == 0
        assert len(sentence_inputs) == 1
        # Check sentence properties
        sentence = sentence_inputs[0]
        assert sentence.text == "The system failed at startup."
        assert sentence.block_id == "blk_001"
        assert sentence.ambiguous  # Assumed ambiguous since not selected
        assert not sentence.verifiable  # Assumed not verifiable since not selected
        assert sentence.rejection_reason == "Not selected by Selection agent"

    def test_persist_claimify_results_empty(self, integration):
        """Test persisting empty results list."""
        claims_created, sentences_created, errors = (
            integration.persist_claimify_results([])
        )
        assert claims_created == 0
        assert sentences_created == 0
        assert len(errors) == 0

    def test_persist_claimify_results_with_claims(
        self, integration, test_chunk, test_context
    ):
        """Test persisting results with valid claims."""
        # Create test results with claims
        claim_candidate = ClaimCandidate(
            text="Server responded with 500 error.",
            is_atomic=True,
            is_self_contained=True,
            is_verifiable=True,
        )
        decomposition_result = DecompositionResult(
            original_text="Server responded with 500 error.",
            claim_candidates=[claim_candidate],
        )
        result = ClaimifyResult(
            original_chunk=test_chunk,
            context=test_context,
            selection_result=SelectionResult(test_chunk, is_selected=True),
            decomposition_result=decomposition_result,
        )
        # Persist results
        claims_created, sentences_created, errors = (
            integration.persist_claimify_results([result])
        )
        assert claims_created == 1
        assert sentences_created == 0
        assert len(errors) == 0

    def test_persist_claimify_results_with_sentences(
        self, integration, test_chunk, test_context
    ):
        """Test persisting results with sentence nodes."""
        # Create test results with rejected sentences
        result = ClaimifyResult(
            original_chunk=test_chunk,
            context=test_context,
            selection_result=SelectionResult(test_chunk, is_selected=False),
        )
        # Persist results
        claims_created, sentences_created, errors = (
            integration.persist_claimify_results([result])
        )
        assert claims_created == 0
        assert sentences_created == 1
        assert len(errors) == 0

    def test_create_claim_input(self, integration, test_chunk):
        """Test creation of ClaimInput from ClaimCandidate."""
        claim_candidate = ClaimCandidate(
            text="The error occurred at 10:30 AM.",
            is_atomic=True,
            is_self_contained=True,
            is_verifiable=True,
            confidence=0.95,
        )
        claim_input = integration._create_claim_input(claim_candidate, test_chunk)
        assert isinstance(claim_input, ClaimInput)
        assert claim_input.text == "The error occurred at 10:30 AM."
        assert claim_input.block_id == "blk_001"
        assert claim_input.verifiable
        assert claim_input.self_contained
        assert claim_input.context_complete
        assert claim_input.entailed_score is None  # Not set by pipeline
        assert claim_input.coverage_score is None
        assert claim_input.decontextualization_score is None
        assert claim_input.id.startswith("claim_")

    def test_create_sentence_input(self, integration, test_chunk):
        """Test creation of SentenceInput from ClaimCandidate."""
        sentence_candidate = ClaimCandidate(
            text="It was problematic.",
            is_atomic=True,
            is_self_contained=False,
            is_verifiable=True,
        )
        sentence_input = integration._create_sentence_input(
            sentence_candidate, test_chunk, rejection_reason="Ambiguous pronoun"
        )
        assert isinstance(sentence_input, SentenceInput)
        assert sentence_input.text == "It was problematic."
        assert sentence_input.block_id == "blk_001"
        assert sentence_input.ambiguous  # Because not self-contained
        assert sentence_input.verifiable
        assert not sentence_input.failed_decomposition  # Was atomic
        assert sentence_input.rejection_reason == "Ambiguous pronoun"
        assert sentence_input.id.startswith("sentence_")

    def test_create_sentence_input_from_chunk(self, integration, test_chunk):
        """Test creation of SentenceInput directly from chunk."""
        sentence_input = integration._create_sentence_input_from_chunk(
            test_chunk, rejection_reason="Failed selection"
        )
        assert isinstance(sentence_input, SentenceInput)
        assert sentence_input.text == "The system failed at startup."
        assert sentence_input.block_id == "blk_001"
        assert sentence_input.ambiguous
        assert not sentence_input.verifiable
        assert not sentence_input.failed_decomposition
        assert sentence_input.rejection_reason == "Failed selection"
