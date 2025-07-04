"""
Tests for the Concept Clustering Job.
These tests validate the clustering logic and configuration handling.
"""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from services.scheduler.aclarai_scheduler.concept_clustering_job import (
    ClusterAssignment,
    ConceptClusteringJob,
    ConceptClusteringJobStats,
)
from shared.aclarai_shared.config import ConceptClusteringJobConfig


class TestClusterAssignment:
    """Test the ClusterAssignment helper class."""

    def test_add_cluster(self):
        """Test adding a cluster."""
        assignment = ClusterAssignment()
        concept_ids = ["concept1", "concept2", "concept3"]

        cluster_id = assignment.add_cluster(concept_ids)

        assert cluster_id == 0
        assert assignment.get_cluster_assignments() == {
            "concept1": 0,
            "concept2": 0,
            "concept3": 0,
        }
        assert assignment.get_clusters() == {0: concept_ids}

    def test_add_outlier(self):
        """Test adding an outlier."""
        assignment = ClusterAssignment()
        assignment.add_outlier("outlier1")

        assert "outlier1" in assignment.outliers
        assert "outlier1" not in assignment.get_cluster_assignments()

    def test_multiple_clusters(self):
        """Test adding multiple clusters."""
        assignment = ClusterAssignment()

        cluster_1 = assignment.add_cluster(["concept1", "concept2"])
        cluster_2 = assignment.add_cluster(["concept3", "concept4"])

        assert cluster_1 == 0
        assert cluster_2 == 1

        assignments = assignment.get_cluster_assignments()
        assert assignments["concept1"] == 0
        assert assignments["concept2"] == 0
        assert assignments["concept3"] == 1
        assert assignments["concept4"] == 1


class TestConceptClusteringJob:
    """Test the ConceptClusteringJob class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock()
        config.scheduler.jobs.concept_clustering = ConceptClusteringJobConfig(
            enabled=True,
            manual_only=False,
            cron="0 2 * * *",
            description="Test clustering job",
            similarity_threshold=0.8,
            min_concepts=2,
            max_concepts=10,
            algorithm="dbscan",
            cache_ttl=3600,
            use_persistent_cache=True,
        )
        return config

    @pytest.fixture
    def mock_neo4j_manager(self):
        """Create a mock Neo4j manager."""
        return Mock()

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        return Mock()

    def test_init(self, mock_config, mock_neo4j_manager, mock_vector_store):
        """Test job initialization."""
        job = ConceptClusteringJob(
            config=mock_config,
            neo4j_manager=mock_neo4j_manager,
            vector_store=mock_vector_store,
        )

        assert job.config == mock_config
        assert job.neo4j_manager == mock_neo4j_manager
        assert job.vector_store == mock_vector_store
        assert job.job_config.similarity_threshold == 0.8

    def test_get_concept_data_success(self, mock_config, mock_neo4j_manager, mock_vector_store):
        """Test successful concept data retrieval."""
        # Mock Neo4j response
        mock_neo4j_manager.execute_query.return_value = [
            {"name": "concept1"},
            {"name": "concept2"},
            {"name": "concept3"},
        ]

        job = ConceptClusteringJob(
            config=mock_config,
            neo4j_manager=mock_neo4j_manager,
            vector_store=mock_vector_store,
        )

        concepts = job._get_concept_data()

        assert len(concepts) == 3
        assert concepts[0] == ("concept1", "concept1")
        mock_neo4j_manager.execute_query.assert_called_once()

    def test_get_concept_data_no_results(self, mock_config, mock_neo4j_manager, mock_vector_store):
        """Test concept data retrieval with no results."""
        mock_neo4j_manager.execute_query.return_value = []

        job = ConceptClusteringJob(
            config=mock_config,
            neo4j_manager=mock_neo4j_manager,
            vector_store=mock_vector_store,
        )

        concepts = job._get_concept_data()

        assert len(concepts) == 0

    def test_get_concept_data_exception(self, mock_config, mock_neo4j_manager, mock_vector_store):
        """Test concept data retrieval with exception."""
        mock_neo4j_manager.execute_query.side_effect = Exception("Neo4j error")

        job = ConceptClusteringJob(
            config=mock_config,
            neo4j_manager=mock_neo4j_manager,
            vector_store=mock_vector_store,
        )

        concepts = job._get_concept_data()

        assert len(concepts) == 0

    def test_perform_clustering_dbscan(self, mock_config, mock_neo4j_manager, mock_vector_store):
        """Test DBSCAN clustering."""
        job = ConceptClusteringJob(
            config=mock_config,
            neo4j_manager=mock_neo4j_manager,
            vector_store=mock_vector_store,
        )

        # Create mock embeddings for clustering
        # Two groups of similar vectors
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # Group 1
            [0.9, 0.1, 0.0],  # Group 1
            [0.0, 1.0, 0.0],  # Group 2
            [0.0, 0.9, 0.1],  # Group 2
            [0.5, 0.5, 0.5],  # Potential outlier
        ])
        concept_names = ["concept1", "concept2", "concept3", "concept4", "concept5"]

        cluster_assignment = job._perform_clustering(embeddings, concept_names)

        # Should have some clusters
        clusters = cluster_assignment.get_clusters()
        assignments = cluster_assignment.get_cluster_assignments()

        # Basic validation that clustering worked
        assert len(assignments) + len(cluster_assignment.outliers) == len(concept_names)

    def test_perform_clustering_hierarchical(self, mock_config, mock_neo4j_manager, mock_vector_store):
        """Test hierarchical clustering."""
        # Update config to use hierarchical clustering
        mock_config.scheduler.jobs.concept_clustering.algorithm = "hierarchical"

        job = ConceptClusteringJob(
            config=mock_config,
            neo4j_manager=mock_neo4j_manager,
            vector_store=mock_vector_store,
        )

        # Create mock embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ])
        concept_names = ["concept1", "concept2", "concept3", "concept4"]

        cluster_assignment = job._perform_clustering(embeddings, concept_names)

        # Should have some clusters
        assignments = cluster_assignment.get_cluster_assignments()
        assert len(assignments) == len(concept_names)

    def test_perform_clustering_empty_embeddings(self, mock_config, mock_neo4j_manager, mock_vector_store):
        """Test clustering with empty embeddings."""
        job = ConceptClusteringJob(
            config=mock_config,
            neo4j_manager=mock_neo4j_manager,
            vector_store=mock_vector_store,
        )

        embeddings = np.array([])
        concept_names = []

        cluster_assignment = job._perform_clustering(embeddings, concept_names)

        assert len(cluster_assignment.get_clusters()) == 0
        assert len(cluster_assignment.get_cluster_assignments()) == 0

    def test_update_cache(self, mock_config, mock_neo4j_manager, mock_vector_store):
        """Test cache update."""
        job = ConceptClusteringJob(
            config=mock_config,
            neo4j_manager=mock_neo4j_manager,
            vector_store=mock_vector_store,
        )

        assignment = ClusterAssignment()
        assignment.add_cluster(["concept1", "concept2"])

        result = job._update_cache(assignment)

        assert result is True
        assert job._cluster_cache is not None
        assert len(job._cluster_cache) == 2

    def test_get_cluster_assignments_cache_hit(self, mock_config, mock_neo4j_manager, mock_vector_store):
        """Test getting cluster assignments from cache."""
        job = ConceptClusteringJob(
            config=mock_config,
            neo4j_manager=mock_neo4j_manager,
            vector_store=mock_vector_store,
        )

        # Set up cache
        job._cluster_cache = {"concept1": 0, "concept2": 0}
        job._cache_timestamp = time.time()

        assignments = job.get_cluster_assignments()

        assert assignments == {"concept1": 0, "concept2": 0}

    def test_get_cluster_assignments_cache_miss(self, mock_config, mock_neo4j_manager, mock_vector_store):
        """Test getting cluster assignments with no cache."""
        job = ConceptClusteringJob(
            config=mock_config,
            neo4j_manager=mock_neo4j_manager,
            vector_store=mock_vector_store,
        )

        assignments = job.get_cluster_assignments()

        assert assignments is None

    def test_get_cluster_assignments_cache_expired(self, mock_config, mock_neo4j_manager, mock_vector_store):
        """Test getting cluster assignments with expired cache."""
        job = ConceptClusteringJob(
            config=mock_config,
            neo4j_manager=mock_neo4j_manager,
            vector_store=mock_vector_store,
        )

        # Set up expired cache
        job._cluster_cache = {"concept1": 0}
        job._cache_timestamp = time.time() - (mock_config.scheduler.jobs.concept_clustering.cache_ttl + 1)

        assignments = job.get_cluster_assignments()

        assert assignments is None

    def test_run_job_no_concepts(self, mock_config, mock_neo4j_manager, mock_vector_store):
        """Test running job with no concepts."""
        mock_neo4j_manager.execute_query.return_value = []

        job = ConceptClusteringJob(
            config=mock_config,
            neo4j_manager=mock_neo4j_manager,
            vector_store=mock_vector_store,
        )

        stats = job.run_job()

        assert stats["success"] is False
        assert stats["concepts_processed"] == 0
        assert "No concepts found for clustering" in stats["error_details"]

    def test_run_job_success(self, mock_config, mock_neo4j_manager, mock_vector_store):
        """Test successful job run."""
        # Mock Neo4j response
        mock_neo4j_manager.execute_query.return_value = [
            {"name": "concept1"},
            {"name": "concept2"},
        ]

        # Mock vector store response
        mock_vector_store.similarity_search.side_effect = [
            [
                ({"embedding": [1.0, 0.0, 0.0]}, 0.95)
            ],
            [
                ({"embedding": [0.0, 1.0, 0.0]}, 0.95)
            ],
        ]

        job = ConceptClusteringJob(
            config=mock_config,
            neo4j_manager=mock_neo4j_manager,
            vector_store=mock_vector_store,
        )

        stats = job.run_job()

        assert stats["success"] is True
        assert stats["concepts_processed"] == 2
        assert stats["cache_updated"] is True
        assert stats["duration"] > 0