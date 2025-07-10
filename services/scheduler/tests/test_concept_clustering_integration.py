import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from services.scheduler.aclarai_scheduler.concept_clustering_job import (
    ConceptClusteringJob,
)
from shared.aclarai_shared import load_config
from shared.aclarai_shared.embedding.models import EmbeddedChunk
from shared.aclarai_shared.embedding.storage import aclaraiVectorStore
from shared.aclarai_shared.graph.models import ConceptInput
from shared.aclarai_shared.graph.neo4j_manager import Neo4jGraphManager


@pytest.mark.integration
class TestConceptClusteringIntegration:
    """
    Integration tests for the ConceptClusteringJob, requiring live
    Neo4j and PostgreSQL (with pgvector) services.
    """

    @pytest.fixture(scope="class")
    def live_services(self):
        """
        Fixture to set up live Neo4j and Vector Store connections
        and populate them with test data.
        """
        config = load_config(validate=True)
        neo4j_manager = Neo4jGraphManager(config)
        vector_store = aclaraiVectorStore(config)

        # Define test concepts and their embeddings
        # Cluster 1: GPU-related errors
        gpu_concepts = [
            "GPU Error",
            "CUDA Failure",
            "VRAM Out of Memory",
            "Nvidia Driver Conflict",
        ]
        gpu_embeddings = [
            np.array([0.9, 0.1, 0.1, 0.0] * 96),  # Base
            np.array([0.91, 0.11, 0.09, 0.01] * 96),  # Similar
            np.array([0.89, 0.09, 0.11, -0.01] * 96),  # Similar
            np.array([0.9, 0.1, 0.12, 0.02] * 96),  # Similar
        ]

        # Cluster 2: Database-related concepts
        db_concepts = ["Database Connection", "PostgreSQL Query", "SQLAlchemy Session"]
        db_embeddings = [
            np.array([0.1, 0.9, 0.1, 0.0] * 96),  # Base
            np.array([0.11, 0.91, 0.09, 0.01] * 96),  # Similar
            np.array([0.09, 0.89, 0.11, -0.01] * 96),  # Similar
        ]

        # Outliers
        outlier_concepts = ["Python Programming", "Gradio UI"]
        outlier_embeddings = [
            np.array([0.0, 0.0, 0.9, 0.1] * 96),
            np.array([0.5, 0.5, 0.5, 0.5] * 96),
        ]

        all_concepts = gpu_concepts + db_concepts + outlier_concepts
        all_embeddings = gpu_embeddings + db_embeddings + outlier_embeddings

        # --- Clean and Populate Neo4j and Vector Store ---
        # 1. Clear previous test data
        neo4j_manager.execute_query(
            "MATCH (n:Concept) DETACH DELETE n", allow_dangerous_operations=True
        )
        vector_store.delete_chunks_by_block_id(
            "concept_GPU Error"
        )  # Example delete to clear old data
        vector_store.delete_chunks_by_block_id(
            "concept_Database Connection"
        )  # Example delete

        # 2. Populate Neo4j with Concept nodes
        concept_inputs = [
            ConceptInput(
                text=name,
                source_candidate_id=f"cand_{i}",
                source_node_id=f"node_{i}",
                source_node_type="test",
                aclarai_id=f"doc_{i}",
            )
            for i, name in enumerate(all_concepts)
        ]
        neo4j_manager.create_concepts(concept_inputs)

        # 3. Populate Vector Store with embeddings
        embedded_chunks = []
        for _i, (name, embedding) in enumerate(
            zip(all_concepts, all_embeddings, strict=False)
        ):
            chunk_meta = vector_store.chunker.chunk_utterance_block(
                name, f"concept_{name.replace(' ', '_')}"
            )[0]
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_metadata=chunk_meta,
                    embedding=embedding.tolist(),
                    model_name="test_model",
                    embedding_dim=384,
                )
            )
        vector_store.store_embeddings(embedded_chunks)

        yield config, neo4j_manager, vector_store

        # --- Teardown ---
        neo4j_manager.execute_query(
            "MATCH (n:Concept) DETACH DELETE n", allow_dangerous_operations=True
        )
        neo4j_manager.close()

    def test_concept_clustering_job_e2e(self, live_services):
        """
        Tests the end-to-end flow of the ConceptClusteringJob.
        It runs the job against live services populated with test data
        and verifies the correctness of the generated clusters.
        """
        config, neo4j_manager, vector_store = live_services

        # Configure the job for the test scenario
        config.scheduler.jobs.concept_clustering.min_concepts = 3
        config.scheduler.jobs.concept_clustering.max_concepts = 10
        config.scheduler.jobs.concept_clustering.similarity_threshold = 0.95
        config.scheduler.jobs.concept_clustering.algorithm = "dbscan"

        # Initialize and run the job
        job = ConceptClusteringJob(
            config=config, neo4j_manager=neo4j_manager, vector_store=vector_store
        )
        stats = job.run_job()

        # 1. Verify job statistics
        assert stats["success"] is True
        assert stats["concepts_processed"] == 9
        assert stats["clusters_formed"] == 2
        assert stats["concepts_clustered"] == 7  # 4 GPU + 3 DB
        assert stats["concepts_outliers"] == 2
        assert stats["cache_updated"] is True

        # 2. Verify cluster assignments from cache
        assignments = job.get_cluster_assignments()
        assert assignments is not None

        # Group concepts by their assigned cluster ID
        clusters = {}
        for concept, cluster_id in assignments.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(concept)

        # 3. Assert cluster composition
        gpu_cluster_found = False
        db_cluster_found = False
        gpu_concepts = {
            "GPU Error",
            "CUDA Failure",
            "VRAM Out of Memory",
            "Nvidia Driver Conflict",
        }
        db_concepts = {"Database Connection", "PostgreSQL Query", "SQLAlchemy Session"}

        for _cluster_id, concepts_in_cluster in clusters.items():
            if set(concepts_in_cluster) == gpu_concepts:
                gpu_cluster_found = True
            elif set(concepts_in_cluster) == db_concepts:
                db_cluster_found = True

        assert gpu_cluster_found, (
            "The GPU-related concept cluster was not formed correctly."
        )
        assert db_cluster_found, (
            "The database-related concept cluster was not formed correctly."
        )

        # 4. Verify outliers are not in any cluster
        assert "Python Programming" not in assignments
        assert "Gradio UI" not in assignments
