"""
Concept Clustering Job for aclarai scheduler.
This module implements the scheduled job for grouping related concepts into
thematic clusters using their embeddings, following the architecture from
docs/arch/on-concepts.md and docs/arch/on-vector_stores.md.
"""

import logging
import time
from typing import Dict, List, Optional, Set, Tuple, TypedDict

import numpy as np
from aclarai_shared import load_config
from aclarai_shared.config import ConceptClusteringJobConfig, aclaraiConfig
from aclarai_shared.embedding.storage import aclaraiVectorStore
from aclarai_shared.graph.neo4j_manager import Neo4jGraphManager
from sklearn.cluster import DBSCAN, AgglomerativeClustering

logger = logging.getLogger(__name__)


class ConceptClusteringJobStats(TypedDict):
    """Type definition for job statistics."""

    success: bool
    concepts_processed: int
    clusters_formed: int
    concepts_clustered: int
    concepts_outliers: int
    cache_updated: bool
    duration: float
    error_details: List[str]


class ClusterAssignment:
    """Represents cluster assignments for concepts."""

    def __init__(self):
        self.concept_to_cluster: Dict[str, int] = {}
        self.cluster_to_concepts: Dict[int, List[str]] = {}
        self.outliers: Set[str] = set()
        self.cluster_counter = 0

    def add_cluster(self, concept_ids: List[str]) -> int:
        """Add a new cluster with the given concept IDs."""
        cluster_id = self.cluster_counter
        self.cluster_counter += 1

        self.cluster_to_concepts[cluster_id] = concept_ids
        for concept_id in concept_ids:
            self.concept_to_cluster[concept_id] = cluster_id

        return cluster_id

    def add_outlier(self, concept_id: str):
        """Mark a concept as an outlier."""
        self.outliers.add(concept_id)

    def get_cluster_assignments(self) -> Dict[str, int]:
        """Get mapping of concept_id to cluster_id."""
        return self.concept_to_cluster.copy()

    def get_clusters(self) -> Dict[int, List[str]]:
        """Get mapping of cluster_id to list of concept_ids."""
        return self.cluster_to_concepts.copy()


class ConceptClusteringJob:
    """
    Job for grouping related concepts into thematic clusters.

    This job:
    1. Retrieves all canonical concepts from Neo4j
    2. Gets their embeddings from the concepts vector store
    3. Uses clustering algorithms (DBSCAN or hierarchical) to form groups
    4. Applies filters (min_concepts, max_concepts, similarity_threshold)
    5. Caches the cluster assignments for the Subject Summary Agent
    """

    def __init__(
        self,
        config: Optional[aclaraiConfig] = None,
        neo4j_manager: Optional[Neo4jGraphManager] = None,
        vector_store: Optional[aclaraiVectorStore] = None,
    ):
        """Initialize concept clustering job."""
        self.config = config or load_config(validate=True)
        self.neo4j_manager = neo4j_manager or Neo4jGraphManager(self.config)
        self.vector_store = vector_store or aclaraiVectorStore(self.config)

        # Get job-specific configuration
        self.job_config: ConceptClusteringJobConfig = (
            self.config.scheduler.jobs.concept_clustering
        )

        # Cache for cluster assignments
        self._cluster_cache: Optional[Dict[str, int]] = None
        self._cache_timestamp: Optional[float] = None

    def _get_concept_data(self) -> List[Tuple[str, str]]:
        """
        Retrieve all canonical concepts from Neo4j.

        Returns:
            List of tuples containing (concept_id, concept_name)
        """
        query = """
        MATCH (c:Concept)
        RETURN c.name as name
        ORDER BY c.name
        """

        try:
            result = self.neo4j_manager.execute_query(query)

            if result:
                concepts = [(record["name"], record["name"]) for record in result]
                logger.info(
                    f"concept_clustering_job._get_concept_data: Retrieved {len(concepts)} concepts from Neo4j",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_clustering_job._get_concept_data",
                        "concepts_count": len(concepts),
                    },
                )
                return concepts
            else:
                logger.warning(
                    "concept_clustering_job._get_concept_data: No concepts found in Neo4j",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_clustering_job._get_concept_data",
                    },
                )
                return []

        except Exception as e:
            logger.error(
                f"concept_clustering_job._get_concept_data: Failed to retrieve concepts from Neo4j: {e}",
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_clustering_job._get_concept_data",
                    "error": str(e),
                },
            )
            return []

    def _get_concept_embeddings(
        self, concept_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Retrieve embeddings for the given concepts from the vector store.

        Args:
            concept_names: List of concept names to get embeddings for

        Returns:
            Tuple of (embeddings_matrix, valid_concept_names)
        """
        embeddings = []
        valid_concepts = []

        for concept_name in concept_names:
            try:
                # Search for the concept in the vector store
                # We use similarity search with the concept name itself to find the embedding
                results = self.vector_store.similarity_search(
                    query_text=concept_name,
                    top_k=1,
                    similarity_threshold=0.9,  # High threshold to find exact matches
                    filter_metadata={
                        "collection": "concepts"
                    },  # Filter to concepts collection
                )

                if results and len(results) > 0:
                    # Extract embedding from the result
                    chunk_data, similarity = results[0]
                    if "embedding" in chunk_data:
                        embedding = np.array(chunk_data["embedding"])
                        embeddings.append(embedding)
                        valid_concepts.append(concept_name)
                    else:
                        logger.warning(
                            f"concept_clustering_job._get_concept_embeddings: No embedding found for concept: {concept_name}",
                            extra={
                                "service": "aclarai-scheduler",
                                "filename.function_name": "concept_clustering_job._get_concept_embeddings",
                                "concept_name": concept_name,
                            },
                        )
                else:
                    logger.warning(
                        f"concept_clustering_job._get_concept_embeddings: Concept not found in vector store: {concept_name}",
                        extra={
                            "service": "aclarai-scheduler",
                            "filename.function_name": "concept_clustering_job._get_concept_embeddings",
                            "concept_name": concept_name,
                        },
                    )

            except Exception as e:
                logger.error(
                    f"concept_clustering_job._get_concept_embeddings: Failed to get embedding for concept {concept_name}: {e}",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_clustering_job._get_concept_embeddings",
                        "concept_name": concept_name,
                        "error": str(e),
                    },
                )

        if embeddings:
            embeddings_matrix = np.vstack(embeddings)
            logger.info(
                f"concept_clustering_job._get_concept_embeddings: Retrieved embeddings for {len(valid_concepts)} concepts",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_clustering_job._get_concept_embeddings",
                    "valid_concepts_count": len(valid_concepts),
                    "embedding_dimension": embeddings_matrix.shape[1],
                },
            )
            return embeddings_matrix, valid_concepts
        else:
            logger.warning(
                "concept_clustering_job._get_concept_embeddings: No valid embeddings retrieved",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_clustering_job._get_concept_embeddings",
                },
            )
            return np.array([]), []

    def _perform_clustering(
        self, embeddings: np.ndarray, concept_names: List[str]
    ) -> ClusterAssignment:
        """
        Perform clustering on concept embeddings using the configured algorithm.

        Args:
            embeddings: Matrix of concept embeddings
            concept_names: List of concept names corresponding to embeddings

        Returns:
            ClusterAssignment object with cluster assignments
        """
        cluster_assignment = ClusterAssignment()

        if len(embeddings) == 0:
            logger.warning(
                "concept_clustering_job._perform_clustering: No embeddings provided for clustering",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_clustering_job._perform_clustering",
                },
            )
            return cluster_assignment

        try:
            if self.job_config.algorithm.lower() == "dbscan":
                # Use DBSCAN clustering
                # Convert similarity threshold to distance for DBSCAN
                eps = 1.0 - self.job_config.similarity_threshold
                clustering = DBSCAN(
                    eps=eps, min_samples=self.job_config.min_concepts, metric="cosine"
                )
                labels = clustering.fit_predict(embeddings)

            elif self.job_config.algorithm.lower() == "hierarchical":
                # Use Agglomerative clustering
                # Calculate number of clusters dynamically based on similarity threshold
                n_concepts = len(embeddings)
                # Start with a reasonable number of clusters
                n_clusters = max(
                    2, min(n_concepts // self.job_config.min_concepts, n_concepts // 2)
                )

                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters, metric="cosine", linkage="average"
                )
                labels = clustering.fit_predict(embeddings)

            else:
                raise ValueError(
                    f"Unsupported clustering algorithm: {self.job_config.algorithm}"
                )

            logger.info(
                f"concept_clustering_job._perform_clustering: Clustering completed using {self.job_config.algorithm}",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_clustering_job._perform_clustering",
                    "algorithm": self.job_config.algorithm,
                    "n_concepts": len(concept_names),
                    "unique_labels": len(set(labels)),
                },
            )

            # Process clustering results
            cluster_map: Dict[int, List[str]] = {}
            outliers = []

            for i, label in enumerate(labels):
                concept_name = concept_names[i]

                if label == -1:  # Outlier in DBSCAN
                    outliers.append(concept_name)
                else:
                    if label not in cluster_map:
                        cluster_map[label] = []
                    cluster_map[label].append(concept_name)

            # Apply size filters and create final cluster assignments
            for label, concepts in cluster_map.items():
                cluster_size = len(concepts)

                # Apply min/max constraints
                if cluster_size < self.job_config.min_concepts:
                    # Too small, treat as outliers
                    outliers.extend(concepts)
                elif cluster_size > self.job_config.max_concepts:
                    # Too large, might need to split or just keep as is
                    # For now, we'll keep large clusters but log a warning
                    logger.warning(
                        f"concept_clustering_job._perform_clustering: Cluster {label} exceeds max_concepts ({cluster_size} > {self.job_config.max_concepts})",
                        extra={
                            "service": "aclarai-scheduler",
                            "filename.function_name": "concept_clustering_job._perform_clustering",
                            "cluster_label": label,
                            "cluster_size": cluster_size,
                            "max_concepts": self.job_config.max_concepts,
                        },
                    )
                    cluster_assignment.add_cluster(concepts)
                else:
                    # Good size, add to clusters
                    cluster_assignment.add_cluster(concepts)

            # Add outliers
            for outlier in outliers:
                cluster_assignment.add_outlier(outlier)

            logger.info(
                f"concept_clustering_job._perform_clustering: Clustering completed with {len(cluster_assignment.get_clusters())} clusters and {len(outliers)} outliers",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_clustering_job._perform_clustering",
                    "clusters_count": len(cluster_assignment.get_clusters()),
                    "outliers_count": len(outliers),
                },
            )

            return cluster_assignment

        except Exception as e:
            logger.error(
                f"concept_clustering_job._perform_clustering: Clustering failed: {e}",
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_clustering_job._perform_clustering",
                    "error": str(e),
                },
            )
            return cluster_assignment

    def _update_cache(self, cluster_assignment: ClusterAssignment) -> bool:
        """
        Update the cluster assignment cache.

        Args:
            cluster_assignment: The cluster assignments to cache

        Returns:
            True if cache was updated successfully, False otherwise
        """
        try:
            self._cluster_cache = cluster_assignment.get_cluster_assignments()
            self._cache_timestamp = time.time()

            logger.info(
                f"concept_clustering_job._update_cache: Cache updated with {len(self._cluster_cache)} concept assignments",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_clustering_job._update_cache",
                    "assignments_count": len(self._cluster_cache),
                    "timestamp": self._cache_timestamp,
                },
            )
            return True

        except Exception as e:
            logger.error(
                f"concept_clustering_job._update_cache: Failed to update cache: {e}",
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_clustering_job._update_cache",
                    "error": str(e),
                },
            )
            return False

    def get_cluster_assignments(self) -> Optional[Dict[str, int]]:
        """
        Get cached cluster assignments for the Subject Summary Agent.

        Returns:
            Dictionary mapping concept_id to cluster_id, or None if cache is empty/expired
        """
        if self._cluster_cache is None or self._cache_timestamp is None:
            logger.debug(
                "concept_clustering_job.get_cluster_assignments: No cached assignments available",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_clustering_job.get_cluster_assignments",
                },
            )
            return None

        # Check if cache is expired
        cache_age = time.time() - self._cache_timestamp
        if cache_age > self.job_config.cache_ttl:
            logger.debug(
                f"concept_clustering_job.get_cluster_assignments: Cache expired (age: {cache_age}s > ttl: {self.job_config.cache_ttl}s)",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_clustering_job.get_cluster_assignments",
                    "cache_age": cache_age,
                    "cache_ttl": self.job_config.cache_ttl,
                },
            )
            return None

        return self._cluster_cache.copy()

    def run_job(self) -> ConceptClusteringJobStats:
        """
        Execute the complete concept clustering job.

        Returns:
            Dictionary with job statistics and results
        """
        start_time = time.time()
        stats: ConceptClusteringJobStats = {
            "success": False,
            "concepts_processed": 0,
            "clusters_formed": 0,
            "concepts_clustered": 0,
            "concepts_outliers": 0,
            "cache_updated": False,
            "duration": 0.0,
            "error_details": [],
        }

        logger.info(
            "concept_clustering_job.run_job: Starting concept clustering job",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "concept_clustering_job.run_job",
                "algorithm": self.job_config.algorithm,
                "similarity_threshold": self.job_config.similarity_threshold,
                "min_concepts": self.job_config.min_concepts,
                "max_concepts": self.job_config.max_concepts,
            },
        )

        try:
            # Step 1: Get concept data from Neo4j
            concept_data = self._get_concept_data()
            stats["concepts_processed"] = len(concept_data)

            if not concept_data:
                logger.warning(
                    "concept_clustering_job.run_job: No concepts found for clustering",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_clustering_job.run_job",
                    },
                )
                stats["error_details"].append("No concepts found for clustering")
                return stats

            # Extract concept names
            concept_names = [name for _, name in concept_data]

            # Step 2: Get embeddings for concepts
            embeddings, valid_concept_names = self._get_concept_embeddings(
                concept_names
            )

            if len(embeddings) == 0:
                logger.warning(
                    "concept_clustering_job.run_job: No embeddings found for concepts",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_clustering_job.run_job",
                    },
                )
                stats["error_details"].append("No embeddings found for concepts")
                return stats

            # Step 3: Perform clustering
            cluster_assignment = self._perform_clustering(
                embeddings, valid_concept_names
            )

            # Step 4: Update statistics
            clusters = cluster_assignment.get_clusters()
            stats["clusters_formed"] = len(clusters)
            stats["concepts_clustered"] = sum(
                len(concepts) for concepts in clusters.values()
            )
            stats["concepts_outliers"] = len(cluster_assignment.outliers)

            # Step 5: Update cache
            if self._update_cache(cluster_assignment):
                stats["cache_updated"] = True
                stats["success"] = True

                logger.info(
                    f"concept_clustering_job.run_job: Successfully completed clustering with {stats['clusters_formed']} clusters",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_clustering_job.run_job",
                        "concepts_processed": stats["concepts_processed"],
                        "clusters_formed": stats["clusters_formed"],
                        "concepts_clustered": stats["concepts_clustered"],
                        "concepts_outliers": stats["concepts_outliers"],
                    },
                )
            else:
                stats["error_details"].append("Failed to update cache")

        except Exception as e:
            error_msg = f"Unexpected error in concept clustering job: {e}"
            logger.error(
                f"concept_clustering_job.run_job: {error_msg}",
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_clustering_job.run_job",
                    "error": str(e),
                },
            )
            stats["error_details"].append(error_msg)

        finally:
            stats["duration"] = time.time() - start_time

        return stats
