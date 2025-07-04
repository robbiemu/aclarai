# Concept Clustering Job

This document describes the concept clustering job implementation for aclarai, which groups related concepts into thematic clusters using their embeddings.

## Overview

The concept clustering job is a scheduled task that:

1. Retrieves all canonical concepts from Neo4j
2. Gets their embeddings from the concepts vector store  
3. Uses clustering algorithms to form semantically coherent groups
4. Applies configurable filters to ensure cluster quality
5. Caches the cluster assignments for use by the Subject Summary Agent

## Architecture

### Components

The concept clustering system consists of several key components:

- **ConceptClusteringJob**: Main job class that orchestrates the clustering process
- **ClusterAssignment**: Helper class for managing cluster assignments and outliers
- **Configuration**: Settings for clustering parameters and job scheduling

### Integration Points

- **Neo4j**: Source of canonical concept data
- **Vector Store**: Source of concept embeddings from the concepts collection
- **Scheduler**: APScheduler integration for periodic execution
- **Subject Summary Agent**: Consumer of cluster assignments

## Configuration

The clustering job is configured through the `subject_summaries` section in `aclarai.config.yaml`:

```yaml
subject_summaries:
  model: "gpt-3.5-turbo"
  similarity_threshold: 0.92
  min_concepts: 3
  max_concepts: 15
  allow_web_search: true
  skip_if_incoherent: false

scheduler:
  jobs:
    concept_clustering:
      enabled: true
      manual_only: false
      cron: "0 2 * * *"  # 2 AM daily
      description: "Group related concepts into thematic clusters"
      similarity_threshold: 0.92
      min_concepts: 3
      max_concepts: 15
      algorithm: "dbscan"
      cache_ttl: 3600
      use_persistent_cache: true
```

### Configuration Parameters

#### Clustering Parameters
- **similarity_threshold**: Minimum similarity for concepts to be clustered together (0.92)
- **min_concepts**: Minimum number of concepts required to form a cluster (3)
- **max_concepts**: Maximum number of concepts allowed in a cluster (15)
- **algorithm**: Clustering algorithm to use ("dbscan" or "hierarchical")

#### Job Parameters
- **enabled**: Whether the job is enabled for execution
- **manual_only**: If true, job only runs when manually triggered
- **cron**: Cron expression for job scheduling (daily at 2 AM by default)
- **cache_ttl**: Time-to-live for cached results in seconds (1 hour)
- **use_persistent_cache**: Whether to use persistent cache storage

## Clustering Algorithms

### DBSCAN (Default)

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is the default algorithm:

- **Advantages**: Automatically determines cluster count, handles outliers well
- **Parameters**: Uses `similarity_threshold` converted to distance (1 - threshold)
- **Min Samples**: Uses `min_concepts` as the minimum samples parameter
- **Metric**: Cosine distance for semantic similarity

### Hierarchical Clustering

Agglomerative hierarchical clustering is available as an alternative:

- **Advantages**: Deterministic results, good for exploring cluster hierarchies
- **Parameters**: Dynamically calculates cluster count based on concept count
- **Linkage**: Average linkage for balanced clusters
- **Metric**: Cosine distance for semantic similarity

## Processing Pipeline

### 1. Concept Retrieval

```cypher
MATCH (c:Concept)
RETURN c.name as name
ORDER BY c.name
```

Retrieves all canonical concepts from Neo4j in alphabetical order.

### 2. Embedding Retrieval

For each concept, searches the vector store using similarity search:

```python
results = vector_store.similarity_search(
    query_text=concept_name,
    top_k=1,
    similarity_threshold=0.9,
    filter_metadata={"collection": "concepts"}
)
```

### 3. Clustering

Applies the configured clustering algorithm to the embedding matrix:

#### DBSCAN
```python
clustering = DBSCAN(
    eps=1.0 - similarity_threshold,
    min_samples=min_concepts,
    metric="cosine"
)
```

#### Hierarchical
```python
clustering = AgglomerativeClustering(
    n_clusters=calculated_clusters,
    metric="cosine",
    linkage="average"
)
```

### 4. Filtering

Applies size constraints to the resulting clusters:

- **Too Small**: Clusters with < `min_concepts` are marked as outliers
- **Too Large**: Clusters with > `max_concepts` generate warnings but are kept
- **Valid Size**: Clusters meeting size criteria are added to final results

### 5. Caching

Stores cluster assignments in memory with timestamp for TTL enforcement:

```python
{
    "concept_name_1": cluster_id_0,
    "concept_name_2": cluster_id_0,
    "concept_name_3": cluster_id_1,
    ...
}
```

## API

### Public Methods

#### `run_job() -> ConceptClusteringJobStats`

Executes the complete clustering pipeline and returns statistics:

```python
{
    "success": bool,
    "concepts_processed": int,
    "clusters_formed": int,
    "concepts_clustered": int,
    "concepts_outliers": int,
    "cache_updated": bool,
    "duration": float,
    "error_details": List[str]
}
```

#### `get_cluster_assignments() -> Optional[Dict[str, int]]`

Returns cached cluster assignments for the Subject Summary Agent:

```python
{
    "concept_name": cluster_id,
    ...
}
```

Returns `None` if cache is empty or expired.

## Error Handling

The job implements comprehensive error handling:

### Database Errors
- Neo4j connection failures are logged and return empty concept list
- Vector store errors for individual concepts are logged but don't stop processing

### Clustering Errors
- Invalid algorithm names raise ValueError with descriptive message
- Empty embedding matrices are handled gracefully
- Clustering failures return empty ClusterAssignment

### Cache Errors
- Cache update failures are logged but don't fail the job
- Cache retrieval handles missing or expired entries

## Performance Considerations

### Scalability
- Memory usage scales with number of concepts and embedding dimensions
- Clustering complexity varies by algorithm:
  - DBSCAN: O(n log n) with efficient implementations
  - Hierarchical: O(n³) for large datasets

### Optimization Strategies
- Batch embedding retrieval where possible
- Use efficient distance metrics (cosine)
- Implement caching to avoid re-clustering unchanged concepts
- Consider incremental clustering for large concept sets

## Monitoring and Logging

### Structured Logging

All log entries follow the structured format with:

```python
{
    "service": "aclarai-scheduler",
    "filename.function_name": "concept_clustering_job.method_name",
    "additional_context": "values"
}
```

### Key Metrics

The job reports several important metrics:

- **Concepts Processed**: Total concepts from Neo4j
- **Clusters Formed**: Number of valid clusters created
- **Concepts Clustered**: Concepts successfully assigned to clusters
- **Outliers**: Concepts that don't fit well in any cluster
- **Duration**: Total job execution time

### Monitoring Alerts

Consider alerting on:

- Job failures (`success: false`)
- High outlier rates (> 50% of concepts)
- Long execution times (> 5 minutes)
- Cache update failures

## Integration with Subject Summary Agent

The clustering job serves as a prerequisite for the Subject Summary Agent:

1. **Cluster Discovery**: Agent calls `get_cluster_assignments()` to find concept groups
2. **Cluster Processing**: For each cluster, agent generates a subject summary page
3. **Cache Management**: Agent respects TTL and handles cache misses gracefully

### Example Usage

```python
clustering_job = ConceptClusteringJob(config)
assignments = clustering_job.get_cluster_assignments()

if assignments:
    for concept_id, cluster_id in assignments.items():
        # Process concept in its cluster context
        pass
else:
    # Handle cache miss - maybe trigger re-clustering
    stats = clustering_job.run_job()
```

## Testing

The implementation includes comprehensive unit tests covering:

- **ClusterAssignment**: Cluster management and outlier tracking
- **ConceptClusteringJob**: All major methods and error conditions
- **Configuration**: Parameter validation and defaults
- **Algorithms**: Both DBSCAN and hierarchical clustering
- **Caching**: TTL enforcement and retrieval logic

### Running Tests

```bash
uv run pytest services/scheduler/tests/test_concept_clustering_job.py -v
```

## Troubleshooting

### Common Issues

#### No Concepts Found
- Check Neo4j connectivity and concept data
- Verify (:Concept) nodes exist in the graph

#### No Embeddings Retrieved  
- Verify concepts vector store is populated
- Check embedding model configuration
- Ensure concept names match between Neo4j and vector store

#### Poor Clustering Results
- Adjust `similarity_threshold` for tighter/looser clusters
- Modify `min_concepts`/`max_concepts` for size constraints
- Try different clustering algorithms

#### Cache Issues
- Check cache TTL settings
- Verify memory availability for caching
- Consider persistent cache for large datasets

### Debug Mode

Enable debug logging for detailed clustering information:

```yaml
logging:
  level: "DEBUG"
```

This provides detailed information about:
- Concept retrieval from Neo4j
- Embedding retrieval from vector store
- Clustering algorithm execution
- Cache operations

## Future Enhancements

Potential improvements for the clustering system:

### Algorithm Enhancements
- Support for additional clustering algorithms (K-means, spectral clustering)
- Ensemble clustering for improved robustness
- Dynamic parameter tuning based on data characteristics

### Performance Improvements
- Incremental clustering for changed concepts only
- Distributed clustering for very large concept sets
- Optimized embedding storage and retrieval

### Quality Improvements
- Cluster quality metrics (silhouette score, Davies-Bouldin index)
- Automatic optimal parameter selection
- Human feedback integration for cluster refinement

### Integration Enhancements
- Real-time clustering updates on concept changes
- Integration with concept lifecycle management
- Cluster visualization and exploration tools