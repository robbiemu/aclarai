# Tutorial: Tuning Concept Clustering

This tutorial guides you through choosing and tuning the clustering algorithm to get the most relevant and coherent thematic groups from your concepts.

## How to Choose and Tune Your Clustering Algorithm

The system supports two distinct clustering algorithms, `dbscan` (default) and `hierarchical`. The choice of algorithm and its parameters significantly impacts the quality and nature of the generated concept clusters. This guide explains when and why you might choose one over the other.

### `dbscan` (Default) - For Automated Discovery

**Use `dbscan` for most day-to-day operations.** It is the best "hands-off" approach because it automatically discovers the natural number of clusters in your data and is excellent at identifying outliers (concepts that don't belong to any thematic group).

**Tuning `dbscan`:**

The primary parameter for tuning `dbscan` is `similarity_threshold`. This value (between 0.0 and 1.0) controls how "tight" or "loose" the clusters are.

-   **If you are getting too many small clusters or too many outliers:**
    -   **Problem:** The similarity threshold is too high, meaning only very similar concepts are being grouped.
    -   **Solution:** **Decrease** the `similarity_threshold` (e.g., from `0.92` to `0.88`). This allows less similar concepts to join a cluster, resulting in fewer, larger, and more inclusive groups.

-   **If your clusters are too large and lack a clear theme:**
    -   **Problem:** The similarity threshold is too low, causing unrelated concepts to be grouped together.
    -   **Solution:** **Increase** the `similarity_threshold` (e.g., from `0.88` to `0.93`). This enforces a stricter requirement for similarity, resulting in smaller, more thematically focused clusters.

### `hierarchical` - For Manual Exploration and Control

**Use `hierarchical` clustering when `dbscan` is not producing the desired results**, especially if you have many outliers that you want to force into groups for analysis. This algorithm is deterministic and gives you a different perspective on the data's structure.

**Key Differences from `dbscan`:**

-   **No Outliers:** Hierarchical clustering will assign *every single concept* to a cluster. This can be useful for seeing how isolated concepts relate to larger groups, but it may also result in less coherent clusters.
-   **Parameter Control:** Instead of a similarity threshold, its primary control is the number of clusters to create. The current implementation calculates this number automatically based on the total number of concepts and the `min_concepts` setting, but it provides a different grouping logic than `dbscan`.

### Recommended Workflow

1.  Start with the default `algorithm: "dbscan"`.
2.  Run the job and review the resulting clusters.
3.  If needed, adjust the `similarity_threshold` up or down to fine-tune the granularity of the clusters.
4.  If you are still not satisfied or want to explore the data's structure differently, switch to `algorithm: "hierarchical"` as an alternative analytical tool.