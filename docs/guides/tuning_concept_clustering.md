# Tuning Concept Clustering

## UI Configuration
Concept clustering parameters can now be adjusted through the Aclarai UI panel under the "Highlight & Summary" section. This provides an easy way to tune the clustering behavior without manually editing configuration files.

### Available UI Controls
- **Similarity Threshold** (0.0-1.0)
  - Controls how similar concepts must be to be grouped together
  - Higher values create more focused but fewer clusters
  - Default: 0.92

- **Min/Max Concepts**
  - Set minimum (1-100) and maximum (1-100) concepts per cluster
  - Helps control cluster size and quality
  - Defaults: min=3, max=15

Changes take effect during the next clustering job run.

## Understanding Clustering Parameters

### Similarity Threshold
The similarity threshold determines how similar concepts must be to be grouped together:

- **Higher values** (e.g., 0.92-0.95)
  - Creates more focused, tightly related clusters
  - Results in fewer but more coherent subjects
  - Best for specialized knowledge bases

- **Lower values** (e.g., 0.80-0.85)
  - Creates broader, more inclusive clusters
  - Results in more subjects with looser relationships
  - Better for general knowledge bases

### Cluster Size Limits
The min/max concept settings help control cluster quality:

- **Minimum Concepts**
  - Prevents creation of subjects from too few concepts
  - Higher values ensure more substantial subjects
  - Consider your knowledge base size when setting

- **Maximum Concepts**
  - Prevents overly broad subject clusters
  - Helps maintain focused, manageable subjects
  - Adjust based on your subject granularity needs

## Best Practices

1. **Start Conservative**
   - Begin with default values (threshold=0.92, min=3, max=15)
   - Review generated subjects for quality and coherence
   - Adjust gradually based on results

2. **Monitor Results**
   - Check subject pages after parameter changes
   - Look for clear thematic connections
   - Ensure subjects aren't too broad or narrow

3. **Consider Scale**
   - Larger knowledge bases may need higher thresholds
   - Adjust cluster sizes based on content volume
   - Balance between quality and coverage

4. **Regular Review**
   - Periodically review clustering results
   - Adjust parameters as your knowledge base grows
   - Fine-tune based on user feedback

## Troubleshooting

### Common Issues

1. **Too Few Subjects**
   - Lower similarity threshold
   - Decrease minimum concept requirement
   - Check concept embedding quality

2. **Incoherent Subjects**
   - Increase similarity threshold
   - Enable "Skip If Incoherent" option
   - Review concept quality

3. **Overlapping Subjects**
   - Increase similarity threshold
   - Decrease maximum concepts
   - Consider concept disambiguation

## Advanced Tuning

For more complex needs, additional parameters can be configured in `settings/aclarai.config.yaml`:

```yaml
concept_clustering:
  algorithm: dbscan
  cache_ttl: 3600
  use_persistent_cache: true
```

These advanced settings should be modified only when the UI controls don't provide sufficient customization for your specific use case.
