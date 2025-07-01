# Evaluation Threshold Implementation

This document describes the evaluation threshold system implemented in aclarai to filter claims based on quality scores for downstream processing.

## Overview

The evaluation threshold system implements the quality filtering logic defined in `docs/arch/on-evaluation_agents.md`. It uses geometric mean calculation to determine claim quality and applies configurable thresholds to control which claims are promoted to various downstream workflows.

## Architecture

### Core Components

1. **Geometric Mean Calculation**: Computes quality score from three evaluation dimensions
2. **Threshold Logic**: Determines inclusion in different workflows based on quality
3. **Configuration**: Configurable quality threshold via `claim_quality` setting
4. **Integration Points**: Applied in claim-concept linking, Tier 2 summaries, and vault promotion

### Files Modified/Created

- `shared/aclarai_shared/evaluation_thresholds.py` - Core threshold logic
- `shared/aclarai_shared/config.py` - Added `claim_quality` threshold configuration
- `shared/aclarai_shared/aclarai.config.default.yaml` - Default threshold value (0.7)
- `shared/aclarai_shared/claim_concept_linking/orchestrator.py` - Applied thresholds to concept linking
- `shared/aclarai_shared/tier2_summary/agent.py` - Applied thresholds to summary generation

## Implementation Details

### Geometric Mean Formula

Following `docs/arch/on-evaluation_agents.md`:

```python
geomean = (entailed_score * coverage_score * decontextualization_score) ** (1/3)
```

### Quality Threshold Rules

A claim meets quality threshold if:
1. All three scores are non-null (no failed evaluations)
2. Geometric mean ≥ configured `claim_quality` threshold (default 0.7)

### Relationship Type Logic

For concept linking, different relationship types have different requirements:

- **SUPPORTS_CONCEPT**: Requires meeting quality threshold
- **CONTRADICTS_CONCEPT**: Requires meeting quality threshold  
- **MENTIONS_CONCEPT**: Allowed below threshold (but not with null scores)

### Integration Points

1. **Claim-Concept Linking**: Applied before creating Neo4j relationships
2. **Tier 2 Summary Generation**: Filters seed claims for summary creation
3. **Vault Promotion**: Controls which claims appear in Tier 2/3 files
4. **Summary Inclusion**: Controls which claims are used in concept pages

## Configuration

### Default Configuration

```yaml
threshold:
  claim_quality: 0.7  # Quality threshold for claim promotion and linking
```

### Customization

Users can override the default threshold in their `settings/aclarai.config.yaml`:

```yaml
threshold:
  claim_quality: 0.8  # Stricter quality requirements
```

## Testing

Comprehensive test coverage includes:

- Unit tests for geometric mean calculation
- Threshold logic validation
- Integration with claim-concept linking
- Integration with Tier 2 summary generation
- End-to-end workflow testing
- Edge cases and error handling

### Test Files

- `shared/tests/test_evaluation_thresholds.py` - Core logic tests
- `shared/tests/test_evaluation_thresholds_integration.py` - Integration tests
- `shared/tests/claim_concept_linking/test_orchestrator_thresholds.py` - Linking integration tests
- Updated `shared/tests/test_tier2_summary.py` - Summary integration tests

## Impact Analysis

### High-Quality Claims (above threshold)
- ✅ Linked to concepts via SUPPORTS_CONCEPT/CONTRADICTS_CONCEPT
- ✅ Included in Tier 2 summaries
- ✅ Promoted to vault files
- ✅ Available for concept pages

### Low-Quality Claims (below threshold, non-null scores)
- ❌ No strong concept linking (SUPPORTS_CONCEPT/CONTRADICTS_CONCEPT)
- ✅ Can still be linked via MENTIONS_CONCEPT
- ❌ Not included in summaries
- ❌ Not promoted to vault files

### Failed Evaluation Claims (null scores)
- ❌ No concept linking of any type
- ❌ Not included in summaries  
- ❌ Not promoted to vault files
- ❌ Retained in graph but filtered from all downstream use

## Benefits

1. **Quality Control**: Ensures only high-quality claims influence knowledge base
2. **Configurable Standards**: Adjustable quality requirements per deployment
3. **Consistent Filtering**: Unified approach across all downstream systems
4. **Performance**: Avoids processing low-quality content in expensive operations
5. **User Experience**: Higher quality summaries and concept relationships

## Monitoring and Debugging

The system provides detailed logging for threshold decisions:

- Debug logs for geometric mean calculations
- Info logs for filtering statistics (e.g., "Retrieved X high-quality claims from Y total")
- Debug logs for individual claim filtering decisions

This enables administrators to:
- Monitor the effectiveness of current thresholds
- Identify if thresholds are too strict/lenient
- Debug why specific claims are included/excluded