"""
Tests for evaluation threshold utilities.
"""

import pytest
from unittest.mock import patch
from aclarai_shared.evaluation_thresholds import (
    compute_geometric_mean,
    meets_quality_threshold, 
    should_allow_concept_linking,
    should_allow_vault_promotion,
    should_include_in_summaries,
)


class TestComputeGeometricMean:
    """Test geometric mean calculation."""
    
    def test_compute_geometric_mean_valid_scores(self):
        """Test geometric mean calculation with valid scores."""
        # Test case: (0.8 * 0.6 * 0.9) ** (1/3) = (0.432) ** (1/3) ≈ 0.755
        result = compute_geometric_mean(0.8, 0.6, 0.9)
        assert result is not None
        assert abs(result - 0.755) < 0.01  # Allow small floating point error
    
    def test_compute_geometric_mean_perfect_scores(self):
        """Test geometric mean with perfect scores."""
        result = compute_geometric_mean(1.0, 1.0, 1.0)
        assert result == 1.0
    
    def test_compute_geometric_mean_zero_scores(self):
        """Test geometric mean with zero scores."""
        result = compute_geometric_mean(0.0, 0.0, 0.0)
        assert result == 0.0
    
    def test_compute_geometric_mean_mixed_scores(self):
        """Test geometric mean with realistic mixed scores."""
        # Test case: (0.85 * 0.7 * 0.6) ** (1/3) = (0.357) ** (1/3) ≈ 0.71
        result = compute_geometric_mean(0.85, 0.7, 0.6)
        assert result is not None
        assert abs(result - 0.71) < 0.02
    
    def test_compute_geometric_mean_null_entailed(self):
        """Test geometric mean returns None when entailed_score is null."""
        result = compute_geometric_mean(None, 0.8, 0.9)
        assert result is None
    
    def test_compute_geometric_mean_null_coverage(self):
        """Test geometric mean returns None when coverage_score is null."""
        result = compute_geometric_mean(0.8, None, 0.9)
        assert result is None
    
    def test_compute_geometric_mean_null_decontextualization(self):
        """Test geometric mean returns None when decontextualization_score is null."""
        result = compute_geometric_mean(0.8, 0.9, None)
        assert result is None
    
    def test_compute_geometric_mean_all_null(self):
        """Test geometric mean returns None when all scores are null."""
        result = compute_geometric_mean(None, None, None)
        assert result is None
    
    def test_compute_geometric_mean_invalid_high_score(self):
        """Test geometric mean returns None for scores above 1.0."""
        result = compute_geometric_mean(1.2, 0.8, 0.9)
        assert result is None
    
    def test_compute_geometric_mean_invalid_low_score(self):
        """Test geometric mean returns None for scores below 0.0."""
        result = compute_geometric_mean(-0.1, 0.8, 0.9)
        assert result is None


class TestMeetsQualityThreshold:
    """Test quality threshold checking."""
    
    def test_meets_quality_threshold_above(self):
        """Test claim meets threshold when geomean is above threshold."""
        # (0.9 * 0.8 * 0.85) ** (1/3) ≈ 0.849 > 0.7
        result = meets_quality_threshold(0.9, 0.8, 0.85, 0.7)
        assert result is True
    
    def test_meets_quality_threshold_equal(self):
        """Test claim meets threshold when geomean equals threshold."""
        # Need scores that produce geomean exactly 0.7
        # 0.7^3 = 0.343, so we need (a*b*c) = 0.343
        result = meets_quality_threshold(0.7, 0.7, 0.7, 0.7)
        assert result is True
    
    def test_meets_quality_threshold_below(self):
        """Test claim does not meet threshold when geomean is below."""
        # (0.5 * 0.6 * 0.4) ** (1/3) ≈ 0.492 < 0.7
        result = meets_quality_threshold(0.5, 0.6, 0.4, 0.7)
        assert result is False
    
    def test_meets_quality_threshold_null_score(self):
        """Test claim does not meet threshold with null scores."""
        result = meets_quality_threshold(None, 0.8, 0.9, 0.7)
        assert result is False
    
    def test_meets_quality_threshold_high_threshold(self):
        """Test with high threshold that good scores don't meet."""
        result = meets_quality_threshold(0.8, 0.7, 0.6, 0.95)
        assert result is False


class TestShouldAllowConceptLinking:
    """Test concept linking threshold logic."""
    
    def test_allow_supports_concept_above_threshold(self):
        """Test SUPPORTS_CONCEPT allowed when above threshold."""
        result = should_allow_concept_linking(0.9, 0.8, 0.85, 0.7, "SUPPORTS_CONCEPT")
        assert result is True
    
    def test_reject_supports_concept_below_threshold(self):
        """Test SUPPORTS_CONCEPT rejected when below threshold."""
        result = should_allow_concept_linking(0.5, 0.6, 0.4, 0.7, "SUPPORTS_CONCEPT")
        assert result is False
    
    def test_reject_supports_concept_null_scores(self):
        """Test SUPPORTS_CONCEPT rejected with null scores."""
        result = should_allow_concept_linking(None, 0.8, 0.9, 0.7, "SUPPORTS_CONCEPT")
        assert result is False
    
    def test_allow_contradicts_concept_above_threshold(self):
        """Test CONTRADICTS_CONCEPT allowed when above threshold."""
        result = should_allow_concept_linking(0.9, 0.8, 0.85, 0.7, "CONTRADICTS_CONCEPT")
        assert result is True
    
    def test_reject_contradicts_concept_below_threshold(self):
        """Test CONTRADICTS_CONCEPT rejected when below threshold."""
        result = should_allow_concept_linking(0.5, 0.6, 0.4, 0.7, "CONTRADICTS_CONCEPT")
        assert result is False
    
    def test_allow_mentions_concept_below_threshold(self):
        """Test MENTIONS_CONCEPT allowed even when below threshold."""
        result = should_allow_concept_linking(0.5, 0.6, 0.4, 0.7, "MENTIONS_CONCEPT")
        assert result is True
    
    def test_reject_mentions_concept_null_scores(self):
        """Test MENTIONS_CONCEPT rejected with null scores."""
        result = should_allow_concept_linking(None, 0.6, 0.4, 0.7, "MENTIONS_CONCEPT")
        assert result is False
    
    def test_unknown_relationship_type_requires_threshold(self):
        """Test unknown relationship types require meeting threshold."""
        result = should_allow_concept_linking(0.5, 0.6, 0.4, 0.7, "UNKNOWN_TYPE")
        assert result is False
        
        result = should_allow_concept_linking(0.9, 0.8, 0.85, 0.7, "UNKNOWN_TYPE")
        assert result is True


class TestShouldAllowVaultPromotion:
    """Test vault promotion threshold logic."""
    
    def test_allow_vault_promotion_above_threshold(self):
        """Test vault promotion allowed when above threshold."""
        result = should_allow_vault_promotion(0.9, 0.8, 0.85, 0.7)
        assert result is True
    
    def test_reject_vault_promotion_below_threshold(self):
        """Test vault promotion rejected when below threshold."""
        result = should_allow_vault_promotion(0.5, 0.6, 0.4, 0.7)
        assert result is False
    
    def test_reject_vault_promotion_null_scores(self):
        """Test vault promotion rejected with null scores."""
        result = should_allow_vault_promotion(None, 0.8, 0.9, 0.7)
        assert result is False


class TestShouldIncludeInSummaries:
    """Test summary inclusion threshold logic."""
    
    def test_allow_summary_inclusion_above_threshold(self):
        """Test summary inclusion allowed when above threshold."""
        result = should_include_in_summaries(0.9, 0.8, 0.85, 0.7)
        assert result is True
    
    def test_reject_summary_inclusion_below_threshold(self):
        """Test summary inclusion rejected when below threshold."""
        result = should_include_in_summaries(0.5, 0.6, 0.4, 0.7)
        assert result is False
    
    def test_reject_summary_inclusion_null_scores(self):
        """Test summary inclusion rejected with null scores."""
        result = should_include_in_summaries(None, 0.8, 0.9, 0.7)
        assert result is False


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @patch('aclarai_shared.evaluation_thresholds.logger')
    def test_logging_behavior_null_scores(self, mock_logger):
        """Test that appropriate debug messages are logged for null scores."""
        compute_geometric_mean(None, 0.8, 0.9)
        mock_logger.debug.assert_called()
        assert "null scores present" in str(mock_logger.debug.call_args)
    
    @patch('aclarai_shared.evaluation_thresholds.logger')
    def test_logging_behavior_valid_computation(self, mock_logger):
        """Test that appropriate debug messages are logged for valid computations."""
        compute_geometric_mean(0.8, 0.7, 0.9)
        mock_logger.debug.assert_called()
        assert "Computed geometric mean" in str(mock_logger.debug.call_args)
    
    def test_geometric_mean_floating_point_precision(self):
        """Test that small floating point differences are handled correctly."""
        # Using numbers that might cause precision issues
        result = compute_geometric_mean(0.1, 0.2, 0.3)
        assert result is not None
        assert 0 < result < 1
    
    def test_threshold_boundary_conditions(self):
        """Test behavior at exact threshold boundaries."""
        # Test scores that should produce geomean very close to threshold
        result = meets_quality_threshold(0.700001, 0.7, 0.7, 0.7)
        assert result is True
        
        result = meets_quality_threshold(0.699999, 0.7, 0.7, 0.7)
        assert result is False


class TestIntegrationWithTypicalValues:
    """Test with realistic evaluation score values."""
    
    def test_high_quality_claim_scenario(self):
        """Test a high-quality claim that should pass all checks."""
        entailed, coverage, decontex = 0.92, 0.85, 0.88
        threshold = 0.7
        
        # Should meet all requirements
        assert meets_quality_threshold(entailed, coverage, decontex, threshold) is True
        assert should_allow_concept_linking(entailed, coverage, decontex, threshold, "SUPPORTS_CONCEPT") is True
        assert should_allow_vault_promotion(entailed, coverage, decontex, threshold) is True
        assert should_include_in_summaries(entailed, coverage, decontex, threshold) is True
    
    def test_marginal_quality_claim_scenario(self):
        """Test a marginal claim that just meets threshold."""
        entailed, coverage, decontex = 0.75, 0.70, 0.72
        threshold = 0.7
        
        # Should just meet requirements
        assert meets_quality_threshold(entailed, coverage, decontex, threshold) is True
        assert should_allow_concept_linking(entailed, coverage, decontex, threshold, "SUPPORTS_CONCEPT") is True
    
    def test_low_quality_claim_scenario(self):
        """Test a low-quality claim that should be filtered out."""
        entailed, coverage, decontex = 0.45, 0.60, 0.55
        threshold = 0.7
        
        # Should fail most checks except MENTIONS_CONCEPT
        assert meets_quality_threshold(entailed, coverage, decontex, threshold) is False
        assert should_allow_concept_linking(entailed, coverage, decontex, threshold, "SUPPORTS_CONCEPT") is False
        assert should_allow_concept_linking(entailed, coverage, decontex, threshold, "MENTIONS_CONCEPT") is True
        assert should_allow_vault_promotion(entailed, coverage, decontex, threshold) is False
        assert should_include_in_summaries(entailed, coverage, decontex, threshold) is False
    
    def test_failed_evaluation_scenario(self):
        """Test a claim with failed evaluation (null scores)."""
        entailed, coverage, decontex = None, 0.85, 0.90
        threshold = 0.7
        
        # Should fail all checks including MENTIONS_CONCEPT
        assert meets_quality_threshold(entailed, coverage, decontex, threshold) is False
        assert should_allow_concept_linking(entailed, coverage, decontex, threshold, "SUPPORTS_CONCEPT") is False
        assert should_allow_concept_linking(entailed, coverage, decontex, threshold, "MENTIONS_CONCEPT") is False
        assert should_allow_vault_promotion(entailed, coverage, decontex, threshold) is False
        assert should_include_in_summaries(entailed, coverage, decontex, threshold) is False