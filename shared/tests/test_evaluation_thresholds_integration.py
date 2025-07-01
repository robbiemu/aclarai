"""
Integration tests for evaluation threshold system.
"""

from aclarai_shared.evaluation_thresholds import (
    compute_geometric_mean,
    should_allow_concept_linking,
    should_allow_vault_promotion,
    should_include_in_summaries,
)


class TestEvaluationThresholdIntegration:
    """Integration tests for the complete evaluation threshold system."""

    def test_end_to_end_claim_processing_workflow(self):
        """Test complete workflow for processing claims with evaluation thresholds."""
        # Test data representing different quality claims
        test_claims = [
            {
                "id": "high_quality_claim",
                "entailed_score": 0.9,
                "coverage_score": 0.85,
                "decontextualization_score": 0.88,
                # Geomean ≈ 0.873 > 0.7 threshold
            },
            {
                "id": "marginal_quality_claim",
                "entailed_score": 0.75,
                "coverage_score": 0.7,
                "decontextualization_score": 0.72,
                # Geomean ≈ 0.724 > 0.7 threshold (just above)
            },
            {
                "id": "low_quality_claim",
                "entailed_score": 0.5,
                "coverage_score": 0.6,
                "decontextualization_score": 0.4,
                # Geomean ≈ 0.492 < 0.7 threshold
            },
            {
                "id": "failed_evaluation_claim",
                "entailed_score": None,
                "coverage_score": 0.8,
                "decontextualization_score": 0.9,
                # Null score = excluded
            },
        ]

        quality_threshold = 0.7
        results = {}

        # Process each claim through the evaluation threshold system
        for claim in test_claims:
            claim_id = claim["id"]
            entailed = claim["entailed_score"]
            coverage = claim["coverage_score"]
            decontex = claim["decontextualization_score"]

            # Calculate geometric mean
            geomean = compute_geometric_mean(entailed, coverage, decontex)

            # Test different relationship types for concept linking
            supports_linking = should_allow_concept_linking(
                entailed, coverage, decontex, quality_threshold, "SUPPORTS_CONCEPT"
            )
            contradicts_linking = should_allow_concept_linking(
                entailed, coverage, decontex, quality_threshold, "CONTRADICTS_CONCEPT"
            )
            mentions_linking = should_allow_concept_linking(
                entailed, coverage, decontex, quality_threshold, "MENTIONS_CONCEPT"
            )

            # Test vault promotion
            vault_promotion = should_allow_vault_promotion(
                entailed, coverage, decontex, quality_threshold
            )

            # Test summary inclusion
            summary_inclusion = should_include_in_summaries(
                entailed, coverage, decontex, quality_threshold
            )

            results[claim_id] = {
                "geomean": geomean,
                "supports_linking": supports_linking,
                "contradicts_linking": contradicts_linking,
                "mentions_linking": mentions_linking,
                "vault_promotion": vault_promotion,
                "summary_inclusion": summary_inclusion,
            }

        # Validate high quality claim - should pass all checks
        high_quality = results["high_quality_claim"]
        assert high_quality["geomean"] is not None
        assert high_quality["geomean"] > quality_threshold
        assert high_quality["supports_linking"] is True
        assert high_quality["contradicts_linking"] is True
        assert high_quality["mentions_linking"] is True
        assert high_quality["vault_promotion"] is True
        assert high_quality["summary_inclusion"] is True

        # Validate marginal quality claim - should pass all checks (just above threshold)
        marginal_quality = results["marginal_quality_claim"]
        assert marginal_quality["geomean"] is not None
        assert marginal_quality["geomean"] > quality_threshold
        assert marginal_quality["supports_linking"] is True
        assert marginal_quality["contradicts_linking"] is True
        assert marginal_quality["mentions_linking"] is True
        assert marginal_quality["vault_promotion"] is True
        assert marginal_quality["summary_inclusion"] is True

        # Validate low quality claim - should fail most checks except MENTIONS_CONCEPT
        low_quality = results["low_quality_claim"]
        assert low_quality["geomean"] is not None
        assert low_quality["geomean"] < quality_threshold
        assert low_quality["supports_linking"] is False
        assert low_quality["contradicts_linking"] is False
        assert low_quality["mentions_linking"] is True  # Still allowed for low quality
        assert low_quality["vault_promotion"] is False
        assert low_quality["summary_inclusion"] is False

        # Validate failed evaluation claim - should fail all checks
        failed_eval = results["failed_evaluation_claim"]
        assert failed_eval["geomean"] is None
        assert failed_eval["supports_linking"] is False
        assert failed_eval["contradicts_linking"] is False
        assert (
            failed_eval["mentions_linking"] is False
        )  # Even MENTIONS_CONCEPT rejected for nulls
        assert failed_eval["vault_promotion"] is False
        assert failed_eval["summary_inclusion"] is False

    def test_configurable_threshold_impact(self):
        """Test that changing the quality threshold affects filtering results."""
        # Test claim with moderate scores
        entailed_score = 0.75
        coverage_score = 0.7
        decontex_score = 0.65
        # Geomean ≈ 0.699

        # With strict threshold (0.8), should fail
        strict_threshold = 0.8
        assert (
            should_include_in_summaries(
                entailed_score, coverage_score, decontex_score, strict_threshold
            )
            is False
        )
        assert (
            should_allow_concept_linking(
                entailed_score,
                coverage_score,
                decontex_score,
                strict_threshold,
                "SUPPORTS_CONCEPT",
            )
            is False
        )

        # With lenient threshold (0.6), should pass
        lenient_threshold = 0.6
        assert (
            should_include_in_summaries(
                entailed_score, coverage_score, decontex_score, lenient_threshold
            )
            is True
        )
        assert (
            should_allow_concept_linking(
                entailed_score,
                coverage_score,
                decontex_score,
                lenient_threshold,
                "SUPPORTS_CONCEPT",
            )
            is True
        )

    def test_geometric_mean_vs_individual_thresholds(self):
        """Test that geometric mean provides better quality assessment than individual thresholds."""
        # Claim with one very low score but others high
        # Individual threshold approach might accept this, geometric mean should reject
        unbalanced_claim = {
            "entailed_score": 0.95,  # Very high
            "coverage_score": 0.9,  # Very high
            "decontextualization_score": 0.1,  # Very low
            # Geomean ≈ 0.459 < 0.7
        }

        # With individual thresholds (hypothetical 0.7 each), 2/3 scores pass
        # But with geometric mean, this claim should be rejected
        geomean = compute_geometric_mean(
            unbalanced_claim["entailed_score"],
            unbalanced_claim["coverage_score"],
            unbalanced_claim["decontextualization_score"],
        )

        assert geomean is not None
        assert geomean < 0.7  # Should fail geometric mean threshold

        # Should be rejected for high-quality uses
        assert (
            should_include_in_summaries(
                unbalanced_claim["entailed_score"],
                unbalanced_claim["coverage_score"],
                unbalanced_claim["decontextualization_score"],
                0.7,
            )
            is False
        )

    def test_edge_case_perfect_and_zero_scores(self):
        """Test edge cases with perfect and zero scores."""
        # Perfect claim
        perfect_geomean = compute_geometric_mean(1.0, 1.0, 1.0)
        assert perfect_geomean == 1.0
        assert should_include_in_summaries(1.0, 1.0, 1.0, 0.99) is True

        # Zero claim (valid but lowest quality)
        zero_geomean = compute_geometric_mean(0.0, 0.0, 0.0)
        assert zero_geomean == 0.0
        assert should_include_in_summaries(0.0, 0.0, 0.0, 0.01) is False

        # Mixed with one zero (should be very low)
        mixed_geomean = compute_geometric_mean(1.0, 1.0, 0.0)
        assert mixed_geomean == 0.0
        assert should_include_in_summaries(1.0, 1.0, 0.0, 0.01) is False

    def test_relationship_type_specific_behavior(self):
        """Test that different relationship types behave correctly with same scores."""
        # Scores that don't meet quality threshold
        entailed = 0.5
        coverage = 0.6
        decontex = 0.4
        threshold = 0.7

        # SUPPORTS_CONCEPT and CONTRADICTS_CONCEPT should be rejected
        assert (
            should_allow_concept_linking(
                entailed, coverage, decontex, threshold, "SUPPORTS_CONCEPT"
            )
            is False
        )
        assert (
            should_allow_concept_linking(
                entailed, coverage, decontex, threshold, "CONTRADICTS_CONCEPT"
            )
            is False
        )

        # MENTIONS_CONCEPT should be allowed (as long as no null scores)
        assert (
            should_allow_concept_linking(
                entailed, coverage, decontex, threshold, "MENTIONS_CONCEPT"
            )
            is True
        )
