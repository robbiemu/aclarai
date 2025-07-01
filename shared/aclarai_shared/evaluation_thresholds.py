"""
Evaluation threshold utilities for aclarai.

This module provides utilities for computing geometric means from evaluation scores
and determining if claims meet quality thresholds for downstream processing.

Following the architecture defined in docs/arch/on-evaluation_agents.md.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_geometric_mean(
    entailed_score: Optional[float],
    coverage_score: Optional[float],
    decontextualization_score: Optional[float],
) -> Optional[float]:
    """
    Compute the geometric mean of three evaluation scores.

    Following the formula from docs/arch/on-evaluation_agents.md:
    geomean = (entailed_score * coverage_score * decontextualization_score) ** (1/3)

    Args:
        entailed_score: Score from entailment evaluation (0.0-1.0 or None)
        coverage_score: Score from coverage evaluation (0.0-1.0 or None)
        decontextualization_score: Score from decontextualization evaluation (0.0-1.0 or None)

    Returns:
        Geometric mean as float, or None if any score is null/missing
    """
    # If any score is null, return None (claim is excluded)
    if (
        entailed_score is None
        or coverage_score is None
        or decontextualization_score is None
    ):
        logger.debug(
            f"Cannot compute geometric mean: null scores present "
            f"(entailed={entailed_score}, coverage={coverage_score}, "
            f"decontextualization={decontextualization_score})"
        )
        return None

    # Validate all scores are in valid range
    for score_name, score_val in [
        ("entailed_score", entailed_score),
        ("coverage_score", coverage_score),
        ("decontextualization_score", decontextualization_score),
    ]:
        if not (0.0 <= score_val <= 1.0):
            logger.warning(
                f"Score {score_name}={score_val} is outside valid range [0.0, 1.0]"
            )
            return None

    try:
        # Compute geometric mean: (a * b * c) ** (1/3)
        product = entailed_score * coverage_score * decontextualization_score
        geomean: float = product ** (1 / 3)

        logger.debug(
            f"Computed geometric mean: {geomean:.3f} from scores "
            f"(entailed={entailed_score:.3f}, coverage={coverage_score:.3f}, "
            f"decontextualization={decontextualization_score:.3f})"
        )

        return geomean

    except Exception as e:
        logger.error(f"Error computing geometric mean: {e}")
        return None


def meets_quality_threshold(
    entailed_score: Optional[float],
    coverage_score: Optional[float],
    decontextualization_score: Optional[float],
    quality_threshold: float,
) -> bool:
    """
    Check if a claim meets the quality threshold for downstream processing.

    A claim meets the threshold if:
    1. All three scores are present (non-null)
    2. Geometric mean of scores >= quality_threshold

    Args:
        entailed_score: Score from entailment evaluation (0.0-1.0 or None)
        coverage_score: Score from coverage evaluation (0.0-1.0 or None)
        decontextualization_score: Score from decontextualization evaluation (0.0-1.0 or None)
        quality_threshold: Minimum geometric mean required (0.0-1.0)

    Returns:
        True if claim meets quality threshold, False otherwise
    """
    geomean = compute_geometric_mean(
        entailed_score, coverage_score, decontextualization_score
    )

    if geomean is None:
        logger.debug("Claim does not meet quality threshold: null scores present")
        return False

    meets_threshold = geomean >= quality_threshold

    logger.debug(
        f"Quality threshold check: geomean={geomean:.3f} {'meets' if meets_threshold else 'below'} "
        f"threshold={quality_threshold:.3f}"
    )

    return meets_threshold


def should_allow_concept_linking(
    entailed_score: Optional[float],
    coverage_score: Optional[float],
    decontextualization_score: Optional[float],
    quality_threshold: float,
    relationship_type: str = "SUPPORTS_CONCEPT",
) -> bool:
    """
    Determine if a claim should be allowed to link to concepts.

    Following the rules from docs/arch/on-linking_claims_to_concepts.md:
    - SUPPORTS_CONCEPT and CONTRADICTS_CONCEPT: Requires meeting quality threshold
    - MENTIONS_CONCEPT: Allowed even if below threshold (but not with null scores)

    Args:
        entailed_score: Score from entailment evaluation (0.0-1.0 or None)
        coverage_score: Score from coverage evaluation (0.0-1.0 or None)
        decontextualization_score: Score from decontextualization evaluation (0.0-1.0 or None)
        quality_threshold: Minimum geometric mean required (0.0-1.0)
        relationship_type: Type of relationship ("SUPPORTS_CONCEPT", "CONTRADICTS_CONCEPT", "MENTIONS_CONCEPT")

    Returns:
        True if linking should be allowed, False otherwise
    """
    # Always reject if any scores are null
    if any(
        score is None
        for score in [entailed_score, coverage_score, decontextualization_score]
    ):
        logger.debug(
            f"Concept linking rejected: null scores present for {relationship_type}"
        )
        return False

    # For MENTIONS_CONCEPT, allow even if below threshold (as long as no nulls)
    if relationship_type == "MENTIONS_CONCEPT":
        logger.debug(
            f"Allowing {relationship_type} relationship despite potential low scores"
        )
        return True

    # For SUPPORTS_CONCEPT and CONTRADICTS_CONCEPT, require quality threshold
    if relationship_type in ["SUPPORTS_CONCEPT", "CONTRADICTS_CONCEPT"]:
        return meets_quality_threshold(
            entailed_score, coverage_score, decontextualization_score, quality_threshold
        )

    # Unknown relationship type - be conservative and require threshold
    logger.warning(
        f"Unknown relationship type: {relationship_type}, requiring quality threshold"
    )
    return meets_quality_threshold(
        entailed_score, coverage_score, decontextualization_score, quality_threshold
    )


def should_allow_vault_promotion(
    entailed_score: Optional[float],
    coverage_score: Optional[float],
    decontextualization_score: Optional[float],
    quality_threshold: float,
) -> bool:
    """
    Determine if a claim should be promoted to vault files (Tier 2/3).

    Following docs/arch/on-evaluation_agents.md promotion logic:
    Claims are promoted only if they meet the quality threshold.

    Args:
        entailed_score: Score from entailment evaluation (0.0-1.0 or None)
        coverage_score: Score from coverage evaluation (0.0-1.0 or None)
        decontextualization_score: Score from decontextualization evaluation (0.0-1.0 or None)
        quality_threshold: Minimum geometric mean required (0.0-1.0)

    Returns:
        True if claim should be promoted to vault, False otherwise
    """
    return meets_quality_threshold(
        entailed_score, coverage_score, decontextualization_score, quality_threshold
    )


def should_include_in_summaries(
    entailed_score: Optional[float],
    coverage_score: Optional[float],
    decontextualization_score: Optional[float],
    quality_threshold: float,
) -> bool:
    """
    Determine if a claim should be included in summaries and concept pages.

    Following docs/arch/on-evaluation_agents.md:
    Claims are included in summaries only if they meet the quality threshold.

    Args:
        entailed_score: Score from entailment evaluation (0.0-1.0 or None)
        coverage_score: Score from coverage evaluation (0.0-1.0 or None)
        decontextualization_score: Score from decontextualization evaluation (0.0-1.0 or None)
        quality_threshold: Minimum geometric mean required (0.0-1.0)

    Returns:
        True if claim should be included in summaries, False otherwise
    """
    return meets_quality_threshold(
        entailed_score, coverage_score, decontextualization_score, quality_threshold
    )
