"""
Trending Topics Job for aclarai scheduler.
This module implements the scheduled job for generating Trending Topics - <date>.md
from concept mention deltas over a configurable time window, following the
architecture from docs/arch/on-linking_claims_to_concepts.md and output format
from docs/arch/on-writing_vault_documents.md.
"""

import contextlib
import logging
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

from aclarai_shared import load_config
from aclarai_shared.config import TrendingTopicsJobConfig, aclaraiConfig
from aclarai_shared.graph.neo4j_manager import Neo4jGraphManager

logger = logging.getLogger(__name__)


class TrendingTopicsJobStats(TypedDict):
    """Type definition for job statistics."""

    success: bool
    concepts_analyzed: int
    trending_concepts_selected: int
    file_written: bool
    window_start: str
    window_end: str
    duration: float
    error_details: List[str]


class TrendingTopicsJob:
    """
    Job for generating Trending Topics - <date>.md from concept mention deltas.

    This job:
    1. Connects to Neo4j and tracks timestamps of SUPPORTS_CONCEPT and MENTIONS_CONCEPT edges
    2. Calculates mention deltas for each concept over the configured time window
    3. Selects top N concepts by growth (configurable count or percent)
    4. Generates markdown content following the "Trending Concepts Agent" format
    5. Writes the file atomically to the vault with aclarai:id tracking
    """

    def __init__(
        self,
        config: Optional[aclaraiConfig] = None,
        neo4j_manager: Optional[Neo4jGraphManager] = None,
    ):
        """Initialize trending topics job."""
        self.config = config or load_config(validate=True)
        self.neo4j_manager = neo4j_manager or Neo4jGraphManager(self.config)

        # Get job-specific configuration
        self.job_config: TrendingTopicsJobConfig = (
            self.config.scheduler.jobs.trending_topics
        )

        # Vault path from config
        self.vault_path = Path(self.config.vault_path)

    def _get_time_windows(self) -> Tuple[datetime, datetime, datetime]:
        """
        Calculate time windows for trending analysis.

        Returns:
            Tuple of (current_time, window_start, comparison_window_start)
        """
        current_time = datetime.now(timezone.utc)
        window_start = current_time - timedelta(days=self.job_config.window_days)
        comparison_window_start = window_start - timedelta(
            days=self.job_config.window_days
        )

        return current_time, window_start, comparison_window_start

    def _get_concept_mention_deltas(self) -> List[Tuple[str, int, int, float]]:
        """
        Calculate mention deltas for each concept over the configured time window.

        Returns:
            List of tuples containing (concept_name, current_mentions, previous_mentions, growth_rate)
        """
        current_time, window_start, comparison_window_start = self._get_time_windows()

        # Convert to ISO format for Neo4j
        current_time_str = current_time.isoformat()
        window_start_str = window_start.isoformat()
        comparison_window_start_str = comparison_window_start.isoformat()

        # Query to get mention counts for current and comparison windows
        query = """
        MATCH (c:Concept)
        OPTIONAL MATCH (c)<-[r_current:SUPPORTS_CONCEPT|MENTIONS_CONCEPT]-(claim_current:Claim)
        WHERE r_current.classified_at >= datetime($window_start) AND r_current.classified_at <= datetime($current_time)
        OPTIONAL MATCH (c)<-[r_prev:SUPPORTS_CONCEPT|MENTIONS_CONCEPT]-(claim_prev:Claim)
        WHERE r_prev.classified_at >= datetime($comparison_window_start) AND r_prev.classified_at < datetime($window_start)

        WITH c,
             count(DISTINCT r_current) as current_mentions,
             count(DISTINCT r_prev) as previous_mentions

        WHERE current_mentions >= $min_mentions OR previous_mentions >= $min_mentions

        WITH c, current_mentions, previous_mentions,
             CASE
                WHEN previous_mentions = 0 AND current_mentions > 0 THEN 1000.0  // New concept, high growth
                WHEN previous_mentions > 0 THEN toFloat(current_mentions - previous_mentions) / previous_mentions
                ELSE 0.0
             END as growth_rate

        RETURN c.name as concept_name, current_mentions, previous_mentions, growth_rate
        ORDER BY growth_rate DESC, current_mentions DESC
        """

        params = {
            "current_time": current_time_str,
            "window_start": window_start_str,
            "comparison_window_start": comparison_window_start_str,
            "min_mentions": self.job_config.min_mentions,
        }

        try:
            result = self.neo4j_manager.execute_query(query, params)

            if result:
                concept_deltas = [
                    (
                        record["concept_name"],
                        record["current_mentions"],
                        record["previous_mentions"],
                        record["growth_rate"],
                    )
                    for record in result
                ]

                logger.info(
                    f"trending_topics_job._get_concept_mention_deltas: Found {len(concept_deltas)} concepts with mention data",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "trending_topics_job._get_concept_mention_deltas",
                        "concepts_count": len(concept_deltas),
                        "window_start": window_start_str,
                        "current_time": current_time_str,
                        "min_mentions": self.job_config.min_mentions,
                    },
                )
                return concept_deltas
            else:
                logger.warning(
                    "trending_topics_job._get_concept_mention_deltas: No concept mention data returned from query",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "trending_topics_job._get_concept_mention_deltas",
                    },
                )
                return []

        except Exception as e:
            logger.error(
                f"trending_topics_job._get_concept_mention_deltas: Failed to retrieve concept mention deltas: {e}",
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "trending_topics_job._get_concept_mention_deltas",
                    "error": str(e),
                },
            )
            return []

    def _select_trending_concepts(
        self, concept_deltas: List[Tuple[str, int, int, float]]
    ) -> List[Tuple[str, int, int, float]]:
        """
        Select top trending concepts based on configuration.

        Args:
            concept_deltas: List of (concept_name, current_mentions, previous_mentions, growth_rate)

        Returns:
            Filtered list of trending concepts
        """
        if not concept_deltas:
            return []

        # Determine limit based on configuration
        if self.job_config.count is not None:
            limit = self.job_config.count
        elif self.job_config.percent is not None:
            limit = max(1, int(len(concept_deltas) * self.job_config.percent / 100))
        else:
            # Default to 10% if neither count nor percent is specified
            limit = max(1, int(len(concept_deltas) * 0.1))

        # Select top concepts by growth rate
        trending_concepts = concept_deltas[:limit]

        logger.info(
            f"trending_topics_job._select_trending_concepts: Selected {len(trending_concepts)} trending concepts",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "trending_topics_job._select_trending_concepts",
                "selected_count": len(trending_concepts),
                "total_analyzed": len(concept_deltas),
                "limit_used": limit,
            },
        )

        return trending_concepts

    def _generate_markdown_content(
        self, trending_concepts: List[Tuple[str, int, int, float]]
    ) -> str:
        """
        Generate markdown content following the "Trending Concepts Agent" format.

        Args:
            trending_concepts: List of (concept_name, current_mentions, previous_mentions, growth_rate)

        Returns:
            Formatted markdown content
        """
        current_time, window_start, _ = self._get_time_windows()

        # Format dates for human reading
        current_date = current_time.strftime("%Y-%m-%d")
        window_start_date = window_start.strftime("%Y-%m-%d")

        if not trending_concepts:
            content = f"""## Trending This Week

*No trending concepts found for the period from {window_start_date} to {current_date}.*

"""
        else:
            content = "## Trending This Week\n\n"
            content += f"*Analysis period: {window_start_date} to {current_date}*\n\n"

            for (
                concept_name,
                current_mentions,
                previous_mentions,
                growth_rate,
            ) in trending_concepts:
                # Create wikilink and description
                if previous_mentions == 0:
                    description = f"New concept with {current_mentions} mentions"
                else:
                    growth_percent = growth_rate * 100
                    if growth_percent > 0:
                        description = f"Mentions up {growth_percent:.0f}% ({previous_mentions} → {current_mentions})"
                    elif growth_percent < 0:
                        description = f"Mentions down {abs(growth_percent):.0f}% ({previous_mentions} → {current_mentions})"
                    else:
                        description = f"Stable at {current_mentions} mentions"

                content += f"- [[{concept_name}]] — {description}\n"

            content += "\n"

        # Add aclarai:id metadata for vault synchronization
        # Use current date to create a unique but consistent ID
        file_date = current_time.strftime("%Y%m%d")
        file_id = f"file_trending_topics_{file_date}"
        content += f"<!-- aclarai:id={file_id} ver=1 -->\n"
        content += f"^{file_id}\n"

        return content

    def _get_target_file_path(self) -> Path:
        """
        Generate the target file path with current date.

        Returns:
            Path object for the target file
        """
        current_time, _, _ = self._get_time_windows()
        current_date = current_time.strftime("%Y-%m-%d")

        # Replace {date} placeholder in target_file configuration
        target_filename = self.job_config.target_file.format(date=current_date)

        return self.vault_path / target_filename

    def _write_file_atomically(self, content: str, target_path: Path) -> bool:
        """
        Write content to target file atomically using temp file and rename.

        Args:
            content: The markdown content to write
            target_path: Path to the target file

        Returns:
            True if write was successful, False otherwise
        """
        try:
            # Ensure target directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Create temporary file in same directory for atomic rename
            temp_dir = target_path.parent
            temp_fd, temp_path_str = tempfile.mkstemp(
                dir=temp_dir, prefix=target_path.name + ".tmp", suffix=".md"
            )
            temp_path = Path(temp_path_str)

            # Write content to temporary file
            with os.fdopen(temp_fd, "w", encoding="utf-8") as tmp_file:
                tmp_file.write(content)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())  # Ensure data is written to disk

            # Atomic rename to target file
            os.replace(temp_path, target_path)

            logger.info(
                f"trending_topics_job._write_file_atomically: Successfully wrote {target_path}",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "trending_topics_job._write_file_atomically",
                    "target_file": str(target_path),
                    "content_length": len(content),
                },
            )
            return True

        except Exception as e:
            logger.error(
                f"trending_topics_job._write_file_atomically: Failed to write {target_path}: {e}",
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "trending_topics_job._write_file_atomically",
                    "target_file": str(target_path),
                    "error": str(e),
                },
            )

            # Clean up temp file if it exists
            if "temp_path" in locals() and temp_path.exists():
                with contextlib.suppress(OSError):
                    temp_path.unlink()

            return False

    def run_job(self) -> TrendingTopicsJobStats:
        """
        Execute the complete trending topics job.

        Returns:
            Dictionary with job statistics and results
        """
        start_time = time.time()
        current_time, window_start, _ = self._get_time_windows()

        stats: TrendingTopicsJobStats = {
            "success": False,
            "concepts_analyzed": 0,
            "trending_concepts_selected": 0,
            "file_written": False,
            "window_start": window_start.isoformat(),
            "window_end": current_time.isoformat(),
            "duration": 0.0,
            "error_details": [],
        }

        logger.info(
            "trending_topics_job.run_job: Starting Trending Topics job",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "trending_topics_job.run_job",
                "window_days": self.job_config.window_days,
                "min_mentions": self.job_config.min_mentions,
                "count": self.job_config.count,
                "percent": self.job_config.percent,
                "window_start": stats["window_start"],
                "window_end": stats["window_end"],
            },
        )

        try:
            # Step 1: Get concept mention deltas
            concept_deltas = self._get_concept_mention_deltas()
            stats["concepts_analyzed"] = len(concept_deltas)

            if not concept_deltas:
                logger.warning(
                    "trending_topics_job.run_job: No concepts found for Trending Topics generation",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "trending_topics_job.run_job",
                    },
                )
                # Still generate file with empty content

            # Step 2: Select trending concepts
            trending_concepts = self._select_trending_concepts(concept_deltas)
            stats["trending_concepts_selected"] = len(trending_concepts)

            # Step 3: Generate markdown content
            markdown_content = self._generate_markdown_content(trending_concepts)

            # Step 4: Write file atomically
            target_path = self._get_target_file_path()
            if self._write_file_atomically(markdown_content, target_path):
                stats["file_written"] = True
                stats["success"] = True
                logger.info(
                    f"trending_topics_job.run_job: Successfully completed Trending Topics job with {len(trending_concepts)} concepts",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "trending_topics_job.run_job",
                        "concepts_analyzed": len(concept_deltas),
                        "trending_concepts": len(trending_concepts),
                        "target_file": str(target_path),
                    },
                )
            else:
                stats["error_details"].append("Failed to write target file")

        except Exception as e:
            error_msg = f"Unexpected error in Trending Topics job: {e}"
            logger.error(
                f"trending_topics_job.run_job: {error_msg}",
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "trending_topics_job.run_job",
                    "error": str(e),
                },
            )
            stats["error_details"].append(error_msg)

        finally:
            stats["duration"] = time.time() - start_time

        return stats
