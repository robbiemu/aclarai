"""
Concept Subject Linking Job for aclarai scheduler.
This module implements the scheduled job for linking concepts to their subjects
by adding footer links in [[Concept]] Markdown files pointing to their parent
[[Subject:XYZ]] pages, following the architecture from
docs/arch/on-writing_vault_documents.md.
"""

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from aclarai_shared import load_config
from aclarai_shared.config import ConceptSubjectLinkingJobConfig, aclaraiConfig
from aclarai_shared.graph.neo4j_manager import Neo4jGraphManager
from aclarai_shared.import_system import write_file_atomically

from .concept_clustering_job import ConceptClusteringJob

logger = logging.getLogger(__name__)


class ConceptSubjectLinkingJobStats(TypedDict):
    """Type definition for job statistics."""

    success: bool
    concepts_processed: int
    concepts_linked: int
    concepts_skipped: int
    files_updated: int
    neo4j_edges_created: int
    duration: float
    error_details: List[str]


class ConceptSubjectLinkingJob:
    """
    Job for linking concepts to their subjects.

    This job:
    1. Gets cluster assignments from the concept clustering job
    2. Finds all concept files in the vault
    3. For each concept, determines its subject(s) from cluster assignments
    4. Adds footer links to concept files pointing to their subject pages
    5. Increments version numbers using atomic writes
    6. Optionally creates Neo4j graph edges (:Concept)-[:PART_OF]->(:Subject)
    """

    def __init__(
        self,
        config: Optional[aclaraiConfig] = None,
        neo4j_manager: Optional[Neo4jGraphManager] = None,
        concept_clustering_job: Optional[ConceptClusteringJob] = None,
    ):
        """Initialize concept subject linking job."""
        self.config = config or load_config(validate=True)
        self.neo4j_manager = neo4j_manager or Neo4jGraphManager(self.config)
        self.concept_clustering_job = concept_clustering_job or ConceptClusteringJob(self.config)

        # Get job-specific configuration
        self.job_config: ConceptSubjectLinkingJobConfig = (
            self.config.scheduler.jobs.concept_subject_linking
        )

        # Get vault path
        self.vault_path = Path(self.config.vault_path)
        self.concepts_dir = self.vault_path / "concepts"

    def _get_cluster_assignments(self) -> Optional[Dict[str, int]]:
        """
        Get cluster assignments from the concept clustering job.

        Returns:
            Dictionary mapping concept_name to cluster_id, or None if not available
        """
        try:
            assignments = self.concept_clustering_job.get_cluster_assignments()
            if assignments:
                logger.info(
                    f"concept_subject_linking_job._get_cluster_assignments: Retrieved {len(assignments)} cluster assignments",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_subject_linking_job._get_cluster_assignments",
                        "assignments_count": len(assignments),
                    },
                )
                return assignments
            else:
                logger.warning(
                    "concept_subject_linking_job._get_cluster_assignments: No cluster assignments available",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_subject_linking_job._get_cluster_assignments",
                    },
                )
                return None

        except Exception as e:
            logger.error(
                f"concept_subject_linking_job._get_cluster_assignments: Failed to get cluster assignments: {e}",
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_subject_linking_job._get_cluster_assignments",
                    "error": str(e),
                },
            )
            return None

    def _get_subject_name_by_cluster_id(self, cluster_id: int) -> Optional[str]:
        """
        Get the subject name for a given cluster ID by querying Neo4j.

        Args:
            cluster_id: The cluster ID to look up

        Returns:
            Subject name or None if not found
        """
        try:
            # Query Neo4j for subject with this cluster_id
            query = """
            MATCH (s:Subject {cluster_id: $cluster_id})
            RETURN s.name as name
            LIMIT 1
            """
            result = self.neo4j_manager.execute_query(query, {"cluster_id": cluster_id})

            if result and len(result) > 0:
                subject_name = result[0]["name"]
                logger.debug(
                    f"concept_subject_linking_job._get_subject_name_by_cluster_id: Found subject {subject_name} for cluster {cluster_id}",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_subject_linking_job._get_subject_name_by_cluster_id",
                        "cluster_id": cluster_id,
                        "subject_name": subject_name,
                    },
                )
                return subject_name
            else:
                logger.warning(
                    f"concept_subject_linking_job._get_subject_name_by_cluster_id: No subject found for cluster {cluster_id}",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_subject_linking_job._get_subject_name_by_cluster_id",
                        "cluster_id": cluster_id,
                    },
                )
                return None

        except Exception as e:
            logger.error(
                f"concept_subject_linking_job._get_subject_name_by_cluster_id: Failed to get subject name for cluster {cluster_id}: {e}",
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_subject_linking_job._get_subject_name_by_cluster_id",
                    "cluster_id": cluster_id,
                    "error": str(e),
                },
            )
            return None

    def _get_concept_files(self) -> List[Path]:
        """
        Get all concept Markdown files in the vault.

        Returns:
            List of paths to concept files
        """
        try:
            if not self.concepts_dir.exists():
                logger.warning(
                    f"concept_subject_linking_job._get_concept_files: Concepts directory does not exist: {self.concepts_dir}",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_subject_linking_job._get_concept_files",
                        "concepts_dir": str(self.concepts_dir),
                    },
                )
                return []

            concept_files = list(self.concepts_dir.glob("*.md"))
            logger.info(
                f"concept_subject_linking_job._get_concept_files: Found {len(concept_files)} concept files",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_subject_linking_job._get_concept_files",
                    "files_count": len(concept_files),
                },
            )
            return concept_files

        except Exception as e:
            logger.error(
                f"concept_subject_linking_job._get_concept_files: Failed to get concept files: {e}",
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_subject_linking_job._get_concept_files",
                    "error": str(e),
                },
            )
            return []

    def _extract_concept_name_from_file(self, file_path: Path) -> Optional[str]:
        """
        Extract concept name from a concept file.

        Args:
            file_path: Path to the concept file

        Returns:
            Concept name without .md extension, or None if invalid
        """
        try:
            # Simple approach: use filename without extension
            concept_name = file_path.stem

            # Validate it's a real concept file by checking for aclarai:id
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if "aclarai:id=concept_" in content:
                    return concept_name
                else:
                    logger.debug(
                        f"concept_subject_linking_job._extract_concept_name_from_file: File {file_path} does not contain concept metadata",
                        extra={
                            "service": "aclarai-scheduler",
                            "filename.function_name": "concept_subject_linking_job._extract_concept_name_from_file",
                            "file_path": str(file_path),
                        },
                    )
                    return None

        except Exception as e:
            logger.error(
                f"concept_subject_linking_job._extract_concept_name_from_file: Failed to extract concept name from {file_path}: {e}",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_subject_linking_job._extract_concept_name_from_file",
                    "file_path": str(file_path),
                    "error": str(e),
                },
            )
            return None

    def _add_subject_footer_links(self, content: str, subject_names: List[str]) -> str:
        """
        Add footer links to subject pages in the concept file content.

        Args:
            content: Original file content
            subject_names: List of subject names to link to

        Returns:
            Updated content with footer links added
        """
        if not subject_names:
            return content

        # Create the footer section
        footer_section = f"\n\n## {self.job_config.footer_section_title}\n\n"
        for subject_name in subject_names:
            footer_section += f"- [[Subject:{subject_name}]]\n"

        # Check if footer section already exists
        footer_pattern = rf"## {re.escape(self.job_config.footer_section_title)}"
        if re.search(footer_pattern, content, re.IGNORECASE):
            logger.debug(
                "concept_subject_linking_job._add_subject_footer_links: Footer section already exists, skipping",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_subject_linking_job._add_subject_footer_links",
                    "subject_names": subject_names,
                },
            )
            return content

        # Find the location to insert the footer (before the aclarai metadata)
        # Look for the pattern: <!-- aclarai:id=... -->
        metadata_pattern = r"(<!-- aclarai:id=.*? -->)"
        match = re.search(metadata_pattern, content, re.DOTALL)

        if match:
            # Insert footer before the metadata
            insert_pos = match.start()
            updated_content = content[:insert_pos] + footer_section + content[insert_pos:]
        else:
            # If no metadata found, append to the end
            updated_content = content + footer_section

        logger.debug(
            f"concept_subject_linking_job._add_subject_footer_links: Added footer links for {len(subject_names)} subjects",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "concept_subject_linking_job._add_subject_footer_links",
                "subject_names": subject_names,
            },
        )
        return updated_content

    def _increment_version(self, content: str) -> str:
        """
        Increment the version number in the aclarai metadata.

        Args:
            content: The file content

        Returns:
            Content with incremented version
        """
        # Pattern to match aclarai:id=... ver=N
        version_pattern = r"(aclarai:id=[^\s]+)\s+ver=(\d+)"

        def increment_match(match: re.Match) -> str:
            aclarai_part = match.group(1)
            current_version = int(match.group(2))
            new_version = current_version + 1
            return f"{aclarai_part} ver={new_version}"

        updated_content = re.sub(version_pattern, increment_match, content)
        return updated_content

    def _update_concept_file(
        self, file_path: Path, concept_name: str, subject_names: List[str]
    ) -> bool:
        """
        Update a concept file with subject footer links.

        Args:
            file_path: Path to the concept file
            concept_name: Name of the concept
            subject_names: List of subject names to link to

        Returns:
            True if file was updated, False otherwise
        """
        try:
            # Read current content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Add footer links
            updated_content = self._add_subject_footer_links(content, subject_names)

            # Only proceed if content changed
            if updated_content != content:
                # Increment version
                updated_content = self._increment_version(updated_content)

                # Write atomically
                write_file_atomically(file_path, updated_content)

                logger.info(
                    f"concept_subject_linking_job._update_concept_file: Updated concept file {file_path.name} with {len(subject_names)} subject links",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_subject_linking_job._update_concept_file",
                        "file_path": str(file_path),
                        "concept_name": concept_name,
                        "subject_names": subject_names,
                    },
                )
                return True
            else:
                logger.debug(
                    f"concept_subject_linking_job._update_concept_file: No changes needed for concept file {file_path.name}",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_subject_linking_job._update_concept_file",
                        "file_path": str(file_path),
                        "concept_name": concept_name,
                    },
                )
                return False

        except Exception as e:
            logger.error(
                f"concept_subject_linking_job._update_concept_file: Failed to update concept file {file_path}: {e}",
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_subject_linking_job._update_concept_file",
                    "file_path": str(file_path),
                    "concept_name": concept_name,
                    "error": str(e),
                },
            )
            return False

    def _create_neo4j_edge(self, concept_name: str, subject_name: str) -> bool:
        """
        Create a Neo4j edge between concept and subject.

        Args:
            concept_name: Name of the concept
            subject_name: Name of the subject

        Returns:
            True if edge was created, False otherwise
        """
        try:
            query = """
            MATCH (c:Concept {name: $concept_name})
            MATCH (s:Subject {name: $subject_name})
            MERGE (c)-[:PART_OF]->(s)
            RETURN c, s
            """

            result = self.neo4j_manager.execute_query(
                query, {"concept_name": concept_name, "subject_name": subject_name}
            )

            if result:
                logger.debug(
                    f"concept_subject_linking_job._create_neo4j_edge: Created PART_OF edge from {concept_name} to {subject_name}",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_subject_linking_job._create_neo4j_edge",
                        "concept_name": concept_name,
                        "subject_name": subject_name,
                    },
                )
                return True
            else:
                logger.warning(
                    f"concept_subject_linking_job._create_neo4j_edge: Failed to create edge from {concept_name} to {subject_name}",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_subject_linking_job._create_neo4j_edge",
                        "concept_name": concept_name,
                        "subject_name": subject_name,
                    },
                )
                return False

        except Exception as e:
            logger.error(
                f"concept_subject_linking_job._create_neo4j_edge: Failed to create Neo4j edge from {concept_name} to {subject_name}: {e}",
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_subject_linking_job._create_neo4j_edge",
                    "concept_name": concept_name,
                    "subject_name": subject_name,
                    "error": str(e),
                },
            )
            return False

    def run_job(self) -> ConceptSubjectLinkingJobStats:
        """
        Execute the complete concept subject linking job.

        Returns:
            Dictionary with job statistics and results
        """
        start_time = time.time()
        stats: ConceptSubjectLinkingJobStats = {
            "success": False,
            "concepts_processed": 0,
            "concepts_linked": 0,
            "concepts_skipped": 0,
            "files_updated": 0,
            "neo4j_edges_created": 0,
            "duration": 0.0,
            "error_details": [],
        }

        logger.info(
            "concept_subject_linking_job.run_job: Starting concept subject linking job",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "concept_subject_linking_job.run_job",
                "create_neo4j_edges": self.job_config.create_neo4j_edges,
                "batch_size": self.job_config.batch_size,
            },
        )

        try:
            # Step 1: Get cluster assignments
            cluster_assignments = self._get_cluster_assignments()
            if not cluster_assignments:
                logger.warning(
                    "concept_subject_linking_job.run_job: No cluster assignments available",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_subject_linking_job.run_job",
                    },
                )
                stats["error_details"].append("No cluster assignments available")
                return stats

            # Step 2: Get concept files
            concept_files = self._get_concept_files()
            if not concept_files:
                logger.warning(
                    "concept_subject_linking_job.run_job: No concept files found",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_subject_linking_job.run_job",
                    },
                )
                stats["error_details"].append("No concept files found")
                return stats

            # Step 3: Process concept files
            for file_path in concept_files:
                try:
                    stats["concepts_processed"] += 1

                    # Extract concept name
                    concept_name = self._extract_concept_name_from_file(file_path)
                    if not concept_name:
                        stats["concepts_skipped"] += 1
                        continue

                    # Get cluster ID for this concept
                    cluster_id = cluster_assignments.get(concept_name)
                    if cluster_id is None:
                        logger.debug(
                            f"concept_subject_linking_job.run_job: No cluster assignment for concept {concept_name}",
                            extra={
                                "service": "aclarai-scheduler",
                                "filename.function_name": "concept_subject_linking_job.run_job",
                                "concept_name": concept_name,
                            },
                        )
                        stats["concepts_skipped"] += 1
                        continue

                    # Get subject name for this cluster
                    subject_name = self._get_subject_name_by_cluster_id(cluster_id)
                    if not subject_name:
                        stats["concepts_skipped"] += 1
                        continue

                    # Update concept file with subject link
                    if self._update_concept_file(file_path, concept_name, [subject_name]):
                        stats["files_updated"] += 1
                        stats["concepts_linked"] += 1

                        # Create Neo4j edge if enabled
                        if self.job_config.create_neo4j_edges and self._create_neo4j_edge(concept_name, subject_name):
                            stats["neo4j_edges_created"] += 1
                    else:
                        stats["concepts_skipped"] += 1

                except Exception as e:
                    error_msg = f"Failed to process concept file {file_path}: {e}"
                    logger.error(
                        f"concept_subject_linking_job.run_job: {error_msg}",
                        exc_info=True,
                        extra={
                            "service": "aclarai-scheduler",
                            "filename.function_name": "concept_subject_linking_job.run_job",
                            "file_path": str(file_path),
                            "error": str(e),
                        },
                    )
                    stats["error_details"].append(error_msg)

            # Mark as successful if we processed at least some concepts
            if stats["concepts_processed"] > 0:
                stats["success"] = True

            logger.info(
                "concept_subject_linking_job.run_job: Completed concept subject linking job",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_subject_linking_job.run_job",
                    "concepts_processed": stats["concepts_processed"],
                    "concepts_linked": stats["concepts_linked"],
                    "concepts_skipped": stats["concepts_skipped"],
                    "files_updated": stats["files_updated"],
                    "neo4j_edges_created": stats["neo4j_edges_created"],
                },
            )

        except Exception as e:
            error_msg = f"Unexpected error in concept subject linking job: {e}"
            logger.error(
                f"concept_subject_linking_job.run_job: {error_msg}",
                exc_info=True,
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_subject_linking_job.run_job",
                    "error": str(e),
                },
            )
            stats["error_details"].append(error_msg)

        finally:
            stats["duration"] = time.time() - start_time

        return stats
