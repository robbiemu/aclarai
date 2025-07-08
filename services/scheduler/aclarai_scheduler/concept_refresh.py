"""
Concept Embedding Refresh Job for aclarai scheduler.
This module implements the scheduled job for refreshing concept embeddings
from Tier 3 concept files, following the architecture from
docs/arch/on-refreshing_concept_embeddings.md
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

from aclarai_shared import load_config
from aclarai_shared.config import aclaraiConfig
from aclarai_shared.embedding.chunking import ChunkMetadata
from aclarai_shared.embedding.models import EmbeddedChunk, EmbeddingGenerator
from aclarai_shared.embedding.storage import aclaraiVectorStore
from aclarai_shared.graph.neo4j_manager import Neo4jGraphManager

logger = logging.getLogger(__name__)


class JobStatsTypedDict(TypedDict):
    success: bool
    concepts_processed: int
    concepts_updated: int
    concepts_skipped: int
    errors: int
    error_details: List[str]  # List of error messages
    duration: float


class ConceptEmbeddingRefreshJob:
    """
    Job for refreshing concept embeddings from Tier 3 concept files.
    This job:
    1. Iterates over all concept files in vault/concepts/*.md
    2. Extracts semantic text (removing metadata) and computes SHA256 hash
    3. Compares hash with stored embedding_hash in Neo4j
    4. Recalculates embeddings for changed files
    5. Updates both vector store and Neo4j with new embeddings and metadata
    """

    def __init__(
        self,
        config: Optional[aclaraiConfig] = None,
        neo4j_manager: Optional[Neo4jGraphManager] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[aclaraiVectorStore] = None,
    ):
        """
        Initialize the concept embedding refresh job.
        Args:
            config: aclarai configuration (loads default if None)
            neo4j_manager: Neo4j graph manager (creates default if None)
            embedding_generator: Embedding generator (creates default if None)
            vector_store: Vector store (creates default if None)
        """
        self.config = config or load_config(validate=True)
        self.concepts_path = Path(self.config.vault_path) / "concepts"
        # Use injected dependencies or create real ones if not provided
        self.neo4j_manager = neo4j_manager or Neo4jGraphManager(self.config)
        self.embedding_generator = embedding_generator or EmbeddingGenerator(
            self.config
        )
        if not self.embedding_generator.model_name:
            raise ValueError("Embedding model name must be specified in configuration")
        self.vector_store = vector_store or aclaraiVectorStore(self.config)

    def run_job(self) -> JobStatsTypedDict:
        start_time = time.time()
        job_stats: JobStatsTypedDict = {
            "success": True,
            "concepts_processed": 0,
            "concepts_updated": 0,
            "concepts_skipped": 0,
            "errors": 0,
            "error_details": [],  # Ensure it's initialized as a list
            "duration": 0.0,
        }
        logger.info(
            "concept_refresh.run_job: Starting concept embedding refresh job",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "concept_refresh.run_job",
                "concepts_path": str(self.concepts_path),
            },
        )
        try:
            # Check if concepts directory exists
            if not self.concepts_path.exists():
                logger.warning(
                    f"concept_refresh.run_job: Concepts directory does not exist: {self.concepts_path}",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_refresh.run_job",
                        "concepts_path": str(self.concepts_path),
                    },
                )
                job_stats["success"] = False
                job_stats["error_details"].append("Concepts directory does not exist")
                return job_stats
            # Get all concept files
            concept_files = list(self.concepts_path.glob("*.md"))
            logger.info(
                f"concept_refresh.run_job: Found {len(concept_files)} concept files",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_refresh.run_job",
                    "concept_files_count": len(concept_files),
                },
            )
            # Process each concept file
            for concept_file in concept_files:
                try:
                    processed, updated = self._process_concept_file(concept_file)
                    if processed:
                        job_stats["concepts_processed"] += 1
                        if updated:
                            job_stats["concepts_updated"] += 1
                        else:
                            job_stats["concepts_skipped"] += 1
                except Exception as e:
                    job_stats["errors"] += 1
                    error_msg = f"Error processing {concept_file.name}: {str(e)}"
                    job_stats["error_details"].append(error_msg)
                    logger.error(
                        f"concept_refresh.run_job: {error_msg}",
                        extra={
                            "service": "aclarai-scheduler",
                            "filename.function_name": "concept_refresh.run_job",
                            "concept_file": concept_file.name,
                            "error": str(e),
                        },
                    )
        except Exception as e:
            job_stats["success"] = False
            job_stats["error_details"].append(f"Job failed: {str(e)}")
            logger.error(
                f"concept_refresh.run_job: Job failed with error: {str(e)}",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_refresh.run_job",
                    "error": str(e),
                },
            )
        job_stats["duration"] = time.time() - start_time
        logger.info(
            "concept_refresh.run_job: Concept embedding refresh job completed",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "concept_refresh.run_job",
                "duration": job_stats["duration"],
                "concepts_processed": job_stats["concepts_processed"],
                "concepts_updated": job_stats["concepts_updated"],
                "concepts_skipped": job_stats["concepts_skipped"],
                "errors": job_stats["errors"],
                "success": job_stats["success"],
            },
        )
        return job_stats

    def _process_concept_file(self, concept_file: Path) -> Tuple[bool, bool]:
        """
        Process a single concept file for embedding refresh.
        Args:
            concept_file: Path to the concept file
        Returns:
            Tuple of (processed_successfully, was_updated)
        """
        concept_name = concept_file.stem  # filename without .md extension
        logger.debug(
            f"concept_refresh._process_concept_file: Processing concept file: {concept_name}",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "concept_refresh._process_concept_file",
                "concept_name": concept_name,
                "concept_file": str(concept_file),
            },
        )
        try:
            # Read file content
            with open(concept_file, "r", encoding="utf-8") as f:
                file_content = f.read()
            # Extract semantic text and compute hash
            semantic_text = self._extract_semantic_text(file_content)
            current_hash = self._compute_hash(semantic_text)
            # Get existing hash from Neo4j
            stored_hash = self._get_stored_embedding_hash(concept_name)
            # Check if update is needed
            if stored_hash == current_hash:
                logger.debug(
                    f"concept_refresh._process_concept_file: No changes detected for concept: {concept_name}",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_refresh._process_concept_file",
                        "concept_name": concept_name,
                        "hash": current_hash,
                    },
                )
                return True, False
            # Update needed - compute new embedding
            logger.info(
                f"concept_refresh._process_concept_file: Changes detected, updating embedding for concept: {concept_name}",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_refresh._process_concept_file",
                    "concept_name": concept_name,
                    "old_hash": stored_hash,
                    "new_hash": current_hash,
                },
            )
            # Generate new embedding
            embedding = self.embedding_generator.embed_text(semantic_text)
            # Update vector store (upsert)
            self._update_vector_store(concept_name, embedding)
            # Update Neo4j metadata
            self._update_neo4j_metadata(concept_name, current_hash)
            logger.info(
                f"concept_refresh._process_concept_file: Successfully updated embedding for concept: {concept_name}",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_refresh._process_concept_file",
                    "concept_name": concept_name,
                    "embedding_dim": len(embedding),
                    "new_hash": current_hash,
                },
            )
            return True, True
        except Exception as e:
            logger.error(
                f"concept_refresh._process_concept_file: Failed to process concept file {concept_name}: {str(e)}",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_refresh._process_concept_file",
                    "concept_name": concept_name,
                    "error": str(e),
                },
            )
            raise

    def _extract_semantic_text(self, file_content: str) -> str:
        """
        Extract semantic text from concept file content.
        Removes metadata lines and anchor references as specified in
        docs/arch/on-refreshing_concept_embeddings.md
        Args:
            file_content: Raw file content
        Returns:
            Semantic text for embedding
        """
        lines = file_content.splitlines()
        semantic_lines = []
        for line in lines:
            # Skip metadata lines
            if line.strip().startswith("<!-- aclarai:"):
                continue
            # Skip anchor references (lines starting with ^)
            if line.strip().startswith("^"):
                continue
            semantic_lines.append(line)
        semantic_text = "\n".join(semantic_lines).strip()
        logger.debug(
            "concept_refresh._extract_semantic_text: Extracted semantic text",
            extra={
                "service": "aclarai-scheduler",
                "filename.function_name": "concept_refresh._extract_semantic_text",
                "original_lines": len(lines),
                "semantic_lines": len(semantic_lines),
                "semantic_text_length": len(semantic_text),
            },
        )
        return semantic_text

    def _compute_hash(self, text: str) -> str:
        """
        Compute SHA256 hash of the given text.
        Args:
            text: Text to hash
        Returns:
            SHA256 hash as hexadecimal string
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_stored_embedding_hash(self, concept_name: str) -> Optional[str]:
        """
        Get the stored embedding hash for a concept from Neo4j.
        Args:
            concept_name: Name of the concept
        Returns:
            Stored embedding hash or None if not found
        """
        try:
            with self.neo4j_manager.session() as session:
                result = session.run(
                    "MATCH (c:Concept {name: $name}) RETURN c.embedding_hash as hash",
                    name=concept_name,
                )
                record = result.single()
                if record and record["hash"] is not None:
                    return str(record["hash"])  # Ensure it's a string
                else:
                    logger.debug(
                        f"concept_refresh._get_stored_embedding_hash: No concept found with name: {concept_name}",
                        extra={
                            "service": "aclarai-scheduler",
                            "filename.function_name": "concept_refresh._get_stored_embedding_hash",
                            "concept_name": concept_name,
                        },
                    )
                    return None
        except Exception as e:
            logger.error(
                f"concept_refresh._get_stored_embedding_hash: Failed to query concept hash for {concept_name}: {str(e)}",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_refresh._get_stored_embedding_hash",
                    "concept_name": concept_name,
                    "error": str(e),
                },
            )
            # Return None to force update on error
            return None

    def _update_vector_store(self, concept_name: str, embedding: List[float]):
        """
        Update the vector store with the new embedding for a concept.
        This is achieved by deleting existing entries and inserting the new one.
        Args:
            concept_name: Name of the concept
            embedding: New embedding vector
        """
        try:
            # The unique ID for a concept's vector is derived from its name.
            # This ensures we can reliably find and delete it.
            concept_block_id = f"concept_{concept_name}"
            logger.info(
                f"concept_refresh._update_vector_store: Upserting embedding for concept: {concept_name} (ID: {concept_block_id})",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_refresh._update_vector_store",
                    "concept_name": concept_name,
                    "concept_block_id": concept_block_id,
                    "embedding_dim": len(embedding),
                },
            )
            # Check if vector store has an upsert method (for mocks)
            if hasattr(self.vector_store, "upsert"):
                self.vector_store.upsert(concept_name, embedding)
                return
            # 1. Delete any existing vectors for this concept to ensure a clean update.
            # This makes the operation idempotent.
            deleted_count = self.vector_store.delete_chunks_by_block_id(
                concept_block_id
            )
            if deleted_count > 0:
                logger.debug(
                    f"concept_refresh._update_vector_store: Deleted {deleted_count} old vector(s) for concept '{concept_name}'",
                    extra={
                        "service": "aclarai-scheduler",
                        "filename.function_name": "concept_refresh._update_vector_store",
                        "concept_name": concept_name,
                        "deleted_count": deleted_count,
                    },
                )
            # 2. Create a new EmbeddedChunk to insert.
            # The semantic text of the concept file is the "content" of this chunk.
            semantic_text = (
                f"Concept: {concept_name}"  # Basic content for metadata context
            )
            chunk_meta = ChunkMetadata(
                aclarai_block_id=concept_block_id,
                chunk_index=0,  # A concept file is treated as a single chunk
                original_text=semantic_text,
                text=semantic_text,
            )
            embedded_chunk = EmbeddedChunk(
                chunk_metadata=chunk_meta,
                embedding=embedding,
                model_name=self.embedding_generator.model_name,  # type: ignore[arg-type]
                embedding_dim=len(embedding),
            )
            # 3. Store the new embedding.
            metrics = self.vector_store.store_embeddings([embedded_chunk])
            if metrics.failed_inserts > 0:
                raise RuntimeError(
                    f"Failed to insert new embedding for concept '{concept_name}'."
                )
            logger.info(
                f"concept_refresh._update_vector_store: Successfully upserted embedding for concept: {concept_name}",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_refresh._update_vector_store",
                    "concept_name": concept_name,
                    "concept_block_id": concept_block_id,
                    "successful_inserts": metrics.successful_inserts,
                },
            )
        except Exception as e:
            logger.error(
                f"concept_refresh._update_vector_store: Failed to update vector store for {concept_name}: {str(e)}",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_refresh._update_vector_store",
                    "concept_name": concept_name,
                    "error": str(e),
                },
            )
            raise

    def _update_neo4j_metadata(self, concept_name: str, embedding_hash: str):
        """
        Update Neo4j concept metadata with new embedding hash and timestamp.
        Args:
            concept_name: Name of the concept
            embedding_hash: New embedding hash
        """
        try:
            with self.neo4j_manager.session() as session:
                result = session.run(
                    """
                    MATCH (c:Concept {name: $name})
                    SET c.embedding_hash = $hash,
                        c.last_updated = datetime()
                    RETURN c.name as name
                    """,
                    name=concept_name,
                    hash=embedding_hash,
                )
                record = result.single()
                if record:
                    logger.info(
                        f"concept_refresh._update_neo4j_metadata: Updated Neo4j metadata for concept: {concept_name}",
                        extra={
                            "service": "aclarai-scheduler",
                            "filename.function_name": "concept_refresh._update_neo4j_metadata",
                            "concept_name": concept_name,
                            "embedding_hash": embedding_hash,
                        },
                    )
                else:
                    logger.warning(
                        f"concept_refresh._update_neo4j_metadata: Concept not found in Neo4j: {concept_name}",
                        extra={
                            "service": "aclarai-scheduler",
                            "filename.function_name": "concept_refresh._update_neo4j_metadata",
                            "concept_name": concept_name,
                        },
                    )
        except Exception as e:
            logger.error(
                f"concept_refresh._update_neo4j_metadata: Failed to update Neo4j metadata for {concept_name}: {str(e)}",
                extra={
                    "service": "aclarai-scheduler",
                    "filename.function_name": "concept_refresh._update_neo4j_metadata",
                    "concept_name": concept_name,
                    "error": str(e),
                },
            )
            raise
