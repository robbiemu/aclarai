"""
Tier 2 Summary Agent implementation.
This agent aggregates and summarizes selected Tier 1 blocks into semantically
coherent groupings, as specified in docs/arch/on-writing_vault_documents.md.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI

from ..claim_concept_linking.neo4j_operations import ClaimConceptNeo4jManager
from ..config import aclaraiConfig
from ..embedding.storage import aclaraiVectorStore
from ..evaluation_thresholds import should_include_in_summaries
from ..graph.neo4j_manager import Neo4jGraphManager
from ..import_system import write_file_atomically
from .data_models import SummaryBlock, SummaryInput, SummaryResult, generate_summary_id

logger = logging.getLogger(__name__)


class Tier2SummaryAgent:
    """
    Agent for generating Tier 2 summaries from grouped Claims and Sentences.
    This agent:
    1. Retrieves grouped Claims/Sentences using vector similarity
    2. Generates coherent summaries using LLM
    3. Creates properly formatted Markdown with links back to Tier 1
    4. Writes files atomically to the vault
    """

    claim_concept_manager: Optional[ClaimConceptNeo4jManager]

    def __init__(
        self,
        config: Optional[aclaraiConfig] = None,
        neo4j_manager: Optional[Neo4jGraphManager] = None,
        embedding_storage: Optional[aclaraiVectorStore] = None,
        llm: Optional[LLM] = None,
    ):
        """
        Initialize the Tier 2 Summary Agent.
        Args:
            config: aclarai configuration
            neo4j_manager: Neo4j graph manager for retrieving claims/sentences
            embedding_storage: Vector storage for similarity search
            llm: Language model for summary generation
        """
        self.config = config or aclaraiConfig()
        # Check if tier2_generation is enabled in configuration
        tier2_enabled = self.config.features.get("tier2_generation", False)
        if not tier2_enabled:
            logger.warning(
                "Tier 2 generation is disabled in configuration",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent.__init__",
                    "tier2_generation": tier2_enabled,
                },
            )
        self.neo4j_manager = neo4j_manager
        self.embedding_storage = embedding_storage
        # Initialize claim-concept linking manager for retrieving linked concepts
        # Only initialize if config is available and has neo4j configuration
        try:
            self.claim_concept_manager = ClaimConceptNeo4jManager(config=self.config)
        except Exception as e:
            logger.warning(
                "Could not initialize claim-concept manager, concept linking disabled",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent.__init__",
                    "error": str(e),
                },
            )
            self.claim_concept_manager = None
        # Initialize LLM
        if llm is not None:
            self.llm = llm
        else:
            # Use configured model or default
            # Use getattr to avoid mypy error for dynamically loaded config
            llm_config = self.config.llm
            model_name = llm_config.model
            self.llm = OpenAI(model=model_name, **llm_config.model_params)
        logger.info(
            "Initialized Tier2SummaryAgent",
            extra={
                "service": "aclarai",
                "filename.function_name": "agent.__init__",
                "model": str(self.llm.model)
                if hasattr(self.llm, "model")
                else "unknown",
            },
        )

    def _is_enabled(self) -> bool:
        """
        Check if Tier 2 summary generation is enabled in configuration.
        Returns:
            True if enabled, False otherwise
        """
        return bool(self.config.features.get("tier2_generation", False))

    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """
        Execute function with retry logic and exponential backoff.
        Following guidelines from docs/arch/on-error-handling-and-resilience.md
        for handling transient errors from LLM APIs and Neo4j.
        """
        max_attempts = (
            getattr(self.config, "processing", {})
            .get("retries", {})
            .get("max_attempts", 3)
        )
        backoff_factor = (
            getattr(self.config, "processing", {})
            .get("retries", {})
            .get("backoff_factor", 2)
        )
        max_wait_time = (
            getattr(self.config, "processing", {})
            .get("retries", {})
            .get("max_wait_time", 60)
        )
        for attempt in range(max_attempts):
            try:
                logger.debug(
                    "Retrying operation",
                    extra={
                        "service": "aclarai",
                        "filename.function_name": f"agent.{func.__name__}",
                        "attempt": attempt + 1,
                        "max_attempts": max_attempts,
                    },
                )
                return func(*args, **kwargs)
            except Exception as e:
                # Check if it's a transient error that should be retried
                is_transient = self._is_transient_error(e)
                if attempt == max_attempts - 1 or not is_transient:
                    error_msg = (
                        "Operation failed after retries"
                        if is_transient
                        else "Non-transient error occurred"
                    )
                    logger.error(
                        error_msg,
                        extra={
                            "service": "aclarai",
                            "filename.function_name": f"agent.{func.__name__}",
                            "attempt": attempt + 1,
                            "error": str(e),
                            "is_transient": is_transient,
                        },
                    )
                    raise
                wait_time = min(backoff_factor**attempt, max_wait_time)
                logger.warning(
                    "Transient error occurred, retrying",
                    extra={
                        "service": "aclarai",
                        "filename.function_name": f"agent.{func.__name__}",
                        "attempt": attempt + 1,
                        "wait_time": wait_time,
                        "error": str(e),
                    },
                )
                time.sleep(wait_time)

    def _is_transient_error(self, error: Exception) -> bool:
        """
        Determine if an error is transient and should be retried.
        Args:
            error: The exception that occurred
        Returns:
            True if error is transient, False otherwise
        """
        # Common transient error patterns
        transient_patterns = [
            "timeout",
            "connection",
            "network",
            "rate limit",
            "throttle",
            "service unavailable",
            "temporary",
            "retry",
            "busy",
        ]
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in transient_patterns)

    def retrieve_grouped_content(
        self,
        similarity_threshold: Optional[float] = None,
        max_groups: int = 10,
        min_group_size: int = 2,
    ) -> List[SummaryInput]:
        """
        Retrieve grouped Claims and Sentences using vector similarity.
        This implements the non-agentic retrieval process described in
        on-writing_vault_documents.md using pg_vector_search and semantic
        neighborhoods based on utterance embeddings.
        Args:
            similarity_threshold: Minimum cosine similarity for grouping (uses config if None)
            max_groups: Maximum number of groups to return
            min_group_size: Minimum items per group
        Returns:
            List of SummaryInput objects representing semantic groups
        """
        if not self.neo4j_manager:
            logger.warning(
                "No Neo4j manager configured - cannot retrieve content",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent.retrieve_grouped_content",
                },
            )
            return []
        if not self.embedding_storage:
            logger.warning(
                "No embedding storage configured - cannot perform similarity search",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent.retrieve_grouped_content",
                },
            )
            return []
        # Use configured threshold if not provided
        if similarity_threshold is None:
            similarity_threshold = self.config.threshold.summary_grouping_similarity
        start_time = time.time()
        try:
            # Get high-quality Claims nodes as seeds for similarity search
            seed_claims = self._retry_with_backoff(self._get_high_quality_claims)
            if not seed_claims:
                logger.warning(
                    "No high-quality claims found to use as seeds",
                    extra={
                        "service": "aclarai",
                        "filename.function_name": "agent.retrieve_grouped_content",
                    },
                )
                return []
            logger.info(
                "Retrieved seed claims for semantic grouping",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent.retrieve_grouped_content",
                    "seed_claims_count": len(seed_claims),
                    "similarity_threshold": similarity_threshold,
                },
            )
            # Build semantic neighborhoods using vector similarity
            groups = self._build_semantic_neighborhoods(
                seed_claims, similarity_threshold, max_groups, min_group_size
            )
            processing_time = time.time() - start_time
            logger.info(
                "Created semantic neighborhoods for summarization",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent.retrieve_grouped_content",
                    "groups_count": len(groups),
                    "processing_time": processing_time,
                },
            )
            return groups
        except Exception as e:
            logger.error(
                "Failed to retrieve grouped content",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent.retrieve_grouped_content",
                    "error": str(e),
                },
            )
            return []

    def _get_high_quality_claims(self) -> List[Dict[str, Any]]:
        """
        Retrieve high-quality Claim nodes from Neo4j to use as seeds.
        High-quality claims are those that meet the evaluation threshold
        defined in docs/arch/on-evaluation_agents.md using geometric mean.
        """
        assert self.neo4j_manager is not None, "Neo4j manager is not initialized"
        try:
            # Get claims with all evaluation scores present
            # The filtering will be done in Python using the threshold logic
            query = """
            MATCH (c:Claim)-[:REFERENCES]->(b:Block)
            WHERE c.entailed_score IS NOT NULL
              AND c.coverage_score IS NOT NULL
              AND c.decontextualization_score IS NOT NULL
            RETURN c.id as id, c.text as text, c.entailed_score as entailed_score,
                   c.coverage_score as coverage_score, c.decontextualization_score as decontextualization_score,
                   c.version as version, c.timestamp as timestamp,
                   b.aclarai_id as source_block_id, b.text as source_block_text
            ORDER BY c.timestamp DESC
            """
            result = self.neo4j_manager.execute_query(query)
            all_claims = []
            for record in result:
                all_claims.append(
                    {
                        "id": record["id"],
                        "text": record["text"],
                        "entailed_score": record["entailed_score"],
                        "coverage_score": record["coverage_score"],
                        "decontextualization_score": record[
                            "decontextualization_score"
                        ],
                        "version": record["version"],
                        "timestamp": record["timestamp"],
                        "source_block_id": record["source_block_id"],
                        "source_block_text": record["source_block_text"],
                        "node_type": "claim",
                    }
                )

            # Apply evaluation threshold filtering using geometric mean
            quality_threshold = self.config.threshold.claim_quality
            high_quality_claims = []

            for claim in all_claims:
                # Check if claim meets quality threshold for summary inclusion
                if should_include_in_summaries(
                    claim["entailed_score"],
                    claim["coverage_score"],
                    claim["decontextualization_score"],
                    quality_threshold,
                ):
                    high_quality_claims.append(claim)
                else:
                    logger.debug(
                        "Excluding claim from summary: does not meet quality threshold",
                        extra={
                            "service": "aclarai",
                            "filename.function_name": "agent._get_high_quality_claims",
                            "claim_id": claim["id"],
                            "entailed_score": claim["entailed_score"],
                            "coverage_score": claim["coverage_score"],
                            "decontextualization_score": claim[
                                "decontextualization_score"
                            ],
                            "quality_threshold": quality_threshold,
                        },
                    )

            # Sort by computed quality (geometric mean) in descending order and limit
            from ..evaluation_thresholds import compute_geometric_mean

            def compute_claim_quality(claim) -> float:
                geomean = compute_geometric_mean(
                    claim["entailed_score"],
                    claim["coverage_score"],
                    claim["decontextualization_score"],
                )
                return geomean if geomean is not None else 0.0

            high_quality_claims.sort(key=compute_claim_quality, reverse=True)
            high_quality_claims = high_quality_claims[:50]  # Limit to top 50

            logger.info(
                "Retrieved high-quality claims as seeds",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent._get_high_quality_claims",
                    "total_claims": len(all_claims),
                    "high_quality_claims": len(high_quality_claims),
                    "quality_threshold": quality_threshold,
                },
            )
            return high_quality_claims
        except Exception as e:
            logger.error(
                "Failed to retrieve high-quality claims from Neo4j",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent._get_high_quality_claims",
                    "error": str(e),
                },
            )
            return []

    def _build_semantic_neighborhoods(
        self,
        seed_claims: List[Dict[str, Any]],
        similarity_threshold: float,
        max_groups: int,
        min_group_size: int,
    ) -> List[SummaryInput]:
        """
        Build semantic neighborhoods using vector similarity search.
        This implements the core strategy described in the comment:
        1. Use high-quality Claims as seeds
        2. For each seed, query utterances vector store for similar content
        3. Group the similar content into semantic neighborhoods
        4. Return groups suitable for summarization
        """
        assert self.embedding_storage is not None, (
            "Embedding storage is not initialized"
        )
        groups: List[SummaryInput] = []
        processed_block_ids = set()  # Track to avoid duplicates
        for seed_claim in seed_claims[:max_groups]:  # Limit seeds to max_groups
            if len(groups) >= max_groups:
                break
            seed_text = seed_claim.get("source_block_text") or seed_claim.get("text")
            seed_block_id = seed_claim.get("source_block_id")
            if not seed_text or seed_block_id in processed_block_ids:
                continue
            try:
                # Perform vector similarity search on utterances vector store
                similar_chunks = self.embedding_storage.similarity_search(
                    query_text=seed_text,
                    top_k=20,  # Get more candidates than needed
                    similarity_threshold=similarity_threshold,
                )
                if not similar_chunks:
                    continue
                # Extract block IDs from similar chunks
                similar_block_ids = []
                for chunk_metadata, _score in similar_chunks:
                    block_id = chunk_metadata.get("aclarai_id")
                    if block_id and block_id not in processed_block_ids:
                        similar_block_ids.append(block_id)
                        processed_block_ids.add(block_id)
                if len(similar_block_ids) < min_group_size:
                    continue
                # Get Claims and Sentences associated with these blocks
                group_claims, group_sentences = (
                    self._get_claims_and_sentences_for_blocks(similar_block_ids)
                )
                # Create the semantic neighborhood group
                if group_claims or group_sentences:
                    summary_input = SummaryInput(
                        claims=group_claims,
                        sentences=group_sentences,
                        group_context=f"Semantic neighborhood for: {seed_claim.get('text', '')[:50]}...",
                    )
                    groups.append(summary_input)
                    logger.debug(
                        "Created semantic neighborhood",
                        extra={
                            "service": "aclarai",
                            "filename.function_name": "agent._build_semantic_neighborhoods",
                            "seed_claim_id": seed_claim.get("id"),
                            "similar_blocks_count": len(similar_block_ids),
                            "group_claims_count": len(group_claims),
                            "group_sentences_count": len(group_sentences),
                        },
                    )
            except Exception as e:
                logger.warning(
                    "Failed to build semantic neighborhood for seed",
                    extra={
                        "service": "aclarai",
                        "filename.function_name": "agent._build_semantic_neighborhoods",
                        "seed_claim_id": seed_claim.get("id"),
                        "error": str(e),
                    },
                )
                continue
        return groups

    def _get_claims_and_sentences_for_blocks(
        self, block_ids: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Retrieve Claims and Sentences associated with specific block IDs.
        Args:
            block_ids: List of aclarai_id values for blocks
        Returns:
            Tuple of (claims, sentences) lists
        """
        assert self.neo4j_manager is not None, "Neo4j manager is not initialized"
        try:
            # Get Claims that reference these blocks
            claims_query = """
            MATCH (c:Claim)-[:REFERENCES]->(b:Block)
            WHERE b.aclarai_id IN $block_ids
            RETURN c.id as id, c.text as text, c.entailed_score as entailed_score,
                   c.coverage_score as coverage_score, c.decontextualization_score as decontextualization_score,
                   c.version as version, c.timestamp as timestamp,
                   b.aclarai_id as source_block_id
            """
            claims_result = self.neo4j_manager.execute_query(
                claims_query, parameters={"block_ids": block_ids}
            )
            claims = []
            for record in claims_result:
                claims.append(
                    {
                        "id": record["id"],
                        "text": record["text"],
                        "entailed_score": record["entailed_score"],
                        "coverage_score": record["coverage_score"],
                        "decontextualization_score": record[
                            "decontextualization_score"
                        ],
                        "version": record["version"],
                        "timestamp": record["timestamp"],
                        "source_block_id": record["source_block_id"],
                        "node_type": "claim",
                    }
                )
            # Get Sentences that reference these blocks
            sentences_query = """
            MATCH (s:Sentence)-[:REFERENCES]->(b:Block)
            WHERE b.aclarai_id IN $block_ids
            RETURN s.id as id, s.text as text, s.ambiguous as ambiguous,
                   s.verifiable as verifiable, s.version as version, s.timestamp as timestamp,
                   b.aclarai_id as source_block_id
            """
            sentences_result = self.neo4j_manager.execute_query(
                sentences_query, parameters={"block_ids": block_ids}
            )
            sentences = []
            for record in sentences_result:
                sentences.append(
                    {
                        "id": record["id"],
                        "text": record["text"],
                        "ambiguous": record["ambiguous"],
                        "verifiable": record["verifiable"],
                        "version": record["version"],
                        "timestamp": record["timestamp"],
                        "source_block_id": record["source_block_id"],
                        "node_type": "sentence",
                    }
                )
            return claims, sentences
        except Exception as e:
            logger.error(
                "Failed to retrieve claims and sentences for blocks",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent._get_claims_and_sentences_for_blocks",
                    "block_ids_count": len(block_ids),
                    "error": str(e),
                },
            )
            return [], []

    def generate_summary(self, summary_input: SummaryInput) -> SummaryResult:
        """
        Generate a summary from grouped Claims and Sentences.
        Args:
            summary_input: Grouped content to summarize
        Returns:
            SummaryResult with generated summary blocks
        """
        start_time = time.time()
        if not summary_input.all_texts:
            return SummaryResult(
                error="No content to summarize",
                processing_time=time.time() - start_time,
            )
        try:
            # Create summary prompt
            prompt = self._create_summary_prompt(summary_input)
            logger.debug(
                "Generating summary for content items",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent.generate_summary",
                    "content_count": len(summary_input.all_texts),
                },
            )

            # Generate summary using LLM with retry logic
            def _generate_llm_response() -> CompletionResponse:
                return self.llm.complete(prompt)

            response = self._retry_with_backoff(_generate_llm_response)
            summary_text = response.text.strip()
            # Get linked concepts for this summary
            linked_concepts = self._get_linked_concepts_for_summary(summary_input)
            # Create summary block
            summary_block = SummaryBlock(
                summary_text=summary_text,
                aclarai_id=generate_summary_id(),
                source_block_ids=summary_input.source_block_ids,
                linked_concepts=linked_concepts,
            )
            processing_time = time.time() - start_time
            result = SummaryResult(
                summary_blocks=[summary_block],
                source_file_context=summary_input.group_context,
                processing_time=processing_time,
                model_used=getattr(self.llm, "model", "unknown"),
            )
            logger.info(
                "Generated summary successfully",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent.generate_summary",
                    "aclarai_id": summary_block.aclarai_id,
                    "processing_time": processing_time,
                },
            )
            return result
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "Failed to generate summary",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent.generate_summary",
                    "error": str(e),
                    "processing_time": processing_time,
                },
            )
            return SummaryResult(error=str(e), processing_time=processing_time)

    def _create_summary_prompt(self, summary_input: SummaryInput) -> str:
        """
        Create a prompt for LLM summary generation.
        Args:
            summary_input: Content to summarize
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "You are a specialized summarization agent for a knowledge management system.",
            "Your task is to create concise, informative summaries of related claims and statements.",
            "",
            "Please summarize the following content into coherent bullet points:",
            "- Focus on the key claims and factual statements",
            "- Group related concepts together",
            "- Keep each point concise but informative",
            "- Maintain the original meaning and context",
            "",
            "Content to summarize:",
        ]
        for i, text in enumerate(summary_input.all_texts, 1):
            prompt_parts.append(f"{i}. {text}")
        prompt_parts.extend(["", "Summary (as bullet points):"])
        return "\n".join(prompt_parts)

    def _get_linked_concepts_for_summary(
        self, summary_input: SummaryInput
    ) -> List[str]:
        """
        Get concept names linked to the claims in this summary.
        Args:
            summary_input: The summary input containing claims
        Returns:
            List of unique concept text values for wikilinks
        """
        if not summary_input.claims or self.claim_concept_manager is None:
            return []
        try:
            # Extract claim IDs
            claim_ids = [
                cid
                for claim in summary_input.claims
                if isinstance(cid := claim.get("id"), str)
            ]
            if not claim_ids:
                return []
            # Get concepts linked to these claims
            concepts_mapping = self.claim_concept_manager.get_concepts_for_claims(
                claim_ids
            )
            # Extract unique concept texts, prioritizing higher strength relationships
            concept_texts = set()
            for _claim_id, concepts in concepts_mapping.items():
                # Sort by strength descending, take top concepts to avoid over-linking
                sorted_concepts = sorted(
                    concepts, key=lambda x: x.get("strength", 0.0), reverse=True
                )
                # Limit to top 3 concepts per claim to keep summaries clean
                for concept in sorted_concepts[:3]:
                    concept_text = concept.get("concept_text", "").strip()
                    if concept_text:
                        concept_texts.add(concept_text)
            result = list(concept_texts)
            logger.debug(
                "Retrieved linked concepts for summary",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent._get_linked_concepts_for_summary",
                    "claim_count": len(claim_ids),
                    "concept_count": len(result),
                },
            )
            return result
        except Exception as e:
            logger.warning(
                "Failed to retrieve linked concepts for summary",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent._get_linked_concepts_for_summary",
                    "error": str(e),
                },
            )
            return []

    def write_summary_file(
        self,
        summary_result: SummaryResult,
        output_path: Union[str, Path],
        title: Optional[str] = None,
    ) -> bool:
        """
        Write summary result to a Tier 2 Markdown file using atomic writes.
        Args:
            summary_result: Generated summary to write
            output_path: Path where to write the file
            title: Optional title for the file
        Returns:
            True if successful, False otherwise
        """
        if not summary_result.is_successful:
            logger.error(
                "Cannot write summary file - generation failed",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent.write_summary_file",
                    "output_path": str(output_path),
                    "error": summary_result.error,
                },
            )
            return False
        try:
            # Generate Markdown content
            markdown_content = summary_result.to_markdown(title=title)
            # Write atomically using existing function
            write_file_atomically(Path(output_path), markdown_content)
            logger.info(
                "Successfully wrote Tier 2 summary file",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent.write_summary_file",
                    "output_path": str(output_path),
                    "summary_blocks_count": len(summary_result.summary_blocks),
                },
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to write summary file",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent.write_summary_file",
                    "output_path": str(output_path),
                    "error": str(e),
                },
            )
            return False

    def process_and_generate_summaries(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        file_prefix: str = "tier2_summary",
    ) -> List[Path]:
        """
        Complete workflow: retrieve content, generate summaries, and write files.
        Args:
            output_dir: Directory to write summary files (uses config if not provided)
            file_prefix: Prefix for generated filenames
        Returns:
            List of paths to successfully written files
        """
        # Check if tier2_generation is enabled
        if not self._is_enabled():
            logger.warning(
                "Tier 2 summary generation is disabled in configuration - skipping",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent.process_and_generate_summaries",
                },
            )
            return []
        if output_dir is None:
            # Use configured Tier 2 path
            vault_path = Path(self.config.paths.vault)
            tier2_path = self.config.paths.tier2
            output_dir = vault_path / tier2_path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        written_files: List[Path] = []
        logger.info(
            "Starting Tier 2 summary generation process",
            extra={
                "service": "aclarai",
                "filename.function_name": "agent.process_and_generate_summaries",
                "output_dir": str(output_dir),
            },
        )
        try:
            # Retrieve grouped content
            groups = self.retrieve_grouped_content()
            if not groups:
                logger.warning(
                    "No content groups found for summarization",
                    extra={
                        "service": "aclarai",
                        "filename.function_name": "agent.process_and_generate_summaries",
                    },
                )
                return written_files
            # Generate summaries for each group
            for i, group in enumerate(groups):
                try:
                    # Generate summary
                    summary_result = self.generate_summary(group)
                    if summary_result.is_successful:
                        # Create filename
                        filename = f"{file_prefix}_{i + 1:03d}.md"
                        output_path = output_dir / filename
                        # Write file
                        title = f"Tier 2 Summary {i + 1}"
                        if self.write_summary_file(summary_result, output_path, title):
                            written_files.append(output_path)
                    else:
                        logger.warning(
                            "Failed to generate summary for group",
                            extra={
                                "service": "aclarai",
                                "filename.function_name": "agent.process_and_generate_summaries",
                                "group_index": i + 1,
                                "error": summary_result.error,
                            },
                        )
                except Exception as e:
                    logger.error(
                        "Error processing group",
                        extra={
                            "service": "aclarai",
                            "filename.function_name": "agent.process_and_generate_summaries",
                            "group_index": i + 1,
                            "error": str(e),
                        },
                    )
            processing_time = time.time() - start_time
            logger.info(
                "Completed Tier 2 summary generation",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent.process_and_generate_summaries",
                    "groups_processed": len(groups),
                    "files_written": len(written_files),
                    "processing_time": processing_time,
                },
            )
            return written_files
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "Failed to complete summary generation process",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "agent.process_and_generate_summaries",
                    "error": str(e),
                    "processing_time": processing_time,
                },
            )
            return written_files
