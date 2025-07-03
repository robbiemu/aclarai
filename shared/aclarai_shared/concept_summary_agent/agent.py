"""
Concept Summary Agent implementation.
This module implements the Concept Summary Agent that generates detailed Markdown
pages for each canonical concept in the graph, following the format specified in
docs/arch/on-writing_vault_documents.md and using the RAG workflow from
docs/arch/on-RAG_workflow.md.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.llms.openai import OpenAI

from aclarai_shared.config import aclaraiConfig
from aclarai_shared.graph.neo4j_manager import Neo4jGraphManager
from aclarai_shared.import_system import write_file_atomically
from aclarai_shared.tools.factory import ToolFactory
from aclarai_shared.tools.implementations.vector_search_tool import VectorSearchTool
from aclarai_shared.tools.vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)


class ConceptSummaryAgent:
    """
    Generates detailed Markdown pages for canonical concepts using RAG workflow.

    This agent follows the specifications from docs/arch/on-writing_vault_documents.md
    for the Concept Summary Agent format and uses the RAG workflow from
    docs/arch/on-RAG_workflow.md to retrieve relevant context.
    """

    def __init__(
        self,
        config: Optional[aclaraiConfig] = None,
        neo4j_manager: Optional[Neo4jGraphManager] = None,
        vector_store_manager: Optional[VectorStoreManager] = None,
    ):
        """
        Initialize the Concept Summary Agent.

        Args:
            config: aclarai configuration (loads default if None)
            neo4j_manager: Neo4j graph manager (creates new if None)
            vector_store_manager: Vector store manager for similarity search (creates new if None)
        """
        if config is None:
            from aclarai_shared.config import load_config

            config = load_config(validate=False)
        self.config = config

        self.neo4j_manager = neo4j_manager or Neo4jGraphManager(config)

        # Initialize vector store manager and tool factory
        if vector_store_manager is None:
            try:
                vector_store_manager = VectorStoreManager(config)
            except Exception as e:
                logger.warning(
                    f"Could not initialize vector store manager: {e}",
                    extra={
                        "service": "aclarai",
                        "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.__init__",
                        "error": str(e),
                    },
                )
                vector_store_manager = None

        self.vector_store_manager = vector_store_manager

        # Initialize tool factory and vector search tool
        if vector_store_manager:
            try:
                tool_factory = ToolFactory(config.model_dump(), vector_store_manager)
                tools = tool_factory.get_tools_for_agent("concept_summary")
                self.vector_search_tool = None
                for tool in tools:
                    if isinstance(tool, VectorSearchTool):
                        self.vector_search_tool = tool
                        break
            except Exception as e:
                logger.warning(
                    f"Could not initialize vector search tool: {e}",
                    extra={
                        "service": "aclarai",
                        "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.__init__",
                        "error": str(e),
                    },
                )
                self.vector_search_tool = None
        else:
            self.vector_search_tool = None

        # Initialize LLM
        llm_config = config.llm
        try:
            if llm_config.provider == "openai":
                self.llm = OpenAI(
                    model=llm_config.model,
                    api_key=llm_config.api_key,
                    temperature=0.3,  # Moderate temperature for creative but consistent content
                )
                self.model_name = llm_config.model
            else:
                raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")
        except Exception as e:
            logger.warning(
                f"Could not initialize LLM: {e}. Falling back to template-based generation.",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.__init__",
                    "error": str(e),
                },
            )
            self.llm = None
            self.model_name = "template-fallback"

        # Initialize concept summaries config with defaults
        concept_summaries_config = getattr(config, "concept_summaries", {})
        if isinstance(concept_summaries_config, dict):
            self.max_examples = concept_summaries_config.get("max_examples", 5)
            self.skip_if_no_claims = concept_summaries_config.get(
                "skip_if_no_claims", True
            )
            self.include_see_also = concept_summaries_config.get(
                "include_see_also", True
            )
        else:
            # Set defaults if concept_summaries not configured
            self.max_examples = 5
            self.skip_if_no_claims = True
            self.include_see_also = True

        # Get the concepts directory from configuration
        vault_path = Path(config.paths.vault)
        concepts_path = config.paths.tier3 or "concepts"
        self.concepts_dir = vault_path / concepts_path

        logger.debug(
            "Initialized ConceptSummaryAgent",
            extra={
                "service": "aclarai",
                "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.__init__",
                "concepts_dir": str(self.concepts_dir),
                "model": self.model_name,
                "max_examples": self.max_examples,
                "skip_if_no_claims": self.skip_if_no_claims,
                "include_see_also": self.include_see_also,
                "vector_search_available": self.vector_search_tool is not None,
                "llm_available": self.llm is not None,
            },
        )

    def get_canonical_concepts(self) -> List[Dict[str, Any]]:
        """
        Retrieve all canonical concepts from the Neo4j graph.

        Returns:
            List of concept dictionaries with id, text, and metadata
        """
        try:
            query = """
            MATCH (c:Concept)
            RETURN c.id as id, c.text as text,
                   c.aclarai_id as aclarai_id,
                   c.version as version, c.timestamp as timestamp
            ORDER BY c.timestamp DESC
            """

            result = self.neo4j_manager.execute_query(query)
            concepts = []

            for record in result:
                concepts.append(
                    {
                        "id": record["id"],
                        "text": record["text"],
                        "aclarai_id": record["aclarai_id"],
                        "version": record["version"],
                        "timestamp": record["timestamp"],
                    }
                )

            logger.debug(
                f"Retrieved {len(concepts)} canonical concepts",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.get_canonical_concepts",
                    "concepts_count": len(concepts),
                },
            )
            return concepts

        except Exception as e:
            logger.error(
                f"Failed to retrieve canonical concepts: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.get_canonical_concepts",
                    "error": str(e),
                },
            )
            return []

    def get_concept_claims(
        self, concept_id: str, limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve claims related to a concept via graph relationships.

        Args:
            concept_id: The concept ID to find claims for
            limit: Maximum number of claims to return

        Returns:
            List of claim dictionaries with text and aclarai_id
        """
        try:
            # Build query with optional limit
            query = """
            MATCH (c:Concept {id: $concept_id})
            MATCH (claim:Claim)-[r:SUPPORTS_CONCEPT|MENTIONS_CONCEPT|CONTRADICTS_CONCEPT]->(c)
            RETURN claim.text as text, claim.id as claim_id, claim.aclarai_id as aclarai_id,
                   type(r) as relationship_type, r.strength as strength
            ORDER BY r.strength DESC, claim.timestamp DESC
            """

            if limit:
                query += f" LIMIT {limit}"

            result = self.neo4j_manager.execute_query(query, concept_id=concept_id)
            claims = []

            for record in result:
                claims.append(
                    {
                        "text": record["text"],
                        "claim_id": record["claim_id"],
                        "aclarai_id": record["aclarai_id"],
                        "relationship_type": record["relationship_type"],
                        "strength": record["strength"],
                    }
                )

            logger.debug(
                f"Retrieved {len(claims)} claims for concept {concept_id}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.get_concept_claims",
                    "concept_id": concept_id,
                    "claims_count": len(claims),
                },
            )
            return claims

        except Exception as e:
            logger.error(
                f"Failed to retrieve claims for concept {concept_id}: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.get_concept_claims",
                    "concept_id": concept_id,
                    "error": str(e),
                },
            )
            return []

    def get_concept_summaries(
        self, concept_id: str, limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve summaries related to a concept via graph relationships.

        Args:
            concept_id: The concept ID to find summaries for
            limit: Maximum number of summaries to return

        Returns:
            List of summary dictionaries with text and aclarai_id
        """
        try:
            # Build query with optional limit
            query = """
            MATCH (c:Concept {id: $concept_id})
            MATCH (summary:Summary)-[r:MENTIONS_CONCEPT|RELATES_TO]->(c)
            RETURN summary.text as text, summary.id as summary_id,
                   summary.aclarai_id as aclarai_id,
                   type(r) as relationship_type
            ORDER BY summary.timestamp DESC
            """

            if limit:
                query += f" LIMIT {limit}"

            result = self.neo4j_manager.execute_query(query, concept_id=concept_id)
            summaries = []

            for record in result:
                summaries.append(
                    {
                        "text": record["text"],
                        "summary_id": record["summary_id"],
                        "aclarai_id": record["aclarai_id"],
                        "relationship_type": record["relationship_type"],
                    }
                )

            logger.debug(
                f"Retrieved {len(summaries)} summaries for concept {concept_id}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.get_concept_summaries",
                    "concept_id": concept_id,
                    "summaries_count": len(summaries),
                },
            )
            return summaries

        except Exception as e:
            logger.error(
                f"Failed to retrieve summaries for concept {concept_id}: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.get_concept_summaries",
                    "concept_id": concept_id,
                    "error": str(e),
                },
            )
            return []

    def get_related_concepts(self, concept_text: str, limit: int = 5) -> List[str]:
        """
        Get related concepts using vector similarity search.

        Uses the VectorSearchTool to find semantically similar concepts in the
        concepts vector store.

        Args:
            concept_text: The concept text to find related concepts for
            limit: Maximum number of related concepts to return

        Returns:
            List of related concept names
        """
        if not self.vector_search_tool:
            logger.debug(
                f"Vector search tool not available for related concepts search: '{concept_text}'",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.get_related_concepts",
                    "concept_text": concept_text,
                },
            )
            return []

        try:
            # Search for related concepts in the concepts collection
            results = self.vector_search_tool.call(
                query=concept_text,
                collections=["concepts"],
                max_results_per_collection=limit,
                similarity_threshold=0.7,
            )

            related_concepts = []
            if "concepts" in results:
                for result in results["concepts"]:
                    # Extract concept name from result text or metadata
                    concept_name = result.get("text", "").strip()
                    if concept_name and concept_name.lower() != concept_text.lower():
                        related_concepts.append(concept_name)

            logger.debug(
                f"Found {len(related_concepts)} related concepts for '{concept_text}'",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.get_related_concepts",
                    "concept_text": concept_text,
                    "related_concepts_count": len(related_concepts),
                },
            )

            return related_concepts[:limit]

        except Exception as e:
            logger.warning(
                f"Vector search failed for related concepts: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.get_related_concepts",
                    "concept_text": concept_text,
                    "error": str(e),
                },
            )
            return []

    def get_related_utterances(
        self, concept_text: str, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get related utterances using vector similarity search.

        Uses the VectorSearchTool to find relevant utterances that mention or
        relate to the concept.

        Args:
            concept_text: The concept text to find related utterances for
            limit: Maximum number of related utterances to return

        Returns:
            List of utterance dictionaries with text and metadata
        """
        if not self.vector_search_tool:
            logger.debug(
                f"Vector search tool not available for related utterances search: '{concept_text}'",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.get_related_utterances",
                    "concept_text": concept_text,
                },
            )
            return []

        try:
            # Search for related utterances in the utterances collection
            results = self.vector_search_tool.call(
                query=concept_text,
                collections=["utterances"],
                max_results_per_collection=limit,
                similarity_threshold=0.6,  # Lower threshold for utterances as they may be less direct
            )

            related_utterances = []
            if "utterances" in results:
                for result in results["utterances"]:
                    utterance_data = {
                        "text": result.get("text", "").strip(),
                        "metadata": result.get("metadata", {}),
                        "score": result.get("score", 0.0),
                    }
                    if utterance_data["text"]:
                        related_utterances.append(utterance_data)

            logger.debug(
                f"Found {len(related_utterances)} related utterances for '{concept_text}'",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.get_related_utterances",
                    "concept_text": concept_text,
                    "related_utterances_count": len(related_utterances),
                },
            )

            return related_utterances[:limit]

        except Exception as e:
            logger.warning(
                f"Vector search failed for related utterances: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.get_related_utterances",
                    "concept_text": concept_text,
                    "error": str(e),
                },
            )
            return []

    def generate_concept_slug(self, concept_text: str) -> str:
        """
        Generate a URL-safe slug for a concept.

        Args:
            concept_text: The concept text

        Returns:
            A safe slug for use in aclarai:id
        """
        # Replace problematic characters and normalize
        slug = re.sub(r"[^\w\s-]", "", concept_text.lower())
        slug = re.sub(r"[-\s]+", "_", slug)
        slug = slug.strip("_")

        # Ensure we have a non-empty slug
        if not slug:
            slug = "unnamed_concept"

        return slug

    def generate_concept_filename(self, concept_text: str) -> str:
        """
        Generate a canonical filename for a concept.

        Args:
            concept_text: The concept text

        Returns:
            A safe filename for the concept
        """
        # Replace problematic characters to make it filesystem-safe
        safe_name = "".join(
            c if c.isalnum() or c in "._- " else "_" for c in concept_text
        )
        # Replace spaces with underscores and limit length
        safe_name = safe_name.replace(" ", "_").strip("_")
        # Remove multiple consecutive underscores
        safe_name = re.sub(r"_+", "_", safe_name).strip("_")

        # Limit the length to avoid filesystem issues
        if len(safe_name) > 200:
            safe_name = safe_name[:200].rstrip("_")

        # Ensure we have a non-empty name
        if not safe_name:
            safe_name = "unnamed_concept"

        return f"{safe_name}.md"

    def generate_concept_content(
        self, concept: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """
        Generate the Markdown content for a concept file using LLM.

        This follows the format specified in docs/arch/on-writing_vault_documents.md
        for the Concept Summary Agent.

        Args:
            concept: The concept dictionary with id, text, etc.
            context: Retrieved context including claims, summaries, related concepts

        Returns:
            The Markdown content as a string
        """
        concept_text = concept["text"]
        self.generate_concept_slug(concept_text)
        concept.get("version", 1)

        if self.llm:
            # Use LLM to generate intelligent concept content
            try:
                content = self._generate_llm_content(concept, context)
                if content:
                    return content
            except Exception as e:
                logger.warning(
                    f"LLM generation failed for concept '{concept_text}', falling back to template: {e}",
                    extra={
                        "service": "aclarai",
                        "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.generate_concept_content",
                        "concept_text": concept_text,
                        "error": str(e),
                    },
                )

        # Fallback to template-based generation
        return self._generate_template_content(concept, context)

    def _generate_llm_content(
        self, concept: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate concept content using LLM with RAG context.

        Args:
            concept: The concept dictionary
            context: Retrieved context from RAG workflow

        Returns:
            Generated content or None if generation fails
        """
        concept_text = concept["text"]
        concept_slug = self.generate_concept_slug(concept_text)
        version = concept.get("version", 1)

        # Build the prompt with RAG context
        prompt_parts = [
            f"Generate a comprehensive concept page for the concept: {concept_text}",
            "",
            "The page should follow this format:",
            f"## Concept: {concept_text}",
            "",
            "[A clear, informative definition of what this concept represents]",
            "",
            "### Examples",
            "[Relevant examples with proper anchor references]",
            "",
            "### See Also",
            "[Related concepts as Obsidian-style wiki links]",
            "",
            "Context information available:",
        ]

        # Add claims context
        claims = context.get("claims", [])
        if claims:
            prompt_parts.append(
                f"\nClaims that support or mention this concept ({len(claims)} total):"
            )
            for _i, claim in enumerate(claims[:5]):  # Show up to 5 claims as examples
                prompt_parts.append(f"- {claim['text']}")

        # Add summaries context
        summaries = context.get("summaries", [])
        if summaries:
            prompt_parts.append(
                f"\nSummaries that mention this concept ({len(summaries)} total):"
            )
            for _i, summary in enumerate(summaries[:3]):  # Show up to 3 summaries
                prompt_parts.append(f"- {summary['text']}")

        # Add related concepts
        related_concepts = context.get("related_concepts", [])
        if related_concepts:
            prompt_parts.append(
                f"\nRelated concepts found: {', '.join(related_concepts)}"
            )

        # Add related utterances
        related_utterances = context.get("related_utterances", [])
        if related_utterances:
            prompt_parts.append(
                f"\nRelated utterances ({len(related_utterances)} total):"
            )
            for utterance in related_utterances:
                prompt_parts.append(f"- {utterance['text']}")

        prompt_parts.extend(
            [
                "",
                "Requirements:",
                "1. Write a clear, informative definition that synthesizes the available context",
                "2. Include up to 5 relevant examples from the claims and summaries, each with a proper ^anchor_id reference",
                "3. Include related concepts as [[wiki-style links]] in the See Also section",
                "4. Use professional, encyclopedia-style language",
                "5. Be comprehensive but concise",
                f"6. End with the metadata: <!-- aclarai:id=concept_{concept_slug} ver={version} --> and ^concept_{concept_slug}",
                "",
                "Generate the complete concept page now:",
            ]
        )

        prompt = "\n".join(prompt_parts)

        try:
            response = self.llm.complete(prompt)
            content = response.text.strip()

            # Ensure proper metadata is included
            aclarai_id = concept.get("aclarai_id") or f"concept_{concept_slug}"
            if f"<!-- aclarai:id={aclarai_id}" not in content:
                content += (
                    f"\n\n<!-- aclarai:id={aclarai_id} ver={version} -->\n^{aclarai_id}"
                )

            logger.debug(
                f"Generated LLM content for concept '{concept_text}'",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent._generate_llm_content",
                    "concept_text": concept_text,
                    "content_length": len(content),
                },
            )

            return content

        except Exception as e:
            logger.error(
                f"LLM content generation failed for concept '{concept_text}': {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent._generate_llm_content",
                    "concept_text": concept_text,
                    "error": str(e),
                },
            )
            return None

    def _generate_template_content(
        self, concept: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """
        Generate concept content using a template (fallback method).

        Args:
            concept: The concept dictionary
            context: Retrieved context from RAG workflow

        Returns:
            Template-generated content
        """
        concept_text = concept["text"]
        concept_slug = self.generate_concept_slug(concept_text)
        version = concept.get("version", 1)

        lines = [
            f"## Concept: {concept_text}",
            "",
            f"This concept represents: {concept_text}",
            "",
        ]

        # Add examples section
        lines.append("### Examples")

        claims = context.get("claims", [])
        summaries = context.get("summaries", [])

        if claims or summaries:
            # Combine claims and summaries for examples, limit to max_examples
            all_examples = []

            # Add claims first (they're typically more direct)
            for claim in claims[: self.max_examples]:
                aclarai_id = claim.get("aclarai_id", claim.get("claim_id", "unknown"))
                lines.append(f"- {claim['text']} ^{aclarai_id}")
                all_examples.append(claim)

            # Add summaries if we have room
            remaining_slots = self.max_examples - len(all_examples)
            for summary in summaries[:remaining_slots]:
                aclarai_id = summary.get(
                    "aclarai_id", summary.get("summary_id", "unknown")
                )
                lines.append(f"- {summary['text']} ^{aclarai_id}")
                all_examples.append(summary)

            if not all_examples:
                lines.append("<!-- No examples available yet -->")
        else:
            lines.append("<!-- No examples available yet -->")

        lines.append("")

        # Add See Also section
        if self.include_see_also:
            lines.append("### See Also")
            related_concepts = context.get("related_concepts", [])
            if related_concepts:
                for related in related_concepts:
                    lines.append(f"- [[{related}]]")
            else:
                lines.append(
                    "<!-- Related concepts will be added through concept linking -->"
                )
            lines.append("")

        # Add aclarai metadata
        aclarai_id = concept.get("aclarai_id") or f"concept_{concept_slug}"
        lines.append(f"<!-- aclarai:id={aclarai_id} ver={version} -->")
        lines.append(f"^{aclarai_id}")

        return "\n".join(lines)

    def should_skip_concept(
        self, concept: Dict[str, Any], context: Dict[str, Any]
    ) -> bool:
        """
        Determine if a concept should be skipped based on configuration.

        Args:
            concept: The concept dictionary
            context: Retrieved context including claims

        Returns:
            True if the concept should be skipped, False otherwise
        """
        if not self.skip_if_no_claims:
            return False

        claims = context.get("claims", [])
        if not claims:
            logger.debug(
                f"Skipping concept '{concept['text']}' - no claims found",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.should_skip_concept",
                    "concept_id": concept.get("id"),
                    "concept_text": concept["text"],
                },
            )
            return True

        return False

    def generate_concept_page(self, concept: Dict[str, Any]) -> bool:
        """
        Generate a complete concept page for a single concept.

        Args:
            concept: The concept dictionary with id, text, etc.

        Returns:
            True if the page was generated successfully, False otherwise
        """
        try:
            concept_id = concept["id"]
            concept_text = concept["text"]

            logger.info(
                f"Generating concept page for '{concept_text}'",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.generate_concept_page",
                    "concept_id": concept_id,
                    "concept_text": concept_text,
                },
            )

            # Retrieve context using RAG workflow
            context = {
                "claims": self.get_concept_claims(
                    concept_id, limit=self.max_examples * 2
                ),
                "summaries": self.get_concept_summaries(
                    concept_id, limit=self.max_examples
                ),
                "related_concepts": self.get_related_concepts(concept_text, limit=5),
                "related_utterances": self.get_related_utterances(
                    concept_text, limit=3
                ),
            }

            # Check if we should skip this concept
            if self.should_skip_concept(concept, context):
                return False

            # Generate the content
            content = self.generate_concept_content(concept, context)

            # Generate filename and write file
            filename = self.generate_concept_filename(concept_text)
            file_path = self.concepts_dir / filename

            # Ensure concepts directory exists
            self.concepts_dir.mkdir(parents=True, exist_ok=True)

            # Write file atomically
            write_file_atomically(file_path, content)

            logger.info(
                f"Successfully generated concept page: {filename}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.generate_concept_page",
                    "concept_id": concept_id,
                    "concept_text": concept_text,
                    "file_path": str(file_path),
                    "claims_found": len(context["claims"]),
                    "summaries_found": len(context["summaries"]),
                    "related_concepts_found": len(context["related_concepts"]),
                },
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to generate concept page for '{concept.get('text', 'unknown')}': {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.generate_concept_page",
                    "concept_id": concept.get("id"),
                    "concept_text": concept.get("text"),
                    "error": str(e),
                },
            )
            return False

    def run_agent(self) -> Dict[str, Any]:
        """
        Run the Concept Summary Agent to generate pages for all canonical concepts.

        Returns:
            Dictionary with execution results and statistics
        """
        logger.info(
            "Starting Concept Summary Agent execution",
            extra={
                "service": "aclarai",
                "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.run_agent",
            },
        )

        try:
            # Get all canonical concepts
            concepts = self.get_canonical_concepts()

            if not concepts:
                logger.warning(
                    "No canonical concepts found in the graph",
                    extra={
                        "service": "aclarai",
                        "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.run_agent",
                    },
                )
                return {
                    "success": True,
                    "concepts_processed": 0,
                    "concepts_generated": 0,
                    "concepts_skipped": 0,
                    "errors": 0,
                    "error_details": [],
                }

            # Process each concept
            concepts_generated = 0
            concepts_skipped = 0
            errors = 0
            error_details = []

            for concept in concepts:
                try:
                    if self.generate_concept_page(concept):
                        concepts_generated += 1
                    else:
                        concepts_skipped += 1
                except Exception as e:
                    errors += 1
                    error_msg = f"Error processing concept '{concept.get('text', 'unknown')}': {e}"
                    error_details.append(error_msg)
                    logger.error(
                        error_msg,
                        extra={
                            "service": "aclarai",
                            "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.run_agent",
                            "concept_id": concept.get("id"),
                            "error": str(e),
                        },
                    )

            result = {
                "success": True,
                "concepts_processed": len(concepts),
                "concepts_generated": concepts_generated,
                "concepts_skipped": concepts_skipped,
                "errors": errors,
                "error_details": error_details,
            }

            logger.info(
                f"Concept Summary Agent completed. Generated {concepts_generated} pages, "
                f"skipped {concepts_skipped}, errors {errors}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.run_agent",
                    **result,
                },
            )

            return result

        except Exception as e:
            logger.error(
                f"Concept Summary Agent failed: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.run_agent",
                    "error": str(e),
                },
            )
            return {
                "success": False,
                "concepts_processed": 0,
                "concepts_generated": 0,
                "concepts_skipped": 0,
                "errors": 1,
                "error_details": [str(e)],
            }
