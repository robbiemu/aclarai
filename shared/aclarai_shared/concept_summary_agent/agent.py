"""
Concept Summary Agent implementation.
This module implements the Concept Summary Agent that generates detailed Markdown
pages for each canonical concept in the graph, following the format specified in
docs/arch/on-writing_vault_documents.md and using the RAG workflow from
docs/arch/on-RAG_workflow.md.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from aclarai_shared.config import aclaraiConfig
from aclarai_shared.graph.models import Concept
from aclarai_shared.graph.neo4j_manager import Neo4jGraphManager

logger = logging.getLogger(__name__)


def write_file_atomically_simple(file_path: Path, content: str):
    """
    Simple atomic file writing without full import_system dependency.
    This implements the atomic write pattern (.tmp -> rename) for safety.
    """
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file first
    temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
    
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        
        # Atomic rename to final location
        if os.name == 'nt':  # Windows
            if file_path.exists():
                file_path.unlink()
        temp_path.rename(file_path)
        
    except Exception:
        # Clean up temp file on failure
        if temp_path.exists():
            temp_path.unlink()
        raise


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
    ):
        """
        Initialize the Concept Summary Agent.
        
        Args:
            config: aclarai configuration (loads default if None)
            neo4j_manager: Neo4j graph manager (creates new if None)
        """
        if config is None:
            from aclarai_shared.config import load_config
            config = load_config(validate=False)
        self.config = config
        
        self.neo4j_manager = neo4j_manager or Neo4jGraphManager(config)
        
        # Get configuration parameters - handle both model object and dict
        model_config = getattr(config, 'model', {})
        if hasattr(model_config, 'concept_summary'):
            self.model = model_config.concept_summary
        elif isinstance(model_config, dict):
            self.model = model_config.get('concept_summary', 'gpt-4')
        else:
            self.model = 'gpt-4'
        
        # Initialize concept summaries config with defaults
        concept_summaries_config = getattr(config, 'concept_summaries', {})
        if isinstance(concept_summaries_config, dict):
            self.max_examples = concept_summaries_config.get('max_examples', 5)
            self.skip_if_no_claims = concept_summaries_config.get('skip_if_no_claims', True)
            self.include_see_also = concept_summaries_config.get('include_see_also', True)
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
                "model": self.model,
                "max_examples": self.max_examples,
                "skip_if_no_claims": self.skip_if_no_claims,
                "include_see_also": self.include_see_also,
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
                concepts.append({
                    "id": record["id"],
                    "text": record["text"],
                    "aclarai_id": record["aclarai_id"],
                    "version": record["version"],
                    "timestamp": record["timestamp"],
                })
            
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

    def get_concept_claims(self, concept_id: str, limit: int = None) -> List[Dict[str, Any]]:
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
                claims.append({
                    "text": record["text"],
                    "claim_id": record["claim_id"],
                    "aclarai_id": record["aclarai_id"],
                    "relationship_type": record["relationship_type"],
                    "strength": record["strength"],
                })
            
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

    def get_concept_summaries(self, concept_id: str, limit: int = None) -> List[Dict[str, Any]]:
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
                summaries.append({
                    "text": record["text"],
                    "summary_id": record["summary_id"],
                    "aclarai_id": record["aclarai_id"],
                    "relationship_type": record["relationship_type"],
                })
            
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
        Get related concepts using vector similarity (placeholder implementation).
        
        In a full implementation, this would use the concepts vector store
        to find semantically similar concepts.
        
        Args:
            concept_text: The concept text to find related concepts for
            limit: Maximum number of related concepts to return
            
        Returns:
            List of related concept names
        """
        # TODO: Implement vector search when vector store integration is available
        # For now, return empty list to avoid blocking concept generation
        logger.debug(
            f"Vector search for related concepts not yet implemented for '{concept_text}'",
            extra={
                "service": "aclarai",
                "filename.function_name": "concept_summary_agent.ConceptSummaryAgent.get_related_concepts",
                "concept_text": concept_text,
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
        slug = re.sub(r'[^\w\s-]', '', concept_text.lower())
        slug = re.sub(r'[-\s]+', '_', slug)
        slug = slug.strip('_')
        
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

    def generate_concept_content(self, concept: Dict[str, Any], context: Dict[str, Any]) -> str:
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
        # For now, generate a basic concept page without LLM to get the structure working
        # TODO: Implement LLM-based generation in a follow-up
        
        concept_text = concept["text"]
        concept_slug = self.generate_concept_slug(concept_text)
        version = concept.get("version", 1)
        
        lines = [
            f"## Concept: {concept_text}",
            "",
            # For now, use a simple placeholder definition
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
            for claim in claims[:self.max_examples]:
                aclarai_id = claim.get("aclarai_id", claim.get("claim_id", "unknown"))
                lines.append(f"- {claim['text']} ^{aclarai_id}")
                all_examples.append(claim)
                
            # Add summaries if we have room
            remaining_slots = self.max_examples - len(all_examples)
            for summary in summaries[:remaining_slots]:
                aclarai_id = summary.get("aclarai_id", summary.get("summary_id", "unknown"))
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
                lines.append("<!-- Related concepts will be added through concept linking -->")
            lines.append("")
        
        # Add aclarai metadata
        aclarai_id = concept.get("aclarai_id") or f"concept_{concept_slug}"
        lines.append(f"<!-- aclarai:id={aclarai_id} ver={version} -->")
        lines.append(f"^{aclarai_id}")
        
        return "\n".join(lines)

    def should_skip_concept(self, concept: Dict[str, Any], context: Dict[str, Any]) -> bool:
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
                "claims": self.get_concept_claims(concept_id, limit=self.max_examples * 2),
                "summaries": self.get_concept_summaries(concept_id, limit=self.max_examples),
                "related_concepts": self.get_related_concepts(concept_text, limit=5),
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
            write_file_atomically_simple(file_path, content)
            
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