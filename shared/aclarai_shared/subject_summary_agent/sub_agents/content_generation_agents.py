"""
Content generation sub-agents for the Subject Summary Agent.
These agents handle the specialized tasks of generating different sections
of the subject summary page using LLM calls with specific prompts.
"""

import logging
from typing import Any, Dict, List, Optional

from aclarai_shared.config import aclaraiConfig
from aclarai_shared.utils.prompt_loader import load_prompt_template
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)


class DefinitionWriterAgent:
    """
    Agent responsible for writing the summary paragraph that defines the subject.
    """
    
    def __init__(self, llm: LLM, config: aclaraiConfig):
        self.llm = llm
        self.config = config
    
    def generate_definition(
        self, 
        subject_name: str, 
        concepts: List[str], 
        context: Dict[str, Any]
    ) -> str:
        """
        Generate a summary paragraph defining the subject.
        
        Args:
            subject_name: The name of the subject
            concepts: List of concept names in the subject
            context: Additional context including claims, summaries, web search
            
        Returns:
            Generated summary paragraph
        """
        try:
            # Format concepts as a bulleted list
            concept_list = "\n".join(f"- {concept}" for concept in concepts)
            
            # Format context information
            context_parts = []
            if context.get("shared_claims"):
                context_parts.append("Shared claims:")
                for claim in context["shared_claims"][:3]:  # Limit to first 3
                    context_parts.append(f"- {claim.get('text', '')}")
            
            if context.get("common_summaries"):
                context_parts.append("Common summaries:")
                for summary in context["common_summaries"][:2]:  # Limit to first 2
                    context_parts.append(f"- {summary.get('text', '')}")
            
            if context.get("web_context"):
                context_parts.append("Web context:")
                context_parts.append(str(context["web_context"])[:500])  # Limit length
            
            context_info = "\n".join(context_parts) if context_parts else "No additional context available."
            
            # Load and format the prompt
            prompt = load_prompt_template(
                "subject_summary_definition",
                subject_name=subject_name,
                concept_list=concept_list,
                context_info=context_info
            )
            
            # Generate the definition
            response = self.llm.complete(prompt)
            
            # Clean up the response
            definition = response.text.strip()
            
            logger.debug(
                f"Generated definition for subject: {subject_name}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "subject_summary_agent.DefinitionWriterAgent.generate_definition",
                    "subject_name": subject_name,
                    "concepts_count": len(concepts),
                }
            )
            
            return definition
            
        except Exception as e:
            logger.error(
                f"Failed to generate definition for subject {subject_name}: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "subject_summary_agent.DefinitionWriterAgent.generate_definition",
                    "subject_name": subject_name,
                    "error": str(e),
                }
            )
            # Return fallback definition
            return f"This subject encompasses {len(concepts)} related concepts including {', '.join(concepts[:3])}{'...' if len(concepts) > 3 else ''}."


class ConceptBlurbAgent:
    """
    Agent responsible for writing short blurbs explaining each concept's role within the subject.
    """
    
    def __init__(self, llm: LLM, config: aclaraiConfig):
        self.llm = llm
        self.config = config
    
    def generate_blurb(
        self, 
        concept_name: str, 
        subject_name: str, 
        context: Dict[str, Any]
    ) -> str:
        """
        Generate a one-sentence blurb explaining the concept's role within the subject.
        
        Args:
            concept_name: The name of the concept
            subject_name: The name of the subject
            context: Additional context including claims and summaries
            
        Returns:
            Generated concept blurb
        """
        try:
            # Format context information relevant to this concept
            context_parts = []
            if context.get("shared_claims"):
                for claim in context["shared_claims"]:
                    if concept_name in claim.get("related_concepts", []):
                        context_parts.append(f"- {claim.get('text', '')}")
            
            if context.get("common_summaries"):
                for summary in context["common_summaries"]:
                    if concept_name in summary.get("related_concepts", []):
                        context_parts.append(f"- {summary.get('text', '')}")
            
            context_info = "\n".join(context_parts) if context_parts else "No specific context available."
            
            # Load and format the prompt
            prompt = load_prompt_template(
                "subject_summary_concept_blurb",
                subject_name=subject_name,
                concept_name=concept_name,
                context_info=context_info
            )
            
            # Generate the blurb
            response = self.llm.complete(prompt)
            
            # Clean up the response
            blurb = response.text.strip()
            
            logger.debug(
                f"Generated blurb for concept: {concept_name}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "subject_summary_agent.ConceptBlurbAgent.generate_blurb",
                    "concept_name": concept_name,
                    "subject_name": subject_name,
                }
            )
            
            return blurb
            
        except Exception as e:
            logger.error(
                f"Failed to generate blurb for concept {concept_name}: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "subject_summary_agent.ConceptBlurbAgent.generate_blurb",
                    "concept_name": concept_name,
                    "error": str(e),
                }
            )
            # Return fallback blurb
            return f"A key component of {subject_name}"


class CommonThreadsAgent:
    """
    Agent responsible for identifying and synthesizing common themes across the concept cluster.
    """
    
    def __init__(self, llm: LLM, config: aclaraiConfig):
        self.llm = llm
        self.config = config
    
    def generate_common_threads(
        self, 
        shared_claims: List[Dict[str, Any]], 
        common_summaries: List[Dict[str, Any]], 
        web_context: Optional[str] = None
    ) -> str:
        """
        Generate the common threads section by analyzing shared claims and summaries.
        
        Args:
            shared_claims: List of claims shared across multiple concepts
            common_summaries: List of summaries mentioning multiple concepts
            web_context: Optional web search context
            
        Returns:
            Generated common threads section as markdown
        """
        try:
            # Format claims information
            claims_info = []
            for claim in shared_claims:
                claims_info.append(f"- {claim.get('text', '')} (relates to: {', '.join(claim.get('related_concepts', []))})")
            claims_text = "\n".join(claims_info) if claims_info else "No shared claims available."
            
            # Format summaries information
            summaries_info = []
            for summary in common_summaries:
                summaries_info.append(f"- {summary.get('text', '')} (relates to: {', '.join(summary.get('related_concepts', []))})")
            summaries_text = "\n".join(summaries_info) if summaries_info else "No common summaries available."
            
            # Load and format the prompt
            prompt = load_prompt_template(
                "subject_summary_common_threads",
                claims_info=claims_text,
                summaries_info=summaries_text,
                web_context=web_context or "No web context available."
            )
            
            # Generate the common threads
            response = self.llm.complete(prompt)
            
            # Clean up the response
            threads = response.text.strip()
            
            logger.debug(
                "Generated common threads section",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "subject_summary_agent.CommonThreadsAgent.generate_common_threads",
                    "claims_count": len(shared_claims),
                    "summaries_count": len(common_summaries),
                }
            )
            
            return threads
            
        except Exception as e:
            logger.error(
                f"Failed to generate common threads: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "subject_summary_agent.CommonThreadsAgent.generate_common_threads",
                    "error": str(e),
                }
            )
            # Return fallback threads
            return "- This subject brings together multiple related concepts\n- The concepts share common themes and applications"