"""
Concept Summary Agent for generating detailed Markdown pages for canonical concepts.

This module implements the Concept Summary Agent that generates detailed concept pages
using a RAG (Retrieval-Augmented Generation) workflow as specified in the architectural
documentation (docs/arch/on-writing_vault_documents.md and docs/arch/on-RAG_workflow.md).
"""

from .agent import ConceptSummaryAgent

__all__ = ["ConceptSummaryAgent"]
