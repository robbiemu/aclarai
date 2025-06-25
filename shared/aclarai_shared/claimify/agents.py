"""
Base agent classes and implementations for the Claimify pipeline.
This module contains the three core agents:
- SelectionAgent: Identifies relevant sentence chunks
- DisambiguationAgent: Rewrites sentences to remove ambiguities
- DecompositionAgent: Breaks sentences into atomic claims
Each agent is designed to accept different LLM instances for future configurability.
"""

import json
import logging
import time
from typing import Optional, Protocol

from .data_models import (
    ClaimCandidate,
    ClaimifyConfig,
    ClaimifyContext,
    DecompositionResult,
    DisambiguationResult,
    SelectionResult,
    SentenceChunk,
)

logger = logging.getLogger(__name__)


class LLMInterface(Protocol):
    """Protocol defining the interface for LLM interactions."""

    def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion for the given prompt."""
        ...


class BaseClaimifyAgent:
    """Base class for all Claimify pipeline agents."""

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        config: Optional[ClaimifyConfig] = None,
    ):
        self.llm = llm
        self.config = config or ClaimifyConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _log_decision(self, stage: str, decision: str, reasoning: str = ""):
        """Log agent decisions if configured."""
        if self.config.log_decisions:
            self.logger.info(
                f"[agents.{self.__class__.__name__}._log_decision] Claimify.{stage} decision: {decision}",
                extra={
                    "service": "aclarai-core",
                    "stage": stage,
                    "decision": decision,
                    "reasoning": reasoning,
                },
            )

    def _log_transformation(
        self, stage: str, original: str, transformed: str, changes: list
    ):
        """Log transformations if configured."""
        if self.config.log_transformations:
            self.logger.info(
                f"[agents.{self.__class__.__name__}._log_transformation] Claimify.{stage} transformation: {len(changes)} changes made",
                extra={
                    "service": "aclarai-core",
                    "stage": stage,
                    "original_length": len(original),
                    "transformed_length": len(transformed),
                    "changes": changes,
                },
            )

    def _log_timing(self, stage: str, processing_time: float):
        """Log processing time if configured."""
        if self.config.log_timing:
            self.logger.debug(
                f"[agents.{self.__class__.__name__}._log_timing] Claimify.{stage} processing time: {processing_time:.3f}s",
                extra={
                    "service": "aclarai-core",
                    "stage": stage,
                    "processing_time": processing_time,
                },
            )


class SelectionAgent(BaseClaimifyAgent):
    """
    Agent responsible for identifying sentence chunks that contain verifiable information
    relevant for claim extraction.
    This is the first stage of the Claimify pipeline.
    """

    def process(self, context: ClaimifyContext) -> SelectionResult:
        """
        Process a sentence chunk to determine if it should be selected for further processing.
        Args:
            context: ClaimifyContext containing the sentence and surrounding context
        Returns:
            SelectionResult with decision and reasoning
        """
        start_time = time.time()
        sentence = context.current_sentence
        try:
            # LLM is required for selection processing
            if self.llm is None:
                raise ValueError("LLM is required for Selection agent processing")
            result = self._llm_selection(context)
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            self._log_decision(
                "selection",
                "selected" if result.is_selected else "rejected",
                result.reasoning or "",
            )
            self._log_timing("selection", processing_time)
            return result
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                f"[agents.{self.__class__.__name__}.process] Error in selection processing: {e}",
                extra={
                    "service": "aclarai-core",
                    "stage": "selection",
                    "error": str(e),
                },
            )
            return SelectionResult(
                sentence_chunk=sentence,
                is_selected=False,
                reasoning=f"Error during processing: {e}",
                processing_time=processing_time,
            )

    def _llm_selection(self, context: ClaimifyContext) -> SelectionResult:
        """
        LLM-based selection following the Claimify approach.
        This implements Stage 1: Selection from the Claimify pipeline, which:
        1. Identifies if the sentence contains verifiable factual information
        2. Uses JSON output format as specified in claimify_selection.yaml prompt
        """
        assert self.llm is not None, "LLM must be initialized for this method"

        sentence = context.current_sentence
        context_text = self._build_context_text(context)
        # Use JSON prompt format matching claimify_selection.yaml
        prompt = f"""You are an expert at identifying verifiable factual content in text. Your task is to determine whether a given sentence contains information that could be extracted as verifiable claims.
Analyze the following sentence within its context to determine if it contains verifiable, factual information.
Context (surrounding sentences):
{context_text}
Target sentence: "{sentence.text}"
Consider these criteria:
1. Does this sentence contain factual, verifiable information?
2. Is it a statement (not a question, command, or exclamation)?
3. Could this information be fact-checked or validated?
4. Does it describe events, relationships, measurements, or properties?
5. Is it specific enough to be meaningful?
Sentences to REJECT:
- Questions ("What should we do?")
- Commands ("Please fix this.")
- Opinions without factual basis ("I think it's bad.")
- Vague statements ("Something happened.")
- Very short fragments ("Yes.", "OK.")
Sentences to SELECT:
- Technical facts ("The system returned error code 500.")
- Event descriptions ("The deployment occurred at 10:30 AM.")
- Measurements ("The response time was 2.3 seconds.")
- Relationships ("User A reported the bug to Team B.")
- Specific observations ("The CPU usage spiked to 95%.")
Respond with valid JSON only:
{{
  "selected": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of decision"
}}"""
        try:
            # Call the LLM with the prompt
            response = self.llm.complete(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens or 500,
            ).strip()
            # Parse JSON response
            try:
                result_data = json.loads(response)
                is_selected = result_data.get("selected", False)
                confidence = result_data.get("confidence", 0.0)
                reasoning = result_data.get("reasoning", "No reasoning provided")
                # Apply confidence threshold - if LLM confidence is below threshold, reject selection
                if (
                    is_selected
                    and confidence < self.config.selection_confidence_threshold
                ):
                    is_selected = False
                    reasoning = f"LLM selected but confidence {confidence:.2f} below threshold {self.config.selection_confidence_threshold:.2f}"
                return SelectionResult(
                    sentence_chunk=sentence,
                    is_selected=is_selected,
                    reasoning=reasoning,
                    confidence=confidence,
                    rewritten_text=sentence.text if is_selected else None,
                )
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response from LLM: {e}") from e
        except Exception as e:
            # If LLM fails, we cannot perform selection without heuristics
            raise ValueError(
                f"LLM selection failed and no fallback available: {e}"
            ) from e

    def _build_context_text(self, context: ClaimifyContext) -> str:
        """Build context text from surrounding sentences."""
        parts = []
        # Add preceding sentences
        for i, sent in enumerate(context.preceding_sentences):
            parts.append(f"[{-len(context.preceding_sentences) + i}] {sent.text}")
        # Add current sentence marker
        parts.append(f"[0] {context.current_sentence.text} ← TARGET")
        # Add following sentences
        for i, sent in enumerate(context.following_sentences):
            parts.append(f"[{i + 1}] {sent.text}")
        return "\n".join(parts)


class DisambiguationAgent(BaseClaimifyAgent):
    """
    Agent responsible for rewriting sentences to remove ambiguities and add context.
    This is the second stage of the Claimify pipeline.
    """

    def process(
        self, sentence: SentenceChunk, context: ClaimifyContext
    ) -> DisambiguationResult:
        """
        Process a selected sentence to remove ambiguities and add inferred subjects.
        Args:
            sentence: The sentence chunk to disambiguate
            context: Surrounding context for disambiguation
        Returns:
            DisambiguationResult with rewritten text and changes
        """
        start_time = time.time()
        try:
            # LLM is required for disambiguation processing
            if self.llm is None:
                raise ValueError("LLM is required for Disambiguation agent processing")
            result = self._llm_disambiguation(sentence, context)
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            self._log_transformation(
                "disambiguation",
                sentence.text,
                result.disambiguated_text,
                result.changes_made,
            )
            self._log_timing("disambiguation", processing_time)
            return result
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                f"[agents.{self.__class__.__name__}.process] Error in disambiguation processing: {e}",
                extra={
                    "service": "aclarai-core",
                    "stage": "disambiguation",
                    "error": str(e),
                },
            )
            return DisambiguationResult(
                original_sentence=sentence,
                disambiguated_text=sentence.text,  # Return original on error
                changes_made=[f"Error during processing: {e}"],
                processing_time=processing_time,
            )

    def _llm_disambiguation(
        self, sentence: SentenceChunk, context: ClaimifyContext
    ) -> DisambiguationResult:
        """
        LLM-based disambiguation following the Claimify approach.
        This implements Stage 2: Disambiguation from the Claimify pipeline, which:
        1. Identifies ambiguities (pronouns, time references, structural ambiguities)
        2. Uses context to resolve ambiguities confidently
        3. Uses JSON output format as specified in claimify_disambiguation.yaml prompt
        """
        assert self.llm is not None, "LLM must be initialized for this method"

        context_text = self._build_context_text(context)
        # Use JSON prompt format matching claimify_disambiguation.yaml
        prompt = f"""You are an expert at disambiguating text by resolving pronouns, adding missing context, and making implicit information explicit. Your goal is to rewrite sentences to be clear and self-contained while preserving their original meaning.
Rewrite the following sentence to remove ambiguities and make it self-contained. Use the surrounding context to resolve pronouns and add missing subjects or objects.
Context (surrounding sentences):
{context_text}
Target sentence to disambiguate: "{sentence.text}"
Disambiguation guidelines:
1. Replace ambiguous pronouns (it, this, that, they) with specific entities
2. Add missing subjects for sentences starting with verbs
3. Clarify vague references ("the error", "the issue", "the problem")
4. Make temporal and causal relationships explicit
5. Preserve the original meaning and factual content
6. Keep the sentence concise but complete
Examples:
- "It failed." → "[The system] failed."
- "This caused problems." → "This [configuration change] caused problems."
- "Reported an error." → "[The application] reported an error."
- "The error occurred when..." → "The [authentication] error occurred when..."
Respond with valid JSON only:
{{
  "disambiguated_text": "The rewritten sentence",
  "changes_made": ["List of specific changes"],
  "confidence": 0.0-1.0
}}"""
        try:
            # Call the LLM with the prompt
            response = self.llm.complete(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens or 500,
            ).strip()
            # Parse JSON response
            try:
                result_data = json.loads(response)
                disambiguated_text = result_data.get(
                    "disambiguated_text", sentence.text
                )
                changes_made = result_data.get("changes_made", [])
                confidence = result_data.get("confidence", 0.8)
                # Apply confidence threshold - if LLM confidence is below threshold, use original text
                if confidence < self.config.disambiguation_confidence_threshold:
                    disambiguated_text = sentence.text
                    changes_made = [
                        f"LLM confidence {confidence:.2f} below threshold {self.config.disambiguation_confidence_threshold:.2f}, using original text"
                    ]
                return DisambiguationResult(
                    original_sentence=sentence,
                    disambiguated_text=disambiguated_text,
                    changes_made=changes_made,
                    confidence=confidence,
                )
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response from LLM: {e}") from e
        except Exception as e:
            # If LLM fails, we cannot perform disambiguation without heuristics
            raise ValueError(
                f"LLM disambiguation failed and no fallback available: {e}"
            ) from e

    def _build_context_text(self, context: ClaimifyContext) -> str:
        """Build context text from surrounding sentences."""
        parts = []
        # Add preceding sentences
        for i, sent in enumerate(context.preceding_sentences):
            parts.append(f"[{-len(context.preceding_sentences) + i}] {sent.text}")
        # Add current sentence marker
        parts.append(f"[0] {context.current_sentence.text} ← TARGET")
        # Add following sentences
        for i, sent in enumerate(context.following_sentences):
            parts.append(f"[{i + 1}] {sent.text}")
        return "\n".join(parts)


class DecompositionAgent(BaseClaimifyAgent):
    """
    Agent responsible for breaking disambiguated sentences into atomic, self-contained claims.
    This is the third and final stage of the Claimify pipeline.
    """

    def process(
        self, disambiguated_text: str, _original_sentence: SentenceChunk
    ) -> DecompositionResult:
        """
        Process a disambiguated sentence to extract atomic claims.
        Args:
            disambiguated_text: The disambiguated sentence text
            original_sentence: The original sentence chunk for reference
        Returns:
            DecompositionResult with claim candidates
        """
        start_time = time.time()
        try:
            # LLM is required for decomposition processing
            if self.llm is None:
                raise ValueError("LLM is required for Decomposition agent processing")
            result = self._llm_decomposition(disambiguated_text)
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            # Log results
            valid_claims = result.valid_claims
            sentence_nodes = result.sentence_nodes
            self.logger.info(
                f"[agents.{self.__class__.__name__}.process] Claimify.decomposition completed: {len(valid_claims)} claims, {len(sentence_nodes)} sentences",
                extra={
                    "service": "aclarai-core",
                    "stage": "decomposition",
                    "claims_count": len(valid_claims),
                    "sentences_count": len(sentence_nodes),
                    "total_candidates": len(result.claim_candidates),
                },
            )
            self._log_timing("decomposition", processing_time)
            return result
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                f"[agents.{self.__class__.__name__}.process] Error in decomposition processing: {e}",
                extra={
                    "service": "aclarai-core",
                    "stage": "decomposition",
                    "error": str(e),
                },
            )
            return DecompositionResult(
                original_text=disambiguated_text,
                claim_candidates=[],
                processing_time=processing_time,
            )

    def _llm_decomposition(self, text: str) -> DecompositionResult:
        """
        LLM-based decomposition following the Claimify approach.
        This implements Stage 3: Decomposition from the Claimify pipeline, which:
        1. Breaks the sentence into atomic, self-contained claims
        2. Uses JSON output format as specified in claimify_decomposition.yaml prompt
        3. Gets quality criteria evaluation from the LLM rather than hardcoding
        """
        assert self.llm is not None, "LLM must be initialized for this method"

        # Use JSON prompt format matching claimify_decomposition.yaml
        prompt = f"""You are an expert at extracting atomic claims from text. Your task is to break down sentences into individual, verifiable claims that meet strict quality criteria. Each claim must be atomic (single fact), self-contained (no ambiguous references), and verifiable (factually checkable).
Analyze the following disambiguated sentence and extract atomic claims that meet the Claimify quality criteria.
Input sentence: "{text}"
Quality Criteria for Claims:
1. ATOMIC: Contains exactly one verifiable fact (no compound statements)
2. SELF-CONTAINED: No ambiguous pronouns or references (all entities clearly identified)
3. VERIFIABLE: Contains specific, factual information that can be fact-checked
Examples of VALID claims:
- "The user received an error from Pylance."
- "In Python, a slice cannot be assigned to a parameter of type int in __setitem__."
- "The error rate increased to 25% after deployment."
Examples of INVALID claims:
- "The error occurred while calling __setitem__ with a slice." (vague reference "the error")
- "The system worked but was slow." (compound statement - not atomic)
- "Something went wrong." (not specific enough to verify)
Instructions:
1. Split compound sentences (connected by "and", "but", "or", "because", etc.)
2. Evaluate each potential claim against the three criteria
3. Only include claims that pass ALL criteria
4. For claims that fail criteria, explain why they should become :Sentence nodes instead
Respond with valid JSON only:
{{
  "claim_candidates": [
    {{
      "text": "The extracted claim text",
      "is_atomic": true/false,
      "is_self_contained": true/false,
      "is_verifiable": true/false,
      "passes_criteria": true/false,
      "confidence": 0.0-1.0,
      "reasoning": "Explanation of evaluation",
      "node_type": "Claim" or "Sentence"
    }}
  ]
}}"""
        try:
            # Call the LLM with the prompt
            response = self.llm.complete(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens or 1000,
            ).strip()
            # Parse JSON response
            try:
                result_data = json.loads(response)
                claim_candidates = []
                for candidate_data in result_data.get("claim_candidates", []):
                    claim_text = candidate_data.get("text", "").strip()
                    if not claim_text:
                        continue
                    # Get quality flags from LLM output
                    is_atomic = candidate_data.get("is_atomic", False)
                    is_self_contained = candidate_data.get("is_self_contained", False)
                    is_verifiable = candidate_data.get("is_verifiable", False)
                    passes_criteria = candidate_data.get("passes_criteria", False)
                    reasoning = candidate_data.get("reasoning", "No reasoning provided")
                    # Get confidence from LLM response or calculate based on quality flags
                    confidence = candidate_data.get("confidence")
                    if confidence is None:
                        # Fallback calculation if LLM doesn't provide confidence
                        if (
                            passes_criteria
                            and is_atomic
                            and is_self_contained
                            and is_verifiable
                        ):
                            confidence = 0.9
                        elif sum([is_atomic, is_self_contained, is_verifiable]) >= 2:
                            confidence = 0.6
                        else:
                            confidence = 0.3
                    # Apply confidence threshold - only include candidates above threshold
                    if confidence >= self.config.decomposition_confidence_threshold:
                        candidate = ClaimCandidate(
                            text=claim_text,
                            is_atomic=is_atomic,
                            is_self_contained=is_self_contained,
                            is_verifiable=is_verifiable,
                            confidence=confidence,
                            reasoning=reasoning,
                        )
                        claim_candidates.append(candidate)
                return DecompositionResult(
                    original_text=text, claim_candidates=claim_candidates
                )
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response from LLM: {e}") from e
        except Exception as e:
            # If LLM fails, we cannot perform decomposition without heuristics
            raise ValueError(
                f"LLM decomposition failed and no fallback available: {e}"
            ) from e
