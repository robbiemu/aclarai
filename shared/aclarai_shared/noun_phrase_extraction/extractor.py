"""
Noun phrase extractor for Claims and Summary nodes.
This module implements the main NounPhraseExtractor class that:
1. Fetches (:Claim) and (:Summary) nodes from Neo4j
2. Extracts noun phrases using spaCy
3. Normalizes and embeds the phrases
4. Stores them in the concept_candidates vector table
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

import spacy

from ..config import aclaraiConfig, load_config
from ..embedding import EmbeddingGenerator
from ..graph import Neo4jGraphManager
from .concept_candidates_store import ConceptCandidatesVectorStore
from .models import ExtractionResult, NounPhraseCandidate

logger = logging.getLogger(__name__)


class NounPhraseExtractor:
    """
    Extracts noun phrases from Claims and Summary nodes and stores them
    in the concept_candidates vector table for future concept detection.
    This class orchestrates the noun phrase extraction workflow by:
    - Fetching (:Claim) and (:Summary) nodes from the Neo4j graph
    - Using spaCy to extract and normalize noun phrases
    - Storing candidates in the concept_candidates vector table with "pending" status
    """

    def __init__(
        self,
        config: Optional[aclaraiConfig] = None,
        spacy_model: str = "en_core_web_sm",
    ):
        """
        Initialize the noun phrase extractor.
        Args:
            config: aclarai configuration (loads default if None)
            spacy_model: spaCy model to use for extraction
        """
        if config is None:
            config = load_config()
        self.config = config
        self.spacy_model_name = spacy_model
        # Initialize dependencies
        self.neo4j_manager = Neo4jGraphManager(config)
        self.embedding_generator = EmbeddingGenerator(config)
        # Initialize spaCy model
        self._nlp: Optional[spacy.language.Language] = None
        self._initialize_spacy()
        # Initialize concept candidates vector store
        self.concept_candidates_store = ConceptCandidatesVectorStore(config)
        logger.info(
            f"Initialized NounPhraseExtractor with spaCy model: {self.spacy_model_name}",
            extra={
                "service": "aclarai",
                "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor.__init__",
            },
        )

    def extract_from_all_nodes(self) -> ExtractionResult:
        """
        Extract noun phrases from all (:Claim) and (:Summary) nodes.
        Returns:
            ExtractionResult with all extracted candidates and processing metrics
        """
        logger.info(
            "Starting noun phrase extraction from all Claims and Summaries",
            extra={
                "service": "aclarai",
                "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor.extract_from_all_nodes",
            },
        )
        start_time = time.time()
        result = ExtractionResult()
        try:
            # Fetch all Claims and Summaries
            claims = self._fetch_claim_nodes()
            summaries = self._fetch_summary_nodes()
            all_nodes = claims + summaries
            result.total_nodes_processed = len(all_nodes)
            logger.info(
                f"Processing {len(claims)} Claims and {len(summaries)} Summaries",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor.extract_from_all_nodes",
                    "claims_count": len(claims),
                    "summaries_count": len(summaries),
                },
            )
            # Extract noun phrases from each node
            for node in all_nodes:
                try:
                    candidates = self._extract_from_node(node)
                    result.candidates.extend(candidates)
                    result.successful_extractions += 1
                    result.total_phrases_extracted += len(candidates)
                except Exception as e:
                    logger.error(
                        f"Failed to extract noun phrases from node {node.get('id')}",
                        extra={
                            "service": "aclarai",
                            "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor.extract_from_all_nodes",
                            "node_id": node.get("id"),
                            "error": str(e),
                        },
                    )
                    result.failed_extractions += 1
            # Store all candidates in vector database
            if result.candidates:
                self._store_candidates(result.candidates)
            result.processing_time = time.time() - start_time
            result.model_used = self.spacy_model_name
            logger.info(
                f"Noun phrase extraction completed: {result.total_phrases_extracted} phrases from {result.successful_extractions} nodes",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor.extract_from_all_nodes",
                    "total_phrases": result.total_phrases_extracted,
                    "successful_nodes": result.successful_extractions,
                    "failed_nodes": result.failed_extractions,
                    "processing_time": result.processing_time,
                },
            )
            return result
        except Exception as e:
            result.error = str(e)
            result.processing_time = time.time() - start_time
            logger.error(
                f"Noun phrase extraction failed: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor.extract_from_all_nodes",
                    "error": str(e),
                },
            )
            return result

    def _fetch_claim_nodes(self) -> List[Dict[str, Any]]:
        """Fetch all (:Claim) nodes from Neo4j."""
        try:
            query = """
            MATCH (c:Claim)
            RETURN c.id as id, c.text as text, 'claim' as node_type
            """
            result = self.neo4j_manager.run_query(query)
            claims = []
            for record in result:
                claims.append(
                    {
                        "id": record["id"],
                        "text": record["text"],
                        "node_type": record["node_type"],
                    }
                )
            logger.debug(
                f"Fetched {len(claims)} Claim nodes",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor._fetch_claim_nodes",
                    "claims_count": len(claims),
                },
            )
            return claims
        except Exception as e:
            logger.error(
                f"Failed to fetch Claim nodes: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor._fetch_claim_nodes",
                    "error": str(e),
                },
            )
            return []

    def _fetch_summary_nodes(self) -> List[Dict[str, Any]]:
        """Fetch all (:Summary) nodes from Neo4j."""
        try:
            query = """
            MATCH (s:Summary)
            RETURN s.id as id, s.text as text, 'summary' as node_type
            """
            result = self.neo4j_manager.run_query(query)
            summaries = []
            for record in result:
                summaries.append(
                    {
                        "id": record["id"],
                        "text": record["text"],
                        "node_type": record["node_type"],
                    }
                )
            logger.debug(
                f"Fetched {len(summaries)} Summary nodes",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor._fetch_summary_nodes",
                    "summaries_count": len(summaries),
                },
            )
            return summaries
        except Exception as e:
            logger.error(
                f"Failed to fetch Summary nodes: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor._fetch_summary_nodes",
                    "error": str(e),
                },
            )
            return []

    def _extract_from_node(self, node: Dict[str, Any]) -> List[NounPhraseCandidate]:
        """
        Extract noun phrases from a single node (Claim or Summary).
        Args:
            node: Dictionary containing node data (id, text, node_type)
        Returns:
            List of NounPhraseCandidate objects
        """
        text = node.get("text", "")
        node_id = node.get("id", "")
        node_type = node.get("node_type", "unknown")
        if not text or not node_id:
            logger.warning(
                f"Skipping node with missing text or ID: {node_id}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor._extract_from_node",
                    "node_id": node_id,
                },
            )
            return []
        # Extract noun phrases using spaCy
        noun_phrases = self._extract_noun_phrases(text)
        # Create candidates with normalization
        candidates = []
        for phrase in noun_phrases:
            normalized = self._normalize_phrase(phrase)
            # Skip if normalization resulted in empty or very short text
            if len(normalized.strip()) < 2:
                continue
            candidate = NounPhraseCandidate(
                text=phrase,
                normalized_text=normalized,
                source_node_id=node_id,
                source_node_type=node_type,
                aclarai_id=node_id,  # Use node_id as aclarai:id for traceability
                status="pending",
            )
            candidates.append(candidate)
        logger.debug(
            f"Extracted {len(candidates)} noun phrases from {node_type} {node_id}",
            extra={
                "service": "aclarai",
                "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor._extract_from_node",
                "node_id": node_id,
                "node_type": node_type,
                "phrases_count": len(candidates),
            },
        )
        return candidates

    def _extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases from text using spaCy's noun_chunks.
        Args:
            text: Input text to process
        Returns:
            List of noun phrase strings
        """
        if not self._nlp:
            logger.error(
                "spaCy model not initialized",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor._extract_noun_phrases",
                },
            )
            return []
        try:
            doc = self._nlp(text)
            noun_phrases = [
                chunk.text.strip() for chunk in doc.noun_chunks if chunk.text.strip()
            ]
            # Filter out very short phrases (single characters, numbers only, etc.)
            filtered_phrases = [
                phrase
                for phrase in noun_phrases
                if len(phrase) > 1 and not phrase.isdigit()
            ]
            return filtered_phrases
        except Exception as e:
            logger.error(
                f"Failed to extract noun phrases from text: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor._extract_noun_phrases",
                    "error": str(e),
                    "text_length": len(text),
                },
            )
            return []

    def _normalize_phrase(self, phrase: str) -> str:
        """
        Normalize a noun phrase according to specifications:
        - Convert to lowercase
        - Lemmatize words
        - Strip punctuation
        - Trim whitespace
        Args:
            phrase: Original noun phrase
        Returns:
            Normalized phrase
        """
        if not phrase or not self._nlp:
            return ""
        try:
            # Process with spaCy for lemmatization
            doc = self._nlp(phrase.lower())
            # Extract lemmatized forms, skip punctuation and stop words
            lemmatized_tokens = []
            for token in doc:
                # Skip punctuation, spaces, and very short tokens
                if not token.is_punct and not token.is_space and len(token.lemma_) > 1:
                    lemmatized_tokens.append(token.lemma_)
            normalized = " ".join(lemmatized_tokens)
            # Additional cleanup - remove extra punctuation
            normalized = re.sub(r"[^\w\s]", "", normalized)
            normalized = re.sub(r"\s+", " ", normalized)  # Collapse multiple spaces
            normalized = normalized.strip()
            return normalized
        except Exception as e:
            logger.error(
                f"Failed to normalize phrase '{phrase}': {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor._normalize_phrase",
                    "phrase": phrase,
                    "error": str(e),
                },
            )
            # Fallback: basic lowercase and punctuation removal
            return re.sub(r"[^\w\s]", "", phrase.lower()).strip()

    def _store_candidates(self, candidates: List[NounPhraseCandidate]):
        """
        Store noun phrase candidates in the concept_candidates vector table.
        Args:
            candidates: List of candidates to store
        """
        if not candidates:
            return
        logger.info(
            f"Storing {len(candidates)} noun phrase candidates in concept_candidates vector table",
            extra={
                "service": "aclarai",
                "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor._store_candidates",
                "candidates_count": len(candidates),
            },
        )
        try:
            # Store using the specialized concept candidates vector store
            successful_count = self.concept_candidates_store.store_candidates(
                candidates
            )
            logger.info(
                f"Successfully stored {successful_count}/{len(candidates)} concept candidates",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor._store_candidates",
                    "successful_count": successful_count,
                    "total_count": len(candidates),
                },
            )
        except Exception as e:
            logger.error(
                f"Failed to store noun phrase candidates: {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor._store_candidates",
                    "candidates_count": len(candidates),
                    "error": str(e),
                },
            )
            raise

    def _initialize_spacy(self):
        """Initialize the spaCy model."""
        if spacy is None:
            raise ImportError(
                "spaCy is not installed. Please install it with: pip install spacy"
            )
        try:
            self._nlp = spacy.load(self.spacy_model_name)
            logger.info(
                f"Loaded spaCy model: {self.spacy_model_name}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor._initialize_spacy",
                    "model": self.spacy_model_name,
                },
            )
        except OSError as e:
            logger.error(
                f"Failed to load spaCy model '{self.spacy_model_name}': {e}",
                extra={
                    "service": "aclarai",
                    "filename.function_name": "noun_phrase_extraction.NounPhraseExtractor._initialize_spacy",
                    "model": self.spacy_model_name,
                    "error": str(e),
                },
            )
            raise ValueError(
                f"spaCy model '{self.spacy_model_name}' not found. "
                f"Please install it with: python -m spacy download {self.spacy_model_name}"
            ) from None
