"""
Concept file writer for Tier 3 Markdown files.
This module implements the creation of Tier 3 Markdown files for promoted
concepts, following the Concept Summary Agent format defined in the
architecture documentation.
"""

import logging
from pathlib import Path
from typing import Optional

from aclarai_shared.config import aclaraiConfig
from aclarai_shared.graph.models import Concept
from aclarai_shared.import_system import write_file_atomically

logger = logging.getLogger(__name__)


class ConceptFileWriter:
    """
    Writes Tier 3 Markdown files for concepts.
    This class handles the creation of concept files in the vault using
    atomic write operations and following the Concept Summary Agent format.
    """

    def __init__(self, config: Optional[aclaraiConfig] = None):
        """Initialize the concept file writer."""
        self.config = config or aclaraiConfig()
        # Get the concepts directory from configuration
        vault_path = Path(self.config.vault_path)
        concepts_path = self.config.paths.tier3 or "concepts"
        self.concepts_dir = vault_path / concepts_path
        logger.debug(
            f"Initialized ConceptFileWriter with concepts directory: {self.concepts_dir}",
            extra={
                "service": "aclarai-core",
                "filename.function_name": "tier3_concept.ConceptFileWriter.__init__",
                "concepts_dir": str(self.concepts_dir),
            },
        )

    def write_concept_file(self, concept: Concept) -> bool:
        """
        Write a Tier 3 Markdown file for a concept.
        Args:
            concept: The Concept object to write
        Returns:
            True if the file was written successfully, False otherwise
        """
        try:
            # Generate the filename based on the concept name
            filename = self._generate_concept_filename(concept.text)
            file_path = self.concepts_dir / filename
            # Generate the Markdown content
            content = self._generate_concept_content(concept)
            # Write the file atomically
            write_file_atomically(file_path, content)
            logger.info(
                f"Successfully wrote concept file: {filename}",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "tier3_concept.ConceptFileWriter.write_concept_file",
                    "concept_id": concept.concept_id,
                    "concept_text": concept.text,
                    "file_path": str(file_path),
                },
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to write concept file for {concept.concept_id}: {e}",
                extra={
                    "service": "aclarai-core",
                    "filename.function_name": "tier3_concept.ConceptFileWriter.write_concept_file",
                    "concept_id": concept.concept_id,
                    "concept_text": concept.text,
                    "error": str(e),
                },
            )
            return False

    def _generate_concept_filename(self, concept_text: str) -> str:
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
        # Remove multiple consecutive underscores and trailing underscores
        import re

        safe_name = re.sub(r"_+", "_", safe_name).strip("_")
        # Limit the length to avoid filesystem issues
        if len(safe_name) > 200:
            safe_name = safe_name[:200].rstrip("_")
        # Ensure we have a non-empty name
        if not safe_name:
            safe_name = "unnamed_concept"
        return f"{safe_name}.md"

    def _generate_concept_content(self, concept: Concept) -> str:
        """
        Generate the Markdown content for a concept file.
        This follows the Concept Summary Agent format defined in
        on-writing_vault_documents.md.
        Args:
            concept: The Concept object
        Returns:
            The Markdown content as a string
        """
        # Extract a clean concept name for the header
        concept_name = concept.text.strip()
        lines = [
            f"## Concept: {concept_name}",
            "",
            "This concept was automatically extracted and promoted from processed content.",
            "",
            "### Examples",
            "<!-- Examples will be added when claims are linked to this concept -->",
            "",
            "### See Also",
            "<!-- Related concepts will be added through concept linking -->",
            "",
            f"<!-- aclarai:id={concept.concept_id} ver={concept.version} -->",
            f"^{concept.concept_id}",
        ]
        return "\n".join(lines)
