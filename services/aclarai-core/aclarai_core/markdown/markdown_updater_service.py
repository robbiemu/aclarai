"""
Service for updating aclarai metadata within Markdown files.
"""

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class MarkdownUpdaterService:
    """
    Handles atomic updates of aclarai metadata in Markdown files.
    """

    ACLARAI_ID_PATTERN = (
        r"<!-- aclarai:id=(?P<id>[a-zA-Z0-9_]+)\s+ver=(?P<ver>\d+)\s*-->"
    )
    ACLARAI_SCORE_PATTERN_TEMPLATE = (
        r"<!-- aclarai:{score_name}=(?P<score>[0-9.]+|null)\s*-->"
    )

    def _atomic_write(self, filepath: Path, content: str) -> bool:
        """
        Writes content to a file atomically using a temporary file and rename.
        """
        try:
            # Create a temporary file in the same directory to ensure rename is atomic
            # (os.rename works atomically if src and dst are on the same filesystem)
            temp_dir = filepath.parent
            temp_file_fd, temp_file_path_str = tempfile.mkstemp(
                dir=temp_dir, prefix=filepath.name + ".tmp"
            )
            temp_file_path = Path(temp_file_path_str)

            with os.fdopen(temp_file_fd, "w", encoding="utf-8") as tmp:
                tmp.write(content)
                tmp.flush()
                os.fsync(tmp.fileno())  # Ensure data is written to disk

            os.replace(temp_file_path, filepath)  # Atomic rename
            logger.debug(f"Atomically wrote updated content to {filepath}")
            return True
        except Exception as e:
            logger.error(
                f"Failed to atomically write to {filepath}: {e}", exc_info=True
            )
            if "temp_file_path" in locals() and temp_file_path.exists():  # type: ignore
                try:
                    os.unlink(temp_file_path)  # type: ignore
                except OSError:  # pragma: no cover
                    logger.error(
                        f"Failed to remove temporary file {temp_file_path}",
                        exc_info=True,
                    )  # type: ignore
            return False

    def _find_block_and_update_score(
        self,
        content: str,
        block_id: str,
        score_name: str,
        score_value: Optional[float],
    ) -> Tuple[Optional[str], str]:
        """
        Finds a specific aclarai block by its ID, updates its version,
        and adds/updates a score comment.

        Returns:
            A tuple: (new_content, status_message).
            new_content is None if the block_id is not found or an error occurs.
        """
        # Regex to find the specific block's metadata comment
        # We need to find the start of the block to insert the score comment after its metadata line
        # This is tricky because block content can be multi-line.
        # For now, we'll assume the aclarai:id comment is at the end of the block's primary line,
        # or that the block we are targeting is a single line followed by its metadata.

        # This simplified version assumes we operate on the line containing the aclarai:id comment.
        # A more robust solution might need to parse Markdown structure or have clearer block delimiters.

        lines = content.splitlines()
        new_lines = []
        block_found = False
        block_id_pattern_compiled = re.compile(self.ACLARAI_ID_PATTERN)
        score_comment_pattern_compiled = re.compile(
            self.ACLARAI_SCORE_PATTERN_TEMPLATE.format(score_name=re.escape(score_name))
        )

        # Construct the new score comment
        score_str = (
            "null"
            if score_value is None
            else f"{score_value:.2f}".rstrip("0").rstrip(".")
        )
        new_score_comment = f"<!-- aclarai:{score_name}={score_str} -->"

        for i, line in enumerate(lines):
            match = block_id_pattern_compiled.search(line)
            if match and match.group("id") == block_id:
                block_found = True
                original_block_meta_comment = match.group(0)
                current_ver = int(match.group("ver"))
                new_ver = current_ver + 1

                # Update version in the block's metadata comment
                updated_block_meta_comment = original_block_meta_comment.replace(
                    f"ver={current_ver}", f"ver={new_ver}"
                )
                updated_line = line.replace(
                    original_block_meta_comment, updated_block_meta_comment
                )

                # Check if the score comment already exists for this block
                # This requires a more complex search if scores can be anywhere relative to the block id.
                # For simplicity, let's assume the score comment is immediately after or before the ID line,
                # or we modify the line containing the ID if it's a combined metadata block.
                # Here, we'll insert the new score comment immediately before the updated block line.
                # And remove any old score comment that might be on the line before it or on the same line.

                # Attempt to remove old score if it's on the line *before* the ID line
                if i > 0 and score_comment_pattern_compiled.search(new_lines[-1]):
                    logger.debug(f"Removing old score comment: {new_lines[-1]}")
                    new_lines.pop()

                # Add new score comment, then the updated line with new version
                new_lines.append(new_score_comment)
                new_lines.append(updated_line)
                logger.info(
                    f"Updated block '{block_id}': incremented version to {new_ver}, added/updated {score_name}={score_str}."
                )

            else:
                # Remove old score comment if it's not associated with the current block_id update pass
                # This is to clean up potentially orphaned score comments if logic changes.
                # However, this might be too aggressive if multiple scores are managed independently.
                # For now, only remove if we are sure it's an old version for *this* block.
                # This part is tricky and might need refinement based on how block content vs metadata is structured.
                # Safest approach: only modify lines directly related to the found block_id.
                new_lines.append(line)

        if not block_found:
            return None, f"Block ID '{block_id}' not found in the Markdown content."

        return "\n".join(
            new_lines
        ), f"Successfully updated score for block '{block_id}'."

    def add_or_update_decontextualization_score(
        self,
        filepath_str: str,
        block_id: str,
        score: Optional[float],
    ) -> bool:
        """
        Adds or updates the decontextualization score for a specific block in a Markdown file.

        It finds the block by its `aclarai:id`, increments its `ver=` number,
        and adds/updates the `<!-- aclarai:decontextualization_score=... -->` comment.
        The operation is performed atomically.

        Args:
            filepath_str: Path to the Markdown file.
            block_id: The unique aclarai:id of the block to update.
            score: The decontextualization score (float or None for null).

        Returns:
            True if the update was successful, False otherwise.
        """
        score_name = "decontextualization_score"
        filepath = Path(filepath_str)
        log_details = {
            "service": "aclarai-core",
            "filename_function_name": "markdown_updater_service.MarkdownUpdaterService.add_or_update_decontextualization_score",
            "filepath": filepath_str,
            "aclarai_id_block": block_id,
            "score_name": score_name,
            "score_value": score,
        }

        if not filepath.exists():
            logger.error(f"Markdown file not found: {filepath_str}", extra=log_details)
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                original_content = f.read()
        except Exception as e:
            logger.error(
                f"Error reading Markdown file {filepath_str}: {e}",
                exc_info=True,
                extra=log_details,
            )
            return False

        new_content, status_msg = self._find_block_and_update_score(
            original_content, block_id, score_name, score
        )

        if new_content is None:
            logger.warning(
                f"Failed to update Markdown for block {block_id} in {filepath_str}: {status_msg}",
                extra=log_details,
            )
            return False

        if new_content == original_content:
            logger.info(
                f"No changes required for decontextualization_score for block {block_id} in {filepath_str}. Content is identical.",
                extra=log_details,
            )
            # Technically a success as the state is as desired, but no write needed.
            # However, version should have incremented if score was re-evaluated, even if value is same.
            # The _find_block_and_update_score should always increment version if block is found.
            # This path might indicate an issue in _find_block_and_update_score's change detection or versioning.
            # For now, let's assume if content is same, something ensured it.
            # A robust check would be if the *intended* state (score + new version) matches current file.
            # This needs careful thought. If score is same, but version should bump, we must write.
            # The current _find_block_and_update_score *will* create new_content with bumped version.
            # So this condition (new_content == original_content) implies block was not found or error.
            # This was a misinterpretation. If block is found, version increments, content *will* change.
            # If new_content is None, it means block not found.
            # Let's proceed to write if new_content is not None.
            pass

        if self._atomic_write(filepath, new_content):
            logger.info(
                f"Successfully updated {score_name} for block {block_id} in {filepath_str}.",
                extra=log_details,
            )
            return True
        else:
            logger.error(
                f"Atomic write failed for {filepath_str} while updating block {block_id}.",
                extra=log_details,
            )
            # Attempt to restore original content if possible? This is complex.
            # For now, the error is logged by _atomic_write.
            return False


# Example Usage (Conceptual)
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     updater = MarkdownUpdaterService()
#
#     # Create a dummy markdown file
#     dummy_md_path = Path("test_markdown_file.md")
#     dummy_content_initial = """
# Some preliminary text.
# This is block one. <!-- aclarai:id=blk_abc123 ver=1 -->
# ^blk_abc123
# Some other text.
# This is block two, with an old score.
# <!-- aclarai:decontextualization_score=0.50 -->
# Line for block two. <!-- aclarai:id=blk_xyz789 ver=3 -->
# ^blk_xyz789
# Final text.
# """
#     with open(dummy_md_path, "w", encoding="utf-8") as f:
#         f.write(dummy_content_initial)
#
#     # Test 1: Add score to blk_abc123
#     print("\nTest 1: Add score to blk_abc123 (should be 0.77, ver=2)")
#     success1 = updater.add_or_update_decontextualization_score(str(dummy_md_path), "blk_abc123", 0.77)
#     print(f"Test 1 Success: {success1}")
#     with open(dummy_md_path, "r", encoding="utf-8") as f:
#         print(f.read())
#
#     # Test 2: Update score for blk_xyz789 (should be null, ver=4)
#     print("\nTest 2: Update score for blk_xyz789 (should be null, ver=4)")
#     success2 = updater.add_or_update_decontextualization_score(str(dummy_md_path), "blk_xyz789", None)
#     print(f"Test 2 Success: {success2}")
#     with open(dummy_md_path, "r", encoding="utf-8") as f:
#         print(f.read())
#
#     # Test 3: Score for non-existent block
#     print("\nTest 3: Score for non-existent block (should fail)")
#     success3 = updater.add_or_update_decontextualization_score(str(dummy_md_path), "blk_nonexistent", 0.5)
#     print(f"Test 3 Success (expected False): {success3}")
#     with open(dummy_md_path, "r", encoding="utf-8") as f:
#         print(f.read())
#
#     # Test 4: Re-score blk_abc123 (should be 0.90, ver=3)
#     print("\nTest 4: Re-score blk_abc123 (should be 0.90, ver=3)")
#     success4 = updater.add_or_update_decontextualization_score(str(dummy_md_path), "blk_abc123", 0.90)
#     print(f"Test 4 Success: {success4}")
#     with open(dummy_md_path, "r", encoding="utf-8") as f:
#         print(f.read())
#
#     # Clean up
#     # dummy_md_path.unlink()
#     print(f"\nTo cleanup, manually delete: {dummy_md_path.resolve()}")
#     print("MarkdownUpdaterService file created. Example usage commented out.")
