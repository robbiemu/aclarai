from aclarai_shared.vault import BlockParser

def test_extract_aclarai_blocks():
    """Test extraction of aclarai blocks from markdown content."""
    content = """
Some text before.
This is a claim. <!-- aclarai:id=clm_abc123 ver=2 -->
^clm_abc123
Another claim here. <!-- aclarai:id=clm_def456 ver=1 -->
^clm_def456
## File-level document
Some content for the whole file.
<!-- aclarai:id=file_summary ver=3 -->
"""
    parser = BlockParser()
    blocks = parser.extract_aclarai_blocks(content)
    assert len(blocks) == 3
    # Check first inline block
    assert blocks[0]["aclarai_id"] == "clm_abc123"
    assert blocks[0]["version"] == 2
    assert "This is a claim." in blocks[0]["semantic_text"]
    # Check second inline block
    assert blocks[1]["aclarai_id"] == "clm_def456"
    assert blocks[1]["version"] == 1
    assert "Another claim here." in blocks[1]["semantic_text"]
    # Check file-level block
    assert blocks[2]["aclarai_id"] == "file_summary"
    assert blocks[2]["version"] == 3
    # File-level block should contain most of the content before the comment
    assert "File-level document" in blocks[2]["semantic_text"]

def test_calculate_content_hash():
    """Test content hash calculation."""
    parser = BlockParser()
    text1 = "This is some text."
    text2 = "This   is    some   text."  # Different whitespace
    text3 = "This is some other text."
    hash1 = parser.calculate_content_hash(text1)
    hash2 = parser.calculate_content_hash(text2)
    hash3 = parser.calculate_content_hash(text3)
    # Same content with different whitespace should have same hash
    assert hash1 == hash2
    # Different content should have different hash
    assert hash1 != hash3
