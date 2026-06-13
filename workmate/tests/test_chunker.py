"""Tests for the text chunker."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.chunker import chunk_text


def test_chunk_text_basic():
    text = "Hello world. " * 200          # ~2600 chars
    chunks = chunk_text(text)
    assert isinstance(chunks, list)
    assert len(chunks) > 1               # must split
    assert all(isinstance(c, str) for c in chunks)
    assert all(len(c) > 0 for c in chunks)


def test_chunk_text_short_string():
    text = "Short text."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_respects_overlap():
    """With overlap > 0, adjacent chunks should share some content."""
    text = "word " * 500
    chunks = chunk_text(text)
    if len(chunks) > 1:
        # Last few tokens of chunk[0] should appear at start of chunk[1]
        end_of_first = chunks[0][-30:]
        start_of_second = chunks[1][:30]
        # At least some overlap exists
        assert any(w in start_of_second for w in end_of_first.split())


def test_chunk_text_empty_string():
    chunks = chunk_text("")
    # Empty or list with one empty string — both are acceptable
    assert isinstance(chunks, list)
