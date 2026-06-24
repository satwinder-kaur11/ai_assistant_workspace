"""
app/ingestion/chunker.py

Splits large parsed documents into smaller, 1000-character chunks using LangChain text splitters.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings
from typing import List


def chunk_text(text: str) -> List[str]:
    """Splits text into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)
