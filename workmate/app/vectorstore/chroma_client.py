import os
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings
import logging

logger = logging.getLogger(__name__)

_chroma_client = None


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        persist_dir = os.path.abspath(settings.CHROMA_PERSIST_DIR)
        os.makedirs(persist_dir, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB client initialized at: {persist_dir}")
    return _chroma_client


def get_document_collection(tenant_id: int):
    """Returns the Chroma collection for document chunks for a specific tenant."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=f"tenant_{tenant_id}_docs",
        metadata={"hnsw:space": "cosine"},
    )


def get_memory_collection(tenant_id: int):
    """Returns the Chroma collection for memory records for a specific tenant."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=f"tenant_{tenant_id}_memory",
        metadata={"hnsw:space": "cosine"},
    )
