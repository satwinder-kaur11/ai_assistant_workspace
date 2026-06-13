from app.config import settings
import logging

logger = logging.getLogger(__name__)

_embedding_model = None


def get_embedding_model():
    """
    Returns the configured embedding model (cached singleton).
    Supports 'local' (sentence-transformers) and 'openai' providers.
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    if settings.EMBEDDING_PROVIDER.lower() == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set when EMBEDDING_PROVIDER is 'openai'")
        logger.info("Using OpenAI embeddings (text-embedding-3-small)")
        _embedding_model = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            model="text-embedding-3-small",
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Using local HuggingFace embeddings (all-MiniLM-L6-v2)")
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    return _embedding_model
