"""
app/rag/rag_service.py

Retrieval-Augmented Generation service. Searches ChromaDB for relevant document chunks 
based on the user's query.
"""

# Searches ChromaDB for chunks matching the user's query.
import logging
from typing import List, Dict, Any, Tuple
from app.vectorstore.chroma_client import get_document_collection
from app.ingestion.embedder import get_embedding_model
from app.config import settings

logger = logging.getLogger(__name__)


def search_documents(
    query: str,
    tenant_id: int,
    user_id: int = None,
    top_k: int = 5,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Searches documents in Chroma for a given tenant/user.
    Returns a tuple of (results, confidence_bucket).
    """
    collection = get_document_collection(tenant_id)
    embedder = get_embedding_model()

    try:
        query_embedding = embedder.embed_query(query)
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return [], "Low"

    # Check if collection is empty first to avoid Chroma errors
    try:
        count = collection.count()
        if count == 0:
            logger.info("Document collection is empty.")
            return [], "Low"
    except Exception:
        pass

    where_filter = None
    if user_id is not None:
        # -1 represents tenant-shared documents
        where_filter = {
            "$or": [
                {"user_id": {"$eq": user_id}},
                {"user_id": {"$eq": -1}},
            ]
        }

    try:
        query_kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        if where_filter:
            query_kwargs["where"] = where_filter

        results = collection.query(**query_kwargs)
    except Exception as e:
        logger.error(f"Error querying Chroma: {e}")
        return [], "Low"

    if not results or not results["documents"] or not results["documents"][0]:
        return [], "Low"

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]  # cosine distance: lower = more similar

    formatted_results = []
    relevant_count = 0
    best_score = 0.0

    for doc, meta, dist in zip(documents, metadatas, distances):
        # Cosine distance → similarity score (0 to 1)
        sim_score = max(0.0, 1.0 - dist)
        if sim_score > best_score:
            best_score = sim_score

        if sim_score >= settings.RAG_CONFIDENCE_THRESHOLD:
            relevant_count += 1

        formatted_results.append({
            "content": doc,
            "filename": meta.get("filename", "Unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "score": round(sim_score, 4),
        })

    # Confidence scoring
    if best_score > 0.75 and relevant_count >= 2:
        confidence = "High"
    elif best_score > 0.45 or relevant_count >= 1:
        confidence = "Medium"
    else:
        confidence = "Low"

    return formatted_results, confidence
