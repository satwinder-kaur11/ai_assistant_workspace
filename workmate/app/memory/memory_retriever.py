"""
app/memory/memory_retriever.py

Searches the ChromaDB vector store to recall relevant past memories when answering a user query.
"""

# Fetches past memories to give the agent context about the user.
import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.vectorstore.chroma_client import get_memory_collection
from app.ingestion.embedder import get_embedding_model
from app.memory.memory_store import update_last_accessed

logger = logging.getLogger(__name__)

def search_memory(query: str, tenant_id: int, user_id: int, db: Session = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Searches for relevant memories in Chroma for a specific user.
    Updates last_accessed_at in the relational DB if session provided.
    """
    collection = get_memory_collection(tenant_id)
    embedder = get_embedding_model()
    
    try:
        query_embedding = embedder.embed_query(query)
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return []
        
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"user_id": user_id},
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        logger.error(f"Error querying Chroma for memory: {e}")
        return []
        
    if not results or not results["documents"] or not results["documents"][0]:
        return []
        
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    
    formatted_results = []
    
    for doc, meta, dist in zip(documents, metadatas, distances):
        memory_id = meta.get("memory_id")
        
        # Update last accessed time
        if db and memory_id:
            try:
                update_last_accessed(db, memory_id)
            except Exception as e:
                logger.error(f"Failed to update last_accessed_at for memory {memory_id}: {e}")
                
        formatted_results.append({
            "content": doc,
            "type": meta.get("type", "semantic"),
            "importance_score": meta.get("importance_score", 0.0),
            "distance": dist
        })
        
    # Hybrid retrieval ranking: distance (lower is better), importance (higher is better)
    # Simple combined score: (1.0 / (1.0 + distance)) * importance_score (or additive)
    for res in formatted_results:
        sim = 1.0 / (1.0 + res["distance"])
        res["hybrid_score"] = (sim * 0.7) + (res["importance_score"] * 0.3)
        
    # Re-sort by hybrid score
    formatted_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
    
    return formatted_results
