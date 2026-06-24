"""
app/agent/subagents/research_agent.py

The Research Sub-Agent.
Handles all knowledge retrieval tasks: Document Q&A (RAG) and Memory Recall.
"""

import logging
from typing import Dict, Any

from app.rag.rag_service import search_documents
from app.memory.memory_retriever import search_memory
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


def research_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Research agent node.
    Looks at `intent` (set by the supervisor) to decide whether to
    run a RAG document search or a memory recall search.
    """
    logger.info("Node: research_agent")
    query     = state.get("current_query", "")
    tenant_id = state.get("tenant_id", 1)
    user_id   = state.get("user_id", 1)
    intent    = state.get("intent", "document_qa")

    db = SessionLocal()
    try:
        if intent == "memory_recall":
            results = search_memory(query, tenant_id, user_id, db=db)
            state["retrieved_context"] = "\n".join(r["content"] for r in results)
            logger.info(f"research_agent: memory recall returned {len(results)} record(s)")
        else:
            # default → document_qa
            results, confidence = search_documents(query, tenant_id, user_id)
            if results:
                context_parts = [
                    f"[Source: {r['filename']}, chunk {r['chunk_index']}]\n{r['content']}"
                    for r in results
                ]
                state["retrieved_context"] = "\n\n".join(context_parts)
            else:
                state["retrieved_context"] = ""
            state["rag_confidence"] = confidence
            logger.info(
                f"research_agent: RAG returned {len(results)} chunk(s) "
                f"with confidence={confidence}"
            )

    except Exception as exc:
        logger.error(f"research_agent error: {exc}", exc_info=True)
        state["error_message"] = str(exc)
    finally:
        db.close()

    return state
