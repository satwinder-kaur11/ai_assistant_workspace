"""
app/agent/subagents/research_agent.py

The Research Sub-Agent.
Handles all knowledge retrieval tasks: Document Q&A (RAG) and Memory Recall.
Includes exponential backoff retry on transient failures (network blips, DB locks).
"""

import time
import logging
from typing import Dict, Any

from app.rag.rag_service import search_documents
from app.memory.memory_retriever import search_memory
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

# Maximum attempts per sub-agent call (independent of the graph-level retry loop)
_MAX_ATTEMPTS  = 3
_BASE_DELAY_S  = 1.0   # seconds — doubles on each retry (1s, 2s, 4s)


def _run_with_backoff(fn, *args, **kwargs):
    """
    Call `fn(*args, **kwargs)` with exponential backoff.
    Raises the last exception if all attempts fail.
    """
    last_exc = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_ATTEMPTS:
                delay = _BASE_DELAY_S * (2 ** (attempt - 1))
                logger.warning(
                    f"research_agent: attempt {attempt}/{_MAX_ATTEMPTS} failed "
                    f"({exc}). Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"research_agent: all {_MAX_ATTEMPTS} internal attempts "
                    f"exhausted. Last error: {exc}"
                )
    raise last_exc


def research_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Research agent node.
    Looks at `intent` (set by the supervisor) to decide whether to
    run a RAG document search or a memory recall search.

    Uses exponential backoff on transient failures. If all internal
    retries fail, sets `error_message` so the graph-level retry loop
    can route back to the supervisor.
    """
    logger.info("Node: research_agent")
    query     = state.get("current_query", "")
    tenant_id = state.get("tenant_id", 1)
    user_id   = state.get("user_id", 1)
    intent    = state.get("intent", "document_qa")

    db = SessionLocal()
    try:
        if intent == "memory_recall":
            results = _run_with_backoff(
                search_memory, query, tenant_id, user_id, db=db
            )
            state["retrieved_context"] = "\n".join(r["content"] for r in results)
            logger.info(
                f"research_agent: memory recall returned {len(results)} record(s)"
            )
        else:
            # default → document_qa
            results, confidence = _run_with_backoff(
                search_documents, query, tenant_id, user_id
            )
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
        # All internal retries exhausted — signal graph-level retry loop
        logger.error(f"research_agent: unrecoverable error: {exc}", exc_info=True)
        state["error_message"] = str(exc)
    finally:
        db.close()

    return state
