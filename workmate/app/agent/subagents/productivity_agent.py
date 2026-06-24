"""
app/agent/subagents/productivity_agent.py

The Productivity Sub-Agent.
Handles all action-staging tasks: task extraction and email drafting.
Both operations use RAG context from uploaded documents where available.
Includes exponential backoff retry on transient failures.
"""

import time
import logging
from typing import Dict, Any

from app.tasks.task_service import create_tasks
from app.email.email_service import draft_email
from app.rag.rag_service import search_documents
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
                    f"productivity_agent: attempt {attempt}/{_MAX_ATTEMPTS} failed "
                    f"({exc}). Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"productivity_agent: all {_MAX_ATTEMPTS} internal attempts "
                    f"exhausted. Last error: {exc}"
                )
    raise last_exc


def productivity_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Productivity agent node.
    Looks at `intent` (set by the supervisor) to decide whether to
    create tasks or draft an email.

    Uses exponential backoff on transient failures. If all internal
    retries fail, sets `error_message` so the graph-level retry loop
    can route back to the supervisor.
    """
    logger.info("Node: productivity_agent")
    query     = state.get("current_query", "")
    tenant_id = state.get("tenant_id", 1)
    user_id   = state.get("user_id", 1)
    intent    = state.get("intent", "task_creation")

    db = SessionLocal()
    try:
        # Both task and email generation benefit from document context first.
        # RAG search failure is non-fatal — we gracefully degrade to raw query.
        try:
            results, _ = _run_with_backoff(search_documents, query, tenant_id, user_id)
            if results:
                source_text = "\n\n".join(
                    f"[Source: {r['filename']}]\n{r['content']}" for r in results
                )
                logger.info(
                    f"productivity_agent: enriching with RAG context "
                    f"from {len(results)} chunk(s)"
                )
            else:
                source_text = query
                logger.info("productivity_agent: no document context, using raw query")
        except Exception as rag_exc:
            # RAG unavailable — degrade gracefully, continue with raw query
            logger.warning(
                f"productivity_agent: RAG search failed ({rag_exc}), "
                "falling back to raw query"
            )
            source_text = query

        if intent == "email_draft":
            draft = _run_with_backoff(draft_email, source_text, tenant_id, user_id, db)
            state["tool_outputs"] = {"email": draft}
            logger.info("productivity_agent: email draft staged as pending approval")
        else:
            # default → task_creation
            tasks = _run_with_backoff(create_tasks, source_text, tenant_id, user_id, db)
            state["tool_outputs"] = {"tasks": tasks}
            logger.info(
                f"productivity_agent: {len(tasks)} task(s) staged as pending approval"
            )

    except Exception as exc:
        # All internal retries exhausted — signal graph-level retry loop
        logger.error(f"productivity_agent: unrecoverable error: {exc}", exc_info=True)
        state["error_message"] = str(exc)
    finally:
        db.close()

    return state
