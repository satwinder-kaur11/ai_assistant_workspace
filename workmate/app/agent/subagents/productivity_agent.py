"""
app/agent/subagents/productivity_agent.py

The Productivity Sub-Agent.
Handles all action-staging tasks: task extraction and email drafting.
Both operations use RAG context from uploaded documents where available.
"""

import logging
from typing import Dict, Any

from app.tasks.task_service import create_tasks
from app.email.email_service import draft_email
from app.rag.rag_service import search_documents
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


def productivity_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Productivity agent node.
    Looks at `intent` (set by the supervisor) to decide whether to
    create tasks or draft an email.
    """
    logger.info("Node: productivity_agent")
    query     = state.get("current_query", "")
    tenant_id = state.get("tenant_id", 1)
    user_id   = state.get("user_id", 1)
    intent    = state.get("intent", "task_creation")

    db = SessionLocal()
    try:
        # Both task and email generation benefit from document context first
        results, _ = search_documents(query, tenant_id, user_id)
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

        if intent == "email_draft":
            draft = draft_email(source_text, tenant_id, user_id, db)
            state["tool_outputs"] = {"email": draft}
            logger.info("productivity_agent: email draft staged as pending approval")
        else:
            # default → task_creation
            tasks = create_tasks(source_text, tenant_id, user_id, db)
            state["tool_outputs"] = {"tasks": tasks}
            logger.info(
                f"productivity_agent: {len(tasks)} task(s) staged as pending approval"
            )

    except Exception as exc:
        logger.error(f"productivity_agent error: {exc}", exc_info=True)
        state["error_message"] = str(exc)
    finally:
        db.close()

    return state
