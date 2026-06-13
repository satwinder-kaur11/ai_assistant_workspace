"""
All LangGraph node functions for the WorkMate agent.
Each node receives the full state dict and returns an updated state dict.

When ANTHROPIC_API_KEY is set:  uses Claude (full AI experience)
When no API key is set:         uses local rule-based fallback (no internet required)
"""
import logging
from typing import Dict, Any

from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

from app.agent.prompts import INTENT_DETECTION_PROMPT, RAG_RESPONSE_PROMPT, CHITCHAT_PROMPT
from app.config import settings
from app.rag.rag_service import search_documents
from app.memory.memory_retriever import search_memory
from app.tasks.task_service import create_tasks
from app.email.email_service import draft_email
from app.db.session import SessionLocal

# Local fallback engine (no API key required)
from app.agent import local_llm as local

logger = logging.getLogger(__name__)


def _has_api_key() -> bool:
    """Check if a valid Anthropic API key is configured."""
    key = (settings.ANTHROPIC_API_KEY or "").strip()
    return bool(key) and key != "your-anthropic-api-key-here"


def get_llm(temperature: float = 0):
    """Returns a configured ChatAnthropic instance (only if API key is set)."""
    if not _has_api_key():
        raise RuntimeError("No ANTHROPIC_API_KEY configured — using local fallback.")
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=settings.ANTHROPIC_API_KEY,
        temperature=temperature,
        max_tokens=4096,
    )


# ── Pydantic models for structured LLM output ──────────────────────────────

class IntentOutput(BaseModel):
    intent: str
    confidence: float
    reasoning: str


# ── Node functions ──────────────────────────────────────────────────────────

def intent_detection(state: Dict[str, Any]) -> Dict[str, Any]:
    """Classify the user's query into a known intent."""
    logger.info("Node: intent_detection")
    query = state.get("current_query", "")

    if not _has_api_key():
        # ── Local fallback ───────────────────────────────────────────────
        logger.info("intent_detection: using local rule-based classifier (no API key)")
        result = local.classify_intent(query)
        state["intent"] = result["intent"]
        state["confidence"] = result["confidence"]
        state["reasoning"] = result["reasoning"]
        return state

    # ── Claude path ──────────────────────────────────────────────────────
    try:
        llm = get_llm().with_structured_output(IntentOutput)
        result: IntentOutput = llm.invoke([
            SystemMessage(content=INTENT_DETECTION_PROMPT),
            HumanMessage(content=query),
        ])
        state["intent"] = result.intent
        state["confidence"] = result.confidence
        state["reasoning"] = result.reasoning
    except Exception as e:
        logger.error(f"Intent detection failed: {e}")
        # Graceful fallback even with a key (network issues, quota, etc.)
        result = local.classify_intent(query)
        state["intent"] = result["intent"]
        state["confidence"] = result["confidence"]
        state["reasoning"] = f"[LLM failed, using local fallback] {result['reasoning']}"
        state["error_message"] = str(e)

    return state


def planning(state: Dict[str, Any]) -> Dict[str, Any]:
    """For multi-step queries, decompose into ordered sub-tasks."""
    logger.info("Node: planning")
    intent = state.get("intent", "chitchat")
    query = state.get("current_query", "")

    if intent == "multi_step":
        # Simplified decomposition: treat as chitchat for prototype
        state["plan"] = [{"step": 1, "intent": "chitchat", "query": query}]
    else:
        state["plan"] = [{"step": 1, "intent": intent, "query": query}]

    state["current_step"] = 0
    return state


def execution(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run the appropriate tool based on detected intent."""
    logger.info("Node: execution")
    intent = state.get("intent", "chitchat")
    query = state.get("current_query", "")
    tenant_id = state.get("tenant_id", 1)
    user_id = state.get("user_id", 1)

    db = SessionLocal()
    try:
        if intent == "document_qa":
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

        elif intent == "memory_recall":
            results = search_memory(query, tenant_id, user_id, db=db)
            state["retrieved_context"] = "\n".join(r["content"] for r in results)

        elif intent == "task_creation":
            tasks = create_tasks(query, tenant_id, user_id, db)
            state["tool_outputs"] = {"tasks": tasks}

        elif intent == "email_draft":
            draft = draft_email(query, tenant_id, user_id, db)
            state["tool_outputs"] = {"email": draft}

        # chitchat and multi_step fall through to response_generation directly

    except Exception as e:
        logger.error(f"Execution error: {e}", exc_info=True)
        state["error_message"] = str(e)
    finally:
        db.close()

    return state


def validation(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate tool outputs before passing to response generation."""
    logger.info("Node: validation")
    intent = state.get("intent", "chitchat")

    # If an error already set, fail immediately
    if state.get("error_message") and intent not in ("chitchat",):
        state["validation_passed"] = False
        return state

    state["validation_passed"] = True

    if intent == "document_qa":
        if not state.get("retrieved_context") or state.get("rag_confidence") == "Low":
            state["validation_passed"] = False
            state["error_message"] = "Retrieval failure or low confidence."

    elif intent in ("task_creation", "email_draft"):
        if not state.get("tool_outputs"):
            state["validation_passed"] = False
            state["error_message"] = "Tool returned empty results."

    return state


def guardrail_check(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    HITL checkpoint: ActionLog entries are already persisted by the service
    with status=pending_approval. Here we just log and pass through.
    """
    logger.info("Node: guardrail_check — action awaiting human approval in UI.")
    return state


def response_generation(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compose the final natural-language response to send to the user."""
    logger.info("Node: response_generation")
    intent = state.get("intent", "chitchat")
    use_local = not _has_api_key()

    try:
        if intent == "document_qa":
            context = state.get("retrieved_context", "")
            rag_confidence = state.get("rag_confidence", "")

            if use_local:
                # ── Local: keyword-based synthesis ──────────────────────
                response = local.generate_rag_response(
                    context, state.get("current_query", ""), rag_confidence
                )
            else:
                # ── Claude: LLM synthesis ────────────────────────────────
                from langchain_core.messages import SystemMessage
                llm = get_llm(temperature=0.3)
                prompt = RAG_RESPONSE_PROMPT.format(
                    context=context,
                    query=state.get("current_query", ""),
                )
                msg = llm.invoke([SystemMessage(content=prompt)])
                response = msg.content
                if rag_confidence == "Medium":
                    response = "⚠️ *I'm not fully confident — please verify with the source.*\n\n" + response

            state["final_response"] = response

        elif intent == "memory_recall":
            context = state.get("retrieved_context", "")
            if not context:
                state["final_response"] = "I couldn't find any relevant memories for that query."
            elif use_local:
                state["final_response"] = (
                    f"🧠 **Here's what I remember:**\n\n{context}\n\n"
                    "*These memories were extracted from your previous conversations.*"
                )
            else:
                from langchain_core.messages import HumanMessage
                llm = get_llm(temperature=0.3)
                prompt = (
                    f"Using these memories as context, answer the user's question:\n\n"
                    f"Memories:\n{context}\n\n"
                    f"User: {state.get('current_query', '')}"
                )
                msg = llm.invoke([HumanMessage(content=prompt)])
                state["final_response"] = msg.content

        elif intent == "task_creation":
            tasks = state.get("tool_outputs", {}).get("tasks", [])
            n = len(tasks)
            task_list = "\n".join(
                f"  {i+1}. **{t.get('title','?')}** ({t.get('priority','?')}) — {t.get('owner','Unassigned')}"
                for i, t in enumerate(tasks)
            )
            state["final_response"] = (
                f"✅ I've drafted **{n} task(s)** for your review:\n{task_list}\n\n"
                "Please approve them in the **Pending Approvals** tab."
            )

        elif intent == "email_draft":
            email = state.get("tool_outputs", {}).get("email", {})
            subject = email.get("subject", "N/A")
            state["final_response"] = (
                f"✉️ Email draft ready — Subject: **{subject}**\n\n"
                "Please review and approve it in the **Pending Approvals** tab."
            )

        else:  # chitchat / multi_step / fallback
            tenant_id = state.get("tenant_id", 1)
            user_id = state.get("user_id", 1)
            db = SessionLocal()
            try:
                memories = search_memory(
                    state.get("current_query", ""), tenant_id, user_id, db=db, top_k=3
                )
            except Exception:
                memories = []
            finally:
                db.close()

            mem_text = "\n".join(m["content"] for m in memories)

            if use_local:
                state["final_response"] = local.generate_chitchat(
                    state.get("current_query", ""), mem_text or "No memories yet."
                )
            else:
                from langchain_core.messages import SystemMessage
                llm = get_llm(temperature=0.3)
                prompt = CHITCHAT_PROMPT.format(
                    memories=mem_text or "No memories yet.",
                    query=state.get("current_query", ""),
                )
                msg = llm.invoke([SystemMessage(content=prompt)])
                state["final_response"] = msg.content

    except Exception as e:
        logger.error(f"Response generation error: {e}", exc_info=True)
        state["final_response"] = (
            "I encountered an error while generating a response. Please try again."
        )

    return state


def error_handling(state: Dict[str, Any]) -> Dict[str, Any]:
    """Gracefully handle validation failures and tool errors."""
    logger.info("Node: error_handling")
    err = state.get("error_message", "An unknown error occurred.")
    intent = state.get("intent", "")

    if "Retrieval failure" in err or "low confidence" in err.lower():
        state["final_response"] = (
            "🔍 I couldn't find relevant information in your documents.\n"
            "Try uploading a document that covers this topic, then ask again."
        )
    elif "empty results" in err.lower():
        state["final_response"] = (
            "I wasn't able to complete that action — the required context was missing.\n"
            "Could you provide more detail?"
        )
    else:
        state["final_response"] = (
            f"⚠️ Something went wrong while processing your request.\n"
            f"*Details:* {err}\n\n"
            "Please try rephrasing or try again in a moment."
        )

    return state
