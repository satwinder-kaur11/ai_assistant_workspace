"""
app/agent/nodes.py

Contains the actual Python functions (nodes) that execute at each step of the LangGraph. 
It handles calling the LLM, triggering tools, and generating the final response.
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

# Centralized LLM factory
from app.agent.llm_factory import get_llm, has_active_llm

# Token counting & cost utilities
from app.observability.token_counter import count_tokens, empty_usage, add_usage

logger = logging.getLogger(__name__)


def _has_api_key() -> bool:
    return has_active_llm()


def _get_model_name() -> str:
    """Return a human-readable model identifier based on current LLM_PROVIDER setting."""
    provider = settings.LLM_PROVIDER.lower()
    if provider == "anthropic":
        return "claude-3-5-sonnet-20241022"
    elif provider == "ollama":
        return f"ollama/{getattr(settings, 'OLLAMA_MODEL', 'llama3')}"
    return "rule-based"


def _extract_llm_tokens(response, prompt_text: str, model_name: str) -> tuple[int, int]:
    """
    Extract real token counts from a LangChain LLM response object.
    Falls back to character-based estimation when metadata is unavailable.
    Returns (prompt_tokens, completion_tokens).
    """
    try:
        # LangChain / Anthropic responses expose usage_metadata
        meta = getattr(response, "usage_metadata", None)
        if meta:
            return (
                int(meta.get("input_tokens", 0)),
                int(meta.get("output_tokens", 0)),
            )
        # LangChain response_metadata (Ollama, OpenAI-compat)
        rmeta = getattr(response, "response_metadata", {})
        if rmeta:
            usage = rmeta.get("usage", {})
            if usage:
                return (
                    int(usage.get("prompt_tokens", 0)),
                    int(usage.get("completion_tokens", 0)),
                )
    except Exception:
        pass
    # Fallback: estimate from text lengths
    prompt_t = count_tokens(prompt_text)
    completion_t = count_tokens(getattr(response, "content", str(response)))
    return prompt_t, completion_t


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
    model_name = _get_model_name()

    # Initialise token_usage if not already present
    usage = state.get("token_usage") or empty_usage(model_name)

    if not _has_api_key():
        # ── Local fallback ───────────────────────────────────────────────
        logger.info("intent_detection: using local rule-based classifier (no API key)")
        result = local.classify_intent(query)
        state["intent"] = result["intent"]
        state["confidence"] = result["confidence"]
        state["reasoning"] = result["reasoning"]
        # Count tokens for the query even in fallback mode (cost = $0)
        prompt_t = count_tokens(INTENT_DETECTION_PROMPT + query)
        usage = add_usage(usage, prompt_t, 0, "rule-based")
        state["token_usage"] = usage
        return state

    # ── Claude / Ollama path ─────────────────────────────────────────────
    prompt_text = INTENT_DETECTION_PROMPT + query
    try:
        raw_llm = get_llm()
        llm = raw_llm.with_structured_output(IntentOutput)
        result: IntentOutput = llm.invoke([
            SystemMessage(content=INTENT_DETECTION_PROMPT),
            HumanMessage(content=query),
        ])
        state["intent"] = result.intent
        state["confidence"] = result.confidence
        state["reasoning"] = result.reasoning
        # Try to get real usage from the underlying response
        # (structured output wraps the raw response, so we call raw_llm directly for tokens)
        prompt_t = count_tokens(prompt_text)
        completion_t = count_tokens(str(result))
        usage = add_usage(usage, prompt_t, completion_t, model_name)
    except Exception as e:
        logger.error(f"Intent detection failed: {e}")
        result = local.classify_intent(query)
        state["intent"] = result["intent"]
        state["confidence"] = result["confidence"]
        state["reasoning"] = f"[LLM failed, using local fallback] {result['reasoning']}"
        state["error_message"] = str(e)
        prompt_t = count_tokens(prompt_text)
        usage = add_usage(usage, prompt_t, 0, "rule-based")

    state["token_usage"] = usage
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
            # Retrieve document context first for accurate task extraction
            results, _ = search_documents(query, tenant_id, user_id)
            if results:
                source_text = "\n\n".join(
                    f"[Source: {r['filename']}]\n{r['content']}" for r in results
                )
                logger.info(f"task_creation: Using RAG context from {len(results)} chunk(s)")
            else:
                source_text = query
                logger.info("task_creation: No document context found, using raw query")
            tasks = create_tasks(source_text, tenant_id, user_id, db)
            state["tool_outputs"] = {"tasks": tasks}

        elif intent == "email_draft":
            # Retrieve document context first for richer email drafting
            results, _ = search_documents(query, tenant_id, user_id)
            if results:
                context = "\n\n".join(
                    f"[Source: {r['filename']}]\n{r['content']}" for r in results
                )
                logger.info(f"email_draft: Using RAG context from {len(results)} chunk(s)")
            else:
                context = query
                logger.info("email_draft: No document context found, using raw query")
            draft = draft_email(context, tenant_id, user_id, db)
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
    model_name = _get_model_name()

    # Carry forward accumulated usage from previous nodes
    usage = state.get("token_usage") or empty_usage(model_name)

    try:
        if intent == "document_qa":
            context = state.get("retrieved_context", "")
            rag_confidence = state.get("rag_confidence", "")

            if use_local:
                # ── Local: keyword-based synthesis ──────────────────────
                response = local.generate_rag_response(
                    context, state.get("current_query", ""), rag_confidence
                )
                prompt_t = count_tokens(context + state.get("current_query", ""))
                completion_t = count_tokens(response)
                usage = add_usage(usage, prompt_t, completion_t, "rule-based")
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
                # Extract real token counts from the LLM response
                prompt_t, completion_t = _extract_llm_tokens(msg, prompt, model_name)
                usage = add_usage(usage, prompt_t, completion_t, model_name)
                if rag_confidence == "Medium":
                    response = "⚠️ *I'm not fully confident — please verify with the source.*\n\n" + response

            state["final_response"] = response

        elif intent == "memory_recall":
            context = state.get("retrieved_context", "")
            if not context:
                state["final_response"] = "I couldn't find any relevant memories for that query."
                usage = add_usage(usage, count_tokens(state.get("current_query", "")), 10, "rule-based")
            elif use_local:
                response = (
                    f"🧠 **Here's what I remember:**\n\n{context}\n\n"
                    "*These memories were extracted from your previous conversations.*"
                )
                state["final_response"] = response
                usage = add_usage(usage, count_tokens(context), count_tokens(response), "rule-based")
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
                prompt_t, completion_t = _extract_llm_tokens(msg, prompt, model_name)
                usage = add_usage(usage, prompt_t, completion_t, model_name)

        elif intent == "task_creation":
            tasks = state.get("tool_outputs", {}).get("tasks", [])
            n = len(tasks)
            task_list = "\n".join(
                f"  {i+1}. **{t.get('title','?')}** ({t.get('priority','?')}) — {t.get('owner','Unassigned')}"
                for i, t in enumerate(tasks)
            )
            response = (
                f"✅ I've drafted **{n} task(s)** for your review:\n{task_list}\n\n"
                "Please approve them in the **Pending Approvals** tab."
            )
            state["final_response"] = response
            usage = add_usage(usage, 0, count_tokens(response), "rule-based")

        elif intent == "email_draft":
            email = state.get("tool_outputs", {}).get("email", {})
            subject = email.get("subject", "N/A")
            response = (
                f"✉️ Email draft ready — Subject: **{subject}**\n\n"
                "Please review and approve it in the **Pending Approvals** tab."
            )
            state["final_response"] = response
            usage = add_usage(usage, 0, count_tokens(response), "rule-based")

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
                response = local.generate_chitchat(
                    state.get("current_query", ""), mem_text or "No memories yet."
                )
                state["final_response"] = response
                prompt_t = count_tokens(state.get("current_query", "") + mem_text)
                completion_t = count_tokens(response)
                usage = add_usage(usage, prompt_t, completion_t, "rule-based")
            else:
                from langchain_core.messages import SystemMessage
                llm = get_llm(temperature=0.3)
                prompt = CHITCHAT_PROMPT.format(
                    memories=mem_text or "No memories yet.",
                    query=state.get("current_query", ""),
                )
                msg = llm.invoke([SystemMessage(content=prompt)])
                state["final_response"] = msg.content
                prompt_t, completion_t = _extract_llm_tokens(msg, prompt, model_name)
                usage = add_usage(usage, prompt_t, completion_t, model_name)

    except Exception as e:
        logger.error(f"Response generation error: {e}", exc_info=True)
        state["final_response"] = (
            "I encountered an error while generating a response. Please try again."
        )

    state["token_usage"] = usage
    return state


def error_handling(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gracefully handle validation failures and tool errors.
    Increments retry_count so the graph can decide whether to retry
    via the supervisor or give up and return a final error message.
    """
    logger.info("Node: error_handling")
    err = state.get("error_message", "An unknown error occurred.")

    # ── Increment retry counter ──────────────────────────────────────────
    retry_count = state.get("retry_count", 0) + 1
    max_retries = state.get("max_retries", 3)
    state["retry_count"] = retry_count

    if retry_count <= max_retries:
        # Still have retries left — clear the error so the graph can loop back
        logger.warning(
            f"error_handling: attempt {retry_count}/{max_retries} failed — "
            f"retrying via supervisor. Error: {err}"
        )
        # Reset transient error state so supervisor gets a clean slate
        state["error_message"]      = None
        state["validation_passed"]  = True
        state["retrieved_context"]  = ""
        state["tool_outputs"]       = {}
        # Keep final_response empty so the user does not see a partial error
        state["final_response"] = ""
        return state

    # ── All retries exhausted — return a human-readable error message ─────
    logger.error(
        f"error_handling: all {max_retries} retries exhausted. "
        f"Returning terminal error to user."
    )
    if "Retrieval failure" in err or "low confidence" in (err or "").lower():
        state["final_response"] = (
            "I couldn't find relevant information in your documents.\n"
            "Try uploading a document that covers this topic, then ask again."
        )
    elif "empty results" in (err or "").lower():
        state["final_response"] = (
            "I wasn't able to complete that action — the required context was missing.\n"
            "Could you provide more detail?"
        )
    else:
        state["final_response"] = (
            f"Something went wrong after {max_retries} attempts.\n"
            f"Details: {err}\n\n"
            "Please try rephrasing or try again in a moment."
        )

    return state
