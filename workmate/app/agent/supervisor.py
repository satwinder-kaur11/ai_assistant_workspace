"""
app/agent/supervisor.py

The Supervisor Agent. Routes the user's request to the correct sub-agent
by asking the LLM a simple routing question and parsing its plain-text reply.

This deliberately avoids .with_structured_output() so that it works with
both Anthropic (cloud) AND Ollama (local) models without modification.
"""

import re
import logging
from typing import Dict, Any

from app.agent.llm_factory import has_active_llm, get_llm
from app.observability.token_counter import count_tokens, empty_usage, add_usage

logger = logging.getLogger(__name__)

# The three sub-agents this supervisor can route to
AGENT_RESEARCH    = "research_agent"
AGENT_PRODUCTIVITY = "productivity_agent"
AGENT_CHITCHAT    = "chitchat_agent"

SUPERVISOR_SYSTEM_PROMPT = """You are a routing supervisor for an AI workspace assistant.
Read the user's message and decide which specialist agent should handle it.

Reply with EXACTLY ONE of these three words and nothing else:
- research_agent     → user wants to SEARCH or READ documents/files, or RECALL a past memory/preference
- productivity_agent → user wants to CREATE tasks/action items/todos, OR DRAFT/WRITE/COMPOSE an email
- chitchat_agent     → user is greeting, asking what you can do, or just having a casual conversation

IMPORTANT RULES:
- "Extract action items", "create tasks", "make a task list" → ALWAYS productivity_agent
- "What does the document say", "search my files", "find in PDF" → ALWAYS research_agent
- "Draft an email", "write an email", "compose a message" → ALWAYS productivity_agent
- "What do you remember", "what did I say" → ALWAYS research_agent
- "Hello", "how are you", "what can you do" → ALWAYS chitchat_agent

User message: {query}

Your answer (one word only):"""


def _parse_agent_from_text(text: str) -> str:
    """Extract the agent name from LLM plain-text response."""
    text = text.strip().lower()
    if "research" in text:
        return AGENT_RESEARCH
    if "productivity" in text:
        return AGENT_PRODUCTIVITY
    # Default to chitchat for any other response
    return AGENT_CHITCHAT


def _local_route(query: str) -> tuple[str, str]:
    """
    Rule-based fallback routing using the existing local classifier.
    Returns (next_agent, intent).
    """
    from app.agent import local_llm as local
    result = local.classify_intent(query)
    intent = result["intent"]

    if intent in ("document_qa", "memory_recall"):
        return AGENT_RESEARCH, intent
    if intent in ("task_creation", "email_draft"):
        return AGENT_PRODUCTIVITY, intent
    return AGENT_CHITCHAT, intent


def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supervisor node: classifies the user query and sets `next_agent` and `intent`
    so that the LangGraph conditional edge knows which sub-agent to invoke.
    """
    logger.info("Node: supervisor")
    query = state.get("current_query", "")
    usage = state.get("token_usage") or empty_usage("supervisor")

    # ── No API key: rule-based routing ──────────────────────────────────────
    if not has_active_llm():
        logger.info("supervisor: using local rule-based router (no API key)")
        next_agent, intent = _local_route(query)
        state["next_agent"] = next_agent
        state["intent"]     = intent
        state["reasoning"]  = "Local fallback routing"
        state["confidence"] = 0.70
        prompt_t = count_tokens(query)
        state["token_usage"] = add_usage(usage, prompt_t, 0, "rule-based")
        return state

    # ── Pre-check: high-confidence local routing for task/email keywords ─
    # This prevents the LLM from misrouting obvious action-item/email queries.
    local_next, local_intent = _local_route(query)
    from app.agent import local_llm as local
    local_result = local.classify_intent(query)
    if local_result["confidence"] >= 0.82 and local_intent in (
        "task_creation", "email_draft"
    ):
        logger.info(
            f"supervisor: high-confidence local pre-check → {local_next} "
            f"(confidence={local_result['confidence']}), skipping LLM"
        )
        state["next_agent"]  = local_next
        state["intent"]      = local_intent
        state["reasoning"]   = "Local high-confidence pre-check"
        state["token_usage"] = add_usage(usage, count_tokens(query), 0, "rule-based")
        return state

    # ── LLM-based routing (Anthropic or Ollama) ──────────────────────────────
    prompt = SUPERVISOR_SYSTEM_PROMPT.format(query=query)
    try:
        llm = get_llm(temperature=0)
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_text = response.content if hasattr(response, "content") else str(response)

        next_agent = _parse_agent_from_text(raw_text)

        # Derive the intent from the agent choice so downstream nodes stay compatible
        if next_agent == AGENT_RESEARCH:
            # Refine: is it a memory recall or document QA?
            _, intent = _local_route(query)
            if intent not in ("document_qa", "memory_recall"):
                intent = "document_qa"
        elif next_agent == AGENT_PRODUCTIVITY:
            _, intent = _local_route(query)
            if intent not in ("task_creation", "email_draft"):
                intent = "task_creation"
        else:
            intent = "chitchat"

        prompt_t   = count_tokens(prompt)
        completion_t = count_tokens(raw_text)
        usage = add_usage(usage, prompt_t, completion_t, "supervisor")

        logger.info(f"supervisor: routed → {next_agent} (intent={intent})")

    except Exception as exc:
        logger.error(f"supervisor: LLM routing failed ({exc}), falling back to local")
        next_agent, intent = _local_route(query)

    state["next_agent"]  = next_agent
    state["intent"]      = intent
    state["token_usage"] = usage
    return state
