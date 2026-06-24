"""
app/agent/graph.py

Defines the LangGraph architecture: Supervisor routes to one of three
specialist sub-agents, then validation + response generation complete the turn.

Architecture:
    Supervisor
      ├─► research_agent     (Document Q&A / Memory Recall)
      ├─► productivity_agent (Task Creation / Email Drafting)
      └─► chitchat_agent     (falls through directly to validation)
           └─► validation → guardrail_check ─► response_generation
                         └─► error_handling
                               ├─► [retries left] → supervisor (loop back)
                               └─► [exhausted]    → END
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.supervisor import supervisor_node, AGENT_RESEARCH, AGENT_PRODUCTIVITY
from app.agent.subagents.research_agent import research_agent_node
from app.agent.subagents.productivity_agent import productivity_agent_node
from app.agent.nodes import (
    validation,
    guardrail_check,
    response_generation,
    error_handling,
)
from app.observability.tracing import trace_node


# ── Routing functions ────────────────────────────────────────────────────────

def _route_after_supervisor(state: Dict[str, Any]) -> str:
    """Route from the supervisor to the appropriate sub-agent."""
    next_agent = state.get("next_agent", "chitchat_agent")
    if next_agent == AGENT_RESEARCH:
        return "research_agent"
    if next_agent == AGENT_PRODUCTIVITY:
        return "productivity_agent"
    # chitchat_agent: skip straight to validation (no tool call needed)
    return "validation"


def _route_after_validation(state: Dict[str, Any]) -> str:
    """After validation, route to error-handling or the correct response node."""
    if not state.get("validation_passed", True):
        return "error_handling"
    intent = state.get("intent", "chitchat")
    if intent in ("task_creation", "email_draft"):
        return "guardrail_check"
    return "response_generation"


def _route_after_error(state: Dict[str, Any]) -> str:
    """
    After error_handling, decide whether to retry (loop back to supervisor)
    or give up and send the terminal error message to the user.

    error_handling already incremented retry_count. If it reset error_message
    to None that means retries remain; otherwise retries are exhausted.
    """
    if state.get("error_message") is None:
        # error_handling cleared the error — retries remain, loop back
        return "supervisor"
    # error_message is still set — retries exhausted, send final error
    return "response_generation"


# ── Graph builder ────────────────────────────────────────────────────────────

def build_graph():
    """Build and compile the multi-agent LangGraph StateGraph."""
    workflow = StateGraph(AgentState)

    # ── Register all nodes ────────────────────────────────────────────────
    workflow.add_node("supervisor",          trace_node(supervisor_node))
    workflow.add_node("research_agent",      trace_node(research_agent_node))
    workflow.add_node("productivity_agent",  trace_node(productivity_agent_node))
    workflow.add_node("validation",          trace_node(validation))
    workflow.add_node("guardrail_check",     trace_node(guardrail_check))
    workflow.add_node("response_generation", trace_node(response_generation))
    workflow.add_node("error_handling",      trace_node(error_handling))

    # ── Entry point ───────────────────────────────────────────────────────
    workflow.set_entry_point("supervisor")

    # ── Edges ─────────────────────────────────────────────────────────────
    workflow.add_conditional_edges("supervisor",         _route_after_supervisor)
    workflow.add_edge("research_agent",     "validation")
    workflow.add_edge("productivity_agent", "validation")
    workflow.add_conditional_edges("validation",         _route_after_validation)
    workflow.add_edge("guardrail_check",     "response_generation")
    workflow.add_edge("response_generation", END)
    # ── Retry loop: error_handling → supervisor (if retries remain) or END ───
    workflow.add_conditional_edges("error_handling",  _route_after_error)

    return workflow.compile()
