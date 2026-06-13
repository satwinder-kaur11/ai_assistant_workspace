"""LangGraph StateGraph wiring for the WorkMate agent."""
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.nodes import (
    intent_detection,
    planning,
    execution,
    validation,
    guardrail_check,
    response_generation,
    error_handling,
)
from app.observability.tracing import trace_node


def _route_after_intent(state: Dict[str, Any]) -> str:
    """After intent detection, always go to planning."""
    return "planning"


def _route_after_validation(state: Dict[str, Any]) -> str:
    """After validation, route to error handling or appropriate response node."""
    if not state.get("validation_passed", True):
        return "error_handling"
    intent = state.get("intent", "chitchat")
    if intent in ("task_creation", "email_draft"):
        return "guardrail_check"
    return "response_generation"


def build_graph():
    """Build and compile the LangGraph StateGraph."""
    workflow = StateGraph(AgentState)

    # Register nodes (wrapped with tracing decorator)
    workflow.add_node("intent_detection",   trace_node(intent_detection))
    workflow.add_node("planning",           trace_node(planning))
    workflow.add_node("execution",          trace_node(execution))
    workflow.add_node("validation",         trace_node(validation))
    workflow.add_node("guardrail_check",    trace_node(guardrail_check))
    workflow.add_node("response_generation",trace_node(response_generation))
    workflow.add_node("error_handling",     trace_node(error_handling))

    # Entry point
    workflow.set_entry_point("intent_detection")

    # Edges
    workflow.add_conditional_edges("intent_detection", _route_after_intent)
    workflow.add_edge("planning",       "execution")
    workflow.add_edge("execution",      "validation")
    workflow.add_conditional_edges("validation", _route_after_validation)
    workflow.add_edge("guardrail_check",    "response_generation")
    workflow.add_edge("response_generation", END)
    workflow.add_edge("error_handling",      END)

    return workflow.compile()
