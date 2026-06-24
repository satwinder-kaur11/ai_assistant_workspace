"""
tests/test_supervisor.py

Unit tests for the new multi-agent Supervisor node.
Tests routing logic without needing any LLM API key (uses local fallback).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.state import AgentState
from app.agent.supervisor import (
    supervisor_node,
    AGENT_RESEARCH,
    AGENT_PRODUCTIVITY,
    AGENT_CHITCHAT,
    _local_route,
    _parse_agent_from_text,
)


# ── Unit tests for helper functions ──────────────────────────────────────────

def test_parse_agent_research():
    assert _parse_agent_from_text("research_agent") == AGENT_RESEARCH

def test_parse_agent_productivity():
    assert _parse_agent_from_text("productivity_agent") == AGENT_PRODUCTIVITY

def test_parse_agent_chitchat():
    assert _parse_agent_from_text("chitchat_agent") == AGENT_CHITCHAT

def test_parse_agent_default_unknown():
    assert _parse_agent_from_text("I don't know") == AGENT_CHITCHAT


# ── Unit tests for local routing ─────────────────────────────────────────────

def test_local_route_document_qa():
    next_agent, intent = _local_route("What does the document say about the launch date?")
    assert next_agent == AGENT_RESEARCH
    assert intent == "document_qa"

def test_local_route_memory_recall():
    next_agent, intent = _local_route("What was my timezone preference?")
    assert next_agent == AGENT_RESEARCH
    assert intent == "memory_recall"

def test_local_route_task_creation():
    next_agent, intent = _local_route("Extract all action items from this meeting transcript.")
    assert next_agent == AGENT_PRODUCTIVITY
    assert intent == "task_creation"

def test_local_route_email_draft():
    next_agent, intent = _local_route("Draft an email to John about the project.")
    assert next_agent == AGENT_PRODUCTIVITY
    assert intent == "email_draft"

def test_local_route_chitchat():
    next_agent, intent = _local_route("Hello, how are you?")
    assert next_agent == AGENT_CHITCHAT
    assert intent == "chitchat"


# ── Integration tests for full supervisor_node ───────────────────────────────

def test_supervisor_node_sets_next_agent():
    state = AgentState(current_query="What is in the uploaded PDF?")
    result = supervisor_node(state)
    assert "next_agent" in result
    assert result["next_agent"] in (AGENT_RESEARCH, AGENT_PRODUCTIVITY, AGENT_CHITCHAT)

def test_supervisor_node_sets_intent():
    state = AgentState(current_query="Create tasks from this text.")
    result = supervisor_node(state)
    assert "intent" in result
    assert result["intent"] != ""

def test_supervisor_node_routes_tasks_to_productivity():
    state = AgentState(current_query="Please create tasks from the meeting notes.")
    result = supervisor_node(state)
    assert result["next_agent"] == AGENT_PRODUCTIVITY

def test_supervisor_node_routes_email_to_productivity():
    state = AgentState(current_query="Draft an email to Sarah summarizing the sprint.")
    result = supervisor_node(state)
    assert result["next_agent"] == AGENT_PRODUCTIVITY

def test_supervisor_node_routes_document_to_research():
    state = AgentState(current_query="What is the deadline according to the project plan?")
    result = supervisor_node(state)
    assert result["next_agent"] == AGENT_RESEARCH

def test_supervisor_node_routes_greeting_to_chitchat():
    state = AgentState(current_query="Hello there!")
    result = supervisor_node(state)
    assert result["next_agent"] == AGENT_CHITCHAT
