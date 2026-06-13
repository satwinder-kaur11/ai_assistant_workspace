"""
Centralized agent state definition for LangGraph.
Using a plain dict-compatible TypedDict for LangGraph compatibility.
"""
from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    """Represents the mutable state flowing through the LangGraph agent."""
    # Core context
    tenant_id: int
    user_id: int
    conversation_id: Optional[int]
    message_id: Optional[int]

    # Input
    current_query: str

    # Intent Detection outputs
    intent: str          # document_qa | memory_recall | task_creation | email_draft | multi_step | chitchat
    confidence: float
    reasoning: str

    # Planning (for multi-step)
    plan: List[Dict[str, str]]
    current_step: int

    # Execution outputs
    selected_tools: List[str]
    tool_outputs: Dict[str, Any]

    # RAG context
    retrieved_context: str
    citations: List[str]
    rag_confidence: str   # High | Medium | Low

    # Validation
    validation_passed: bool
    error_message: Optional[str]

    # Final response
    final_response: str
