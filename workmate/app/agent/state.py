"""
app/agent/state.py

Defines the `AgentState` dictionary structure. This state object is passed from node to node 
throughout the LangGraph execution, carrying the user query, tool outputs, and token counts.
"""

# Defines the "State" dictionary that gets passed between nodes in the graph
#  (e.g., current query, detected intent, tool outputs, final response).
from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    """Represents the mutable state flowing through the LangGraph agent."""
    # Multi-Agent Routing: which sub-agent the supervisor selected
    next_agent: str  # research_agent | productivity_agent | chitchat_agent

    # Fault-tolerance: retry loop counter
    retry_count: int   # how many retries have been attempted so far
    max_retries: int   # maximum retries allowed before giving up (default 3)

    # Core context
    tenant_id: int
    user_id: int
    conversation_id: Optional[int]
    message_id: Optional[int]

    # Input
    current_query: str

    # Intent Detection outputs (kept for compatibility with validation/response nodes)
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

    # Token usage — accumulated across all LLM calls in this turn
    token_usage: Dict[str, Any]   # {prompt_tokens, completion_tokens, model_name}
