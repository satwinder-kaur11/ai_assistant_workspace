"""
app/tasks/task_service.py

Uses the LLM to extract actionable tasks from text and stages them as pending approvals in the DB.
"""

#  Extracts action items from text/documents. 
# Instead of creating them directly, it saves an ActionLog to SQLite for human approval.
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session
from app.db.models import ActionLog, ActionStatus
from app.agent import local_llm as local
from app.agent.llm_factory import get_llm, has_active_llm

logger = logging.getLogger(__name__)

# ── Echo detection ─────────────────────────────────────────────────────────────
_ECHO_PHRASES = [
    "create high priority", "create tasks", "extract tasks",
    "based on the meeting", "based on the notes", "all project",
    "deliverables based", "actionable tasks from",
]

def _is_echoed(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in _ECHO_PHRASES)


class TaskItem(BaseModel):
    title: str = Field(
        description="Short, specific task title (e.g. 'Deploy auth module to staging'). "
                    "MUST be derived from document facts. NEVER echo the user instruction."
    )
    description: str = Field(
        description="One or two sentences expanding on what needs to be done, "
                    "taken strictly from the source document. No meta-language."
    )
    priority: str = Field(
        description="High, Medium, or Low — inferred from urgency signals in the document."
    )
    owner: str = Field(
        description="Person assigned in the document. Default to 'Unassigned' if not mentioned."
    )

    @model_validator(mode="after")
    def no_echo_in_fields(self):
        # Log a warning but don't raise — prevents pydantic from dropping
        # the entire task list when the LLM slightly echoes wording.
        if _is_echoed(self.title):
            logger.warning(
                f"Task title may echo user instruction: '{self.title}'. "
                "Will be caught by echo guard if all tasks are affected."
            )
        if _is_echoed(self.description):
            logger.warning(
                f"Task description may echo user instruction: '{self.description}'."
            )
        return self


class TaskList(BaseModel):
    tasks: List[TaskItem] = Field(
        description="List of concrete, individual tasks extracted from the document."
    )


# ── Prompt (system + human split for Ollama compatibility) ────────────────────
_SYSTEM_PROMPT = """You are a project management assistant that extracts tasks from documents.

HARD RULES:
1. Read the DOCUMENT CONTEXT and extract each concrete, actionable item as a separate task.
2. Each task title must name a SPECIFIC action (verb + object), e.g. "Deploy auth service to staging".
3. NEVER write a title or description that mirrors or paraphrases the user's instruction.
4. NEVER produce a single generic task summarising everything — split into individual tasks.
5. Priority must be inferred from words like "urgent", "blocker", "by EOD", "critical" in the document.
6. If no owner is named in the document, set owner to "Unassigned".
"""

_USER_TEMPLATE = """--- DOCUMENT CONTEXT START ---
{source_text}
--- DOCUMENT CONTEXT END ---

Extract every individual actionable task from the document above.
Return ONLY tasks that are explicitly stated or clearly implied in the document.
Do NOT summarise. Do NOT echo this instruction in the output."""


def _invoke_structured(llm, source_text: str) -> TaskList:
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", _USER_TEMPLATE),
    ])
    chain = prompt | llm.with_structured_output(TaskList)
    return chain.invoke({"source_text": source_text})


def _invoke_pydantic_parser(llm, source_text: str) -> TaskList:
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    parser = PydanticOutputParser(pydantic_object=TaskList)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT + "\n\nOutput format:\n{format_instructions}"),
        ("human", _USER_TEMPLATE),
    ]).partial(format_instructions=parser.get_format_instructions())
    chain = prompt | llm | parser
    return chain.invoke({"source_text": source_text})


def _llm_extract(source_text: str) -> List[Dict[str, Any]]:
    """Run LLM pipeline with structured output → pydantic parser fallback + echo retry.
    Short-circuits to local rule-based extraction when no active LLM is configured.
    """
    if not has_active_llm():
        logger.info("_llm_extract: No active LLM configured — using local rule-based fallback.")
        return local.extract_tasks(source_text)

    llm = get_llm(temperature=0)  # Zero temp for deterministic task extraction

    try:
        result = _invoke_structured(llm, source_text)
    except Exception as e:
        logger.warning(f"Structured output failed ({e}), trying PydanticOutputParser.")
        result = _invoke_pydantic_parser(llm, source_text)

    # ── Echo guard: if ALL tasks look echoed, retry once ──────────────────
    echoed_count = sum(
        1 for t in result.tasks
        if _is_echoed(t.title) or _is_echoed(t.description)
    )
    if echoed_count == len(result.tasks) and result.tasks:
        logger.warning("All tasks appear echoed — retrying with reinforced prompt.")
        reinforced = (
            "[CORRECTION: Your previous output echoed the user’s instruction. "
            "This time read the document facts below and extract SPECIFIC tasks only.]\n\n"
            + source_text
        )
        try:
            result = _invoke_structured(llm, reinforced)
        except Exception:
            result = _invoke_pydantic_parser(llm, reinforced)

        # ── Second echo guard: still all echoed → local fallback ───────────
        echoed_count = sum(
            1 for t in result.tasks
            if _is_echoed(t.title) or _is_echoed(t.description)
        )
        if echoed_count == len(result.tasks) and result.tasks:
            logger.warning("Echo persists after retry — falling back to local rule-based extraction.")
            return local.extract_tasks(source_text)

    return [t.model_dump() for t in result.tasks]


def create_tasks(
    source_text: str,
    tenant_id: int,
    user_id: int,
    db: Session,
) -> List[Dict[str, Any]]:
    """
    Extracts actionable tasks from document text using LLM (cloud or local Ollama).
    Always creates an ActionLog for HITL approval — never writes Task records directly.

    Args:
        source_text: Extracted document content (NOT the raw user instruction).
        tenant_id:   Tenant identifier.
        user_id:     User initiating the action.
        db:          SQLAlchemy session.

    Returns:
        List of task dicts with keys: title, description, priority, owner.
    """
    tasks_data: List[Dict[str, Any]] = []

    try:
        tasks_data = _llm_extract(source_text)
        logger.info(f"create_tasks: LLM extracted {len(tasks_data)} task(s).")
    except Exception as e:
        logger.error(f"create_tasks: LLM pipeline failed entirely, using local fallback. Error: {e}")
        tasks_data = local.extract_tasks(source_text)

    if not tasks_data:
        return []

    action_log = ActionLog(
        tenant_id=tenant_id,
        user_id=user_id,
        action_type="create_tasks",
        payload_json={"tasks": tasks_data},
        status=ActionStatus.pending_approval,
    )
    db.add(action_log)
    db.commit()
    db.refresh(action_log)

    return tasks_data