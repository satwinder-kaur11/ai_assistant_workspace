import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.config import settings
from app.db.models import ActionLog, ActionStatus
from app.agent import local_llm as local

logger = logging.getLogger(__name__)


def _has_api_key() -> bool:
    key = (settings.ANTHROPIC_API_KEY or "").strip()
    return bool(key) and key != "your-anthropic-api-key-here"


class TaskItem(BaseModel):
    title: str = Field(description="Title of the task")
    description: str = Field(description="Detailed description of the task")
    priority: str = Field(description="High, Medium, or Low")
    owner: str = Field(description="Assignee of the task")

class TaskList(BaseModel):
    tasks: List[TaskItem]


def create_tasks(source_text: str, tenant_id: int, user_id: int, db: Session) -> List[Dict[str, Any]]:
    """
    Extracts tasks from text using Claude (if API key set) or local rule-based extractor.
    Creates an ActionLog entry for HITL approval instead of immediately creating Task records.
    """
    tasks_data: List[Dict[str, Any]] = []

    if _has_api_key():
        # ── Claude path ──────────────────────────────────────────────────
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.prompts import PromptTemplate

            llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                api_key=settings.ANTHROPIC_API_KEY,
                temperature=0,
            )
            structured_llm = llm.with_structured_output(TaskList)
            prompt = PromptTemplate.from_template(
                "Extract a list of actionable tasks from the following text.\n"
                "If someone is assigned a task, note them as the owner, otherwise default to 'Unassigned'.\n\n"
                "Source Text:\n{text}\n"
            )
            chain = prompt | structured_llm
            result: TaskList = chain.invoke({"text": source_text})
            if result and result.tasks:
                tasks_data = [t.model_dump() for t in result.tasks]
        except Exception as e:
            logger.error(f"Claude task extraction failed, falling back to local: {e}")
            tasks_data = local.extract_tasks(source_text)
    else:
        # ── Local fallback ───────────────────────────────────────────────
        logger.info("create_tasks: using local rule-based extractor (no API key)")
        tasks_data = local.extract_tasks(source_text)

    if not tasks_data:
        return []

    # Create ActionLog for HITL approval (regardless of extraction method)
    action_log = ActionLog(
        tenant_id=tenant_id,
        user_id=user_id,
        action_type="create_tasks",
        payload_json={"tasks": tasks_data},
        status=ActionStatus.pending_approval,
    )
    db.add(action_log)
    db.commit()

    return tasks_data
