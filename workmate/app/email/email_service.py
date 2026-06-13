import logging
from typing import Dict, Any
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.config import settings
from app.db.models import ActionLog, ActionStatus
from app.agent import local_llm as local

logger = logging.getLogger(__name__)


def _has_api_key() -> bool:
    key = (settings.ANTHROPIC_API_KEY or "").strip()
    return bool(key) and key != "your-anthropic-api-key-here"


class EmailDraft(BaseModel):
    subject: str = Field(description="The email subject line")
    body: str = Field(description="The body of the email in a professional tone")
    suggested_recipients: str = Field(description="Comma-separated list of suggested recipient names or emails")


def draft_email(context: str, tenant_id: int, user_id: int, db: Session) -> Dict[str, Any]:
    """
    Drafts an email using Claude (if API key set) or local template-based composer.
    Creates an ActionLog entry for HITL approval instead of sending.
    """
    email_data: Dict[str, Any] = {}

    if _has_api_key():
        # ── Claude path ──────────────────────────────────────────────────
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.prompts import PromptTemplate

            llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                api_key=settings.ANTHROPIC_API_KEY,
                temperature=0.7,
            )
            structured_llm = llm.with_structured_output(EmailDraft)
            prompt = PromptTemplate.from_template(
                "Draft a professional email based on the following context. "
                "Identify the subject, the body content, and any suggested recipients from the context if mentioned.\n\n"
                "Context:\n{context}\n"
            )
            chain = prompt | structured_llm
            result: EmailDraft = chain.invoke({"context": context})
            if result:
                email_data = result.model_dump()
        except Exception as e:
            logger.error(f"Claude email drafting failed, falling back to local: {e}")
            email_data = local.draft_email(context)
    else:
        # ── Local fallback ───────────────────────────────────────────────
        logger.info("draft_email: using local template-based composer (no API key)")
        email_data = local.draft_email(context)

    if not email_data:
        return {}

    # Ensure required field for HITL display
    if "suggested_recipients" not in email_data:
        email_data["suggested_recipients"] = ""

    # Create ActionLog for HITL approval (regardless of drafting method)
    action_log = ActionLog(
        tenant_id=tenant_id,
        user_id=user_id,
        action_type="draft_email",
        payload_json=email_data,
        status=ActionStatus.pending_approval,
    )
    db.add(action_log)
    db.commit()

    return email_data
