"""
app/email/email_service.py

Uses the LLM to draft professional emails and stages them as pending approvals in the DB.
"""

# Drafts emails based on document context. Also saves to SQLite for approval.
import logging
from typing import Dict, Any
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session
from app.db.models import ActionLog, ActionStatus
from app.agent import local_llm as local
from app.agent.llm_factory import get_llm, has_active_llm

logger = logging.getLogger(__name__)

# ── Prompt echo detection patterns ────────────────────────────────────────────
_ECHO_PHRASES = [
    "draft a", "write a", "follow-up email", "compose an email",
    "generate an email", "create an email", "send an email",
]

def _is_echoed(text: str) -> bool:
    """Detect if LLM echoed the instruction instead of synthesizing content."""
    lowered = text.lower()
    return any(phrase in lowered for phrase in _ECHO_PHRASES)


class EmailDraft(BaseModel):
    to: str = Field(
        default="the recipient",
        description="Recipient name or email found in the document. Default: 'the recipient'."
    )
    subject: str = Field(
        description="A concise, specific subject line derived ONLY from document facts. "
                    "NEVER echo the user instruction here."
    )
    body: str = Field(
        description="Complete professional email body synthesized from document facts. "
                    "NEVER repeat the user's raw instruction inside the body."
    )
    suggested_recipients: str = Field(
        default="",
        description="Comma-separated recipient names/emails found in the document."
    )

    @field_validator("subject")
    @classmethod
    def subject_must_not_echo(cls, v: str) -> str:
        if _is_echoed(v):
            raise ValueError(
                f"Subject echoes the user instruction: '{v}'. "
                "Derive a subject from the document content instead."
            )
        return v


# ── Core prompt (shared by all LLM paths) ─────────────────────────────────────
_SYSTEM_PROMPT = """You are a corporate email drafting assistant.
Your ONLY job is to read the DOCUMENT CONTEXT and write a professional email FROM it.

HARD RULES — violating any rule makes the output invalid:
1. The subject line MUST be derived from facts in the document (project names, dates, decisions).
   It MUST NOT contain phrases like "draft", "write", "follow-up email", or any meta-instruction.
2. The body MUST NOT repeat or paraphrase the user's raw instruction.
   Write as the sender — not as someone describing what to write.
3. If the document mentions specific people, use their names as recipients.
4. If the document has no clear recipient, address it to "the team" or "the relevant stakeholder".
5. Keep the tone professional and concise.
"""

_USER_TEMPLATE = """--- DOCUMENT CONTEXT START ---
{context}
--- DOCUMENT CONTEXT END ---

Now produce the email. Remember: synthesize from the document above.
Do NOT echo the instruction. Do NOT use placeholder subject lines."""


def _invoke_structured(llm, context: str) -> EmailDraft:
    """Try structured output binding (works with Claude, GPT-4, newer Ollama models)."""
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", _USER_TEMPLATE),
    ])
    chain = prompt | llm.with_structured_output(EmailDraft)
    return chain.invoke({"context": context})


def _invoke_pydantic_parser(llm, context: str) -> EmailDraft:
    """Fallback for older local models without native JSON tool binding."""
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    parser = PydanticOutputParser(pydantic_object=EmailDraft)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT + "\n\nOutput format:\n{format_instructions}"),
        ("human", _USER_TEMPLATE),
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    return chain.invoke({"context": context})


def _llm_draft(context: str) -> Dict[str, Any]:
    """
    Run the LLM pipeline with structured output → pydantic parser fallback.
    Short-circuits to local rule-based draft when no active LLM is configured.
    Returns email_data dict, or raises on total failure.
    """
    if not has_active_llm():
        logger.info("_llm_draft: No active LLM configured — using local rule-based fallback.")
        return local.draft_email(context)

    llm = get_llm(temperature=0.2)  # Low temp = less hallucination / echoing

    try:
        result = _invoke_structured(llm, context)
    except (NotImplementedError, Exception) as e:
        logger.warning(f"Structured output failed ({e}), trying PydanticOutputParser fallback.")
        result = _invoke_pydantic_parser(llm, context)

    # ── Echo guard: retry once with stricter prompt if echo detected ──────────
    if _is_echoed(result.subject) or _is_echoed(result.body[:120]):
        logger.warning("Echo detected in LLM output — retrying with reinforced prompt.")
        stricter_context = (
            "[IMPORTANT: Your previous attempt echoed the user instruction. "
            "This time, write ONLY from the facts below. No meta-language.]\n\n"
            + context
        )
        try:
            result = _invoke_structured(llm, stricter_context)
        except Exception:
            result = _invoke_pydantic_parser(llm, stricter_context)

        # ── Second echo guard: if still echoed after retry, use local fallback ─
        if _is_echoed(result.subject) or _is_echoed(result.body[:120]):
            logger.warning("Echo persists after retry — falling back to local rule-based draft.")
            return local.draft_email(context)

    return result.model_dump()


def draft_email(
    context: str,
    tenant_id: int,
    user_id: int,
    db: Session,
) -> Dict[str, Any]:
    """
    Drafts an email from document context using LLM (cloud or local Ollama).
    Always creates an ActionLog for HITL approval — never sends directly.

    Args:
        context:   Extracted document text (NOT the raw user instruction).
        tenant_id: Tenant identifier for multi-tenancy.
        user_id:   User initiating the action.
        db:        SQLAlchemy session.

    Returns:
        Dict with keys: to, subject, body, suggested_recipients.
    """
    email_data: Dict[str, Any] = {}

    try:
        email_data = _llm_draft(context)
        logger.info("draft_email: LLM pipeline succeeded.")
    except Exception as e:
        logger.error(f"draft_email: LLM pipeline failed entirely, using local fallback. Error: {e}")
        email_data = local.draft_email(context)

    if not email_data:
        return {}

    email_data.setdefault("suggested_recipients", "")
    email_data.setdefault("to", "the recipient")

    action_log = ActionLog(
        tenant_id=tenant_id,
        user_id=user_id,
        action_type="draft_email",
        payload_json=email_data,
        status=ActionStatus.pending_approval,
    )
    db.add(action_log)
    db.commit()
    db.refresh(action_log)

    return email_data