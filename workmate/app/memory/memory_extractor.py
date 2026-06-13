import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from app.config import settings
from app.agent import local_llm as local

logger = logging.getLogger(__name__)


def _has_api_key() -> bool:
    key = (settings.ANTHROPIC_API_KEY or "").strip()
    return bool(key) and key != "your-anthropic-api-key-here"


class MemoryItem(BaseModel):
    type: str = Field(description="Must be exactly one of: semantic, episodic, preference")
    content: str = Field(description="The extracted fact, event, or preference in clear language")
    importance_score: float = Field(description="Score from 0.0 to 1.0 indicating long-term importance")

class MemoryExtractionResult(BaseModel):
    memories: List[MemoryItem]


def extract_memories(conversation_history: str) -> List[Dict[str, Any]]:
    """
    Extracts memories (semantic, episodic, preference) from conversation history.
    Uses Claude when API key is set, falls back to local rule-based extractor otherwise.
    Only returns memories above the MEMORY_IMPORTANCE_THRESHOLD.
    """
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
            structured_llm = llm.with_structured_output(MemoryExtractionResult)
            prompt = PromptTemplate.from_template(
                "Extract any important semantic facts, episodic events, or user preferences from the following conversation.\n"
                "Only extract items that are genuinely worth remembering long-term. Rate their importance from 0.0 to 1.0.\n\n"
                "Conversation:\n{conversation}\n"
            )
            chain = prompt | structured_llm
            result: MemoryExtractionResult = chain.invoke({"conversation": conversation_history})

            valid_memories = []
            if result and result.memories:
                for m in result.memories:
                    if m.type not in ["semantic", "episodic", "preference"]:
                        continue
                    if m.importance_score >= settings.MEMORY_IMPORTANCE_THRESHOLD:
                        valid_memories.append({
                            "type": m.type,
                            "content": m.content,
                            "importance_score": m.importance_score,
                        })
            return valid_memories

        except Exception as e:
            logger.error(f"Claude memory extraction failed, using local fallback: {e}")
            # Fall through to local extractor

    # ── Local fallback ───────────────────────────────────────────────────
    logger.info("extract_memories: using local rule-based extractor")
    memories = local.extract_memories_local(conversation_history)
    return [
        m for m in memories
        if m.get("importance_score", 0) >= settings.MEMORY_IMPORTANCE_THRESHOLD
    ]
