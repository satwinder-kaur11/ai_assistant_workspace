"""
app/memory/memory_store.py

Handles saving extracted memories into the SQLite database.
"""

from sqlalchemy.orm import Session
from app.db.models import Memory, MemoryType
from typing import List, Optional
from datetime import datetime, timezone

def create_memory(db: Session, tenant_id: int, user_id: int, memory_type: MemoryType, content: str, importance_score: float, embedding_id: str, source_message_id: Optional[int] = None) -> Memory:
    """Creates a new memory record in the relational database."""
    memory = Memory(
        tenant_id=tenant_id,
        user_id=user_id,
        type=memory_type,
        content=content,
        importance_score=importance_score,
        embedding_id=embedding_id,
        source_message_id=source_message_id
    )
    db.add(memory)
    db.commit()
    db.refresh(memory)
    return memory

def get_memories(db: Session, tenant_id: int, user_id: int, memory_type: Optional[MemoryType] = None) -> List[Memory]:
    """Retrieves memories for a user, optionally filtered by type."""
    query = db.query(Memory).filter(Memory.tenant_id == tenant_id, Memory.user_id == user_id)
    if memory_type:
        query = query.filter(Memory.type == memory_type)
    return query.all()

def update_last_accessed(db: Session, memory_id: int):
    """Updates the last_accessed_at timestamp for a memory."""
    memory = db.query(Memory).filter(Memory.id == memory_id).first()
    if memory:
        memory.last_accessed_at = datetime.now(timezone.utc)
        db.commit()
