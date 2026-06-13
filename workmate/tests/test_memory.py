"""Tests for memory CRUD operations."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.session import Base
from app.db.models import MemoryType
from app.memory.memory_store import create_memory, get_memories, update_last_accessed

# ── Use an in-memory SQLite DB for isolation ───────────────────────────────
TEST_DB_URL = "sqlite://"  # pure in-memory

@pytest.fixture(scope="module")
def db_session():
    engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


def test_create_memory(db_session):
    mem = create_memory(
        db=db_session,
        tenant_id=1,
        user_id=1,
        memory_type=MemoryType.semantic,
        content="The project launch date is Q4.",
        importance_score=0.9,
        embedding_id="test-emb-001",
    )
    assert mem.id is not None
    assert mem.content == "The project launch date is Q4."
    assert mem.type == MemoryType.semantic


def test_get_memories(db_session):
    mems = get_memories(db_session, tenant_id=1, user_id=1)
    assert len(mems) >= 1
    assert any(m.content == "The project launch date is Q4." for m in mems)


def test_get_memories_filtered_by_type(db_session):
    # Create a preference memory
    create_memory(db_session, 1, 1, MemoryType.preference, "Prefers dark mode.", 0.8, "emb-002")

    semantic = get_memories(db_session, 1, 1, MemoryType.semantic)
    preference = get_memories(db_session, 1, 1, MemoryType.preference)

    assert all(m.type == MemoryType.semantic for m in semantic)
    assert all(m.type == MemoryType.preference for m in preference)


def test_update_last_accessed(db_session):
    mems = get_memories(db_session, 1, 1)
    mem = mems[0]
    old_time = mem.last_accessed_at
    update_last_accessed(db_session, mem.id)
    db_session.refresh(mem)
    assert mem.last_accessed_at >= old_time


def test_memory_json_extraction_schema():
    """Validate that a raw extraction result matches expected schema."""
    import json
    raw = json.dumps({
        "memories": [
            {"type": "semantic",    "content": "test",  "importance_score": 0.8},
            {"type": "episodic",    "content": "event", "importance_score": 0.9},
            {"type": "preference",  "content": "pref",  "importance_score": 0.75},
        ]
    })
    data = json.loads(raw)
    valid_types = {"semantic", "episodic", "preference"}
    for m in data["memories"]:
        assert m["type"] in valid_types
        assert isinstance(m["content"], str)
        assert 0.0 <= m["importance_score"] <= 1.0
