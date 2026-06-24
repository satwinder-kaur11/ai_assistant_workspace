"""
app/db/models.py

Defines the database schema using SQLAlchemy ORM classes (Users, Documents, ActionLogs, TokenUsage).
"""

#  Defines the SQLite schema (Users, Tenants, Memories, ActionLogs).
from datetime import datetime, timezone
import enum
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float, JSON, Enum, Boolean
from sqlalchemy.orm import relationship
from app.db.session import Base

def utcnow():
    return datetime.now(timezone.utc)

class MemoryType(str, enum.Enum):
    semantic = "semantic"
    episodic = "episodic"
    preference = "preference"

class ActionStatus(str, enum.Enum):
    pending_approval = "pending_approval"
    approved = "approved"
    rejected = "rejected"
    executed = "executed"

class TaskSource(str, enum.Enum):
    manual = "manual"
    ai_generated = "ai_generated"

class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    created_at = Column(DateTime, default=utcnow)

    users = relationship("User", back_populates="tenant")
    documents = relationship("Document", back_populates="tenant")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"))
    name = Column(String)
    email = Column(String, unique=True, index=True)
    timezone = Column(String, default="UTC")
    created_at = Column(DateTime, default=utcnow)

    tenant = relationship("Tenant", back_populates="users")
    documents = relationship("Document", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True) # null = tenant-shared
    filename = Column(String)
    file_type = Column(String)
    status = Column(String) # uploaded, parsing, chunking, embedding, ready, failed
    raw_path = Column(String)
    uploaded_at = Column(DateTime, default=utcnow)

    tenant = relationship("Tenant", back_populates="documents")
    user = relationship("User", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    chunk_index = Column(Integer)
    content = Column(Text)
    token_count = Column(Integer)
    chroma_id = Column(String, index=True)

    document = relationship("Document", back_populates="chunks")

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String) # user, assistant, system
    content = Column(Text)
    created_at = Column(DateTime, default=utcnow)

    conversation = relationship("Conversation", back_populates="messages")

class Memory(Base):
    __tablename__ = "memories"
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    type = Column(Enum(MemoryType))
    content = Column(Text)
    source_message_id = Column(Integer, ForeignKey("messages.id"), nullable=True)
    embedding_id = Column(String) # For chroma vector storage mapping
    importance_score = Column(Float)
    created_at = Column(DateTime, default=utcnow)
    last_accessed_at = Column(DateTime, default=utcnow)

class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String)
    description = Column(Text)
    priority = Column(String) # High, Medium, Low
    owner = Column(String)
    status = Column(String, default="pending")
    source = Column(Enum(TaskSource))
    created_at = Column(DateTime, default=utcnow)

class ActionLog(Base):
    __tablename__ = "action_logs"
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    action_type = Column(String) # create_tasks, draft_email
    payload_json = Column(JSON)
    status = Column(Enum(ActionStatus), default=ActionStatus.pending_approval)
    created_at = Column(DateTime, default=utcnow)

class AgentTrace(Base):
    __tablename__ = "agent_traces"
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"))
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=True)
    node_name = Column(String)
    input_json = Column(JSON)
    output_json = Column(JSON)
    latency_ms = Column(Float)
    tokens_used = Column(Integer)
    created_at = Column(DateTime, default=utcnow)


class TokenUsage(Base):
    """Tracks token consumption and cost per chat message."""
    __tablename__ = "token_usage"
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=True)
    model_name = Column(String, default="rule-based")   # e.g. claude-3-5-sonnet, llama3, rule-based
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)               # computed cost in US dollars
    created_at = Column(DateTime, default=utcnow)
