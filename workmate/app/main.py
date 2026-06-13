"""
FastAPI entrypoint for WorkMate.
Handles document upload, chat, and HITL action approval endpoints.
"""
import os
import sys
import uuid
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Depends, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel

# ── ensure project root is on sys.path ─────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.db.session import get_db
from app.db.init_db import init_db
from app.db.models import Document, Conversation, Message, MemoryType, ActionLog
from app.agent.graph import build_graph
from app.ingestion.loaders import parse_document
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import get_embedding_model
from app.vectorstore.chroma_client import get_document_collection, get_memory_collection
from app.safety.guardrails import get_pending_actions, approve_action, reject_action
from app.memory.memory_extractor import extract_memories
from app.memory.memory_store import create_memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Build the agent graph once at startup ──────────────────────────────────
agent_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_graph
    logger.info("WorkMate API starting up…")
    init_db()
    agent_graph = build_graph()
    logger.info("LangGraph agent compiled successfully.")
    yield
    logger.info("WorkMate API shutting down.")


app = FastAPI(title="WorkMate API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    tenant_id: int = 1
    user_id: int = 1
    query: str
    conversation_id: Optional[int] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: int
    intent: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    rag_confidence: Optional[str] = None


# ── Background helpers ─────────────────────────────────────────────────────

def _process_document_bg(document_id: int, tenant_id: int, user_id: Optional[int], raw_path: str):
    """Parse → chunk → embed → store in Chroma. Runs in background."""
    db = next(get_db())
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        return

    try:
        doc.status = "parsing"
        db.commit()
        text = parse_document(raw_path)

        doc.status = "chunking"
        db.commit()
        chunks = chunk_text(text)

        doc.status = "embedding"
        db.commit()
        embedder = get_embedding_model()
        collection = get_document_collection(tenant_id)

        if chunks:
            embeddings = embedder.embed_documents(chunks)
            ids = [str(uuid.uuid4()) for _ in chunks]
            u_id = user_id if user_id is not None else -1
            metadatas = [
                {
                    "document_id": document_id,
                    "chunk_index": i,
                    "filename": doc.filename,
                    "user_id": u_id,
                    "tenant_id": tenant_id,
                }
                for i in range(len(chunks))
            ]
            collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

            # Save chunks to relational DB
            from app.db.models import Chunk
            for i, (chunk_text_val, cid) in enumerate(zip(chunks, ids)):
                chunk_row = Chunk(
                    document_id=document_id,
                    chunk_index=i,
                    content=chunk_text_val,
                    token_count=len(chunk_text_val) // 4,
                    chroma_id=cid,
                )
                db.add(chunk_row)
            db.commit()

        doc.status = "ready"
        db.commit()
        logger.info(f"Document {document_id} processed: {len(chunks)} chunks embedded.")
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}", exc_info=True)
        try:
            doc.status = "failed"
            db.commit()
        except Exception:
            pass
    finally:
        db.close()


def _extract_memories_bg(tenant_id: int, user_id: int, message_id: int, query: str, response: str):
    """Extract memories from conversation turn and store in Chroma + DB."""
    db = next(get_db())
    try:
        history = f"User: {query}\nAssistant: {response}"
        memories = extract_memories(history)
        if not memories:
            return

        embedder = get_embedding_model()
        collection = get_memory_collection(tenant_id)

        for mem in memories:
            db_mem = create_memory(
                db=db,
                tenant_id=tenant_id,
                user_id=user_id,
                memory_type=MemoryType(mem["type"]),
                content=mem["content"],
                importance_score=mem["importance_score"],
                embedding_id="",
                source_message_id=message_id,
            )
            emb = embedder.embed_query(mem["content"])
            c_id = str(uuid.uuid4())
            collection.add(
                ids=[c_id],
                embeddings=[emb],
                documents=[mem["content"]],
                metadatas=[{
                    "memory_id": db_mem.id,
                    "user_id": user_id,
                    "type": mem["type"],
                    "importance_score": float(mem["importance_score"]),
                }],
            )
            db_mem.embedding_id = c_id
            db.commit()
        logger.info(f"Extracted {len(memories)} memories for user {user_id}.")
    except Exception as e:
        logger.error(f"Memory extraction failed: {e}", exc_info=True)
    finally:
        db.close()


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "WorkMate API"}


@app.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tenant_id: int = Form(1),
    user_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
):
    """Upload a PDF/DOCX/TXT document and trigger async processing."""
    uploads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    safe_name = f"{uuid.uuid4()}_{file.filename}"
    raw_path = os.path.join(uploads_dir, safe_name)

    content = await file.read()
    with open(raw_path, "wb") as f:
        f.write(content)

    doc = Document(
        tenant_id=tenant_id,
        user_id=user_id,
        filename=file.filename,
        file_type=file.content_type or "application/octet-stream",
        status="uploaded",
        raw_path=raw_path,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    background_tasks.add_task(_process_document_bg, doc.id, tenant_id, user_id, raw_path)
    return {"message": "Upload accepted. Processing in background.", "document_id": doc.id}


@app.get("/documents/status/{document_id}")
def document_status(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"document_id": doc.id, "filename": doc.filename, "status": doc.status}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Main chat endpoint: runs the LangGraph agent and returns a response."""
    global agent_graph
    if agent_graph is None:
        raise HTTPException(status_code=503, detail="Agent not ready yet.")

    # Get or create conversation
    if req.conversation_id:
        conv_id = req.conversation_id
    else:
        conv = Conversation(tenant_id=req.tenant_id, user_id=req.user_id)
        db.add(conv)
        db.commit()
        db.refresh(conv)
        conv_id = conv.id

    # Persist user message
    u_msg = Message(conversation_id=conv_id, role="user", content=req.query)
    db.add(u_msg)
    db.commit()
    db.refresh(u_msg)

    # Build initial state as plain dict (TypedDict compatible)
    initial_state = {
        "tenant_id": req.tenant_id,
        "user_id": req.user_id,
        "conversation_id": conv_id,
        "message_id": u_msg.id,
        "current_query": req.query,
        "intent": "",
        "confidence": 0.0,
        "reasoning": "",
        "plan": [],
        "current_step": 0,
        "selected_tools": [],
        "tool_outputs": {},
        "retrieved_context": "",
        "citations": [],
        "rag_confidence": "",
        "validation_passed": True,
        "error_message": None,
        "final_response": "",
    }

    # Run the agent
    try:
        final_state = agent_graph.invoke(initial_state)
    except Exception as e:
        logger.error(f"Agent graph error: {e}", exc_info=True)
        final_state = {**initial_state, "final_response": "I encountered an internal error. Please try again."}

    response_text = final_state.get("final_response") or "I'm sorry, I couldn't generate a response."

    # Persist assistant message
    a_msg = Message(conversation_id=conv_id, role="assistant", content=response_text)
    db.add(a_msg)
    db.commit()

    # Extract memories in background
    background_tasks.add_task(
        _extract_memories_bg,
        req.tenant_id, req.user_id, u_msg.id, req.query, response_text
    )

    return ChatResponse(
        response=response_text,
        conversation_id=conv_id,
        intent=final_state.get("intent"),
        confidence=final_state.get("confidence"),
        reasoning=final_state.get("reasoning"),
        rag_confidence=final_state.get("rag_confidence"),
    )


@app.get("/actions/pending")
def list_pending(tenant_id: int = 1, user_id: int = 1, db: Session = Depends(get_db)):
    actions = get_pending_actions(db, tenant_id, user_id)
    return [
        {
            "id": a.id,
            "action_type": a.action_type,
            "payload": a.payload_json,
            "created_at": str(a.created_at),
        }
        for a in actions
    ]


@app.post("/actions/{action_id}/approve")
def approve(action_id: int, db: Session = Depends(get_db)):
    success = approve_action(db, action_id)
    return {"success": success}


@app.post("/actions/{action_id}/reject")
def reject(action_id: int, db: Session = Depends(get_db)):
    success = reject_action(db, action_id)
    return {"success": success}


@app.get("/memories")
def list_memories(tenant_id: int = 1, user_id: int = 1, db: Session = Depends(get_db)):
    from app.memory.memory_store import get_memories
    mems = get_memories(db, tenant_id, user_id)
    return [
        {
            "id": m.id,
            "type": m.type,
            "content": m.content,
            "importance_score": m.importance_score,
            "created_at": str(m.created_at),
        }
        for m in mems
    ]


@app.get("/tasks")
def list_tasks(tenant_id: int = 1, db: Session = Depends(get_db)):
    from app.db.models import Task
    tasks = db.query(Task).filter(Task.tenant_id == tenant_id).all()
    return [
        {
            "id": t.id,
            "title": t.title,
            "description": t.description,
            "priority": t.priority,
            "owner": t.owner,
            "status": t.status,
            "source": t.source,
        }
        for t in tasks
    ]


@app.get("/documents")
def list_documents(tenant_id: int = 1, db: Session = Depends(get_db)):
    docs = db.query(Document).filter(Document.tenant_id == tenant_id).all()
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "status": d.status,
            "uploaded_at": str(d.uploaded_at),
        }
        for d in docs
    ]
