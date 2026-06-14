"""
WorkMate Streamlit Frontend
Tabs: Chat | Pending Approvals | Memory | Tasks | Debug Trace
"""
import os
import sys
import json
import sqlite3
import requests
import pandas as pd
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FRONTEND_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

API_URL   = os.getenv("API_URL", "http://localhost:8000")
DB_PATH   = os.path.join(ROOT_DIR, "workmate.db")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WorkMate AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    section[data-testid="stSidebar"] { background-color: #1a1d2e; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .metric-card {
        background: linear-gradient(135deg, #1e2140, #252a45);
        border-radius: 10px; padding: 16px; text-align: center;
        border: 1px solid #3d4270;
    }
    h1,h2,h3 { color: #e0e6ff !important; }
    .stButton>button {
        border-radius: 8px; font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover { transform: translateY(-1px); }
    .approve-btn>button { background:#1a7a4a; color:white; border:none; }
    .reject-btn>button  { background:#7a1a1a; color:white; border:none; }
</style>
""", unsafe_allow_html=True)

# ── DB helpers (read-only from Streamlit side) ─────────────────────────────
def _db():
    if not os.path.exists(DB_PATH):
        return None
    return sqlite3.connect(DB_PATH)

def fetch_df(query: str, params=()):
    conn = _db()
    if conn is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query(query, conn, params=params)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def api_post(path: str, **kwargs):
    try:
        r = requests.post(f"{API_URL}{path}", timeout=120, **kwargs)
        return r
    except requests.exceptions.ConnectionError:
        return None

def api_get(path: str, params=None):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=10)
        return r
    except requests.exceptions.ConnectionError:
        return None

# ── Session state defaults ─────────────────────────────────────────────────
if "messages"        not in st.session_state: st.session_state.messages = []
if "conversation_id" not in st.session_state: st.session_state.conversation_id = None
if "last_trace"      not in st.session_state: st.session_state.last_trace = {}

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🤖 WorkMate AI")
    st.markdown("*Your AI Workspace Assistant*")
    st.divider()

    # Mock login
    st.subheader("👤 Session")
    tenant_id = int(st.number_input("Tenant ID", min_value=1, value=1, step=1))
    user_id   = int(st.number_input("User ID",   min_value=1, value=1, step=1))

    st.divider()
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "PDF / DOCX / TXT",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
    )
    if st.button("⬆️ Upload", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please select a file first.")
        else:
            with st.spinner("Uploading…"):
                resp = api_post(
                    "/documents/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    data={"tenant_id": tenant_id, "user_id": user_id},
                )
            if resp and resp.status_code == 200:
                st.success(f"✅ Uploaded! Processing in background.")
            elif resp:
                st.error(f"Upload failed ({resp.status_code}): {resp.text[:200]}")
            else:
                st.error("Cannot reach API. Is the backend running?")

    st.divider()
    st.subheader("📁 Your Documents")
    docs_df = fetch_df(
        "SELECT filename, status, uploaded_at FROM documents WHERE tenant_id=? ORDER BY id DESC",
        (tenant_id,),
    )
    if docs_df.empty:
        st.caption("No documents yet.")
    else:
        st.dataframe(docs_df, use_container_width=True, hide_index=True)

    st.divider()
    # API health
    health = api_get("/health")
    if health and health.status_code == 200:
        st.success("🟢 API Online")
    else:
        st.error("🔴 API Offline — start the backend first")

    # Show API mode indicator
    try:
        import sys, os as _os
        _root = _os.path.dirname(FRONTEND_DIR)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from app.config import settings as _s
        provider = _s.LLM_PROVIDER.lower()
        if provider == "ollama":
            _mode = "ollama"
        elif provider == "anthropic":
            _key = (_s.ANTHROPIC_API_KEY or "").strip()
            _mode = "claude" if (bool(_key) and _key != "your-anthropic-api-key-here") else "rule-based"
        else:
            _mode = "rule-based"
    except Exception:
        _mode = "rule-based"

    if _mode == "ollama":
        st.success("🦙 AI Mode: **Ollama** (Local LLM)")
    elif _mode == "claude":
        st.success("🧠 AI Mode: **Claude** (Full Intelligence)")
    else:
        st.warning("⚙️ **Rule-based Mode** — no API key set.\nAll features work! Set `LLM_PROVIDER=ollama` or add `ANTHROPIC_API_KEY` to `.env` for AI responses.")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.session_state.last_trace = {}
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════
tab_chat, tab_approvals, tab_memory, tab_tasks, tab_debug = st.tabs([
    "💬 Chat", "🔔 Pending Approvals", "🧠 Memory", "✅ Tasks", "🔍 Debug Trace"
])

# ── TAB 1: CHAT ─────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown("### 💬 Chat with WorkMate")

    # Render history
    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("meta"):
                with st.expander("ℹ️ Intent info", expanded=False):
                    cols = st.columns(3)
                    cols[0].metric("Intent",     msg["meta"].get("intent", "–"))
                    cols[1].metric("Confidence", f"{msg['meta'].get('confidence', 0):.0%}")
                    cols[2].metric("RAG Conf",   msg["meta"].get("rag_confidence", "–") or "–")

    if prompt := st.chat_input("Ask WorkMate anything…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Thinking…"):
                resp = api_post(
                    "/chat",
                    json={
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "query": prompt,
                        "conversation_id": st.session_state.conversation_id,
                    },
                )
            if resp is None:
                st.error("❌ Cannot reach the backend. Is `uvicorn app.main:app` running?")
            elif resp.status_code == 200:
                data = resp.json()
                st.session_state.conversation_id = data["conversation_id"]
                response_text = data["response"]
                meta = {
                    "intent":        data.get("intent"),
                    "confidence":    data.get("confidence") or 0.0,
                    "rag_confidence":data.get("rag_confidence"),
                }
                st.session_state.last_trace = meta
                st.markdown(response_text)
                with st.expander("ℹ️ Intent info", expanded=False):
                    cols = st.columns(3)
                    cols[0].metric("Intent",     meta["intent"] or "–")
                    cols[1].metric("Confidence", f"{meta['confidence']:.0%}")
                    cols[2].metric("RAG Conf",   meta["rag_confidence"] or "–")
                st.session_state.messages.append({
                    "role": "assistant", "content": response_text, "meta": meta
                })
            else:
                st.error(f"API error {resp.status_code}: {resp.text[:300]}")

# ── TAB 2: PENDING APPROVALS ────────────────────────────────────────────────
with tab_approvals:
    st.markdown("### 🔔 Pending Approvals (Human-in-the-Loop)")
    st.caption("Actions created by the AI that require your approval before executing.")

    if st.button("🔄 Refresh", key="refresh_approvals"):
        st.rerun()

    actions_df = fetch_df(
        "SELECT id, action_type, payload_json, created_at FROM action_logs "
        "WHERE status='pending_approval' AND tenant_id=? ORDER BY id DESC",
        (tenant_id,),
    )

    if actions_df.empty:
        st.info("✅ No pending actions — you're all caught up!")
    else:
        for _, row in actions_df.iterrows():
            with st.expander(
                f"{'📋' if row['action_type']=='create_tasks' else '✉️'} "
                f"**{row['action_type']}** — ID {row['id']} | {row['created_at']}",
                expanded=True,
            ):
                try:
                    payload = json.loads(row["payload_json"]) if isinstance(row["payload_json"], str) else row["payload_json"]
                    st.json(payload)
                except Exception:
                    st.text(row["payload_json"])

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Approve", key=f"app_{row['id']}", use_container_width=True):
                        r = api_post(f"/actions/{row['id']}/approve")
                        if r and r.json().get("success"):
                            st.success("Approved and executed!")
                        else:
                            st.error("Approval failed.")
                        st.rerun()
                with col2:
                    if st.button("❌ Reject", key=f"rej_{row['id']}", use_container_width=True):
                        r = api_post(f"/actions/{row['id']}/reject")
                        if r and r.json().get("success"):
                            st.warning("Action rejected.")
                        else:
                            st.error("Rejection failed.")
                        st.rerun()

# ── TAB 3: MEMORY ───────────────────────────────────────────────────────────
with tab_memory:
    st.markdown("### 🧠 Long-Term Memory")
    st.caption("Facts, events, and preferences automatically extracted from your conversations.")

    col1, col2 = st.columns([3, 1])
    with col2:
        mem_filter = st.selectbox("Filter by type", ["All", "semantic", "episodic", "preference"])
    with col1:
        if st.button("🔄 Refresh memories"):
            st.rerun()

    q = "SELECT type, content, importance_score, created_at FROM memories WHERE tenant_id=? AND user_id=? ORDER BY importance_score DESC"
    mem_df = fetch_df(q, (tenant_id, user_id))

    if mem_df.empty:
        st.info("No memories stored yet. Start chatting to build your memory bank!")
    else:
        if mem_filter != "All":
            mem_df = mem_df[mem_df["type"] == mem_filter]

        # Type badges
        type_colors = {"semantic": "🔵", "episodic": "🟡", "preference": "🟢"}
        mem_df["type"] = mem_df["type"].apply(lambda t: f"{type_colors.get(t,'⚪')} {t}")
        st.dataframe(mem_df, use_container_width=True, hide_index=True)
        st.caption(f"Total memories: **{len(mem_df)}**")

# ── TAB 4: TASKS ────────────────────────────────────────────────────────────
with tab_tasks:
    st.markdown("### ✅ Tasks")
    st.caption("AI-generated tasks approved through the Human-in-the-Loop workflow.")

    if st.button("🔄 Refresh tasks"):
        st.rerun()

    tasks_df = fetch_df(
        "SELECT title, description, priority, owner, status, source, created_at "
        "FROM tasks WHERE tenant_id=? ORDER BY id DESC",
        (tenant_id,),
    )

    if tasks_df.empty:
        st.info("No tasks yet. Ask WorkMate to extract tasks from text or a meeting transcript!")
    else:
        # Priority color coding
        priority_icons = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
        tasks_df["priority"] = tasks_df["priority"].apply(lambda p: f"{priority_icons.get(p,'⚪')} {p}")
        st.dataframe(tasks_df, use_container_width=True, hide_index=True)
        st.caption(f"Total tasks: **{len(tasks_df)}**")

# ── TAB 5: DEBUG TRACE ──────────────────────────────────────────────────────
with tab_debug:
    st.markdown("### 🔍 Agent Debug Trace")
    st.caption("Step-by-step execution log of the last agent run.")

    if st.session_state.last_trace:
        m = st.session_state.last_trace
        cols = st.columns(3)
        cols[0].metric("Intent",     m.get("intent") or "–")
        cols[1].metric("Confidence", f"{m.get('confidence', 0):.0%}")
        cols[2].metric("RAG Conf",   m.get("rag_confidence") or "–")

    conv_id = st.session_state.conversation_id
    if conv_id:
        trace_df = fetch_df(
            "SELECT node_name, latency_ms, tokens_used, input_json, output_json, created_at "
            "FROM agent_traces WHERE conversation_id=? ORDER BY id DESC LIMIT 20",
            (conv_id,),
        )
        if not trace_df.empty:
            st.dataframe(trace_df, use_container_width=True, hide_index=True)

            # Latency bar chart
            st.markdown("#### ⏱️ Latency per Node (ms)")
            chart_df = trace_df[["node_name", "latency_ms"]].set_index("node_name")
            st.bar_chart(chart_df)
        else:
            st.info("No traces recorded yet for this conversation.")
    else:
        st.info("Start a chat to see the trace here.")
