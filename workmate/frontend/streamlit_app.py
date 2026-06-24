"""
frontend/streamlit_app.py

This file contains the entire Streamlit User Interface. It handles all visual rendering 
(Chat, Memory, Approvals, Cost Analytics) and makes REST API calls to the FastAPI backend.
"""

import os
import sys
import json
import sqlite3
import requests
import pandas as pd
import streamlit as st

# Path setup 
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FRONTEND_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

API_URL   = os.getenv("API_URL", "http://localhost:8000")
DB_PATH   = os.path.join(ROOT_DIR, "workmate.db")

#  Page config 
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

# ── Session state defaults ────────────────────────────────────────────────
if "messages"        not in st.session_state: st.session_state.messages = []
if "conversation_id" not in st.session_state: st.session_state.conversation_id = None
if "last_trace"      not in st.session_state: st.session_state.last_trace = {}
if "session_tokens"  not in st.session_state: st.session_state.session_tokens = 0
if "session_cost"    not in st.session_state: st.session_state.session_cost = 0.0


# SIDEBAR
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
        st.session_state.session_tokens = 0
        st.session_state.session_cost = 0.0
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════
tab_chat, tab_approvals, tab_memory, tab_tasks, tab_costs, tab_debug = st.tabs([
    "💬 Chat", "🔔 Pending Approvals", "🧠 Memory", "✅ Tasks", "💰 Cost Analytics", "🔍 Debug Trace"
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
                # Token & cost data from backend
                p_tok   = data.get("prompt_tokens", 0)
                c_tok   = data.get("completion_tokens", 0)
                t_tok   = data.get("total_tokens", 0)
                cost    = data.get("cost_usd", 0.0)
                mdl     = data.get("model_name", "rule-based")
                meta = {
                    "intent":         data.get("intent"),
                    "confidence":     data.get("confidence") or 0.0,
                    "rag_confidence": data.get("rag_confidence"),
                    "prompt_tokens":  p_tok,
                    "completion_tokens": c_tok,
                    "total_tokens":   t_tok,
                    "cost_usd":       cost,
                    "model_name":     mdl,
                }
                st.session_state.last_trace = meta
                # Accumulate session totals
                st.session_state.session_tokens += t_tok
                st.session_state.session_cost   += cost

                st.markdown(response_text)

                # ── Inline cost badge ─────────────────────────────────────
                cost_color = "#4ade80" if cost == 0 else "#facc15"
                cost_str   = f"${cost:.6f}" if cost > 0 else "$0.0000 (free)"
                st.markdown(
                    f"""
                    <div style="
                        display:inline-flex; align-items:center; gap:10px;
                        background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1);
                        border-radius:20px; padding:4px 12px; margin-top:6px;
                        font-size:0.78rem; color:#94a3b8;
                    ">
                        <span>🪙 <b style='color:{cost_color}'>{cost_str}</b></span>
                        <span>|</span>
                        <span>↑ {p_tok} prompt</span>
                        <span>↓ {c_tok} completion</span>
                        <span>|</span>
                        <span>⚡ {t_tok} total tokens</span>
                        <span>|</span>
                        <span style='color:#7c3aed'>{mdl}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

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

# ── TAB 5: COST ANALYTICS ───────────────────────────────────────────────────
with tab_costs:
    st.markdown("### 💰 Cost & Token Analytics")
    st.caption("Real-time token consumption and API cost tracking for every prompt.")

    col_r, _ = st.columns([1, 3])
    with col_r:
        if st.button("🔄 Refresh", key="refresh_costs"):
            st.rerun()

    # ── Session summary cards ────────────────────────────────────────────
    st.markdown("#### 📊 This Session")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tokens Used",   f"{st.session_state.session_tokens:,}")
    c2.metric("Total Cost",          f"${st.session_state.session_cost:.6f}")
    c3.metric("Messages Sent",       str(sum(1 for m in st.session_state.messages if m["role"] == "user")))

    # ── Budget progress bar ──────────────────────────────────────────────
    BUDGET_USD = 1.0   # configurable budget cap in USD
    budget_pct = min(st.session_state.session_cost / BUDGET_USD, 1.0)
    bar_color  = "normal" if budget_pct < 0.7 else "inverse"
    st.markdown(f"**💳 Session Budget** (cap: ${BUDGET_USD:.2f})")
    st.progress(budget_pct)
    if budget_pct >= 0.9:
        st.warning("⚠️ Approaching session budget cap!")

    st.divider()

    # ── Historical cost data from backend ────────────────────────────────
    st.markdown("#### 📜 Full Cost History (from DB)")
    analytics_resp = api_get("/analytics/costs", params={"tenant_id": tenant_id, "user_id": user_id})

    if analytics_resp and analytics_resp.status_code == 200:
        analytics_data = analytics_resp.json()
        summary        = analytics_data.get("summary", {})
        records        = analytics_data.get("records", [])

        # Lifetime summary cards
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("📨 Total Messages",    str(summary.get("message_count", 0)))
        lc2.metric("🔢 Lifetime Tokens",   f"{summary.get('total_tokens', 0):,}")
        lc3.metric("↑ Prompt Tokens",      f"{summary.get('total_prompt_tokens', 0):,}")
        lc4.metric("↓ Completion Tokens",  f"{summary.get('total_completion_tokens', 0):,}")

        st.markdown(
            f"""
            <div style="
                background:linear-gradient(135deg,#1e2140,#252a45);
                border:1px solid #3d4270; border-radius:10px;
                padding:16px 24px; margin:12px 0; text-align:center;
            ">
                <span style='font-size:1.6rem; font-weight:700; color:#4ade80'>
                    💵 Lifetime Cost: ${summary.get('total_cost_usd', 0.0):.6f}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if records:
            cost_df = pd.DataFrame(records)
            cost_df["cost_usd"] = cost_df["cost_usd"].apply(lambda x: f"${x:.6f}")
            cost_df = cost_df.rename(columns={
                "model_name":         "Model",
                "prompt_tokens":      "↑ Prompt",
                "completion_tokens":  "↓ Completion",
                "total_tokens":       "Total Tokens",
                "cost_usd":           "Cost (USD)",
                "created_at":         "Timestamp",
            })
            st.dataframe(
                cost_df[["Timestamp", "Model", "↑ Prompt", "↓ Completion", "Total Tokens", "Cost (USD)"]],
                use_container_width=True,
                hide_index=True,
            )

            # Daily cost bar chart
            st.markdown("#### 📅 Token Usage by Message")
            chart_raw = analytics_data["records"]
            if chart_raw:
                chart_df = pd.DataFrame(chart_raw)[["created_at", "total_tokens"]].copy()
                chart_df["created_at"] = pd.to_datetime(chart_df["created_at"]).dt.strftime("%m-%d %H:%M")
                chart_df = chart_df.set_index("created_at")
                st.bar_chart(chart_df, color="#7c3aed")
        else:
            st.info("No cost history yet. Send a message to start tracking!")
    else:
        st.warning("Could not load cost history. Is the backend running?")


# ── TAB 6: DEBUG TRACE ──────────────────────────────────────────────────────
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
