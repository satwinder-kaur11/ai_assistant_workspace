# WorkMate Architecture Diagrams

> **v3 — Updated:** Now includes fault-tolerant retry loop with exponential backoff.

---

## 1. Full System Architecture

```mermaid
graph TD
    classDef frontend  fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff
    classDef backend   fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff
    classDef super     fill:#7c3aed,stroke:#5b21b6,stroke-width:2px,color:#fff
    classDef agent     fill:#2563eb,stroke:#1d4ed8,stroke-width:2px,color:#fff
    classDef shared    fill:#0891b2,stroke:#0e7490,stroke-width:2px,color:#fff
    classDef retry     fill:#dc2626,stroke:#b91c1c,stroke-width:2px,color:#fff
    classDef db        fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:#fff
    classDef llm       fill:#ef4444,stroke:#b91c1c,stroke-width:2px,color:#fff

    User((User))
    UI[Streamlit Frontend UI]:::frontend
    API[FastAPI Backend]:::backend

    subgraph "LangGraph Multi-Agent Orchestrator"
        SUP[Supervisor Agent\nRoutes query to sub-agent]:::super

        subgraph "Specialist Sub-Agents with Backoff"
            RA[Research Agent\nDocument QA + Memory Recall\nBackoff: 1s → 2s → 4s]:::agent
            PA[Productivity Agent\nTask Creation + Email Draft\nBackoff: 1s → 2s → 4s]:::agent
        end

        VAL[Validation Node]:::shared
        GRD[Guardrail HITL Node]:::shared
        RES[Response Generation]:::shared
        ERR[Error Handling\nRetry Counter]:::retry
    end

    SQLite[(SQLite\nLogs + Memory + Actions)]:::db
    Chroma[(ChromaDB\nVector Search)]:::db
    LLM{{LLM Engine\nOllama / Anthropic}}:::llm

    User --> UI --> API --> SUP
    SUP -->|document_qa / memory_recall| RA
    SUP -->|task_creation / email_draft| PA
    SUP -->|chitchat| VAL

    RA -->|Searches| Chroma
    RA -->|Recalls| SQLite
    PA -->|Stages tasks/emails| SQLite

    RA --> VAL
    PA --> VAL
    VAL -->|Tasks or Email| GRD
    VAL -->|QA or Chitchat| RES
    VAL -->|Error| ERR

    GRD --> RES
    RES --> API --> UI --> User

    ERR -->|Retries left\nreset state| SUP
    ERR -->|Exhausted\nfinal error| RES

    SUP -.->|Route query| LLM
    RES -.->|Synthesize| LLM
```

---

## 2. Fault-Tolerant Retry Architecture

The system has **two independent layers** of fault tolerance to handle any transient failure.

```mermaid
flowchart TD
    U([User Message]) --> SUP

    subgraph "Layer 1 — Internal Backoff per Sub-Agent"
        SUP[Supervisor Agent] --> SA[Sub-Agent\nResearch or Productivity]
        SA -->|Attempt 1| T1{Success?}
        T1 -->|No — wait 1s| T2{Attempt 2}
        T2 -->|No — wait 2s| T3{Attempt 3}
        T3 -->|No — wait 4s| FAIL[Signal failure\nset error_message]
        T1 -->|Yes| OK([Continue to Validation])
        T2 -->|Yes| OK
        T3 -->|Yes| OK
    end

    subgraph "Layer 2 — Graph-Level Retry Loop"
        FAIL --> ERR[error_handling\nincrement retry_count]
        ERR -->|retry_count <= max_retries\nclear error state| SUP
        ERR -->|retry_count > max_retries\nretries exhausted| FINAL([Final Error to User])
    end
```

---

## 3. Multi-Agent Supervisor Decision Flow

How the Supervisor decides which sub-agent handles the request —
using a two-layer routing strategy.

```mermaid
flowchart TD
    Q([User Query]) --> PRE{Local Pre-Check\nConfidence >= 82pct?}

    PRE -->|YES — task or email\nkeyword matched| SKIP[Skip LLM\nRoute directly]
    PRE -->|NO — ambiguous query| LLM[Ask LLM\nOllama or Anthropic]

    SKIP --> PA[productivity_agent]
    LLM --> PARSE[Parse plain-text\nLLM response]

    PARSE --> RA[research_agent]
    PARSE --> PA
    PARSE --> CA[chitchat — direct\nto validation]

    RA --> OUT([Sub-Agent Executes\nwith backoff])
    PA --> OUT
    CA --> OUT2([Validation Node])
```

---

## 4. LangGraph State Machine (Full Node Graph)

The exact nodes and edges registered in `app/agent/graph.py`.

```mermaid
stateDiagram-v2
    [*] --> Supervisor

    Supervisor --> ResearchAgent     : next_agent = research_agent
    Supervisor --> ProductivityAgent : next_agent = productivity_agent
    Supervisor --> Validation        : next_agent = chitchat_agent

    ResearchAgent     --> Validation
    ProductivityAgent --> Validation

    Validation --> GuardrailCheck     : intent = task_creation or email_draft
    Validation --> ResponseGeneration : intent = document_qa / memory_recall / chitchat
    Validation --> ErrorHandling      : validation_passed = False

    GuardrailCheck    --> ResponseGeneration

    note right of ErrorHandling
        Increments retry_count.
        If retries remain: loops back.
        If exhausted: sends final error.
    end note

    ErrorHandling --> Supervisor        : retry_count <= max_retries
    ErrorHandling --> ResponseGeneration: retry_count > max_retries

    ResponseGeneration --> [*]
```

---

## 5. Error State Machine

Exactly what happens inside `error_handling` every time it is triggered.

```mermaid
flowchart TD
    ERR([error_handling triggered]) --> INC[Increment\nretry_count]
    INC --> CHK{retry_count\n<= max_retries?}

    CHK -->|YES — retries remain| CLR[Clear transient state\nerror_message = None\nretrieved_context = empty\ntool_outputs = empty]
    CLR --> BACK[Route back\nto Supervisor]
    BACK --> SUP([Supervisor retries\nfull pipeline])

    CHK -->|NO — exhausted| MSG[Write final\nuser-facing error message]
    MSG --> RES([Response Generation\nshows error to user])
```

---

## 6. Document Processing Pipeline

How uploaded documents are processed and stored for semantic search.

```mermaid
graph LR
    Upload[Document Upload\nPDF / DOCX / TXT]
    Parsers[Parsers\nPyPDF2 + docx + txt]
    Chunker[LangChain\nRecursiveCharacterTextSplitter\nchunk=1000 overlap=200]
    Embedder[Embeddings\nSentenceTransformers\nall-MiniLM-L6-v2]
    Chroma[(ChromaDB\nPer-tenant collection)]

    Upload --> Parsers --> Chunker --> Embedder --> Chroma
```

---

## 7. Memory System Architecture

How WorkMate builds and retrieves long-term memory across conversations.

```mermaid
graph TD
    Chat[Chat Completion] --> Extractor[LLM Memory Extractor\napp/memory/memory_extractor.py]
    Extractor -->|importance >= 0.7| SQLite[(SQLite\nSource of Truth)]
    Extractor -->|importance >= 0.7| Embedder[Embed Content\nSentenceTransformers]
    Embedder --> Chroma[(ChromaDB\nSemantic Search)]

    Query[User Query] --> Retriever[Memory Retriever\napp/memory/memory_retriever.py]
    Retriever --> Chroma
    Retriever --> SQLite
```

---

## 8. Human-In-The-Loop Safety Flow

How WorkMate safely stages dangerous actions for human approval.

```mermaid
sequenceDiagram
    actor User
    participant Supervisor
    participant ProductivityAgent
    participant GuardrailNode
    participant SQLite
    participant StreamlitUI

    User->>Supervisor: "Create tasks from this transcript"
    Supervisor->>ProductivityAgent: Routes (intent=task_creation)
    ProductivityAgent->>SQLite: Save tasks as status=pending_approval
    ProductivityAgent->>GuardrailNode: Passes to guardrail check
    GuardrailNode->>StreamlitUI: "Tasks staged. Review in Pending Approvals tab."
    StreamlitUI->>User: Shows approval dashboard
    User->>StreamlitUI: Clicks Approve
    StreamlitUI->>SQLite: Update status=approved
```

---

## 9. Observability and Cost Tracking

How token usage is tracked across every LLM call.

```mermaid
graph LR
    Node[Any LangGraph Node] -->|LLM call completes| Counter[token_counter.py\ncount_tokens + add_usage]
    Counter -->|Accumulates| State[AgentState\ntoken_usage dict]
    State -->|Final node writes| SQLite[(SQLite\nTokenUsage table)]
    SQLite -->|Read by dashboard| UI[Streamlit\nCost Analytics Tab]
```

---

## 10. File Structure Map

```
workmate/
├── app/
│   ├── agent/
│   │   ├── graph.py            ← LangGraph wiring + retry conditional edge
│   │   ├── state.py            ← AgentState (retry_count, max_retries added)
│   │   ├── supervisor.py       ← Supervisor routing agent
│   │   ├── nodes.py            ← Shared nodes (error_handling has retry logic)
│   │   ├── llm_factory.py      ← Returns Anthropic or Ollama client
│   │   ├── local_llm.py        ← Rule-based fallback engine
│   │   ├── prompts.py          ← Prompt templates
│   │   └── subagents/
│   │       ├── research_agent.py     ← RAG + Memory with exponential backoff
│   │       └── productivity_agent.py ← Tasks + Email with exponential backoff
│   ├── db/                     ← SQLAlchemy models + session + init
│   ├── ingestion/              ← Document loaders, chunker, embedder
│   ├── memory/                 ← Memory extractor, retriever, store
│   ├── observability/
│   │   ├── tracing.py          ← @trace_node decorator
│   │   └── token_counter.py    ← Token usage + cost tracking
│   ├── rag/                    ← RAG service (search_documents)
│   ├── safety/                 ← Guardrails + HITL action logging
│   ├── tasks/                  ← Task extraction service
│   ├── email/                  ← Email drafting service
│   └── main.py                 ← FastAPI app + all API routes
├── frontend/
│   └── streamlit_app.py        ← Full UI (chat, uploads, approvals, analytics)
├── tests/
│   ├── test_agent.py           ← Graph init test
│   ├── test_chunker.py         ← Text splitter tests
│   ├── test_memory.py          ← Memory CRUD tests
│   ├── test_supervisor.py      ← 15 supervisor routing tests
│   └── eval_agent.py           ← Routing accuracy eval (6/6 = 100%)
├── ARCHITECTURE.md             ← This file (10 diagrams)
├── DESIGN_DOC.md               ← Architectural decisions and trade-offs
├── README.md                   ← Setup + feature overview
└── requirements.txt            ← Python dependencies
```

---

## Summary: What Makes This Production-Grade

| Feature | Implementation |
|---|---|
| **Multi-Agent Routing** | Supervisor with LLM + local pre-check |
| **Specialist Sub-Agents** | Research Agent + Productivity Agent |
| **Internal Backoff** | 3 attempts per tool call: 1s → 2s → 4s |
| **Graph-Level Retry** | error_handling loops back to Supervisor (max 3x) |
| **Graceful Degradation** | RAG failure falls back to raw query, not hard crash |
| **Human-In-The-Loop** | All actions staged for approval before execution |
| **Long-Term Memory** | Dual storage: SQLite (relational) + ChromaDB (semantic) |
| **Observability** | Token counting on every LLM call |
| **Test Coverage** | 25/25 tests passing, 100% supervisor routing accuracy |
