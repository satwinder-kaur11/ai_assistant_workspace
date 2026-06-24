# WorkMate Architecture Diagrams

> **Updated:** Reflects the new Multi-Agent Supervisor architecture (v2).

---

## 1. Full System Architecture

The top-level view of the entire WorkMate system — from the user typing a message
all the way through the multi-agent backend and back to the UI.

```mermaid
graph TD
    %% Styles
    classDef frontend  fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff
    classDef backend   fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff
    classDef super     fill:#7c3aed,stroke:#5b21b6,stroke-width:2px,color:#fff
    classDef agent     fill:#2563eb,stroke:#1d4ed8,stroke-width:2px,color:#fff
    classDef shared    fill:#0891b2,stroke:#0e7490,stroke-width:2px,color:#fff
    classDef db        fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:#fff
    classDef llm       fill:#ef4444,stroke:#b91c1c,stroke-width:2px,color:#fff

    %% User & Presentation Layer
    User((User))
    UI[Streamlit Frontend UI]:::frontend
    API[FastAPI Backend]:::backend

    %% Multi-Agent Orchestrator
    subgraph "LangGraph Multi-Agent Orchestrator"
        SUP[Supervisor Agent\nRoutes query to sub-agent]:::super

        subgraph "Specialist Sub-Agents"
            RA[Research Agent\nDocument QA + Memory Recall]:::agent
            PA[Productivity Agent\nTask Creation + Email Draft]:::agent
        end

        VAL[Validation Node]:::shared
        GRD[Guardrail HITL Node]:::shared
        RES[Response Generation]:::shared
        ERR[Error Handling]:::shared
    end

    %% Services
    subgraph "Backend Services"
        RAGSvc[RAG Service]:::backend
        TaskSvc[Task Service]:::backend
        EmailSvc[Email Service]:::backend
        MemSvc[Memory Service]:::backend
    end

    %% Data Stores
    SQLite[(SQLite\nLogs + Memory + Actions)]:::db
    Chroma[(ChromaDB\nVector Search)]:::db
    LLM{{LLM Engine\nOllama / Anthropic}}:::llm

    %% User flow
    User -->|Prompts & Files| UI
    UI -->|REST API Calls| API
    API -->|Sends Query| SUP

    %% Supervisor routing
    SUP -->|document_qa / memory_recall| RA
    SUP -->|task_creation / email_draft| PA
    SUP -->|chitchat| VAL

    %% Sub-agent service calls
    RA -->|Semantic Search| RAGSvc
    RA -->|Recall Memories| MemSvc
    PA -->|Extract Tasks| TaskSvc
    PA -->|Compose Email| EmailSvc

    %% Service → DB
    RAGSvc <-->|Embeddings| Chroma
    MemSvc <-->|CRUD| SQLite
    MemSvc <-->|Embeddings| Chroma
    TaskSvc -->|Save Pending Action| SQLite
    EmailSvc -->|Save Pending Action| SQLite

    %% LLM calls
    SUP -.->|Route query| LLM
    RAGSvc -.->|Synthesize answer| LLM
    TaskSvc -.->|Extract tasks| LLM
    EmailSvc -.->|Draft email| LLM
    RES -.->|Final synthesis| LLM

    %% Back to shared nodes
    RA --> VAL
    PA --> VAL
    VAL -->|Task or Email| GRD
    VAL -->|QA or Chitchat| RES
    VAL -->|Error| ERR
    GRD --> RES
    RES -->|Final Answer| API
    API -->|Response| UI
    UI -->|Visual Output| User
```

---

## 2. Multi-Agent Supervisor Decision Flow

How the Supervisor decides which sub-agent handles the request,
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

    RA --> OUT([Continues to Validation])
    PA --> OUT
    CA --> OUT
```

---

## 3. LangGraph State Machine (Full Node Graph)

The exact nodes and edges registered in `app/agent/graph.py`.

```mermaid
stateDiagram-v2
    [*] --> Supervisor

    Supervisor --> ResearchAgent    : next_agent = research_agent
    Supervisor --> ProductivityAgent: next_agent = productivity_agent
    Supervisor --> Validation       : next_agent = chitchat_agent

    ResearchAgent     --> Validation
    ProductivityAgent --> Validation

    Validation --> GuardrailCheck      : intent = task_creation or email_draft
    Validation --> ResponseGeneration  : intent = document_qa / memory_recall / chitchat
    Validation --> ErrorHandling       : validation_passed = False

    GuardrailCheck    --> ResponseGeneration
    ResponseGeneration --> [*]
    ErrorHandling      --> [*]
```

---

## 4. Document Processing Pipeline (RAG Ingestion)

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

## 5. Memory System Architecture

How WorkMate builds and retrieves long-term memory across conversations.

```mermaid
graph TD
    Chat[Chat Completion] --> Extractor[LLM Memory Extractor\napp/memory/memory_extractor.py]
    Extractor -->|Importance above threshold| Router{Threshold Check\nimportance >= 0.7}
    Router -->|Passes| SQLite[(SQLite\nSource of Truth)]
    Router -->|Passes| Embedder[Embed Content\nSentenceTransformers]
    Embedder --> Chroma[(ChromaDB\nSemantic Search)]

    Query[User Query] --> Retriever[Memory Retriever\napp/memory/memory_retriever.py]
    Retriever --> Chroma
    Retriever --> SQLite
```

---

## 6. Human-In-The-Loop (HITL) Safety Flow

How WorkMate safely stages dangerous actions for human approval
instead of executing them autonomously.

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

## 7. Observability & Cost Tracking

How token usage is tracked across every LLM call in the pipeline.

```mermaid
graph LR
    Node[Any LangGraph Node] -->|LLM call completes| Counter[token_counter.py\ncount_tokens + add_usage]
    Counter -->|Accumulates| State[AgentState\ntoken_usage dict]
    State -->|Final node writes to DB| SQLite[(SQLite\nTokenUsage table)]
    SQLite -->|Read by dashboard| UI[Streamlit\nCost Analytics Tab]
```

---

## 8. File Structure Map

```
workmate/
├── app/
│   ├── agent/
│   │   ├── graph.py            ← LangGraph wiring (Supervisor entry point)
│   │   ├── state.py            ← AgentState shared across all nodes
│   │   ├── supervisor.py       ← NEW: Supervisor routing agent
│   │   ├── nodes.py            ← Shared nodes (validation, response, error)
│   │   ├── llm_factory.py      ← Returns Anthropic or Ollama client
│   │   ├── local_llm.py        ← Rule-based fallback engine
│   │   ├── prompts.py          ← Prompt templates
│   │   └── subagents/
│   │       ├── research_agent.py     ← NEW: RAG + Memory sub-agent
│   │       └── productivity_agent.py ← NEW: Tasks + Email sub-agent
│   ├── db/                     ← SQLAlchemy models + session + init
│   ├── ingestion/              ← Document loaders, chunker, embedder
│   ├── memory/                 ← Memory extractor, retriever, store
│   ├── observability/
│   │   ├── tracing.py          ← @trace_node decorator
│   │   └── token_counter.py    ← NEW: Token usage + cost tracking
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
│   ├── test_supervisor.py      ← NEW: 15 supervisor routing tests
│   └── eval_agent.py           ← Routing accuracy evaluation (6/6 = 100%)
├── data/
│   ├── chroma/                 ← Local ChromaDB storage
│   └── uploads/                ← Uploaded document staging
├── ARCHITECTURE.md             ← This file
├── DESIGN_DOC.md               ← Architectural decisions & trade-offs
├── README.md                   ← Setup + feature overview
└── requirements.txt            ← Python dependencies
```
