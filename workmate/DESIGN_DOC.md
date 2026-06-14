# WorkMate AI: System Design Document

This document outlines the architectural decisions, trade-offs, future improvements, and production considerations for the WorkMate AI Workspace Assistant.

## 1. Architectural Decisions

**State Machine Orchestration (LangGraph)**
Instead of relying on rigid, linear LLM chains, the core agent is orchestrated using LangGraph as a cyclic state machine. This decision was made because an effective workspace assistant must handle complex routing. By separating the flow into distinct nodes (Intent Detection → Execution → Validation → Response), the system can dynamically route a user query to a specialized tool (like RAG or Email Drafting), validate the output, and even loop back if a tool fails, providing deterministic control over non-deterministic LLM outputs.

**Decoupled Client-Server Architecture**
The system strictly separates the presentation layer (Streamlit) from the business logic layer (FastAPI), communicating exclusively via REST API. This ensures that the core AI engine is client-agnostic. The decision guarantees that if the product scales to a Chrome Extension, a Desktop App (Electron), or a native Mobile App, the entire LangGraph backend remains untouched.

**Dual-Database Memory System**
The architecture utilizes two distinct databases to handle different types of data:
*   **ChromaDB (Vector Store):** Used for semantic retrieval. It handles the high-dimensional embeddings required for RAG (Retrieval-Augmented Generation) and semantic memory searches.
*   **SQLite (Relational Store):** Used for structured logging, user preferences, and managing the state of Human-in-the-Loop (HITL) actions via the `ActionLog` table.

**Local-First & Privacy-Centric Design**
Enterprise AI adoption is heavily bottlenecked by data privacy concerns. The architecture was specifically designed to support local, air-gapped execution. By supporting local embeddings (`SentenceTransformers`) and local LLM inference (`Ollama`), users can process highly sensitive internal documents without sending data to third-party cloud providers like OpenAI or Anthropic.

---

## 2. Trade-offs

**Streamlit vs. React/Next.js**
*   **Decision:** Streamlit was chosen for the frontend.
*   **Trade-off:** Streamlit allowed for incredibly rapid prototyping, enabling the entire UI (including chat, file uploads, and data tables) to be built in hours using only Python. The trade-off is a lack of deep UI customization and sub-optimal state management, as Streamlit re-runs the entire script top-to-bottom on every user interaction, which can cause minor latency compared to a highly optimized React SPA.

**SQLite vs. PostgreSQL**
*   **Decision:** SQLite is used as the default relational database.
*   **Trade-off:** SQLite requires zero configuration, meaning the application can be cloned and run instantly without spinning up Docker containers or database servers. The trade-off is that SQLite struggles with high concurrent write loads. In a multi-user production environment, this would become a bottleneck, requiring a migration to PostgreSQL (which the SQLAlchemy ORM easily supports via a `DB_URL` change).

**Local LLM Inference vs. Cloud AI**
*   **Decision:** Supporting local execution via Ollama (`llama3.2`) as the primary alternative to Cloud AI (Anthropic).
*   **Trade-off:** When the Anthropic API key is omitted or the cloud service is unreachable, the system seamlessly routes execution to a local LLM rather than crashing or relying on hardcoded scripts. This guarantees the application continues to provide intelligent, contextual answers, task extraction, and email drafting without sending data off-device. The trade-off is that local models require sufficient user hardware (RAM/GPU) to run at speeds comparable to cloud providers, but the massive gain in data privacy and offline capability makes this the superior architectural choice. (Note: A secondary regex-based engine exists strictly to prevent 500 Server Errors if the local LLM daemon itself crashes).

---

## 3. Future Improvements

**True API Execution Integrations**
Currently, the Human-in-the-Loop (HITL) system safely intercepts dangerous actions (like sending emails or creating tasks) and logs them as "Pending Approvals" in the database. The most critical next step is wiring the "Approve" button to execute actual external API calls (e.g., the Gmail API for sending the draft, or the Jira/Linear API for posting the task).

**Proactive Background Agents**
The current paradigm is purely reactive: the user speaks, the agent responds. Future iterations will introduce background cron-agents that monitor specific environments (e.g., watching a designated Slack channel or a local incoming folder) and proactively trigger LangGraph workflows to stage task lists or draft summaries before the user even asks.

**Advanced Document Parsing (Vision Models)**
The current RAG pipeline uses standard text extraction (`PyPDF2`), which often destroys complex formatting like tables, columns, and charts. Upgrading the ingestion pipeline to use Vision-Language Models (VLMs) or advanced parsers (like LlamaParse) will retain structural fidelity, dramatically improving the accuracy of document-based QA.

---

## 4. Production Considerations

**Security, Authentication, and Multi-Tenancy**
While the database schema natively supports `tenant_id` to separate data between organizations, the current prototype assumes these IDs are passed harmlessly from the frontend. For production, the FastAPI backend must be secured behind an Identity Provider (Auth0, AWS Cognito, etc.). The `user_id` and `tenant_id` must be cryptographically extracted and verified from incoming JWT (JSON Web Tokens) headers to prevent privilege escalation.

**Asynchronous Task Queues**
Operations like document embedding (chunking large PDFs and writing to ChromaDB) and background memory extraction are currently handled using FastAPI's lightweight `BackgroundTasks`. In a production environment with heavy traffic, these CPU-intensive operations would block API workers. They must be offloaded to a dedicated distributed task queue (such as Celery with RabbitMQ, or Redis Queue) running on separate worker nodes.

**Enterprise Observability**
While the prototype includes basic `@trace_node` logging for debugging, a production AI system requires deep, token-level observability. The LangGraph orchestration must be hooked into an enterprise telemetry system (like LangSmith or Datadog) to monitor LLM hallucination rates, exact token costs per tenant, and latency bottlenecks at the granular node level.
