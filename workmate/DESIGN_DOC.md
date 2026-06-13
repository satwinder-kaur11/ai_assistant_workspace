# WorkMate Design Document

## Architectural Decisions & Trade-offs

### 1. Vector Database: ChromaDB vs. Alternatives
**Decision**: Used ChromaDB in Persistent local mode.
**Trade-off**: ChromaDB is lightweight, runs locally without extra infrastructure, and provides sufficient performance for prototyping. Qdrant or Pinecone might offer better scalability for billions of vectors, but for a prototype, avoiding a separate Docker service for the vector DB significantly reduces setup friction.

### 2. Memory System: Dual Storage
**Decision**: Memories are stored in both ChromaDB and PostgreSQL.
**Trade-off**: Storing in ChromaDB allows fast semantic retrieval. Storing in PostgreSQL allows for relational queries (e.g., fetching all memories for a user, tracking `last_accessed_at` for decay logic) and acts as the true source of record. Keeping them synchronized adds slight application complexity but provides the optimal retrieval capabilities.

### 3. Agent Orchestration: LangGraph vs. Pure Prompts
**Decision**: Used LangGraph with explicit StateGraph and conditional edges.
**Trade-off**: Pure prompt routing (e.g., using ReAct) is easier to setup but heavily relies on LLM determinism, which can fail. LangGraph allows explicit, deterministic routing (e.g., validation failure always routes to error handling). It is more verbose but significantly more reliable, observable, and maintainable for production.

### 4. Background Tasks: FastAPI vs. Celery
**Decision**: Used FastAPI `BackgroundTasks` for the prototype.
**Trade-off**: Simplifies the prototype by eliminating the need for Redis/RabbitMQ. However, if the server restarts, pending tasks are lost. A true production system requires a distributed task queue.

## Scaling Plan for 100K Users / 10M Docs / 50M Memories

### 1. Vector Store Sharding & Migration
- **Strategy**: Migrate from local ChromaDB to a managed vector database (e.g., Qdrant, Milvus, or Pinecone). Shard the collections by `tenant_id`.
- **Why**: 10M documents require robust HNSW indexing and memory management that standalone local Chroma cannot easily distribute across multiple nodes.

### 2. Async Ingestion Pipeline
- **Strategy**: Introduce a robust task queue (Celery or Temporal) backed by Redis/RabbitMQ.
- **Why**: Document parsing and embedding are compute-intensive. Moving them to dedicated worker nodes prevents API saturation and ensures reliable retries on failures.

### 3. Relational Database Scaling
- **Strategy**: Move from SQLite to a highly available managed PostgreSQL instance (e.g., AWS Aurora). Implement read replicas to offload heavy read loads (e.g., fetching documents or traces). Add connection pooling (PgBouncer).

### 4. Embedding Cache and Batch Optimization
- **Strategy**: Implement Redis caching for embeddings of common/repeated queries. Batch embeddings during ingestion (e.g., process 100 chunks per API call) instead of processing chunk-by-chunk to maximize throughput.

### 5. Memory Archival (Cold Storage)
- **Strategy**: Implement a cron worker that checks the `last_accessed_at` and `importance_score` fields in the `Memory` table. Memories that are rarely accessed and have low importance can be archived to cold storage (e.g., S3 + Athena) and removed from the active vector index to reduce latency and memory bloat.

### 6. Model Routing & Prompt Caching
- **Strategy**: Route simple intents and memory extraction tasks to a smaller, faster model (e.g., Claude 3.5 Haiku) to save costs. Use Anthropic's Prompt Caching for large RAG contexts in final response generation to significantly lower Time To First Token (TTFT) and token costs. Implement multi-tenancy PII redaction prior to LLM calls.

## Product Thinking

### Question 1: Why do most AI workplace assistants fail?
Most AI workplace assistants fail because they operate in isolation from the user's actual workflows and lack true context. Key reasons include:
- **Lack of Actionability:** Many assistants are just chatbots that can summarize text but cannot execute actions (like creating a ticket or sending an email) on the user's behalf.
- **Context Amnesia:** They treat every session as a blank slate. If an assistant doesn't remember a user's preferences, teammates, or past decisions, it becomes frustrating to use.
- **Trust and Safety Issues:** Black-box execution without Human-In-The-Loop (HITL) guardrails leads to mistakes (e.g., sending an incorrect email or deleting records).
- **Poor Integration:** A lack of deep integration into existing systems of record (Jira, Salesforce, Google Workspace) limits utility.

### Question 2: How would you differentiate this solution from ChatGPT, Claude, and Notion AI?
- **vs. ChatGPT / Claude:** While ChatGPT and Claude are generalized horizontal LLMs, WorkMate is a deeply integrated **Agentic Workspace Assistant**. It has a dedicated memory architecture (Semantic, Episodic, Preference) that persists across sessions. It also features an explicit Action Execution Layer with Human-In-The-Loop workflows, meaning it doesn't just generate text—it prepares concrete tasks and emails for user approval.
- **vs. Notion AI:** Notion AI excels at text manipulation within documents. WorkMate differentiates itself by acting as an orchestrator across the entire "Work Operating System" (Tasks, Emails, Documents). It is intent-driven and capable of complex multi-step reasoning, retrieving past interactions, and taking external actions beyond simple document editing.

### Question 3: If given three additional months, what would you build next and why?
If given three more months, I would build:
1. **Third-Party Integrations Layer:** Connect the `task_service` to Jira/Asana and `email_service` to Gmail/Outlook APIs. Without this, the assistant remains a silo.
2. **Advanced Multi-Agent Collaboration:** Break down the single LangGraph agent into specialized sub-agents (e.g., a dedicated "Research Agent" for deep document analysis, a "Scheduling Agent" for calendar management). This improves reliability for complex tasks.
3. **Proactive Assistance (Background Agents):** Shift from purely reactive (waiting for a prompt) to proactive. The system could scan incoming emails or documents asynchronously and surface proposed tasks to the "Pending Approvals" queue before the user even asks.
4. **Graph-Based Memory (Knowledge Graph):** Upgrade the memory system from simple vector chunks to a Knowledge Graph (e.g., Neo4j). This would allow the AI to understand complex relationships (e.g., "Sarah manages the Engineering team, which owns Sprint 1").
