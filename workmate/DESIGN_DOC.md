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
