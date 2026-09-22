# Graph Report - production_rag  (2026-09-22)

## Corpus Check
- Corpus is ~16,421 words - fits in a single context window. You may not need a graph.

## Summary
- 339 nodes · 442 edges · 22 communities (19 shown, 3 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 19 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Production Agent
- Security Pipeline
- Observability
- Cost Optimization
- Security Pipeline
- Security Pipeline
- Observability
- Hybrid Retrieval
- Chunking Pipeline
- Cost Optimization
- Observability
- Community 11
- Chunking Pipeline
- Chunking Pipeline
- Cost Optimization
- Vector Storage
- Community 20

## God Nodes (most connected - your core abstractions)
1. `PGVectorSemanticCache` - 11 edges
2. `TokenBudget` - 10 edges
3. `ResponseCache` - 9 edges
4. `chat()` - 9 edges
5. `get_settings()` - 8 edges
6. `lifespan()` - 8 edges
7. `EffortModelRouter` - 8 edges
8. `CustomBM25Retriever` - 8 edges
9. `ProductionAgent` - 7 edges
10. `SecurityPipeline` - 7 edges

## Surprising Connections (you probably didn't know these)
- `lifespan()` --calls--> `ResponseCache`  [EXTRACTED]
  app/main.py → app/cache.py
- `lifespan()` --calls--> `SecurityPipeline`  [EXTRACTED]
  app/main.py → app/security.py
- `chat()` --uses--> `ChatRequest`  [INFERRED]
  app/main.py → app/models.py
- `chat()` --uses--> `ChatResponse`  [INFERRED]
  app/main.py → app/models.py
- `chat()` --calls--> `RequestTimer`  [EXTRACTED]
  app/main.py → app/monitoring.py

## Import Cycles
- None detected.

## Communities (22 total, 3 thin omitted)

### Community 0 - "Production Agent"
Cohesion: 0.06
Nodes (40): AgentState, ProductionAgent, observe, TypedDict, Invoke the agent with the user message, state for the production agent. uses annotated with add_messages reducer for…, Production LangGraph agent with : - Retry on failure (model fallback) -…, Build the LangGraph state machine (+32 more)

### Community 1 - "Security Pipeline"
Cohesion: 0.07
Nodes (28): demo_secure_pipeline(), _extract_text(), GuardResult, InputSanitizer, OutputValidator, PIIDetector, ProcessResult, observe (+20 more)

### Community 2 - "Observability"
Cohesion: 0.06
Nodes (26): Exception, main(), observe, TypedDict, Token-budgeted LLM wrapper around ChatGroq, instrumented with Langfuse., Check whether `text` fits within the configured token budget., Record token usage for a request that actually reached the LLM. Only call this…, Record a request that was blocked before the LLM was ever called. Deliberately… (+18 more)

### Community 3 - "Cost Optimization"
Cohesion: 0.08
Nodes (25): CacheStats, EffortModelRouter, _extract_text(), get_sample_documents(), ProductionCachedLLM, ProductionRAGPipeline, observe, TypedDict (+17 more)

### Community 4 - "Security Pipeline"
Cohesion: 0.10
Nodes (20): _extract_text(), GuardResult, OutputValidator, PIIDetector, ProcessResult, observe, TypedDict, Detect and mask personally identifiable information and secrets. These are… (+12 more)

### Community 5 - "Security Pipeline"
Cohesion: 0.08
Nodes (19): InputSanitizer, Remove potentially dangerous content and format control sequences., Use LLM to detect malicious intent and unsafe prompts., Sanitize user input and detect prompt injection attempts. This is a cheap…, Check if input contains suspicious injection patterns., SecurityGuard, ChatGroq, create_base_vectorstore() (+11 more)

### Community 6 - "Observability"
Cohesion: 0.13
Nodes (12): demo_monitoring(), InstrumentedLLM, JSONFormatter, MetricsCollector, observe, Monitoring and Logging for Production Structured logging, metrics, and alerts…, LLM with full instrumentation., Demonstrate monitoring. (+4 more)

### Community 7 - "Hybrid Retrieval"
Cohesion: 0.14
Nodes (13): BaseRetriever, CallbackManagerForRetrieverRun, create_retrievers(), Need to focus on only get what matters for the asked question. Quality over…, Create vector, BM25, and ensemble retrievers, Test a retriever with a query, test_query(), Config (+5 more)

### Community 8 - "Chunking Pipeline"
Cohesion: 0.13
Nodes (13): Chroma, HNSW, IVFFlat, pgvector, demo_parent_document_retriever(), Parent Document Retriever: small chunks for search, large for context. with…, get_vector_store(), Document (+5 more)

### Community 9 - "Cost Optimization"
Cohesion: 0.16
Nodes (9): PGVectorSemanticCache, Document, Two-tier cache: an in-process exact-match lookup by query hash, then a PGVector…, Create an MD5 hash for fast exact-match checks, scoped by namespace., Check the cache: exact-hash first, then semantic similarity., Store a query/response pair in both cache tiers. Silently skips the write…, Remove a specific query from both cache tiers. PGVector.delete() only deletes…, Clear the cache entirely: drops the collection and recreates it empty, so the… (+1 more)

### Community 10 - "Observability"
Cohesion: 0.14
Nodes (6): JSONFormatter, Context manager for timing requests, Create a structured JSON logger, Formate log records as JSON for log aggregration (ELK, Datadog, etc.), RequestTimer, Logger

### Community 11 - "Community 11"
Cohesion: 0.22
Nodes (5): Create a cache key from the normalized query, Get cached response if it hits and hasn't expired. return None on cache miss., cache performance statistics, In Memory response cache with TTL(time-to-live) In production, replace this…, ResponseCache

### Community 13 - "Chunking Pipeline"
Cohesion: 0.40
Nodes (3): Text splitters and chunking strategies Optimizing document chunks for RAG, You can use separators to guide how it should split the text. Overlap is…, recursive_splitter()

### Community 14 - "Chunking Pipeline"
Cohesion: 0.67
Nodes (4): Chunking, Embedding Model, RAG, Retrieval Quality

## Knowledge Gaps
- **6 isolated node(s):** `production-rag`, `Config`, `Chunking`, `Observability`, `Cost Optimization` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EffortModelRouter` connect `Cost Optimization` to `Cost Optimization`?**
  _High betweenness centrality (0.157) - this node is a cross-community bridge._
- **Why does `main()` connect `Observability` to `Security Pipeline`?**
  _High betweenness centrality (0.155) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `ChatGroq` (e.g. with `.__init__()` and `.__init__()`) actually correct?**
  _`ChatGroq` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `chat()` (e.g. with `ChatRequest` and `ChatResponse`) actually correct?**
  _`chat()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `production-rag`, `Config`, `Chunking` to the rest of the system?**
  _6 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Production Agent` be split into smaller, more focused modules?**
  _Cohesion score 0.06471631205673758 - nodes in this community are weakly interconnected._
- **Should `Security Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.06707317073170732 - nodes in this community are weakly interconnected._