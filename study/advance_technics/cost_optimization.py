import hashlib
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import NotRequired, TypedDict

# Must run before any library builds an SSL context (httpx, Langfuse's OTLP
# exporter, etc.). Makes Python verify TLS via the OS certificate store
# (Windows CryptoAPI) instead of parsing certifi's bundled cafile - works
# around a known CPython bug where ssl.create_default_context(cafile=...)
# does ~143,000 disk operations on Windows and can look like a hang/freeze.
# See: https://github.com/python/cpython/pull/137596
import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler

load_dotenv()
logger = logging.getLogger(__name__)


def _extract_text(content: str | list[str | dict]) -> str:
    """Normalize a LangChain message's `.content` into plain text.

    `BaseMessage.content` is typed as `str | list[str | dict]` to allow for
    multimodal/structured content blocks (images, tool-call fragments,
    reasoning segments). For a plain-text chat completion it's almost always
    a plain `str`, but code that does `result: str = response.content`
    without narrowing is exactly what type checkers (Pyrefly, mypy, Pyright)
    correctly flag as a bad-return / bad-argument-type error - the
    declared type doesn't match what the attribute can actually hold.
    This makes the narrowing explicit instead of relying on the assumption
    silently continuing to hold.
    """
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


class TierConfig(TypedDict):
    """Execution parameters for one effort tier."""

    reasoning_effort: str
    temperature: float
    retrieval_k: int
    cost_multiplier: float


class CacheStats(TypedDict):
    """Cache hit/miss counters. hit_rate is a raw fraction (0.0-1.0) -
    format it for display where you print/log it, not in the data itself.
    """

    hits: int
    misses: int
    hit_rate: float


class RAGResult(TypedDict):
    """Result of ProductionRAGPipeline.run(). retrieved_docs_count is only
    present on a cache miss (nothing was retrieved on a cache hit).
    """

    response: str
    from_cache: bool
    effort: str
    retrieved_docs_count: NotRequired[int]


# === Effort-Based Model Router & LLM Factory ===
class EffortModelRouter:
    """Maps an effort tier ('low'/'medium'/'high') to execution parameters,
    and hands out a ChatGroq client configured for that tier.

    Clients are cached per tier (get_llm) rather than rebuilt on every call -
    constructing a ChatGroq instance builds its own HTTP client/connection
    pool, so recreating one per request is wasted work and adds latency.
    """

    def __init__(self, model_name: str = "openai/gpt-oss-20b") -> None:
        self.model_name = model_name
        self._llm_cache: dict[str, ChatGroq] = {}
        # One handler is enough; LangChain callbacks are cheap and stateless.
        self._langfuse_handler = CallbackHandler()

    @staticmethod
    def get_tier_config(effort: str) -> TierConfig:
        """Map effort level ('low', 'medium', 'high') to execution parameters.

        Static because it's a pure function of `effort` - it doesn't read or
        write any instance state, so it doesn't need `self` (this is what
        your type checker's "may be 'static'" hint is telling you: the
        method never touches `self`, so tying it to an instance is
        misleading - it implies a dependency on instance state that isn't
        actually there. You can still call it as `router.get_tier_config(x)`
        on an instance; `@staticmethod` just documents that it's pure).
        """
        effort = effort.lower().strip()
        if effort == "low":
            return {
                "reasoning_effort": "low",
                "temperature": 0.1,
                "retrieval_k": 2,
                "cost_multiplier": 1.0,
            }
        elif effort == "high":
            return {
                "reasoning_effort": "high",
                "temperature": 0.0,
                "retrieval_k": 6,
                "cost_multiplier": 3.5,
            }
        else:  # medium / default
            return {
                "reasoning_effort": "medium",
                "temperature": 0.0,
                "retrieval_k": 4,
                "cost_multiplier": 1.8,
            }

    def get_llm(self, effort: str) -> ChatGroq:
        """Return a cached ChatGroq client configured for this effort tier."""
        cache_key = effort.lower().strip()
        if cache_key not in self._llm_cache:
            config = self.get_tier_config(cache_key)
            # api_key intentionally omitted: ChatGroq reads GROQ_API_KEY from
            # the environment itself as a proper pydantic SecretStr. Passing
            # a plain `str` here is what triggers "str is not assignable to
            # SecretStr | None" from static type checkers.
            self._llm_cache[cache_key] = ChatGroq(
                model=self.model_name,
                temperature=config["temperature"],
                reasoning_effort=config["reasoning_effort"],
                timeout=30,
                max_retries=2,
            )
        return self._llm_cache[cache_key]

    @observe(name="effort_routed_invoke", capture_input=False, capture_output=False)
    def invoke(self, query: str, effort: str = "medium") -> tuple[str, TierConfig]:
        """Invoke the LLM with dynamic effort configuration."""
        config = self.get_tier_config(effort)
        langfuse = get_client()
        langfuse.update_current_span(input=query, metadata={"effort": effort, **config})

        llm = self.get_llm(effort)
        try:
            response = llm.invoke(query, config={"callbacks": [self._langfuse_handler]})
        except Exception as e:
            raise RuntimeError(f"LLM invocation failed (effort={effort}): {e}") from e

        result = _extract_text(response.content)
        langfuse.update_current_span(output=result)
        return result, config


# === PGVector Semantic Cache (with real exact-hash lookup + tier scoping) ===
class PGVectorSemanticCache:
    """Two-tier cache: an in-process exact-match lookup by query hash, then
    a PGVector semantic similarity search as a fallback.

    `namespace` scopes cache entries (e.g. by effort tier, and/or a prompt
    version - see ProductionRAGPipeline) so, for example, a cheap "low"
    effort answer is never served for a "high" effort request just because
    the query text matches, and a response cached under an old, buggy
    prompt template stops being served the moment you bump the version.

    `is_cacheable` is a predicate checked before every write; it exists so a
    known failure-mode response (an empty string, a "the context doesn't
    contain enough information" fallback, an error message) never gets
    cached in the first place - since that's the actual cause a bad answer
    can outlive the bug that produced it. Defaults to "non-empty".

    `ttl`, if set, expires entries after that duration as a backstop against
    anything the above two don't catch (a stale fact, an updated knowledge
    base). Entries written before ttl was set (or by a version of this class
    without it) have no `cached_at` and are treated as never-expiring.

    Note: the exact-match tier is a plain in-process dict, so it is NOT
    shared across worker processes in a multi-process deployment (gunicorn
    with several workers, etc.) - each process warms its own. For a shared
    exact-match tier across processes, back it with Redis instead.
    """

    def __init__(
        self,
        connection_string: str,
        collection_name: str = "semantic_cache",
        similarity_threshold: float = 0.92,
        is_cacheable: Callable[[str], bool] | None = None,
        ttl: timedelta | None = None,
    ):
        self.embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
        self.threshold = similarity_threshold
        self.is_cacheable = is_cacheable or (lambda response: bool(response.strip()))
        self.ttl = ttl
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=connection_string,
            # Explicit rather than relying on PGVector's default: the score
            # -> similarity conversion below (1 - distance) is only valid
            # for cosine distance, so pin it instead of assuming it silently.
            distance_strategy=DistanceStrategy.COSINE,
        )
        self._exact_cache: dict[str, str] = {}

    @staticmethod
    def _hash_query(query: str, namespace: str = "") -> str:
        """Create an MD5 hash for fast exact-match checks, scoped by namespace."""
        return hashlib.md5(f"{namespace}:{query.lower().strip()}".encode()).hexdigest()

    def _is_expired(self, doc: Document) -> bool:
        if self.ttl is None:
            return False
        cached_at_str = doc.metadata.get("cached_at")
        if not cached_at_str:
            return False  # written before TTL tracking existed - don't break it
        cached_at = datetime.fromisoformat(cached_at_str)
        return datetime.now(UTC) - cached_at > self.ttl

    def get(self, query: str, namespace: str = "") -> str | None:
        """Check the cache: exact-hash first, then semantic similarity."""
        query_hash = self._hash_query(query, namespace)

        exact = self._exact_cache.get(query_hash)
        if exact is not None:
            logger.info("Cache HIT (exact match)")
            return exact

        try:
            filter_ = {"namespace": namespace} if namespace else None
            results = self.vector_store.similarity_search_with_score(
                query, k=1, filter=filter_
            )
            if not results:
                return None

            doc, distance = results[0]
            if self._is_expired(doc):
                logger.info("Cache entry expired; treating as a miss")
                return None

            # Cosine distance is in [0, 2]; similarity = 1 - distance can go
            # negative for near-opposite vectors, which we clamp to 0 rather
            # than treat as a match.
            similarity = max(0.0, 1.0 - distance)

            if similarity >= self.threshold:
                logger.info("Cache HIT (semantic, similarity=%.4f)", similarity)
                return doc.metadata.get("response")
        except Exception:
            logger.error("Cache lookup failed", exc_info=True)

        return None

    def set(self, query: str, response: str, namespace: str = "") -> None:
        """Store a query/response pair in both cache tiers.

        Silently skips the write (logging why) if `is_cacheable(response)`
        is False - e.g. a "don't know" fallback should never poison the
        cache for every semantically-similar question asked afterward.
        """
        if not self.is_cacheable(response):
            logger.info("Skipping cache write: response failed is_cacheable check")
            return

        query_hash = self._hash_query(query, namespace)
        self._exact_cache[query_hash] = response

        try:
            doc = Document(
                page_content=query,
                metadata={
                    "response": response,
                    "query_hash": query_hash,
                    "namespace": namespace,
                    "cached_at": datetime.now(UTC).isoformat(),
                },
            )
            self.vector_store.add_documents([doc])
        except Exception:
            logger.error("Cache write failed", exc_info=True)

    def invalidate_query(self, query: str, namespace: str = "") -> None:
        """Remove a specific query from both cache tiers.

        PGVector.delete() only deletes by id - there's no metadata `filter`
        parameter on it (passing one is silently ignored). So this looks up
        the id(s) of matching entries via similarity_search_with_score's
        filter first, then deletes those ids. Note this still costs an
        embedding call (embed_query), since that method always embeds the
        query before searching - there's no id-free metadata-only lookup in
        PGVector's public API.
        """
        query_hash = self._hash_query(query, namespace)
        self._exact_cache.pop(query_hash, None)

        try:
            matches = self.vector_store.similarity_search_with_score(
                query, k=5, filter={"query_hash": query_hash}
            )
            ids = [doc.id for doc, _ in matches if doc.id is not None]
            if ids:
                self.vector_store.delete(ids=ids)
                logger.info(
                    "Invalidated %d cached entr%s",
                    len(ids),
                    "y" if len(ids) == 1 else "ies",
                )
            else:
                logger.info("No cached entry found to invalidate for this query")
        except Exception:
            logger.error(
                "Failed to invalidate query from PGVector cache", exc_info=True
            )

    def clear(self) -> None:
        """Clear the cache entirely: drops the collection and recreates it
        empty, so the cache stays usable immediately afterward (skipping
        create_collection() here would make every subsequent set() raise
        ValueError: Collection not found).
        """
        self._exact_cache.clear()
        try:
            self.vector_store.delete_collection()
            self.vector_store.create_collection()
            logger.info("Cleared PGVector semantic cache collection.")
        except Exception:
            logger.error("Failed to clear PGVector cache collection", exc_info=True)


# === Production Cached LLM Wrapper ===
class ProductionCachedLLM:
    """LLM wrapper integrating persistent PGVector caching and Langfuse tracing."""

    def __init__(
        self,
        connection_string: str,
        router: EffortModelRouter | None = None,
        effort: str = "medium",
    ):
        self.router = router or EffortModelRouter()
        self.effort = effort
        self.cache = PGVectorSemanticCache(connection_string=connection_string)
        self.hits = 0
        self.misses = 0

    @observe(name="production_cached_invoke", capture_input=False, capture_output=False)
    def invoke(self, query: str) -> tuple[str, bool]:
        """Invoke LLM with a semantic cache check."""
        langfuse = get_client()
        langfuse.update_current_span(input=query, metadata={"effort": self.effort})

        cached_response = self.cache.get(query, namespace=self.effort)
        if cached_response:
            self.hits += 1
            langfuse.update_current_span(
                output=cached_response, metadata={"cache_hit": True}
            )
            return cached_response, True

        self.misses += 1
        llm = self.router.get_llm(self.effort)
        try:
            response = llm.invoke(
                query, config={"callbacks": [self.router._langfuse_handler]}
            )
        except Exception as e:
            raise RuntimeError(f"LLM invocation failed: {e}") from e

        result = _extract_text(response.content)
        self.cache.set(query, result, namespace=self.effort)

        langfuse.update_current_span(output=result, metadata={"cache_hit": False})
        return result, False

    def get_stats(self) -> CacheStats:
        """Get current cache hit/miss statistics.

        Note: hits/misses are plain ints with no locking. Fine for a single
        thread/process; under concurrent requests (async server, thread
        pool, multiple workers) these counters can race, and won't be
        aggregated across worker processes either way. For real production
        metrics, prefer emitting these as Langfuse/Prometheus counters over
        relying on in-process state.
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": hit_rate}


# === RAG Pipeline with Sample Documents ===
def get_sample_documents() -> list[Document]:
    """Generate clear test documents for the RAG pipeline."""
    return [
        Document(
            page_content=(
                "LangGraph is a library for building stateful, multi-agent applications "
                "with LLMs, built on top of LangChain. It allows you to define agent "
                "workflows as graphs with nodes and edges."
            ),
            metadata={"source": "langgraph_overview.txt", "category": "architecture"},
        ),
        Document(
            page_content=(
                "PGVector extends PostgreSQL with vector datatypes and efficient similarity "
                "search. In LangChain, you can use langchain-postgres to store embeddings "
                "and perform vector search directly in your Postgres database."
            ),
            metadata={"source": "pgvector_integration.txt", "category": "storage"},
        ),
        Document(
            page_content=(
                "Langfuse is an open-source LLM engineering platform that provides tracing, "
                "prompt management, evaluations, and cost tracking for applications "
                "built with LangChain and LangGraph."
            ),
            metadata={
                "source": "langfuse_observability.txt",
                "category": "observability",
            },
        ),
    ]


class ProductionRAGPipeline:
    """End-to-end RAG pipeline using PGVector and Langfuse.

    `prompt_version` is folded into the cache namespace alongside the effort
    tier. Bump it whenever the prompt template changes: old cache entries
    (written under the previous version) simply stop matching - no manual
    cleanup needed, and a prompt bug like an accidentally-empty context can
    never permanently poison the cache for that query's neighborhood again.
    Old entries become harmless orphaned rows; clear() or a TTL removes them
    eventually.
    """

    def __init__(self, connection_string: str, prompt_version: str = "v1"):
        self.prompt_version = prompt_version
        self.router = EffortModelRouter()
        self.cache = PGVectorSemanticCache(
            connection_string=connection_string,
            # Never cache the "I couldn't find an answer" fallback - if it's
            # showing up, something upstream (retrieval, an empty context,
            # a template bug) likely misfired, and caching it would make the
            # cache actively wrong for every similar question afterward.
            is_cacheable=lambda r: (
                bool(r.strip())
                and "don't know" not in r.lower()
                and "do not know" not in r.lower()
            ),
            ttl=timedelta(days=7),
        )
        self.embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name="production_kb",
            connection=connection_string,
            distance_strategy=DistanceStrategy.COSINE,
        )
        self.prompt = ChatPromptTemplate.from_template(
            "Answer the user's question using only the provided context below.\n"
            "If you do not know the answer based on the context, state that "
            "you don't know.\n\nContext:\n{context}\n\nQuestion: {question}"
        )

    def _cache_namespace(self, effort: str) -> str:
        return f"{effort}:{self.prompt_version}"

    def initialize_kb(self) -> None:
        """Populate the vector store with test documents."""
        docs = get_sample_documents()
        self.vector_store.add_documents(docs)
        logger.info("Initialized RAG knowledge base with sample documents.")

    @observe(name="rag_pipeline_execution", capture_input=False, capture_output=False)
    def run(self, query: str, effort: str = "medium") -> RAGResult:
        """Execute RAG flow with caching, effort routing, and telemetry."""
        langfuse = get_client()
        langfuse.update_current_span(input=query, metadata={"effort": effort})
        namespace = self._cache_namespace(effort)

        # 1. Check the cache, scoped to this effort tier and prompt version -
        # see PGVectorSemanticCache's and this class's docstrings for why
        # both matter.
        cached_result = self.cache.get(query, namespace=namespace)
        if cached_result:
            langfuse.update_current_span(
                output=cached_result, metadata={"cache_hit": True}
            )
            return {"response": cached_result, "from_cache": True, "effort": effort}

        # 2. Get dynamic configuration based on the effort tier
        config = self.router.get_tier_config(effort)

        # 3. Retrieve documents dynamically based on effort (e.g., k=2 for low, k=6 for high)
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": config["retrieval_k"]}
        )
        try:
            retrieved_docs = retriever.invoke(query)
        except Exception as e:
            raise RuntimeError(f"Document retrieval failed: {e}") from e

        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # 4. Invoke the (tier-cached) LLM with effort parameters
        llm = self.router.get_llm(effort)
        chain = self.prompt | llm
        try:
            response = chain.invoke(
                {"context": context, "question": query},
                config={"callbacks": [self.router._langfuse_handler]},
            )
        except Exception as e:
            raise RuntimeError(f"LLM invocation failed (effort={effort}): {e}") from e

        result_text = _extract_text(response.content)

        # 5. Save to the cache, scoped to this effort tier and prompt version.
        # set() itself will skip the write if result_text looks like a
        # "don't know" fallback (see is_cacheable above).
        self.cache.set(query, result_text, namespace=namespace)

        langfuse.update_current_span(
            output=result_text,
            metadata={"cache_hit": False, "retrieved_docs_count": len(retrieved_docs)},
        )
        return {
            "response": result_text,
            "from_cache": False,
            "effort": effort,
            "retrieved_docs_count": len(retrieved_docs),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY environment variable not set")

    DB_CONNECTION = os.environ.get(
        "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/vectordb"
    )

    langfuse = get_client()
    if not langfuse.auth_check():
        logger.warning(
            "Langfuse credentials missing or invalid - traces will not be sent. "
            "Set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL."
        )

    # 1. Test RAG Pipeline
    rag_system = ProductionRAGPipeline(connection_string=DB_CONNECTION)
    # rag_system.initialize_kb()  # Run once to ingest sample docs

    # Test with low vs high effort - these are now genuinely independent,
    # since the cache is scoped per effort tier.
    res_low = rag_system.run(
        "How exactly LangGraph state management works?", effort="low"
    )
    print(f"Low Effort Result: {res_low}")

    res_high = rag_system.run(
        "How exactly LangGraph state management works?", effort="high"
    )
    print(f"High Effort Result: {res_high}")

    langfuse.flush()
