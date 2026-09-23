import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from langfuse import observe
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.agent import ProductionAgent
from app.cache import ResponseCache
from app.config import get_settings
from app.models import ChatRequest, ChatResponse, HealthResponse, MetricsResponse
from app.monitoring import MetricsCollector, RequestTimer
from app.security import SecurityPipeline

logger = logging.getLogger(__name__)

security: SecurityPipeline | None = None
cache: ResponseCache | None = None
metrics: MetricsCollector | None = None
agent: ProductionAgent | None = None

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per day", "10 per hour"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application components and clean them up on shutdown."""
    del app
    global security, cache, metrics, agent

    settings = get_settings()
    logger.info(
        "Starting up...",
        extra={
            "extra_data": {
                "environment": settings.APP_ENV,
                "primary_model": settings.PRIMARY_MODEL,
                "tracing_enabled": True,
            }
        },
    )

    if security is None:
        security = SecurityPipeline()
    if cache is None:
        cache = ResponseCache(ttl_seconds=settings.CACHE_TTL_SECONDS)
    if metrics is None:
        metrics = MetricsCollector()
    if agent is None:
        agent = ProductionAgent()

    logger.info("Components initialized")
    yield

    logger.info("Shutting down...", extra={"extra_data": metrics.summary})


app = FastAPI(
    title="Production RAG Agent API",
    description="A production-ready RAG agent API",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().RATE_LIMIT)
@observe(name="chat_endpoint")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Run security checks, cache lookup, generation, and output validation.
        Flow:
        1.Security check (injection + PII masking)
        2.Cache lookup
        3.LangGraph agent invoke ( if cache missed)
        4.Output validation
        5.Cache store
        6.Return response
    """
    del request
    if security is None or cache is None or metrics is None or agent is None:
        raise HTTPException(status_code=503, detail="Service is starting")

    with RequestTimer() as timer:

         # --- Step 1: Security Check ---
        security_notes: list[str] = []
        is_allowed, cleaned_message, notes = security.check_input(body.message)
        security_notes.extend(notes)

        if not is_allowed:
            metrics.record_request(
                latency_ms=timer.elapsed_ms,
                input_tokens=0,
                output_tokens=0,
                error=True,
            )
            raise HTTPException(
                status_code=400,
                detail="Your message contains security issues",
            )

        # --- Step 2: Cache Lookup ---
        cached_response = cache.get(cleaned_message)
        if cached_response is not None:
            metrics.record_request(
                latency_ms=timer.elapsed_ms,
                input_tokens=0,
                output_tokens=0,
                cache_hit=True,
            )
            return ChatResponse(
                response=cached_response,
                thread_id=body.thread_id,
                model_used="cache",
                cached=True,
                processing_time=timer.elapsed_ms,
            )

        # --- Step 3: Invoke Langgraph Agent ---
        try:
            result = agent.invoke(cleaned_message)
        except Exception as ex:
            logger.error(
                "Error invoking agent",
                extra={
                    "extra_data": {
                        "thread_id": body.thread_id,
                        "error": str(ex),
                    }
                },
            )
            metrics.record_request(
                latency_ms=timer.elapsed_ms,
                input_tokens=0,
                output_tokens=0,
                error=True,
            )
            raise HTTPException(status_code=500, detail="Error invoking agent") from ex

        # --- Step 4: Output Validation ---
        response_text = result["response"]
        model_used = result["model_used"]
        validated_response, output_warning = security.check_output(response_text)
        if output_warning:
            security_notes.append(output_warning)

         # --- Step 5: Cache Store ---
        cache.set(cleaned_message, validated_response)
        input_tokens = int(len(cleaned_message.split()) * 1.3)
        output_tokens = int(len(validated_response.split()) * 1.3)
        metrics.record_request(
            latency_ms=timer.elapsed_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        if security_notes:
            logger.info(
                "Security notes found",
                extra={
                    "extra_data": {
                        "notes": security_notes,
                        "thread_id": body.thread_id,
                    }
                },
            )

         # --- Step 6: Return Response ---
        return ChatResponse(
            response=validated_response,
            thread_id=body.thread_id,
            model_used=model_used,
            cached=False,
            processing_time=timer.elapsed_ms,
        )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return component readiness for Docker or Kubernetes probes."""
    settings = get_settings()
    check = {
        "agent": agent is not None,
        "security": security is not None,
        "cache": cache is not None,
    }
    return HealthResponse(
        status="healthy" if all(check.values()) else "degraded",
        environment=settings.APP_ENV,
        check=check,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Return application metrics for monitoring dashboards."""
    if metrics is None:
        raise HTTPException(status_code=503, detail="Metrics are unavailable")
    summary = metrics.summary
    return MetricsResponse(
        total_requests=int(summary["total_requests"]),
        total_errors=int(summary["total_errors"]),
        error_rate=str(summary["error_rate"]),
        avg_latency_ms=float(summary["avg_latency_ms"]),
        cache_hit_rate=str(summary["cache_hit_rate"]),
        total_input_tokens=int(summary["total_input_tokens"]),
        total_output_tokens=int(summary["total_output_tokens"]),
    )


@app.get("/cache/status")
async def cache_status() -> dict[str, int | float | str]:
    """Return cache performance statistics."""
    if cache is None:
        raise HTTPException(status_code=503, detail="Cache is unavailable")
    return cache.status
