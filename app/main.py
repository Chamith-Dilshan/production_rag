import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langfuse import observe
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.agent import ProductionAgent
from app.cache import ResponseCache
from app.config import get_settings
from app.models import ChatRequest, ChatResponse, HealthResponse, MetricsResponse
from app.monitoring import MetricsCollector
from app.security import SecurityPipeline

logger = logging.getLogger(__name__)

# Rate Limiter
limiter = Limiter(
    key_func=get_remote_address, default_limits=["100 per day", "10 per hour"]
)


# Lifespan (startup/shutdown)
async def lifespan(app: FastAPI):
    """Initialize all components on startup, clean up on shutdown
    This is the modern FastApi pattern to replace @app.on_event()
    """
    global security, cache, matrics, agent

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

    # Initialize components
    security = SecurityPipeline()
    cache = ResponseCache(ttl_seconds=settings.CACHE_TTL_SECONDS)
    matrics = MetricsCollector()
    agent = ProductionAgent()

    logger.info("Components initialized")

    yield  # app is running

    # shutdown
    logger.info("shutting down...", extra={"extra_data": matrics.summary})


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
async def chat(request: Request, body: ChatRequest):
    """Main chat endpoint

    Flow:
    1.Security check(injection + PII masking)
    2.Cache lookup
    3.LangGraph agent invoke( if cache missed)
    4.Output validation
    5.Cache store
    6.Return response
    """

    with RequestTimer() as timer:
        security_notes = []

        # --- Step 1: Security Check ---
        is_allowed, cleaned_message, notes = security.check_input(body.message)
        security_notes.extend(notes)

        if not is_allowed:
            logger.warning(
                "Security check failed",
                extra={"extra_data": {"reason": notes, "thread_id": body.thread_id}},
            )
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=400, detail="Your message contains security issues"
            )

        # --- Step 2: Cache Lookup ---
        cached_response = cache.get(cleaned_message)
        if cached_response is not None:
            metrics.record_request(latency_ms=0, cache_hit=True)
            logger.info(
                "Cache hit", extra={"extra_data": {"thread_id": body.thread_id}}
            )

            return ChatResponse(
                response=cached_response,
                thread_id=body.thread_id,
                model_used="cache",
                cached=True,
                processing_time_ms=0,
            )

        # --- Step 3: Invoke Langgraph Agent ---
        try:
            result: agent.invoke(cleaned_message)
        except Exception as ex:
            logger.error(
                "Error invoking agent",
                extra={"extra_data": {"thread_id": body.thread_id, "error": str(ex)}},
            )
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(status_code=500, detail="Error invoking agent")

        response_text = result["response"]
        model_used = result["model_used"]

        # --- Step 4: Output Validation ---
        validated_response, output_warnings = security.check_ouput(response_text)
        security_notes.extend(output_warnings)

        # --- Step 5: Chache Store ---
        cache.set(cleaned_message, validated_response)

        # --- Step 6: Log And Record Matrics ---
        input_tokens = int(len(cleaned_message.split()) * 1.3)
        output_tokens = int(len(validated_response.split()) * 1.3)

        metrics.record_request(
            latency_ms=0,
            cache_hit=False,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        if security_notes:
            logger.info(
                "Security notes found in response",
                extra={
                    "extra_data": {
                        "notes": security_notes,
                        "thread_id": body.thread_id,
                    }
                },
            )

        logger.info(
            "Chat response generated",
            extra={
                "extra_data": {
                    "thread_id": body.thread_id,
                    "model_used": model_used,
                    "latency_ms": round(timer.elapsed_ms, 2),
                }
            },
        )

        return ChatResponse(
            response=validated_response,
            thread_id=body.thread_id,
            model_used=model_used,
            cached=False,
            processing_time_ms=round(timer.elapsed_ms, 2),
        )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for Docker/Kubernetes"""
    settings = get_settings()

    check = {
        "agent": agent is not None,
        "security": security is not None,
        "cache": cache is not None,
    }

    all_healthy = all(check.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        environment=settings.APP_ENV,
        check=check,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Metrics for monitoring dashboards"""
    summary = metrics.summary
    return MetricsResponse(**summary)


@app.get("/cache/status")
async def cache_status():
    """Cache performance statistics"""
    return cache.status
