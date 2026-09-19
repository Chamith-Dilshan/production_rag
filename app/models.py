from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat requests"""

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The user's message to the agent",
    )
    thread_id: str = Field(default="default", description="Conversation thread ID")


class ChatResponse(BaseModel):
    """Chat response returned to the client"""

    response: str
    thread_id: str
    model_used: str
    cached: bool = False
    processing_time: float
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of the response",
    )


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = "healthy"
    environment: str
    version: str = "1.0.0"
    checks: dict = {}


class MetricsResponse(BaseModel):
    """Metrics response returned to the client"""

    total_requests: int
    total_errors: int
    error_rate: str
    avg_latency_ms: float
    cache_hit_rate: str
    total_input_tokens: int
    total_output_tokens: int


class ErrorResponse(BaseModel):
    """Error response"""

    error: str
    detail: str | None = None
    request_id: str | None = None
