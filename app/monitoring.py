import json
import logging
import time
from datetime import UTC, datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formate log records as JSON for log aggregration (ELK, Datadog, etc.)"""

    def format(self, record: logging.LogRecord) -> str:
        log_object = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        # Merge any extra data attached to the record
        extra_data = getattr(record, "extra_data", None)
        if isinstance(extra_data, dict):
            log_object.update(extra_data)
        return json.dumps(log_object)

    def get_logger(self, name: str = "production-api") -> logging.Logger:
        """Create a structured JSON logger"""
        logger = logging.getLogger(name)

        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JSONFormatter())
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

class MetricsCollector:
     """ Collects and aggregates application metrics.
        In production, replace it with Prometheus client:
        from prometheus_client import Counter, Histogram
        """

     def __init__(self):
        self.metrics: dict[str, int | float] = {
            "requests_total": 0,
            "errors_total": 0,
            "latency_sum": 0.0,
            "latency_count": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

     def record_request(
        self,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        error: bool = False,
        cache_hit: bool = False,
    ) -> None:
        self.metrics["requests_total"] += 1
        self.metrics["latency_sum"] += latency_ms
        self.metrics["latency_count"] += 1
        self.metrics["tokens_input"] += input_tokens
        self.metrics["tokens_output"] += output_tokens

        if error:
            self.metrics["errors_total"] += 1

        if cache_hit:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1

     @property
     def summary(self) -> dict[str, int | float | str]:
        avg_latency = (
            self.metrics["latency_sum"] / self.metrics["latency_count"]
            if self.metrics["latency_count"] > 0
            else 0
        )
        error_rate = (
            self.metrics["errors_total"] / self.metrics["requests_total"]
            if self.metrics["requests_total"] > 0
            else 0
        )
        cache_hit_rate = (
            self.metrics["cache_hits"]
            / (self.metrics["cache_hits"] + self.metrics["cache_misses"])
            if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0
            else 0
        )

        return {
            "total_requests": self.metrics["requests_total"],
            "total_errors": self.metrics["errors_total"],
            "error_rate": f"{error_rate:.2%}",
            "avg_latency_ms": round(avg_latency, 2),
            "total_input_tokens": self.metrics["tokens_input"],
            "total_output_tokens": self.metrics["tokens_output"],
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
        }


class RequestTimer:
    """Context manager for measuring elapsed request time in milliseconds."""

    def __enter__(self) -> RequestTimer:
        self._start = time.perf_counter()
        return self

    @property
    def elapsed_ms(self) -> float:
        """Return elapsed time while active and after context exit."""
        return (time.perf_counter() - self._start) * 1000

    def __exit__(self, *args: Any) -> None:
        return None
