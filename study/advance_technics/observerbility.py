"""
Monitoring and Logging for Production
Structured logging, metrics, and alerts
Later you can easily add these logs to track. such as datadog
structered logs are more easy to monitor and reson with than plain strings.
"""

import json
import logging
import time
from datetime import UTC, datetime

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler

load_dotenv()


# === Structured Logging ===
class JSONFormatter(logging.Formatter):
    """Format logs as JSON for log aggregation."""

    def format(self, record):
        log_obj = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)

        return json.dumps(log_obj)


def setup_logging():
    """Set up structured JSON logging."""

    logger = logging.getLogger("langgraph_app")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    return logger


# === Metrics Collection ===
class MetricsCollector:
    """Collect and aggregate metrics."""

    def __init__(self):
        self.metrics = {
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
    ):
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

    def get_summary(self) -> dict:
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


# === Instrumented LLM ===
class InstrumentedLLM:
    """LLM with full instrumentation."""

    def __init__(self):
        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0,
            reasoning_effort="medium",
            timeout=30,
            max_retries=2,
        )
        self.metrics = MetricsCollector()
        self.logger = setup_logging()
        self._langfuse_handler = CallbackHandler()

    @observe(name="instrumented_invoke", capture_input=False, capture_output=False)
    def invoke(self, query: str) -> str:
        start_time = time.time()
        error = False

        try:
            langfuse = get_client()
            langfuse.update_current_span(input=query, metadata=self.metrics)

            response = self.llm.invoke(
                query, config={"callbacks": [self._langfuse_handler]}
            )
            result = response.content

            # Estimate tokens
            input_tokens = len(query.split()) * 4 // 3
            output_tokens = len(result.split()) * 4 // 3

            self.metrics.record_request(
                latency_ms=(time.time() - start_time) * 1000,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=False,
                cache_hit=False,
            )

            langfuse.update_current_span(output=result, metadata=self.metrics)

            self.logger.info(
                "LLM request completed",
                extra={
                    "extra_data": {
                        "latency_ms": (time.time() - start_time) * 1000,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                },
            )

            return result

        except Exception as e:
            error = True
            self.metrics.record_request(
                latency_ms=(time.time() - start_time) * 1000,
                input_tokens=0,
                output_tokens=0,
                error=True,
                cache_hit=False,
            )

            self.logger.error(
                f"LLM request failed: {e}", extra={"extra_data": {"error": str(e)}}
            )

            raise


def demo_monitoring():
    """Demonstrate monitoring."""

    llm = InstrumentedLLM()

    print("Monitoring Demo:\n")

    queries = [
        "What is Python?",
        "Explain machine learning.",
        "What is 2 + 2?",
    ]

    for query in queries:
        result = llm.invoke(query)
        print(f"Query: {query[:30]}... -> {result[:30]}...")

    print("\nMetrics Summary:")
    summary = llm.metrics.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # logger = setup_logging()
    # logger.info("Logging setup complete", extra={"extra_data": {"app": "langgraph"}})
    demo_monitoring()
