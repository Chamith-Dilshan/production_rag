import logging
import time

from app.monitoring import JSONFormatter, MetricsCollector, RequestTimer


def test_metrics_summary_aggregates_values() -> None:
    metrics = MetricsCollector()

    metrics.record_request(25.0, 10, 20)
    metrics.record_request(75.0, 30, 40, error=True)
    metrics.record_request(50.0, 5, 15, cache_hit=True)

    summary = metrics.summary

    assert summary["total_requests"] == 3
    assert summary["total_errors"] == 1
    assert summary["total_input_tokens"] == 45
    assert summary["total_output_tokens"] == 75
    assert summary["cache_hit_rate"] == "33.33%"


def test_request_timer_measurement_is_positive() -> None:
    with RequestTimer() as timer:
        time.sleep(0.01)

    assert timer.elapsed_ms >= 5.0


def test_json_formatter_and_logger_include_extra_data() -> None:
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="demo",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.extra_data = {"request_id": "abc"}

    payload = formatter.format(record)

    assert "hello" in payload
    assert "request_id" in payload
    assert "abc" in payload

    logger = formatter.get_logger("demo.logger")
    assert logger.name == "demo.logger"
    assert logger.handlers
