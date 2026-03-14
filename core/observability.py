"""Observability: Prometheus metrics, OpenTelemetry tracing."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

# ── Prometheus Metrics ────────────────────────────────────────────────────────

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server

    TASK_COUNTER = Counter(
        "codegen_tasks_total",
        "Total tasks processed",
        ["task_type", "status"],
    )
    TASK_DURATION = Histogram(
        "codegen_task_duration_seconds",
        "Task execution duration",
        ["task_type"],
    )
    LLM_TOKENS = Counter(
        "codegen_llm_tokens_total",
        "Total LLM tokens used",
        ["direction"],  # input/output
    )
    ACTIVE_AGENTS = Gauge(
        "codegen_active_agents",
        "Currently active agents",
    )
    SANDBOX_ERRORS = Counter(
        "codegen_sandbox_errors_total",
        "Sandbox execution errors",
    )

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


def start_metrics_server(port: int = 9090) -> None:
    if _PROMETHEUS_AVAILABLE:
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
    else:
        logger.warning("prometheus_client not installed, metrics disabled")


def record_task_completion(task_type: str, status: str, duration: float) -> None:
    if _PROMETHEUS_AVAILABLE:
        TASK_COUNTER.labels(task_type=task_type, status=status).inc()
        TASK_DURATION.labels(task_type=task_type).observe(duration)


def record_llm_usage(input_tokens: int, output_tokens: int) -> None:
    if _PROMETHEUS_AVAILABLE:
        LLM_TOKENS.labels(direction="input").inc(input_tokens)
        LLM_TOKENS.labels(direction="output").inc(output_tokens)


def record_agent_start() -> None:
    if _PROMETHEUS_AVAILABLE:
        ACTIVE_AGENTS.inc()


def record_agent_end() -> None:
    if _PROMETHEUS_AVAILABLE:
        ACTIVE_AGENTS.dec()


def record_sandbox_error() -> None:
    if _PROMETHEUS_AVAILABLE:
        SANDBOX_ERRORS.inc()


# ── OpenTelemetry Tracing ─────────────────────────────────────────────────────

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


def setup_tracing(otlp_endpoint: str = "http://localhost:4317") -> None:
    if not _OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not installed, tracing disabled")
        return

    provider = TracerProvider()
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    logger.info(f"OpenTelemetry tracing configured with endpoint {otlp_endpoint}")


def get_tracer(name: str = "codegen") -> Any:
    if _OTEL_AVAILABLE:
        return trace.get_tracer(name)
    return _NoOpTracer()


class _NoOpTracer:
    """Fallback tracer when OpenTelemetry is not available."""

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any) -> Generator[Any, None, None]:
        yield _NoOpSpan()


class _NoOpSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exc: BaseException) -> None:
        pass
