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

    def record_exception(self, exception: BaseException) -> None:
        pass


# ── Decorators for automatic span instrumentation ────────────────────────────

def traced(
    span_name: str | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> Any:
    """Decorator that wraps an async function in an OpenTelemetry span.

    Usage::

        @traced("llm_generate", attributes={"model": "claude"})
        async def generate(self, prompt):
            ...

    When OpenTelemetry is unavailable, the function runs unmodified.
    """
    import functools

    def decorator(fn: Any) -> Any:
        name = span_name or f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer(fn.__module__)
            with tracer.start_as_current_span(name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)
                try:
                    result = await fn(*args, **kwargs)
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    raise

        return wrapper
    return decorator


# ── Pipeline-level metrics helpers ───────────────────────────────────────────

_file_timings: dict[str, dict[str, float]] = {}


def record_file_phase_start(file_path: str, phase: str) -> None:
    """Record when a file enters a pipeline phase (generate, fix, build, test)."""
    import time
    key = f"{file_path}:{phase}"
    _file_timings[key] = {"start": time.monotonic()}


def record_file_phase_end(file_path: str, phase: str, success: bool = True) -> None:
    """Record when a file exits a pipeline phase and emit metrics."""
    import time
    key = f"{file_path}:{phase}"
    timing = _file_timings.pop(key, None)
    if timing:
        duration = time.monotonic() - timing["start"]
        record_task_completion(phase, "success" if success else "failure", duration)
        logger.debug(
            "File %s phase %s completed in %.1fs (success=%s)",
            file_path, phase, duration, success,
        )


def check_cost_warning(
    spent_usd: float,
    limit_usd: float,
    warning_pct: float = 0.8,
) -> bool:
    """Log a warning if spending exceeds the warning threshold.

    Returns True if the warning was triggered (so callers can take action).
    """
    if limit_usd <= 0:
        return False
    if spent_usd >= limit_usd * warning_pct:
        logger.warning(
            "COST WARNING: $%.2f spent of $%.2f limit (%.0f%% — threshold: %.0f%%)",
            spent_usd, limit_usd, 100 * spent_usd / limit_usd, 100 * warning_pct,
        )
        return True
    return False
