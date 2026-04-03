"""Structured error hierarchy for the multi-agent pipeline.

Replaces generic ``except Exception`` swallowing with typed errors that
carry actionable context.  Every module should raise or catch these
instead of bare ``Exception``.

Hierarchy::

    PipelineError
    ├── LLMError
    │   ├── LLMRetryableError      (transient — retry with backoff)
    │   ├── LLMRateLimitError      (rate-limited — longer backoff)
    │   └── LLMConfigError         (bad API key / model — do not retry)
    ├── BuildError                  (build checkpoint failure)
    ├── ToolExecutionError          (agent tool dispatch failure)
    ├── PermissionDeniedError       (agent requested forbidden operation)
    ├── SessionError                (checkpoint save/load failure)
    ├── ConfigValidationError       (bad settings from env / file)
    └── AgentError                  (agent-level failure with context)
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for all pipeline errors."""

    def __init__(self, message: str, *, context: dict | None = None) -> None:
        super().__init__(message)
        self.context = context or {}


# ── LLM errors ───────────────────────────────────────────────────────────────

class LLMError(PipelineError):
    """Base for LLM call failures."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        is_retryable: bool = False,
        context: dict | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.provider = provider
        self.model = model
        self.is_retryable = is_retryable


class LLMRetryableError(LLMError):
    """Transient error — retry with exponential backoff."""

    def __init__(self, message: str, **kwargs: object) -> None:
        super().__init__(message, is_retryable=True, **kwargs)  # type: ignore[arg-type]


class LLMRateLimitError(LLMError):
    """Rate-limited — use longer backoff."""

    def __init__(self, message: str, *, retry_after: float = 0, **kwargs: object) -> None:
        super().__init__(message, is_retryable=True, **kwargs)  # type: ignore[arg-type]
        self.retry_after = retry_after


class LLMConfigError(LLMError):
    """Configuration error — do NOT retry (bad key, invalid model)."""

    def __init__(self, message: str, **kwargs: object) -> None:
        super().__init__(message, is_retryable=False, **kwargs)  # type: ignore[arg-type]


# ── Build / checkpoint errors ────────────────────────────────────────────────

class BuildError(PipelineError):
    """Build checkpoint failure with per-file attribution."""

    def __init__(
        self,
        message: str,
        *,
        affected_files: list[str] | None = None,
        errors_by_file: dict[str, list[str]] | None = None,
        context: dict | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.affected_files = affected_files or []
        self.errors_by_file = errors_by_file or {}


# ── Tool / agent errors ─────────────────────────────────────────────────────

class ToolExecutionError(PipelineError):
    """Agent tool dispatch failure."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        agent_name: str = "",
        context: dict | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.tool_name = tool_name
        self.agent_name = agent_name


class AgentError(PipelineError):
    """Agent-level failure carrying agent metadata."""

    def __init__(
        self,
        message: str,
        *,
        agent_role: str = "",
        file_path: str = "",
        phase: str = "",
        context: dict | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.agent_role = agent_role
        self.file_path = file_path
        self.phase = phase


# ── Permission errors ────────────────────────────────────────────────────────

class PermissionDeniedError(PipelineError):
    """Agent attempted an operation that was denied by the permission layer."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        path: str = "",
        context: dict | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.tool_name = tool_name
        self.path = path


# ── Session / persistence errors ─────────────────────────────────────────────

class SessionError(PipelineError):
    """Checkpoint save/load failure."""
    pass


# ── Configuration errors ─────────────────────────────────────────────────────

class ConfigValidationError(PipelineError):
    """Invalid settings detected at startup."""
    pass


class CostLimitExceededError(PipelineError):
    """Session cost has exceeded the configured ``max_cost_usd`` budget."""

    def __init__(
        self,
        message: str,
        *,
        spent_usd: float = 0.0,
        limit_usd: float = 0.0,
        context: dict | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd
