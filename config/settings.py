"""Global configuration for the multi-agent code generation platform."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from core.errors import ConfigValidationError


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


class SandboxType(str, Enum):
    DOCKER = "docker"
    LOCAL = "local"  # For development/testing only


class SandboxTier(str, Enum):
    """Isolation level for Docker sandboxes.

    BUILD — network access allowed (needs to fetch dependencies).
    TEST  — no network, read-only rootfs, tmpfs for /tmp.  Prevents
            LLM-generated test code from exfiltrating data or mutating
            the host.
    """
    BUILD = "build"
    TEST = "test"


class SkippableAgent(str, Enum):
    """Pipeline phases that may be skipped via ``Settings.skip_agents``."""
    TESTER = "tester"
    SECURITY = "security"
    INTEGRATION = "integration"


@dataclass(frozen=True)
class ExecutionConfig:
    """Tunable constants for the pipeline executor, agents, and context builder.

    Previously scattered as module-level ``_MAX_*`` constants across
    ``pipeline_executor.py``, ``context_builder.py``, ``coder_agent.py``,
    and ``base_agent.py``.  Centralising them here lets operators tune
    behaviour per-project without code changes.
    """

    # ── Pipeline executor ────────────────────────────────────────────────
    # Max re-verification depth per file before suppressing cascades.
    max_reverify_depth: int = 5

    # ── Context builder budgets ──────────────────────────────────────────
    max_context_files: int = 20
    max_context_chars: int = 120_000
    max_semantic_hits: int = 3
    max_direct_deps: int = 10
    dep_truncate_large: int = 4_000   # per-dep char budget for projects >50 files
    dep_truncate_small: int = 8_000   # per-dep char budget for projects ≤50 files
    max_same_layer: int = 3

    # ── Coder agent thresholds ───────────────────────────────────────────
    # Files with more lines than this are modified via diff instead of full rewrite.
    large_file_threshold: int = 200
    # Max chars of compiler output included in fix prompts.
    max_error_chars: int = 3_000
    # Max chars of a single related file included in fix context.
    max_related_file_chars: int = 2_000
    # Max allowed content growth factor for fix/modify rewrites.
    max_content_growth: float = 1.35

    # ── Base agent ───────────────────────────────────────────────────────
    # Timeout in seconds for individual tool handler calls.
    tool_timeout_seconds: float = 60.0
    # Seconds between "still waiting" heartbeat log lines during LLM calls.
    heartbeat_interval: int = 15

    # ── LLM client ───────────────────────────────────────────────────────
    # Per-request timeout in seconds (individual API call, not full retry loop).
    request_timeout: int = 180
    # Max retry attempts for transient LLM errors.
    retry_count: int = 4
    # Base delay in seconds for exponential backoff (doubles each attempt).
    backoff_base: float = 2.0

    # ── Circuit breaker ──────────────────────────────────────────────────
    # Consecutive failures before the circuit opens.
    circuit_failure_threshold: int = 5
    # Seconds to wait before allowing a half-open probe.
    circuit_recovery_timeout: float = 60.0


@dataclass(frozen=True)
class LLMConfig:
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 16384
    temperature: float = 0.2
    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", ""))
    gemini_api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))


@dataclass(frozen=True)
class SandboxConfig:
    sandbox_type: SandboxType = SandboxType.DOCKER
    image: str = ""  # Auto-detected from language profile if empty
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    timeout_seconds: int = 300
    network_enabled: bool = False


@dataclass(frozen=True)
class MemoryConfig:
    chroma_persist_dir: str = ".chroma"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_context_tokens: int = 6000
    max_related_files: int = 10


@dataclass(frozen=True)
class ObservabilityConfig:
    prometheus_port: int = 9090
    otlp_endpoint: str = "http://localhost:4317"
    enable_tracing: bool = True


@dataclass
class Settings:
    workspace_dir: Path = field(default_factory=lambda: Path("workspace"))
    llm: LLMConfig = field(default_factory=LLMConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    max_concurrent_agents: int = 4
    max_debug_iterations: int = 5
    # Maximum wall-clock seconds allowed for a single lifecycle phase (generate,
    # fix, build, test). Phases that exceed this budget are cancelled
    # and the file is marked FAILED so the rest of the pipeline can proceed.
    phase_timeout_seconds: int = 600
    figma_token: str = field(default_factory=lambda: os.environ.get("FIGMA_TOKEN", ""))
    allow_host_execution: bool = False  # Must be True to run without Docker
    require_plan_approval: bool = False  # If True, pause for human review after change planning
    require_architecture_approval: bool = True  # If True, pause for human review at architecture checkpoints
    # Number of build attempts per tier checkpoint (1 initial + retries).
    # Increase for flaky compilers; decrease to fail-fast during development.
    build_checkpoint_retries: int = 3
    # Agent phases the user wants to skip entirely. Recognised values:
    # "tester", "security", "integration".
    # Phases listed here are bypassed in the executor.
    skip_agents: frozenset[str] = field(default_factory=frozenset)
    # Per-session cost cap in USD.  When total LLM spend exceeds this value,
    # CostLimitExceededError is raised.  0 (default) means unlimited.
    max_cost_usd: float = 0.0
    
    # Optional command to start a local embedded MCP server (e.g. ['npx', '-y', '@figma/mcp-server'])
    mcp_server_command: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> Settings:
        settings = cls(
            workspace_dir=Path(os.environ.get("WORKSPACE_DIR", "workspace")),
            llm=LLMConfig(
                provider=LLMProvider(os.environ.get("LLM_PROVIDER", "anthropic")),
                model=os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514"),
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
                openai_base_url=os.environ.get("OPENAI_BASE_URL", ""),
                gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
            ),
            sandbox=SandboxConfig(
                sandbox_type=SandboxType(os.environ.get("SANDBOX_TYPE", "docker")),
            ),
            max_concurrent_agents=int(os.environ.get("MAX_CONCURRENT_AGENTS", "4")),
            build_checkpoint_retries=int(os.environ.get("BUILD_CHECKPOINT_RETRIES", "3")),
            mcp_server_command=os.environ.get("MCP_SERVER_COMMAND", "").split() if os.environ.get("MCP_SERVER_COMMAND") else [],
            max_cost_usd=float(os.environ.get("MAX_COST_USD", "0")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Validate settings and raise ConfigValidationError on problems."""
        errors: list[str] = []

        # API key must be set for the selected provider
        provider = self.llm.provider
        if provider == LLMProvider.ANTHROPIC and not self.llm.api_key:
            errors.append("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        elif provider == LLMProvider.OPENAI and not self.llm.openai_api_key:
            errors.append("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        elif provider == LLMProvider.GEMINI and not self.llm.gemini_api_key:
            errors.append("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")

        # Numeric ranges
        if self.max_concurrent_agents < 1:
            errors.append(f"max_concurrent_agents must be >= 1, got {self.max_concurrent_agents}")
        if self.build_checkpoint_retries < 1:
            errors.append(f"build_checkpoint_retries must be >= 1, got {self.build_checkpoint_retries}")
        if self.phase_timeout_seconds < 30:
            errors.append(f"phase_timeout_seconds must be >= 30, got {self.phase_timeout_seconds}")
        if self.execution.max_reverify_depth < 1:
            errors.append(f"max_reverify_depth must be >= 1, got {self.execution.max_reverify_depth}")

        # Validate skip_agents against the SkippableAgent enum
        _valid_agents = {a.value for a in SkippableAgent}
        _bad = self.skip_agents - _valid_agents
        if _bad:
            errors.append(
                f"skip_agents contains unrecognised values: {sorted(_bad)}. "
                f"Valid: {sorted(_valid_agents)}"
            )

        if errors:
            raise ConfigValidationError(
                f"Invalid configuration: {'; '.join(errors)}",
                context={"errors": errors},
            )


# ── CLI / API entry-point helper ──────────────────────────────────────────────
# Call this ONLY at process entry points (CLI startup, FastAPI lifespan).
# Internal components receive a Settings instance via their constructors —
# never call get_settings() from library code.


def get_settings() -> Settings:
    """Build and return a fresh Settings from environment variables.

    Intentionally NOT cached: each call reads the current environment so
    there is no global mutable state and no need to reset between tests.
    Entry points (CLI, API) should call this once and pass the result
    through constructors everywhere else.
    """
    return Settings.from_env()
