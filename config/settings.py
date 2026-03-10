"""Global configuration for the multi-agent code generation platform."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


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


@dataclass(frozen=True)
class LLMConfig:
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 8192
    temperature: float = 0.2
    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
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
    max_concurrent_agents: int = 4
    max_debug_iterations: int = 5
    review_levels: list[str] = field(
        default_factory=lambda: ["file", "module", "architecture"]
    )
    allow_host_execution: bool = False  # Must be True to run without Docker
    require_plan_approval: bool = False  # If True, pause for human review after change planning

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            workspace_dir=Path(os.environ.get("WORKSPACE_DIR", "workspace")),
            llm=LLMConfig(
                provider=LLMProvider(os.environ.get("LLM_PROVIDER", "anthropic")),
                model=os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514"),
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
                gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
            ),
            sandbox=SandboxConfig(
                sandbox_type=SandboxType(os.environ.get("SANDBOX_TYPE", "docker")),
            ),
            max_concurrent_agents=int(os.environ.get("MAX_CONCURRENT_AGENTS", "4")),
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
