"""Domain models for the multi-agent code generation platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Blueprint Models ──────────────────────────────────────────────────────────


@dataclass
class FileBlueprint:
    path: str
    purpose: str
    depends_on: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    language: str = ""  # Resolved from tech_stack at blueprint parse time
    layer: str = ""  # controller, service, repository, model, test, config


@dataclass
class RepositoryBlueprint:
    name: str
    description: str
    architecture_style: str  # REST, GraphQL, gRPC, etc.
    tech_stack: dict[str, str] = field(default_factory=dict)  # e.g. {"db": "postgresql"}
    folder_structure: list[str] = field(default_factory=list)
    file_blueprints: list[FileBlueprint] = field(default_factory=list)
    architecture_doc: str = ""


# ── Task Models ───────────────────────────────────────────────────────────────


class TaskType(str, Enum):
    GENERATE_FILE = "generate_file"
    REVIEW_FILE = "review_file"
    REVIEW_MODULE = "review_module"
    REVIEW_ARCHITECTURE = "review_architecture"
    GENERATE_TEST = "generate_test"
    SECURITY_SCAN = "security_scan"
    GENERATE_DEPLOY = "generate_deploy"
    GENERATE_DOCS = "generate_docs"
    FIX_CODE = "fix_code"


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Task:
    task_id: int
    task_type: TaskType
    file: str
    description: str
    dependencies: list[int] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: str | None = None
    result: TaskResult | None = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    success: bool
    output: str = ""
    errors: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


# ── Agent Models ──────────────────────────────────────────────────────────────


class AgentRole(str, Enum):
    ARCHITECT = "architect"
    PLANNER = "planner"
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"
    SECURITY = "security"
    DEPLOYER = "deployer"
    WRITER = "writer"


@dataclass
class AgentContext:
    """Context provided to an agent for task execution."""
    task: Task
    blueprint: RepositoryBlueprint
    file_blueprint: FileBlueprint | None = None
    related_files: dict[str, str] = field(default_factory=dict)  # path -> content
    architecture_summary: str = ""
    dependency_info: dict[str, Any] = field(default_factory=dict)


# ── Repository Knowledge Models ──────────────────────────────────────────────


@dataclass
class FileIndex:
    path: str
    exports: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    checksum: str = ""


@dataclass
class RepositoryIndex:
    files: list[FileIndex] = field(default_factory=list)

    def get_file(self, path: str) -> FileIndex | None:
        for f in self.files:
            if f.path == path:
                return f
        return None

    def add_or_update(self, file_index: FileIndex) -> None:
        for i, f in enumerate(self.files):
            if f.path == file_index.path:
                self.files[i] = file_index
                return
        self.files.append(file_index)


# ── Review Models ─────────────────────────────────────────────────────────────


class ReviewLevel(str, Enum):
    FILE = "file"
    MODULE = "module"
    ARCHITECTURE = "architecture"


@dataclass
class ReviewFinding:
    level: ReviewLevel
    severity: str  # critical, warning, info
    file: str
    line: int | None = None
    message: str = ""
    suggestion: str = ""


@dataclass
class ReviewResult:
    level: ReviewLevel
    passed: bool
    findings: list[ReviewFinding] = field(default_factory=list)
    summary: str = ""
