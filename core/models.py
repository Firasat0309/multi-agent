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
    # ── Modification workflow task types ──
    ANALYZE_REPO = "analyze_repo"
    PLAN_CHANGES = "plan_changes"
    MODIFY_FILE = "modify_file"
    # ── Quality & validation task types ──
    GENERATE_INTEGRATION_TEST = "generate_integration_test"


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
    PATCH_AGENT = "patch_agent"
    REVIEWER = "reviewer"
    TESTER = "tester"
    SECURITY = "security"
    DEPLOYER = "deployer"
    WRITER = "writer"
    REPO_ANALYZER = "repo_analyzer"
    CHANGE_PLANNER = "change_planner"
    INTEGRATION_TESTER = "integration_tester"


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
    # O(1) lookup cache — rebuilt lazily when needed
    _path_index: dict[str, int] = field(default_factory=dict, repr=False)

    def _rebuild_index(self) -> None:
        self._path_index = {f.path: i for i, f in enumerate(self.files)}

    def get_file(self, path: str) -> FileIndex | None:
        if not self._path_index and self.files:
            self._rebuild_index()
        idx = self._path_index.get(path)
        return self.files[idx] if idx is not None else None

    def add_or_update(self, file_index: FileIndex) -> None:
        if not self._path_index and self.files:
            self._rebuild_index()
        idx = self._path_index.get(file_index.path)
        if idx is not None:
            self.files[idx] = file_index
        else:
            self._path_index[file_index.path] = len(self.files)
            self.files.append(file_index)


# ── Review Models ─────────────────────────────────────────────────────────────


# ── Modification Workflow Models ──────────────────────────────────────────────


@dataclass
class ModuleInfo:
    """Summary of a module/file discovered during repository analysis."""
    name: str
    file: str
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    layer: str = ""  # controller, service, repository, model, etc.


@dataclass
class RepoAnalysis:
    """Complete analysis of an existing repository."""
    modules: list[ModuleInfo] = field(default_factory=list)
    tech_stack: dict[str, str] = field(default_factory=dict)
    architecture_style: str = ""
    entry_points: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class FilePatch:
    """A unified diff patch for a single file."""
    file_path: str
    original_checksum: str   # SHA-1 of file before patch — for safety check
    unified_diff: str        # Standard unified diff format
    description: str         # Human-readable description of change


class ChangeActionType(str, Enum):
    ADD_FUNCTION = "add_function"
    ADD_METHOD = "add_method"
    ADD_ENDPOINT = "add_endpoint"
    ADD_CLASS = "add_class"
    ADD_IMPORT = "add_import"
    MODIFY_FUNCTION = "modify_function"
    ADD_FIELD = "add_field"
    CREATE_FILE = "create_file"


@dataclass
class ChangeAction:
    """A single planned change to the repository."""
    type: ChangeActionType
    file: str
    description: str
    function: str = ""  # target function/method name
    class_name: str = ""  # target class (for method additions)
    depends_on: list[str] = field(default_factory=list)  # file paths this change depends on
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChangePlan:
    """Complete plan for modifying an existing repository."""
    summary: str
    changes: list[ChangeAction] = field(default_factory=list)
    new_files: list[FileBlueprint] = field(default_factory=list)
    affected_tests: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)


# ── Review Models ─────────────────────────────────────────────────────────────


@dataclass
class TokenCost:
    """Aggregated token usage and estimated USD cost for one pipeline run."""

    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float  # Calculated from MODEL_PRICING in llm_client.py


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
