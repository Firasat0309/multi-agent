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
    tech_stack: dict[str, str] = field(default_factory=dict)  # e.g. {"db": "h2"}
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
    SECURITY_FIX = "security_fix"
    DESIGN_ARCHITECTURE = "design_architecture"
    CREATE_PLAN = "create_plan"
    VERIFY_BUILD = "verify_build"
    # ── Fullstack / Frontend task types ──
    PLAN_PRODUCT = "plan_product"
    GENERATE_API_CONTRACT = "generate_api_contract"
    PARSE_DESIGN = "parse_design"
    PLAN_COMPONENTS = "plan_components"
    BUILD_COMPONENT_DAG = "build_component_dag"
    GENERATE_COMPONENT = "generate_component"
    INTEGRATE_API = "integrate_api"
    MANAGE_STATE = "manage_state"
    FIX_COMPONENT = "fix_component"


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
    BUILD_VERIFIER = "build_verifier"
    # ── Fullstack / Frontend agent roles ──
    PRODUCT_PLANNER = "product_planner"
    API_CONTRACT_GENERATOR = "api_contract_generator"
    DESIGN_PARSER = "design_parser"
    COMPONENT_PLANNER = "component_planner"
    COMPONENT_DAG_BUILDER = "component_dag_builder"
    COMPONENT_GENERATOR = "component_generator"
    API_INTEGRATOR = "api_integrator"
    STATE_MANAGER = "state_manager"


@dataclass
class AgentContext:
    """Context provided to an agent for task execution."""
    task: Task
    blueprint: RepositoryBlueprint
    file_blueprint: FileBlueprint | None = None
    related_files: dict[str, str] = field(default_factory=dict)  # path -> content
    architecture_summary: str = ""
    dependency_info: dict[str, Any] = field(default_factory=dict)
    # Optional API contract — populated in fullstack mode so backend agents
    # (CoderAgent, ReviewerAgent) can see the exact endpoint schemas they must
    # implement / verify rather than relying on implicit conventions.
    api_contract: "APIContract | None" = None


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

    def _ensure_index(self) -> None:
        """Rebuild the index if it is empty or out of sync with ``files``.

        Catches the common mutation patterns (append, extend, slice-assign,
        full replacement) by comparing dict length to list length.  In-place
        path changes via ``files[i] = x`` should always go through
        ``add_or_update`` to keep the index consistent.
        """
        if len(self._path_index) != len(self.files):
            self._rebuild_index()

    def get_file(self, path: str) -> FileIndex | None:
        self._ensure_index()
        idx = self._path_index.get(path)
        return self.files[idx] if idx is not None else None

    def add_or_update(self, file_index: FileIndex) -> None:
        self._ensure_index()
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


# ── Fullstack / Frontend Models ───────────────────────────────────────────────


@dataclass
class ProductRequirements:
    """High-level product requirements extracted from a user prompt."""
    title: str
    description: str
    user_stories: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    # e.g. {"frontend": "React/Next.js", "backend": "FastAPI", "db": "PostgreSQL"}
    tech_preferences: dict[str, str] = field(default_factory=dict)
    has_frontend: bool = True
    has_backend: bool = True


@dataclass
class APIEndpoint:
    """A single REST or GraphQL endpoint contract."""
    path: str
    method: str  # GET, POST, PUT, DELETE, PATCH
    description: str
    request_schema: dict[str, Any] = field(default_factory=dict)
    response_schema: dict[str, Any] = field(default_factory=dict)
    auth_required: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass
class APIContract:
    """API contract (OpenAPI or GraphQL) shared between frontend and backend."""
    title: str
    version: str = "1.0.0"
    base_url: str = "/api/v1"
    endpoints: list[APIEndpoint] = field(default_factory=list)
    # Named reusable schemas (JSON Schema objects)
    schemas: dict[str, Any] = field(default_factory=dict)
    openapi_spec: str = ""   # Full OpenAPI 3.x YAML/JSON string
    contract_format: str = "openapi"  # "openapi" | "graphql"


@dataclass
class UIComponent:
    """Blueprint for a single UI component."""
    name: str
    file_path: str
    component_type: str  # "page", "layout", "feature", "ui", "shared"
    description: str
    figma_node_id: str | None = None
    props: list[str] = field(default_factory=list)
    state_needs: list[str] = field(default_factory=list)   # state slices needed
    api_calls: list[str] = field(default_factory=list)     # endpoint paths consumed
    depends_on: list[str] = field(default_factory=list)    # other component names
    children: list[str] = field(default_factory=list)
    layer: str = ""  # pages, components/feature, components/ui, layouts


@dataclass
class UIDesignSpec:
    """Parsed representation of a UI design (from Figma URL or text description)."""
    framework: str = "react"   # "react", "nextjs", "vue", "angular"
    design_description: str = ""
    figma_url: str = ""        # Original Figma URL if provided
    pages: list[str] = field(default_factory=list)
    global_styles: dict[str, str] = field(default_factory=dict)
    design_tokens: dict[str, Any] = field(default_factory=dict)  # colors, spacing, etc.


@dataclass
class ComponentPlan:
    """Complete plan for all frontend components."""
    components: list[UIComponent] = field(default_factory=list)
    framework: str = "react"
    state_solution: str = "zustand"   # "redux", "zustand", "context"
    api_base_url: str = "/api/v1"
    routing_solution: str = ""        # "react-router", "nextjs", "vue-router"
    package_json: dict[str, Any] = field(default_factory=dict)


@dataclass
class FullstackBlueprint:
    """Combined blueprint for the entire fullstack application."""
    product_requirements: ProductRequirements | None = None
    backend_blueprint: RepositoryBlueprint | None = None
    frontend_blueprint: RepositoryBlueprint | None = None
    api_contract: APIContract | None = None
    component_plan: ComponentPlan | None = None
    workspace_root: str = ""

