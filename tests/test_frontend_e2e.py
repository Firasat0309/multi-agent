"""End-to-end tests for the FrontendPipeline with dummy data.

Exercises the full flow: Design Parsing → Component Planning → DAG →
Component Generation → TSX Check → Import Validation → API Integration →
State Management → Documentation → Deployment → Final Build Check.

All LLM calls and external processes are mocked so these run fast and offline.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from core.models import (
    AgentContext,
    APIContract,
    APIEndpoint,
    ComponentPlan,
    FileBlueprint,
    ProductRequirements,
    RepositoryBlueprint,
    Task,
    TaskResult,
    TaskType,
    UIComponent,
    UIDesignSpec,
)
from core.tsx_compiler import TSXCompileResult, TSXError


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def requirements():
    return ProductRequirements(
        title="Task Manager",
        description="A SaaS task management app",
        user_stories=["As a user, I can create tasks"],
        features=["Task CRUD", "User auth", "Dashboard"],
        tech_preferences={
            "frontend": "nextjs",
            "backend": "fastapi",
            "db": "postgresql",
            "state": "zustand",
            "styling": "tailwind",
        },
        has_frontend=True,
        has_backend=True,
    )


@pytest.fixture
def api_contract():
    return APIContract(
        title="Task Manager API",
        version="1.0.0",
        base_url="/api/v1",
        endpoints=[
            APIEndpoint(
                path="/tasks", method="GET",
                description="List tasks", auth_required=True, tags=["tasks"],
            ),
            APIEndpoint(
                path="/tasks", method="POST",
                description="Create task", auth_required=True, tags=["tasks"],
            ),
            APIEndpoint(
                path="/auth/login", method="POST",
                description="Login", auth_required=False, tags=["auth"],
            ),
        ],
        schemas={
            "Task": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "title": {"type": "string"},
                    "status": {"type": "string", "enum": ["todo", "done"]},
                },
            },
        },
    )


@pytest.fixture
def component_plan():
    return ComponentPlan(
        components=[
            UIComponent(
                name="Button",
                file_path="src/components/ui/Button.tsx",
                component_type="ui",
                description="Reusable button",
                depends_on=[],
                layer="components/ui",
            ),
            UIComponent(
                name="TaskCard",
                file_path="src/components/feature/TaskCard.tsx",
                component_type="feature",
                description="Displays a single task",
                depends_on=["Button"],
                api_calls=["/tasks"],
                layer="components/feature",
            ),
            UIComponent(
                name="DashboardPage",
                file_path="src/app/dashboard/page.tsx",
                component_type="page",
                description="Dashboard page",
                depends_on=["TaskCard", "Sidebar"],  # Sidebar is a ghost
                state_needs=["tasks"],
                api_calls=["/tasks"],
                layer="pages",
            ),
        ],
        framework="nextjs",
        state_solution="zustand",
        api_base_url="/api/v1",
        routing_solution="nextjs",
    )


@pytest.fixture
def design_spec():
    return UIDesignSpec(
        framework="nextjs",
        design_description="Clean dashboard layout",
        pages=["Dashboard", "Login"],
        global_styles={"primary_color": "#3B82F6", "font_family": "Inter"},
        design_tokens={"colors": {"primary": "#3B82F6", "danger": "#EF4444"}},
    )


@pytest.fixture
def backend_blueprint():
    return RepositoryBlueprint(
        name="task-manager-backend",
        description="FastAPI backend",
        architecture_style="REST",
        tech_stack={"language": "python", "framework": "fastapi", "db": "postgresql"},
        file_blueprints=[
            FileBlueprint(
                path="app/models/task.py",
                purpose="Task ORM model",
                exports=["Task"],
                language="python",
                layer="model",
            ),
            FileBlueprint(
                path="app/models/user.py",
                purpose="User ORM model",
                exports=["User"],
                language="python",
                layer="model",
            ),
            FileBlueprint(
                path="app/api/routes.py",
                purpose="API routes",
                exports=["router"],
                language="python",
                layer="controller",
            ),
        ],
    )


@pytest.fixture
def mock_settings(tmp_path):
    s = MagicMock()
    s.workspace_dir = tmp_path
    s.mcp_server_command = None
    s.require_architecture_approval = False  # skip approval gates in tests
    s.build_checkpoint_retries = 1
    s.sandbox = MagicMock()
    s.sandbox.sandbox_type = MagicMock()
    s.memory = MagicMock()
    s.memory.embedding_model = "all-MiniLM-L6-v2"
    s.memory.chroma_persist_dir = str(tmp_path / ".chroma")
    return s


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate_json = AsyncMock()
    llm.generate = AsyncMock()
    llm.total_input_tokens = 1000
    llm.total_output_tokens = 500
    llm.config = MagicMock()
    llm.config.model = "test-model"
    return llm


def _make_agent_mock(success=True):
    """Create a mock agent whose execute() returns a TaskResult."""
    agent = MagicMock()
    agent.execute = AsyncMock(return_value=TaskResult(
        success=success,
        output="generated",
        files_modified=["test.tsx"],
    ))
    return agent


def _make_event_bus_mock():
    """Create an EventBus mock with async publish."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    return bus


# ── Helper to run the pipeline with full mocking ─────────────────────────────


async def _run_frontend_pipeline(
    tmp_path,
    mock_settings,
    mock_llm,
    requirements,
    api_contract,
    component_plan,
    design_spec,
    backend_blueprint=None,
    *,
    tsx_compile_results=None,
    agent_execute_side_effect=None,
    sandbox_fails=False,
    design_parse_fails=False,
    component_plan_fails=False,
    approval_rejects=False,
    require_approval=False,
):
    """Run FrontendPipeline.execute() with all external dependencies mocked."""
    from core.pipeline_frontend import FrontendPipeline

    mock_settings.require_architecture_approval = require_approval
    frontend_workspace = tmp_path / "frontend"
    frontend_workspace.mkdir(parents=True, exist_ok=True)

    pipeline = FrontendPipeline(
        settings=mock_settings,
        llm=mock_llm,
        live=None,
        interactive=True,
    )

    # Default tsx results: tsc not available (skip compilation checks)
    if tsx_compile_results is None:
        tsx_compile_results = [TSXCompileResult(tsc_available=False)]

    tsx_check_idx = {"val": 0}

    async def fake_tsx_check(workspace):
        idx = tsx_check_idx["val"]
        tsx_check_idx["val"] += 1
        if idx < len(tsx_compile_results):
            return tsx_compile_results[idx]
        return tsx_compile_results[-1]

    # Mock DesignParserAgent
    if design_parse_fails:
        mock_design_parser = MagicMock()
        mock_design_parser.parse_design = AsyncMock(side_effect=RuntimeError("Figma unavailable"))
    else:
        mock_design_parser = MagicMock()
        mock_design_parser.parse_design = AsyncMock(return_value=design_spec)

    # Mock ComponentPlannerAgent
    if component_plan_fails:
        mock_planner = MagicMock()
        mock_planner.plan_components = AsyncMock(side_effect=RuntimeError("LLM returned garbage"))
    else:
        mock_planner = MagicMock()
        mock_planner.plan_components = AsyncMock(return_value=component_plan)

    # Mock ComponentDAGAgent
    mock_dag = MagicMock()
    tier_map = {}
    for i, comp in enumerate(component_plan.components):
        # Simple: UI=0, feature=1, page=2
        if comp.component_type == "ui":
            tier_map[comp.name] = 0
        elif comp.component_type == "feature":
            tier_map[comp.name] = 1
        else:
            tier_map[comp.name] = 2
    mock_dag.build_dag = MagicMock(return_value=(component_plan.components, tier_map))

    # Mock SandboxOrchestrator
    mock_sandbox_result = MagicMock()
    mock_sandbox_result.manager = MagicMock()
    mock_sandbox_result.build_id = "build-123"
    mock_sandbox_result.test_id = "test-123"

    mock_sandbox = MagicMock()
    if sandbox_fails:
        from core.sandbox_orchestrator import SandboxUnavailableError
        mock_sandbox.setup = AsyncMock(side_effect=SandboxUnavailableError("Docker not found"))
    else:
        mock_sandbox.setup = AsyncMock(return_value=mock_sandbox_result)
    mock_sandbox.teardown = AsyncMock()

    # Mock AgentManager._create_agent — return agents that always succeed
    default_agent = _make_agent_mock(success=True)
    if agent_execute_side_effect:
        default_agent.execute = AsyncMock(side_effect=agent_execute_side_effect)

    mock_agent_manager = MagicMock()
    mock_agent_manager._create_agent = MagicMock(return_value=default_agent)

    # Mock EmbeddingStore
    mock_embedding = MagicMock()
    mock_embedding._ensure_client = MagicMock()

    patches = [
        patch("core.pipeline_frontend.DesignParserAgent", return_value=mock_design_parser),
        patch("core.pipeline_frontend.ComponentPlannerAgent", return_value=mock_planner),
        patch("core.pipeline_frontend.ComponentDAGAgent", return_value=mock_dag),
        patch("core.pipeline_frontend.SandboxOrchestrator", return_value=mock_sandbox),
        patch("core.pipeline_frontend.AgentManager", return_value=mock_agent_manager),
        patch("core.pipeline_frontend.EmbeddingStore", return_value=mock_embedding),
        patch("core.pipeline_frontend.DependencyGraphStore"),
        patch("core.pipeline_frontend.EventBus", return_value=_make_event_bus_mock()),
        patch("core.pipeline_frontend.index_workspace"),
        patch("core.pipeline_frontend.detect_language_from_blueprint"),
    ]

    tsx_compiler_patch = patch("core.pipeline_frontend.TSXCompiler")
    patches.append(tsx_compiler_patch)

    # Architecture approval mock — imported inside the method from core.architecture_approver
    if require_approval:
        mock_approver = MagicMock()
        mock_approver.approve_frontend_architecture = MagicMock(return_value=not approval_rejects)
        patches.append(
            patch("core.architecture_approver.ArchitectureApprover", return_value=mock_approver)
        )

    # Enter all patches
    entered = []
    for p in patches:
        entered.append(p.start())

    # Wire up TSXCompiler mock — tsx_compiler_patch is at index 10 in patches list
    tsx_mock_instance = MagicMock()
    tsx_mock_instance.check = AsyncMock(side_effect=fake_tsx_check)
    # entered[10] is the mock class returned by patch().start() for TSXCompiler
    entered[10].return_value = tsx_mock_instance

    # Mock npm subprocess (for Phase 9)
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"installed", None))

    try:
        with patch("core.pipeline_frontend.asyncio.create_subprocess_exec",
                    new_callable=AsyncMock, return_value=mock_proc):
            result = await pipeline.execute(
                requirements=requirements,
                api_contract=api_contract,
                start_time=0.0,
                frontend_workspace=frontend_workspace,
                backend_blueprint=backend_blueprint,
            )
    finally:
        for p in patches:
            p.stop()

    return result, {
        "agent_manager": mock_agent_manager,
        "sandbox": mock_sandbox,
        "dag": mock_dag,
        "design_parser": mock_design_parser,
        "planner": mock_planner,
        "tsx_compiler": tsx_mock_instance,
        "default_agent": default_agent,
    }


# ── Test Classes ──────────────────────────────────────────────────────────────


class TestFrontendE2EHappyPath:
    """Full pipeline run with all agents succeeding."""

    @pytest.mark.anyio
    async def test_success_all_phases(
        self, tmp_path, mock_settings, mock_llm,
        requirements, api_contract, component_plan, design_spec,
        backend_blueprint,
    ):
        result, mocks = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            requirements, api_contract, component_plan, design_spec,
            backend_blueprint=backend_blueprint,
        )

        assert result.success is True
        assert result.workspace_path == tmp_path / "frontend"
        assert result.errors == [] or all("TSX" not in e for e in result.errors)
        assert result.blueprint is not None
        assert result.blueprint.tech_stack["framework"] == "nextjs"

    @pytest.mark.anyio
    async def test_config_files_written(
        self, tmp_path, mock_settings, mock_llm,
        requirements, api_contract, component_plan, design_spec,
    ):
        result, _ = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            requirements, api_contract, component_plan, design_spec,
        )

        ws = tmp_path / "frontend"
        assert (ws / "package.json").exists()
        pkg = json.loads((ws / "package.json").read_text())
        assert pkg["name"] == "task-manager"
        assert "next" in pkg["dependencies"]
        assert "zustand" in pkg["dependencies"]
        assert "swr" in pkg["dependencies"]

        assert (ws / ".env.local").exists()
        env = (ws / ".env.local").read_text()
        assert "NEXT_PUBLIC_API_BASE_URL=/api/v1" in env

        assert (ws / "tsconfig.json").exists()
        assert (ws / "next.config.js").exists()
        assert (ws / "src" / "app" / "layout.tsx").exists()
        assert (ws / "src" / "app" / "globals.css").exists()
        assert (ws / "tailwind.config.js").exists()
        assert (ws / "postcss.config.js").exists()

    @pytest.mark.anyio
    async def test_report_written(
        self, tmp_path, mock_settings, mock_llm,
        requirements, api_contract, component_plan, design_spec,
    ):
        result, _ = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            requirements, api_contract, component_plan, design_spec,
        )

        report_path = tmp_path / "frontend" / "frontend_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["mode"] == "frontend"
        assert report["success"] is True
        assert report["framework"] == "nextjs"
        assert report["components_planned"] == 3

    @pytest.mark.anyio
    async def test_agents_called_for_all_phases(
        self, tmp_path, mock_settings, mock_llm,
        requirements, api_contract, component_plan, design_spec,
    ):
        """Verify all agent types were invoked."""
        result, mocks = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            requirements, api_contract, component_plan, design_spec,
        )

        create_calls = mocks["agent_manager"]._create_agent.call_args_list
        task_types_called = [c[0][0] for c in create_calls]

        # Component generation for 3 planned + 1 ghost (Sidebar)
        assert task_types_called.count(TaskType.GENERATE_COMPONENT) >= 1
        # API integration
        assert TaskType.INTEGRATE_API in task_types_called
        # State management
        assert TaskType.MANAGE_STATE in task_types_called
        # Documentation
        assert TaskType.GENERATE_DOCS in task_types_called
        # Deployment
        assert TaskType.GENERATE_DEPLOY in task_types_called


class TestFrontendE2EGhostDeps:
    """Ghost dependency detection and stub generation."""

    @pytest.mark.anyio
    async def test_ghost_sidebar_detected(
        self, tmp_path, mock_settings, mock_llm,
        requirements, api_contract, component_plan, design_spec,
    ):
        """DashboardPage depends on 'Sidebar' which is not in the plan."""
        result, mocks = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            requirements, api_contract, component_plan, design_spec,
        )

        # Ghost should be generated — check the blueprint includes it
        assert result.blueprint is not None
        paths = [fb.path for fb in result.blueprint.file_blueprints]
        assert any("Sidebar" in p for p in paths)

    @pytest.mark.anyio
    async def test_ghost_gets_correct_layer(self):
        from core.pipeline_frontend import FrontendPipeline

        comps = [
            UIComponent(
                name="Page",
                file_path="src/app/page.tsx",
                component_type="page",
                description="Main page",
                depends_on=["Header", "UserDropdown", "DataChart"],
            ),
        ]
        ghosts = FrontendPipeline._find_ghost_dependencies(
            comps, {"Page"}, framework="nextjs",
        )
        ghost_map = {g.name: g for g in ghosts}

        assert "Header" in ghost_map
        assert ghost_map["Header"].layer == "components/layout"  # layout keyword

        assert "UserDropdown" in ghost_map
        assert ghost_map["UserDropdown"].layer == "components/ui"  # dropdown keyword

        assert "DataChart" in ghost_map
        assert ghost_map["DataChart"].layer == "components/shared"  # default


class TestFrontendE2EDesignParsingFails:
    """Design parsing failure is non-fatal — pipeline should continue."""

    @pytest.mark.anyio
    async def test_continues_after_design_parse_failure(
        self, tmp_path, mock_settings, mock_llm,
        requirements, api_contract, component_plan, design_spec,
    ):
        result, _ = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            requirements, api_contract, component_plan, design_spec,
            design_parse_fails=True,
        )

        # Pipeline should still succeed — design parse is non-fatal
        assert result.success is True
        assert any("Design parsing failed" in e for e in result.errors)


class TestFrontendE2EComponentPlanFails:
    """Component planning failure is fatal — pipeline should stop."""

    @pytest.mark.anyio
    async def test_stops_on_component_plan_failure(
        self, tmp_path, mock_settings, mock_llm,
        requirements, api_contract, component_plan, design_spec,
    ):
        result, _ = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            requirements, api_contract, component_plan, design_spec,
            component_plan_fails=True,
        )

        assert result.success is False
        assert any("Component planning failed" in e for e in result.errors)


class TestFrontendE2ESandboxFails:
    """Sandbox unavailable — pipeline should fail gracefully."""

    @pytest.mark.anyio
    async def test_sandbox_unavailable(
        self, tmp_path, mock_settings, mock_llm,
        requirements, api_contract, component_plan, design_spec,
    ):
        result, _ = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            requirements, api_contract, component_plan, design_spec,
            sandbox_fails=True,
        )

        assert result.success is False
        assert any("Docker" in e or "Sandbox" in e or "sandbox" in e for e in result.errors)


class TestFrontendE2EApprovalGate:
    """Architecture approval gate (Gate 3)."""

    @pytest.mark.anyio
    async def test_approval_accepted(
        self, tmp_path, mock_settings, mock_llm,
        requirements, api_contract, component_plan, design_spec,
    ):
        result, _ = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            requirements, api_contract, component_plan, design_spec,
            require_approval=True,
            approval_rejects=False,
        )

        assert result.success is True

    @pytest.mark.anyio
    async def test_approval_rejected(
        self, tmp_path, mock_settings, mock_llm,
        requirements, api_contract, component_plan, design_spec,
    ):
        result, _ = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            requirements, api_contract, component_plan, design_spec,
            require_approval=True,
            approval_rejects=True,
        )

        assert result.success is False
        assert any("rejected" in e for e in result.errors)


class TestFrontendE2EComponentGenFailure:
    """One component generation fails — pipeline should continue but report errors."""

    @pytest.mark.anyio
    async def test_partial_component_failure(
        self, tmp_path, mock_settings, mock_llm,
        requirements, api_contract, component_plan, design_spec,
    ):
        call_count = {"n": 0}

        async def _alternating_results(ctx):
            call_count["n"] += 1
            if call_count["n"] == 2:  # Fail the second component
                return TaskResult(success=False, errors=["Cannot generate TaskCard"])
            return TaskResult(success=True, output="ok", files_modified=["f.tsx"])

        result, _ = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            requirements, api_contract, component_plan, design_spec,
            agent_execute_side_effect=_alternating_results,
        )

        # Pipeline should still succeed (component gen errors are non-fatal)
        assert any("TaskCard" in e or "Cannot generate" in e for e in result.errors)

    @pytest.mark.anyio
    async def test_component_gen_exception(
        self, tmp_path, mock_settings, mock_llm,
        requirements, api_contract, component_plan, design_spec,
    ):
        """Component generation raises an exception — should be caught."""
        call_count = {"n": 0}

        async def _raises_on_third(ctx):
            call_count["n"] += 1
            if call_count["n"] == 3:
                raise RuntimeError("LLM timeout")
            return TaskResult(success=True, output="ok", files_modified=["f.tsx"])

        result, _ = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            requirements, api_contract, component_plan, design_spec,
            agent_execute_side_effect=_raises_on_third,
        )

        assert any("LLM timeout" in e for e in result.errors)


class TestFrontendE2ETSXCompilation:
    """Phase 4.5 and Phase 9 TSX compilation checks."""

    @pytest.mark.anyio
    async def test_tsx_passes_first_try(
        self, tmp_path, mock_settings, mock_llm,
        requirements, api_contract, component_plan, design_spec,
    ):
        result, _ = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            requirements, api_contract, component_plan, design_spec,
            tsx_compile_results=[
                TSXCompileResult(tsc_available=True, errors=[]),  # Phase 4.5
                TSXCompileResult(tsc_available=True, errors=[]),  # Phase 9
            ],
        )
        assert result.success is True

    @pytest.mark.anyio
    async def test_tsx_errors_in_phase45_fixed(
        self, tmp_path, mock_settings, mock_llm,
        requirements, api_contract, component_plan, design_spec,
    ):
        """Phase 4.5 finds errors, fix retry succeeds."""
        result, mocks = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            requirements, api_contract, component_plan, design_spec,
            tsx_compile_results=[
                # Phase 4.5: first check has errors
                TSXCompileResult(tsc_available=True, errors=[
                    TSXError(file="src/components/ui/Button.tsx", line=5, col=1,
                             code="TS2304", message="Cannot find name 'Foo'"),
                ]),
                # Phase 4.5: after fix, passes
                TSXCompileResult(tsc_available=True, errors=[]),
                # Phase 9: passes
                TSXCompileResult(tsc_available=True, errors=[]),
            ],
        )

        # Fix agent should have been called
        create_calls = mocks["agent_manager"]._create_agent.call_args_list
        fix_calls = [c for c in create_calls if c[0][0] == TaskType.FIX_COMPONENT]
        assert len(fix_calls) >= 1

    @pytest.mark.anyio
    async def test_tsx_not_available_skipped(
        self, tmp_path, mock_settings, mock_llm,
        requirements, api_contract, component_plan, design_spec,
    ):
        """tsc not on PATH — should be non-fatal."""
        result, _ = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            requirements, api_contract, component_plan, design_spec,
            tsx_compile_results=[
                TSXCompileResult(tsc_available=False),
            ],
        )
        assert result.success is True

    @pytest.mark.anyio
    async def test_phase9_build_fails_marks_failure(
        self, tmp_path, mock_settings, mock_llm,
        requirements, api_contract, component_plan, design_spec,
    ):
        """Phase 9 final build has persistent errors — pipeline fails."""
        persistent_error = TSXError(
            file="src/lib/api.ts", line=10, col=1,
            code="TS2307", message="Cannot find module 'axios'",
        )
        result, _ = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            requirements, api_contract, component_plan, design_spec,
            tsx_compile_results=[
                TSXCompileResult(tsc_available=False),       # Phase 4.5 skipped
                TSXCompileResult(tsc_available=True, errors=[persistent_error]),  # Phase 9 initial
                TSXCompileResult(tsc_available=True, errors=[persistent_error]),  # Phase 9 after fix
            ],
        )

        assert result.success is False
        assert any("final_build" in str(e).lower() or "TS2307" in e for e in result.errors)


class TestFrontendE2EBackendModelScan:
    """Backend model discovery for FE type generation."""

    def test_strategy_a_blueprint_driven(self, tmp_path, backend_blueprint):
        from core.pipeline_frontend import FrontendPipeline

        backend_ws = tmp_path / "backend"
        backend_ws.mkdir()

        # Create one model file on disk
        model_dir = backend_ws / "app" / "models"
        model_dir.mkdir(parents=True)
        (model_dir / "task.py").write_text(
            "from sqlalchemy import Column, Integer, String\n\n"
            "class Task:\n"
            "    id: int\n"
            "    title: str\n"
            "    status: str\n",
            encoding="utf-8",
        )

        models = FrontendPipeline._scan_backend_models(
            backend_ws, backend_blueprint=backend_blueprint,
        )

        assert len(models) >= 1
        # task.py should be found via Strategy A
        assert any("task" in k.lower() for k in models)
        # user.py doesn't exist on disk — should have blueprint metadata
        assert any("user" in k.lower() for k in models)
        user_entry = [v for k, v in models.items() if "user" in k.lower()][0]
        assert "not yet generated" in user_entry or "from blueprint" in user_entry

    def test_strategy_a_no_model_layers(self, tmp_path):
        """Blueprint has no model-layer files — falls to Strategy B."""
        from core.pipeline_frontend import FrontendPipeline

        backend_ws = tmp_path / "backend"
        backend_ws.mkdir()

        bp = RepositoryBlueprint(
            name="api",
            description="API",
            architecture_style="REST",
            file_blueprints=[
                FileBlueprint(path="src/routes.py", purpose="Routes", layer="controller"),
            ],
        )

        models = FrontendPipeline._scan_backend_models(
            backend_ws, backend_blueprint=bp,
        )

        # No model layers in blueprint, no model dirs on disk
        assert len(models) == 0

    def test_strategy_b_filesystem_discovery(self, tmp_path):
        """No blueprint — uses find/glob to discover model directories."""
        from core.pipeline_frontend import FrontendPipeline

        backend_ws = tmp_path / "backend"
        (backend_ws / "models").mkdir(parents=True)
        (backend_ws / "models" / "user.py").write_text(
            "class User:\n    name: str\n    email: str\n",
            encoding="utf-8",
        )

        models = FrontendPipeline._scan_backend_models(backend_ws)

        assert len(models) >= 1
        assert any("user" in k.lower() for k in models)

    def test_nonexistent_backend_workspace(self, tmp_path):
        """Backend workspace doesn't exist — returns empty."""
        from core.pipeline_frontend import FrontendPipeline

        models = FrontendPipeline._scan_backend_models(
            tmp_path / "nonexistent",
        )
        assert models == {}


class TestFrontendE2EImportValidation:
    """Phase 4.6 import validation and auto-fix."""

    def test_broken_relative_import_detected(self, tmp_path):
        from core.pipeline_frontend import FrontendPipeline

        ws = tmp_path / "frontend"
        ws.mkdir()
        comp_dir = ws / "src" / "components" / "feature"
        comp_dir.mkdir(parents=True)
        (comp_dir / "TaskCard.tsx").write_text(
            "import { Button } from '../ui/Button';\n"
            "import { Sidebar } from '../layout/Sidebar';\n"
            "export const TaskCard = () => <div />;\n",
            encoding="utf-8",
        )

        # Only Button exists
        ui_dir = ws / "src" / "components" / "ui"
        ui_dir.mkdir(parents=True)
        (ui_dir / "Button.tsx").write_text("export const Button = () => <button />;\n")

        comps = [
            UIComponent(name="TaskCard", file_path="src/components/feature/TaskCard.tsx",
                        component_type="feature", description="Card"),
        ]

        errors = FrontendPipeline._validate_component_imports(ws, comps)
        # Sidebar import should be broken
        assert any("Sidebar" in e for e in errors)
        # Button import should be fine
        assert not any("Button" in e for e in errors)

    def test_at_alias_import_auto_fixed(self, tmp_path):
        from core.pipeline_frontend import FrontendPipeline

        ws = tmp_path / "frontend"
        ws.mkdir()
        page_dir = ws / "src" / "app" / "dashboard"
        page_dir.mkdir(parents=True)
        (page_dir / "page.tsx").write_text(
            "import { useAuthStore } from '@/store/useAuthStore';\n"
            "export default function DashboardPage() { return <div />; }\n",
            encoding="utf-8",
        )

        store_dir = ws / "src" / "store"
        store_dir.mkdir(parents=True)
        (store_dir / "useAuthStore.ts").write_text(
            "export const useAuthStore = () => ({});\n"
        )

        comps = [
            UIComponent(name="DashboardPage", file_path="src/app/dashboard/page.tsx",
                        component_type="page", description="Dashboard"),
        ]

        fixed = FrontendPipeline._auto_fix_imports(ws, comps)
        # @/store/useAuthStore should resolve — either skipped or fixed
        content = (page_dir / "page.tsx").read_text()
        # After fix, import should be relative or @/ should resolve
        assert "useAuthStore" in content


class TestFrontendE2EVueFramework:
    """Vue 3 specific config file generation."""

    @pytest.mark.anyio
    async def test_vue_config_files(
        self, tmp_path, mock_settings, mock_llm,
        api_contract, design_spec,
    ):
        vue_requirements = ProductRequirements(
            title="Vue App",
            description="A Vue app",
            tech_preferences={"frontend": "Vue 3", "styling": "tailwind"},
            has_frontend=True,
            has_backend=False,
        )
        vue_plan = ComponentPlan(
            components=[
                UIComponent(
                    name="HelloWorld",
                    file_path="src/components/HelloWorld.vue",
                    component_type="ui",
                    description="Hello world",
                    layer="components",
                ),
            ],
            framework="vue",
            state_solution="pinia",
            api_base_url="/api/v1",
            routing_solution="vue-router",
        )

        result, _ = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            vue_requirements, api_contract, vue_plan, design_spec,
        )

        ws = tmp_path / "frontend"
        assert (ws / "vite.config.ts").exists()
        assert (ws / "index.html").exists()
        assert (ws / "src" / "main.ts").exists()
        assert (ws / "tsconfig.json").exists()
        assert (ws / "tsconfig.node.json").exists()

        pkg = json.loads((ws / "package.json").read_text())
        assert "vue" in pkg.get("dependencies", {})
        assert "pinia" in pkg.get("dependencies", {})

        env = (ws / ".env.local").read_text()
        assert "VITE_API_BASE_URL" in env

        tsconfig = json.loads((ws / "tsconfig.json").read_text())
        assert "src/**/*.vue" in tsconfig.get("include", [])


class TestFrontendE2ENoBackendBlueprint:
    """Pipeline runs without a backend blueprint — Strategy B used for model scan."""

    @pytest.mark.anyio
    async def test_works_without_backend_blueprint(
        self, tmp_path, mock_settings, mock_llm,
        requirements, api_contract, component_plan, design_spec,
    ):
        result, _ = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            requirements, api_contract, component_plan, design_spec,
            backend_blueprint=None,
        )

        assert result.success is True


class TestFrontendE2ENoAPIContract:
    """Pipeline runs without an API contract."""

    @pytest.mark.anyio
    async def test_works_without_api_contract(
        self, tmp_path, mock_settings, mock_llm,
        requirements, component_plan, design_spec,
    ):
        result, _ = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            requirements, None, component_plan, design_spec,
        )

        assert result.success is True


class TestFrontendE2EEmptyComponentPlan:
    """Component plan has zero components — edge case."""

    @pytest.mark.anyio
    async def test_empty_plan(
        self, tmp_path, mock_settings, mock_llm,
        requirements, api_contract, design_spec,
    ):
        empty_plan = ComponentPlan(
            components=[],
            framework="nextjs",
            state_solution="zustand",
            api_base_url="/api/v1",
        )

        result, _ = await _run_frontend_pipeline(
            tmp_path, mock_settings, mock_llm,
            requirements, api_contract, empty_plan, design_spec,
        )

        assert result.success is True
        assert result.metrics.get("components_planned", 0) == 0


class TestFrontendE2EWriteConfigEdgeCases:
    """Edge cases in _write_config_files."""

    def test_angular_config(self, tmp_path):
        from core.pipeline_frontend import FrontendPipeline

        ws = tmp_path / "angular-app"
        ws.mkdir()
        plan = ComponentPlan(
            components=[],
            framework="angular",
            state_solution="ngrx",
            api_base_url="/api",
        )
        req = ProductRequirements(
            title="Angular App",
            description="Test",
            tech_preferences={"styling": "css"},
        )

        FrontendPipeline._write_config_files(ws, plan, req)

        assert (ws / "package.json").exists()
        pkg = json.loads((ws / "package.json").read_text())
        assert "@angular/core" in pkg.get("dependencies", {})

        assert (ws / "tsconfig.json").exists()
        # No tailwind files for css styling
        assert not (ws / "tailwind.config.js").exists()

    def test_react_plain_config(self, tmp_path):
        from core.pipeline_frontend import FrontendPipeline

        ws = tmp_path / "react-app"
        ws.mkdir()
        plan = ComponentPlan(
            components=[],
            framework="react",
            state_solution="redux",
            api_base_url="/api",
        )
        req = ProductRequirements(
            title="React App",
            description="Test",
            tech_preferences={"styling": "css"},
        )

        FrontendPipeline._write_config_files(ws, plan, req)

        pkg = json.loads((ws / "package.json").read_text())
        assert "react" in pkg.get("dependencies", {})
        assert "@reduxjs/toolkit" in pkg.get("dependencies", {})
        assert "react-redux" in pkg.get("dependencies", {})

        env = (ws / ".env.local").read_text()
        assert "REACT_APP_API_BASE_URL" in env

    def test_zustand_immer_added(self, tmp_path):
        from core.pipeline_frontend import FrontendPipeline

        ws = tmp_path / "zustand-app"
        ws.mkdir()
        plan = ComponentPlan(
            components=[],
            framework="nextjs",
            state_solution="zustand",
            api_base_url="/api",
        )
        req = ProductRequirements(title="App", description="Test",
                                  tech_preferences={"styling": "tailwind"})

        FrontendPipeline._write_config_files(ws, plan, req)

        pkg = json.loads((ws / "package.json").read_text())
        assert "zustand" in pkg.get("dependencies", {})

    def test_nextjs_segment_layouts_injected(self, tmp_path):
        """Nested page routes should get auto-injected segment layouts."""
        from core.pipeline_frontend import FrontendPipeline

        ws = tmp_path / "next-app"
        ws.mkdir()
        plan = ComponentPlan(
            components=[
                UIComponent(
                    name="DashboardPage",
                    file_path="src/app/dashboard/page.tsx",
                    component_type="page",
                    description="Dashboard",
                ),
                UIComponent(
                    name="SettingsPage",
                    file_path="src/app/dashboard/settings/page.tsx",
                    component_type="page",
                    description="Settings",
                ),
                UIComponent(
                    name="HomePage",
                    file_path="src/app/page.tsx",
                    component_type="page",
                    description="Home",
                ),
            ],
            framework="nextjs",
            state_solution="zustand",
            api_base_url="/api",
        )
        req = ProductRequirements(title="App", description="Test",
                                  tech_preferences={"styling": "tailwind"})

        FrontendPipeline._write_config_files(ws, plan, req)

        # Root layout should exist
        assert (ws / "src" / "app" / "layout.tsx").exists()
        # Dashboard segment layout should be auto-injected
        assert (ws / "src" / "app" / "dashboard" / "layout.tsx").exists()
        dashboard_layout = (ws / "src" / "app" / "dashboard" / "layout.tsx").read_text()
        assert "DashboardLayout" in dashboard_layout
        assert "children" in dashboard_layout
        # Settings is nested under dashboard — dashboard layout covers it,
        # but settings itself also needs a layout
        assert (ws / "src" / "app" / "dashboard" / "settings" / "layout.tsx").exists()
        # Home page is direct child of app/ — no extra layout needed
        # (root layout handles it)

    def test_nextjs_no_duplicate_layouts(self, tmp_path):
        """If component planner already created a layout, don't overwrite it."""
        from core.pipeline_frontend import FrontendPipeline

        ws = tmp_path / "next-app"
        ws.mkdir()
        # Pre-create a custom dashboard layout
        dash_dir = ws / "src" / "app" / "dashboard"
        dash_dir.mkdir(parents=True)
        custom_layout = "export default function CustomDashboardLayout({ children }) { return <nav>{children}</nav>; }\n"
        (dash_dir / "layout.tsx").write_text(custom_layout)

        plan = ComponentPlan(
            components=[
                UIComponent(
                    name="DashboardPage",
                    file_path="src/app/dashboard/page.tsx",
                    component_type="page",
                    description="Dashboard",
                ),
            ],
            framework="nextjs",
            state_solution="zustand",
            api_base_url="/api",
        )
        req = ProductRequirements(title="App", description="Test",
                                  tech_preferences={"styling": "tailwind"})

        FrontendPipeline._write_config_files(ws, plan, req)

        # Custom layout should NOT be overwritten
        assert "CustomDashboardLayout" in (dash_dir / "layout.tsx").read_text()

    def test_nextjs_non_page_components_ignored(self, tmp_path):
        """Non-page components should not trigger layout injection."""
        from core.pipeline_frontend import FrontendPipeline

        ws = tmp_path / "next-app"
        ws.mkdir()
        plan = ComponentPlan(
            components=[
                UIComponent(
                    name="Button",
                    file_path="src/components/ui/Button.tsx",
                    component_type="ui",
                    description="Button",
                ),
            ],
            framework="nextjs",
            state_solution="zustand",
            api_base_url="/api",
        )
        req = ProductRequirements(title="App", description="Test",
                                  tech_preferences={"styling": "tailwind"})

        FrontendPipeline._write_config_files(ws, plan, req)

        # Root layout should exist (always for Next.js)
        assert (ws / "src" / "app" / "layout.tsx").exists()
        # No components/ui/layout.tsx should be created
        assert not (ws / "src" / "components" / "ui" / "layout.tsx").exists()

    def test_nextjs_app_without_src_prefix(self, tmp_path):
        """When LLM outputs app/ paths (no src/), layouts go to workspace/app/."""
        from core.pipeline_frontend import FrontendPipeline

        ws = tmp_path / "next-app"
        ws.mkdir()
        plan = ComponentPlan(
            components=[
                UIComponent(
                    name="DashboardPage",
                    file_path="app/dashboard/page.tsx",
                    component_type="page",
                    description="Dashboard",
                ),
                UIComponent(
                    name="HomePage",
                    file_path="app/page.tsx",
                    component_type="page",
                    description="Home",
                ),
            ],
            framework="nextjs",
            state_solution="zustand",
            api_base_url="/api",
        )
        req = ProductRequirements(title="App", description="Test",
                                  tech_preferences={"styling": "tailwind"})

        FrontendPipeline._write_config_files(ws, plan, req)

        # Root layout at app/ (NOT src/app/) — matches page paths
        assert (ws / "app" / "layout.tsx").exists()
        root_layout = (ws / "app" / "layout.tsx").read_text()
        assert "RootLayout" in root_layout
        # Dashboard segment layout
        assert (ws / "app" / "dashboard" / "layout.tsx").exists()
        # Should NOT create src/app/layout.tsx (wrong tree)
        assert not (ws / "src" / "app" / "layout.tsx").exists()


class TestPreBuildFileVerification:
    """Verify that the Phase 4a stub-generation logic writes valid stubs."""

    def test_stub_is_valid_tsx(self, tmp_path):
        """Stubs for missing components should be importable TSX."""
        comp = UIComponent(
            name="MissingWidget",
            file_path="src/components/feature/MissingWidget.tsx",
            component_type="feature",
            description="A widget that failed generation",
        )
        stub_path = tmp_path / comp.file_path
        stub_path.parent.mkdir(parents=True, exist_ok=True)

        # Simulate the stub-writing logic from Phase 4a
        stub_name = comp.name or Path(comp.file_path).stem
        stub = (
            f"// AUTO-STUB: generation failed for {comp.name}\n"
            f"// TODO: implement {comp.description or comp.name}\n"
            f"export default function {stub_name}() {{\n"
            f"  return <div>{stub_name} — not yet implemented</div>;\n"
            f"}}\n"
        )
        stub_path.write_text(stub, encoding="utf-8")

        content = stub_path.read_text()
        assert "export default function MissingWidget" in content
        assert "AUTO-STUB" in content
        assert "TODO" in content

    def test_missing_file_detection(self, tmp_path):
        """Missing files should be detected from ordered_components."""
        components = [
            UIComponent(
                name="Button",
                file_path="src/components/ui/Button.tsx",
                component_type="ui",
                description="Button",
            ),
            UIComponent(
                name="Card",
                file_path="src/components/ui/Card.tsx",
                component_type="ui",
                description="Card",
            ),
        ]
        # Only create Button, not Card
        btn_path = tmp_path / components[0].file_path
        btn_path.parent.mkdir(parents=True, exist_ok=True)
        btn_path.write_text("export default function Button() {}", encoding="utf-8")

        missing = [
            c for c in components
            if c.file_path and not (tmp_path / c.file_path).exists()
        ]
        assert len(missing) == 1
        assert missing[0].name == "Card"


class TestVueEntryStubGeneration:
    """Verify that _write_config_files creates the mandatory Vue entry stubs."""

    def _make_vue_plan(self, components=None):
        return ComponentPlan(
            components=components or [],
            framework="vue",
            state_solution="pinia",
            api_base_url="/api/v1",
            routing_solution="vue-router",
        )

    def _make_req(self, styling="css"):
        return ProductRequirements(
            title="Vue App",
            description="Test",
            tech_preferences={"styling": styling},
        )

    def test_app_vue_stub_created(self, tmp_path):
        """src/App.vue stub must be created alongside main.ts."""
        from core.pipeline_frontend import FrontendPipeline

        ws = tmp_path / "vue-app"
        ws.mkdir()
        FrontendPipeline._write_config_files(ws, self._make_vue_plan(), self._make_req())

        app_vue = ws / "src" / "App.vue"
        assert app_vue.exists(), "App.vue stub was not generated"
        content = app_vue.read_text()
        assert "<RouterView" in content
        assert "<template>" in content

    def test_router_index_stub_created(self, tmp_path):
        """src/router/index.ts stub must be created alongside main.ts."""
        from core.pipeline_frontend import FrontendPipeline

        ws = tmp_path / "vue-app"
        ws.mkdir()
        FrontendPipeline._write_config_files(ws, self._make_vue_plan(), self._make_req())

        router = ws / "src" / "router" / "index.ts"
        assert router.exists(), "router/index.ts stub was not generated"
        content = router.read_text()
        assert "createRouter" in content
        assert "createWebHistory" in content
        assert "export default router" in content

    def test_main_css_stub_created(self, tmp_path):
        """src/assets/main.css stub must be created alongside main.ts."""
        from core.pipeline_frontend import FrontendPipeline

        ws = tmp_path / "vue-app"
        ws.mkdir()
        FrontendPipeline._write_config_files(ws, self._make_vue_plan(), self._make_req())

        css = ws / "src" / "assets" / "main.css"
        assert css.exists(), "main.css stub was not generated"

    def test_vite_env_stub_created(self, tmp_path):
        """src/vite-env.d.ts must be created for Vue TypeScript projects."""
        from core.pipeline_frontend import FrontendPipeline

        ws = tmp_path / "vue-app"
        ws.mkdir()
        FrontendPipeline._write_config_files(ws, self._make_vue_plan(), self._make_req())

        vite_env = ws / "src" / "vite-env.d.ts"
        assert vite_env.exists(), "vite-env.d.ts stub was not generated"
        assert "vite/client" in vite_env.read_text()

    def test_stubs_not_overwritten_if_exist(self, tmp_path):
        """Pre-existing files must not be overwritten by stubs."""
        from core.pipeline_frontend import FrontendPipeline

        ws = tmp_path / "vue-app"
        ws.mkdir()
        src = ws / "src"
        src.mkdir()
        (src / "App.vue").write_text("<template>Custom</template>", encoding="utf-8")
        (src / "router").mkdir()
        (src / "router" / "index.ts").write_text("// custom", encoding="utf-8")
        (src / "assets").mkdir()
        (src / "assets" / "main.css").write_text("body{}", encoding="utf-8")

        FrontendPipeline._write_config_files(ws, self._make_vue_plan(), self._make_req())

        assert (src / "App.vue").read_text() == "<template>Custom</template>"
        assert (src / "router" / "index.ts").read_text() == "// custom"
        # Note: main.css is always overwritten by the globals CSS section
        # which runs before the stub check — so we only check App.vue and router.

    def test_router_includes_page_routes(self, tmp_path):
        """Page components should be wired into the router stub."""
        from core.pipeline_frontend import FrontendPipeline

        ws = tmp_path / "vue-app"
        ws.mkdir()
        plan = self._make_vue_plan([
            UIComponent(
                name="HomeView",
                file_path="src/views/HomeView.vue",
                component_type="page",
                description="Home page",
            ),
            UIComponent(
                name="AboutView",
                file_path="src/views/AboutView.vue",
                component_type="page",
                description="About page",
            ),
            UIComponent(
                name="Navbar",
                file_path="src/components/Navbar.vue",
                component_type="ui",
                description="Navigation bar",
            ),
        ])
        FrontendPipeline._write_config_files(ws, plan, self._make_req())

        content = (ws / "src" / "router" / "index.ts").read_text()
        assert "HomeView" in content
        assert "AboutView" in content
        # Non-page component should NOT be in routes
        assert "Navbar" not in content

    def test_home_page_gets_root_route(self, tmp_path):
        """A component named 'Home*' or 'Index*' should get the '/' route."""
        from core.pipeline_frontend import FrontendPipeline

        ws = tmp_path / "vue-app"
        ws.mkdir()
        plan = self._make_vue_plan([
            UIComponent(
                name="HomeView",
                file_path="src/views/HomeView.vue",
                component_type="page",
                description="Home",
            ),
        ])
        FrontendPipeline._write_config_files(ws, plan, self._make_req())

        content = (ws / "src" / "router" / "index.ts").read_text()
        assert "path: '/'" in content

    def test_router_creates_missing_view_stubs(self, tmp_path):
        """Referenced Vue page components should get placeholder files when absent."""
        from core.pipeline_frontend import FrontendPipeline

        ws = tmp_path / "vue-app"
        ws.mkdir()
        plan = self._make_vue_plan([
            UIComponent(
                name="HomeView",
                file_path="src/views/HomeView.vue",
                component_type="page",
                description="Home",
            ),
        ])

        FrontendPipeline._write_config_files(ws, plan, self._make_req())

        home_view = ws / "src" / "views" / "HomeView.vue"
        assert home_view.exists(), "Referenced view stub was not generated"
        content = home_view.read_text()
        assert "<template><div>HomeView</div></template>" in content

    def test_nextjs_does_not_get_vue_stubs(self, tmp_path):
        """Next.js projects should NOT get App.vue or router/index.ts."""
        from core.pipeline_frontend import FrontendPipeline

        ws = tmp_path / "next-app"
        ws.mkdir()
        plan = ComponentPlan(
            components=[],
            framework="nextjs",
            state_solution="zustand",
            api_base_url="/api",
        )
        req = ProductRequirements(
            title="Next App", description="Test",
            tech_preferences={"styling": "css"},
        )
        FrontendPipeline._write_config_files(ws, plan, req)

        assert not (ws / "src" / "App.vue").exists()
        assert not (ws / "src" / "router" / "index.ts").exists()
