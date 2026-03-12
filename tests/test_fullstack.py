"""Tests for fullstack agent system — models, agents, pipelines, and DAG logic."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.models import (
    AgentContext,
    AgentRole,
    APIContract,
    APIEndpoint,
    ComponentPlan,
    FullstackBlueprint,
    ProductRequirements,
    RepositoryBlueprint,
    Task,
    TaskStatus,
    TaskType,
    UIComponent,
    UIDesignSpec,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate_json = AsyncMock()
    llm.generate = AsyncMock()
    return llm


@pytest.fixture
def mock_repo_manager(tmp_path):
    rm = MagicMock()
    rm.workspace = tmp_path
    return rm


@pytest.fixture
def sample_requirements():
    return ProductRequirements(
        title="Task Manager",
        description="A SaaS task management application.",
        user_stories=["As a user, I want to create tasks"],
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
def sample_api_contract():
    return APIContract(
        title="Task Manager API",
        version="1.0.0",
        base_url="/api/v1",
        endpoints=[
            APIEndpoint(
                path="/tasks",
                method="GET",
                description="List tasks",
                auth_required=True,
                tags=["tasks"],
            ),
            APIEndpoint(
                path="/tasks",
                method="POST",
                description="Create task",
                auth_required=True,
                tags=["tasks"],
            ),
            APIEndpoint(
                path="/users/me",
                method="GET",
                description="Get current user",
                auth_required=True,
                tags=["users"],
            ),
        ],
        schemas={"Task": {"type": "object"}, "User": {"type": "object"}},
        openapi_spec="openapi: 3.0.3\ninfo:\n  title: Task Manager API",
    )


@pytest.fixture
def sample_component_plan():
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
                api_calls=["/api/v1/tasks"],
                state_needs=["taskStore"],
                layer="components/feature",
            ),
            UIComponent(
                name="TaskList",
                file_path="src/components/feature/TaskList.tsx",
                component_type="feature",
                description="List of tasks",
                depends_on=["TaskCard"],
                api_calls=["/api/v1/tasks"],
                state_needs=["taskStore"],
                layer="components/feature",
            ),
            UIComponent(
                name="Dashboard",
                file_path="src/pages/Dashboard.tsx",
                component_type="pages",
                description="Dashboard page",
                depends_on=["TaskList"],
                layer="pages",
            ),
        ],
        framework="nextjs",
        state_solution="zustand",
        api_base_url="/api/v1",
        routing_solution="nextjs",
    )


# ── Model tests ───────────────────────────────────────────────────────────────


class TestFullstackModels:
    def test_product_requirements_defaults(self):
        req = ProductRequirements(title="Test", description="Desc")
        assert req.has_frontend is True
        assert req.has_backend is True
        assert req.user_stories == []
        assert req.features == []
        assert req.tech_preferences == {}

    def test_api_contract_defaults(self):
        contract = APIContract(title="My API")
        assert contract.version == "1.0.0"
        assert contract.base_url == "/api/v1"
        assert contract.contract_format == "openapi"
        assert contract.endpoints == []

    def test_ui_component_defaults(self):
        comp = UIComponent(
            name="Button",
            file_path="src/Button.tsx",
            component_type="ui",
            description="A button",
        )
        assert comp.depends_on == []
        assert comp.props == []
        assert comp.api_calls == []
        assert comp.state_needs == []

    def test_fullstack_blueprint_all_none_defaults(self):
        fb = FullstackBlueprint()
        assert fb.product_requirements is None
        assert fb.backend_blueprint is None
        assert fb.frontend_blueprint is None
        assert fb.api_contract is None
        assert fb.component_plan is None

    def test_new_task_types_in_enum(self):
        assert TaskType.PLAN_PRODUCT == "plan_product"
        assert TaskType.GENERATE_API_CONTRACT == "generate_api_contract"
        assert TaskType.PARSE_DESIGN == "parse_design"
        assert TaskType.PLAN_COMPONENTS == "plan_components"
        assert TaskType.BUILD_COMPONENT_DAG == "build_component_dag"
        assert TaskType.GENERATE_COMPONENT == "generate_component"
        assert TaskType.INTEGRATE_API == "integrate_api"
        assert TaskType.MANAGE_STATE == "manage_state"

    def test_new_agent_roles_in_enum(self):
        assert AgentRole.PRODUCT_PLANNER == "product_planner"
        assert AgentRole.API_CONTRACT_GENERATOR == "api_contract_generator"
        assert AgentRole.DESIGN_PARSER == "design_parser"
        assert AgentRole.COMPONENT_PLANNER == "component_planner"
        assert AgentRole.COMPONENT_DAG_BUILDER == "component_dag_builder"
        assert AgentRole.COMPONENT_GENERATOR == "component_generator"
        assert AgentRole.API_INTEGRATOR == "api_integrator"
        assert AgentRole.STATE_MANAGER == "state_manager"


# ── ProductPlannerAgent tests ─────────────────────────────────────────────────


class TestProductPlannerAgent:
    @pytest.mark.anyio
    async def test_plan_product_returns_requirements(self, mock_llm, mock_repo_manager):
        from agents.product_planner_agent import ProductPlannerAgent

        mock_llm.generate_json.return_value = {
            "title": "Task Manager",
            "description": "A task manager app.",
            "user_stories": ["As a user, I want tasks"],
            "features": ["Task CRUD", "Auth"],
            "tech_preferences": {"frontend": "nextjs", "backend": "fastapi"},
            "has_frontend": True,
            "has_backend": True,
        }

        agent = ProductPlannerAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        req = await agent.plan_product("Build a task manager")

        assert req.title == "Task Manager"
        assert req.has_frontend is True
        assert req.has_backend is True
        assert "Task CRUD" in req.features

    @pytest.mark.anyio
    async def test_execute_returns_task_result(self, mock_llm, mock_repo_manager):
        from agents.product_planner_agent import ProductPlannerAgent

        mock_llm.generate_json.return_value = {
            "title": "App",
            "description": "An app",
            "has_frontend": True,
            "has_backend": False,
        }

        agent = ProductPlannerAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        context = AgentContext(
            task=Task(
                task_id=1,
                task_type=TaskType.PLAN_PRODUCT,
                file="",
                description="Plan product",
                metadata={"user_prompt": "Build a static site"},
            ),
            blueprint=MagicMock(),
        )
        result = await agent.execute(context)

        assert result.success is True
        assert "App" in result.output

    @pytest.mark.anyio
    async def test_execute_handles_llm_failure(self, mock_llm, mock_repo_manager):
        from agents.product_planner_agent import ProductPlannerAgent

        mock_llm.generate_json.side_effect = RuntimeError("LLM unavailable")

        agent = ProductPlannerAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        context = AgentContext(
            task=Task(
                task_id=2,
                task_type=TaskType.PLAN_PRODUCT,
                file="",
                description="Plan",
                metadata={},
            ),
            blueprint=MagicMock(),
        )
        result = await agent.execute(context)

        assert result.success is False
        assert len(result.errors) > 0


# ── APIContractAgent tests ────────────────────────────────────────────────────


class TestAPIContractAgent:
    @pytest.mark.anyio
    async def test_generate_contract(
        self, mock_llm, mock_repo_manager, sample_requirements
    ):
        from agents.api_contract_agent import APIContractAgent

        mock_llm.generate_json.return_value = {
            "title": "Task Manager API",
            "version": "1.0.0",
            "base_url": "/api/v1",
            "contract_format": "openapi",
            "endpoints": [
                {
                    "path": "/tasks",
                    "method": "GET",
                    "description": "List tasks",
                    "auth_required": True,
                    "tags": ["tasks"],
                }
            ],
            "schemas": {"Task": {"type": "object"}},
            "openapi_spec": "openapi: 3.0.3",
        }

        bp = MagicMock(spec=RepositoryBlueprint)
        bp.name = "backend"
        bp.architecture_style = "REST"
        bp.tech_stack = {"language": "python"}
        bp.file_blueprints = []

        agent = APIContractAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        contract = await agent.generate_contract(sample_requirements, bp)

        assert contract.title == "Task Manager API"
        assert len(contract.endpoints) == 1
        assert contract.endpoints[0].method == "GET"
        assert "Task" in contract.schemas

    @pytest.mark.anyio
    async def test_parse_contract_handles_missing_fields(
        self, mock_llm, mock_repo_manager
    ):
        from agents.api_contract_agent import APIContractAgent

        agent = APIContractAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        contract = agent._parse_contract({})

        assert contract.title == "API"
        assert contract.endpoints == []
        assert contract.version == "1.0.0"


# ── DesignParserAgent tests ───────────────────────────────────────────────────


class TestDesignParserAgent:
    @pytest.mark.anyio
    async def test_parse_design_returns_spec(
        self, mock_llm, mock_repo_manager, sample_requirements
    ):
        from agents.design_parser_agent import DesignParserAgent

        mock_llm.generate_json.return_value = {
            "framework": "nextjs",
            "design_description": "Modern dashboard UI",
            "figma_url": "",
            "pages": ["Home", "Dashboard", "Profile"],
            "global_styles": {"primary_color": "#3B82F6"},
            "design_tokens": {"colors": {"brand": "#3B82F6"}},
        }

        agent = DesignParserAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        spec = await agent.parse_design(sample_requirements, "")

        assert spec.framework == "nextjs"
        assert "Dashboard" in spec.pages
        assert len(spec.pages) == 3

    def test_parse_spec_respects_vue_preference(
        self, mock_llm, mock_repo_manager
    ):
        from agents.design_parser_agent import DesignParserAgent

        agent = DesignParserAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        req = ProductRequirements(
            title="App",
            description="Test",
            tech_preferences={"frontend": "Vue 3"},
        )
        spec = agent._parse_spec({"pages": ["Home"]}, req, "")
        assert spec.framework == "vue"


# ── ComponentPlannerAgent tests ───────────────────────────────────────────────


class TestComponentPlannerAgent:
    @pytest.mark.anyio
    async def test_plan_components(
        self,
        mock_llm,
        mock_repo_manager,
        sample_requirements,
        sample_api_contract,
    ):
        from agents.component_planner_agent import ComponentPlannerAgent

        mock_llm.generate_json.return_value = {
            "framework": "nextjs",
            "state_solution": "zustand",
            "api_base_url": "/api/v1",
            "routing_solution": "nextjs",
            "package_json": {},
            "components": [
                {
                    "name": "Button",
                    "file_path": "src/components/ui/Button.tsx",
                    "component_type": "ui",
                    "description": "Button",
                    "props": ["label: string"],
                    "depends_on": [],
                }
            ],
        }

        design_spec = UIDesignSpec(framework="nextjs", pages=["Home", "Dashboard"])

        agent = ComponentPlannerAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        plan = await agent.plan_components(design_spec, sample_api_contract, sample_requirements)

        assert plan.framework == "nextjs"
        assert plan.state_solution == "zustand"
        assert len(plan.components) == 1
        assert plan.components[0].name == "Button"


# ── ComponentDAGAgent tests ───────────────────────────────────────────────────


class TestComponentDAGAgent:
    def _make_agent(self, mock_llm, mock_repo_manager):
        from agents.component_dag_agent import ComponentDAGAgent
        return ComponentDAGAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)

    def test_build_dag_linear_chain(
        self, mock_llm, mock_repo_manager, sample_component_plan
    ):
        agent = self._make_agent(mock_llm, mock_repo_manager)
        ordered, tier_map = agent.build_dag(sample_component_plan)

        # Button has no deps → tier 0
        assert tier_map["Button"] == 0
        # TaskCard depends on Button → tier 1
        assert tier_map["TaskCard"] == 1
        # TaskList depends on TaskCard → tier 2
        assert tier_map["TaskList"] == 2
        # Dashboard depends on TaskList → tier 3
        assert tier_map["Dashboard"] == 3

    def test_build_dag_preserves_all_components(
        self, mock_llm, mock_repo_manager, sample_component_plan
    ):
        agent = self._make_agent(mock_llm, mock_repo_manager)
        ordered, _ = agent.build_dag(sample_component_plan)
        assert len(ordered) == len(sample_component_plan.components)

    def test_build_dag_detects_cycle(self, mock_llm, mock_repo_manager):
        agent = self._make_agent(mock_llm, mock_repo_manager)
        cyclic_plan = ComponentPlan(
            components=[
                UIComponent(
                    name="A",
                    file_path="A.tsx",
                    component_type="ui",
                    description="A",
                    depends_on=["B"],
                ),
                UIComponent(
                    name="B",
                    file_path="B.tsx",
                    component_type="ui",
                    description="B",
                    depends_on=["A"],
                ),
            ]
        )
        with pytest.raises(ValueError, match="Cycle detected"):
            agent.build_dag(cyclic_plan)

    def test_build_dag_independent_components_all_tier_zero(
        self, mock_llm, mock_repo_manager
    ):
        agent = self._make_agent(mock_llm, mock_repo_manager)
        flat_plan = ComponentPlan(
            components=[
                UIComponent(
                    name=f"Comp{i}",
                    file_path=f"src/Comp{i}.tsx",
                    component_type="ui",
                    description=f"Component {i}",
                )
                for i in range(5)
            ]
        )
        _, tier_map = agent.build_dag(flat_plan)
        assert all(v == 0 for v in tier_map.values())

    @pytest.mark.anyio
    async def test_execute_returns_task_result(
        self, mock_llm, mock_repo_manager, sample_component_plan
    ):
        from agents.component_dag_agent import ComponentDAGAgent

        agent = ComponentDAGAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        blueprint = MagicMock(spec=RepositoryBlueprint)
        context = AgentContext(
            task=Task(
                task_id=1,
                task_type=TaskType.BUILD_COMPONENT_DAG,
                file="",
                description="Build DAG",
                metadata={"component_plan": sample_component_plan},
            ),
            blueprint=blueprint,
        )
        result = await agent.execute(context)

        assert result.success is True
        assert "tiers" in result.metrics

    @pytest.mark.anyio
    async def test_execute_fails_without_plan(self, mock_llm, mock_repo_manager):
        from agents.component_dag_agent import ComponentDAGAgent

        agent = ComponentDAGAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        blueprint = MagicMock(spec=RepositoryBlueprint)
        context = AgentContext(
            task=Task(
                task_id=2,
                task_type=TaskType.BUILD_COMPONENT_DAG,
                file="",
                description="Build DAG",
                metadata={},  # No component_plan
            ),
            blueprint=blueprint,
        )
        result = await agent.execute(context)
        assert result.success is False


# ── FrontendPipeline tests ────────────────────────────────────────────────────


class TestFrontendPipeline:
    @pytest.mark.anyio
    async def test_infer_framework_vue(self):
        from core.pipeline_frontend import FrontendPipeline

        req = ProductRequirements(
            title="App",
            description="Test",
            tech_preferences={"frontend": "Vue 3"},
        )
        assert FrontendPipeline._infer_framework(req) == "vue"

    @pytest.mark.anyio
    async def test_infer_framework_angular(self):
        from core.pipeline_frontend import FrontendPipeline

        req = ProductRequirements(
            title="App",
            description="Test",
            tech_preferences={"frontend": "Angular 17"},
        )
        assert FrontendPipeline._infer_framework(req) == "angular"

    @pytest.mark.anyio
    async def test_infer_framework_defaults_nextjs(self):
        from core.pipeline_frontend import FrontendPipeline

        req = ProductRequirements(title="App", description="Test")
        assert FrontendPipeline._infer_framework(req) == "nextjs"

    @pytest.mark.anyio
    async def test_make_frontend_blueprint(
        self, mock_llm, sample_requirements, sample_component_plan, tmp_path
    ):
        from core.pipeline_frontend import FrontendPipeline

        settings = MagicMock()
        settings.workspace_dir = tmp_path
        pipeline = FrontendPipeline(settings=settings, llm=mock_llm)
        bp = pipeline._make_frontend_blueprint(sample_requirements, sample_component_plan)

        assert bp.name == "Task Manager-frontend"
        assert bp.tech_stack["framework"] == "nextjs"
        assert len(bp.file_blueprints) == len(sample_component_plan.components)


# ── pipeline_definition tests ─────────────────────────────────────────────────


class TestFrontendPipelineDefinition:
    def test_frontend_pipeline_defined(self):
        from core.pipeline_definition import FRONTEND_PIPELINE

        assert FRONTEND_PIPELINE.name == "frontend"
        assert len(FRONTEND_PIPELINE.phases) == 1
        assert FRONTEND_PIPELINE.phases[0].name == "component_generation"
        assert TaskType.GENERATE_COMPONENT in [
            ft.task_type for ft in FRONTEND_PIPELINE.phases[0].file_tasks
        ]

    def test_frontend_pipeline_global_tasks(self):
        from core.pipeline_definition import FRONTEND_PIPELINE

        global_task_types = FRONTEND_PIPELINE.global_tasks
        assert TaskType.INTEGRATE_API in global_task_types
        assert TaskType.MANAGE_STATE in global_task_types

    def test_existing_pipelines_unchanged(self):
        from core.pipeline_definition import GENERATE_PIPELINE, ENHANCE_PIPELINE

        assert GENERATE_PIPELINE.name == "generate"
        assert ENHANCE_PIPELINE.name == "enhance"
        # Confirm existing phases and global tasks still intact
        assert any(
            ft.task_type == TaskType.GENERATE_FILE
            for phase in GENERATE_PIPELINE.phases
            for ft in phase.file_tasks
        )


# ── AgentManager TASK_AGENT_MAP tests ─────────────────────────────────────────


class TestAgentManagerTaskMap:
    def test_all_new_task_types_registered(self):
        from core.agent_manager import TASK_AGENT_MAP
        from agents.product_planner_agent import ProductPlannerAgent
        from agents.api_contract_agent import APIContractAgent
        from agents.design_parser_agent import DesignParserAgent
        from agents.component_planner_agent import ComponentPlannerAgent
        from agents.component_dag_agent import ComponentDAGAgent
        from agents.component_generator_agent import ComponentGeneratorAgent
        from agents.api_integration_agent import APIIntegrationAgent
        from agents.state_management_agent import StateManagementAgent

        assert TASK_AGENT_MAP[TaskType.PLAN_PRODUCT] is ProductPlannerAgent
        assert TASK_AGENT_MAP[TaskType.GENERATE_API_CONTRACT] is APIContractAgent
        assert TASK_AGENT_MAP[TaskType.PARSE_DESIGN] is DesignParserAgent
        assert TASK_AGENT_MAP[TaskType.PLAN_COMPONENTS] is ComponentPlannerAgent
        assert TASK_AGENT_MAP[TaskType.BUILD_COMPONENT_DAG] is ComponentDAGAgent
        assert TASK_AGENT_MAP[TaskType.GENERATE_COMPONENT] is ComponentGeneratorAgent
        assert TASK_AGENT_MAP[TaskType.INTEGRATE_API] is APIIntegrationAgent
        assert TASK_AGENT_MAP[TaskType.MANAGE_STATE] is StateManagementAgent

    def test_existing_task_types_still_registered(self):
        from core.agent_manager import TASK_AGENT_MAP
        from agents.coder_agent import CoderAgent
        from agents.reviewer_agent import ReviewerAgent

        assert TASK_AGENT_MAP[TaskType.GENERATE_FILE] is CoderAgent
        assert TASK_AGENT_MAP[TaskType.REVIEW_FILE] is ReviewerAgent


# ── FullstackPipeline helpers tests ──────────────────────────────────────────


class TestFullstackPipelineHelpers:
    def test_enrich_prompt_includes_features(self):
        from core.pipeline_fullstack import FullstackPipeline

        req = ProductRequirements(
            title="App",
            description="Desc",
            features=["Auth", "Dashboard"],
            tech_preferences={"frontend": "nextjs"},
        )
        enriched = FullstackPipeline._enrich_prompt("Build an app", req)
        assert "Auth" in enriched
        assert "Dashboard" in enriched
        assert "Build an app" in enriched

    def test_override_workspace(self, tmp_path):
        from core.pipeline_fullstack import _override_workspace

        settings = MagicMock()
        settings.workspace_dir = tmp_path / "original"
        new_ws = tmp_path / "new"
        cloned = _override_workspace(settings, new_ws)

        assert cloned.workspace_dir == new_ws
        # Original must be unmodified (shallow copy — different object)
        assert cloned is not settings


# ── ComponentGeneratorAgent tests ────────────────────────────────────────────


class TestComponentGeneratorAgent:
    """Tests for the previously-untested ComponentGeneratorAgent."""

    def _make_agent(self, mock_llm, mock_repo_manager):
        from agents.component_generator_agent import ComponentGeneratorAgent
        return ComponentGeneratorAgent(
            llm_client=mock_llm, repo_manager=mock_repo_manager
        )

    @pytest.mark.anyio
    async def test_execute_success(self, mock_llm, mock_repo_manager):
        from agents.component_generator_agent import ComponentGeneratorAgent
        from core.models import TaskResult

        agent = self._make_agent(mock_llm, mock_repo_manager)
        component = UIComponent(
            name="Button",
            file_path="src/components/ui/Button.tsx",
            component_type="ui",
            description="A reusable button",
        )
        context = AgentContext(
            task=Task(
                task_id=1,
                task_type=TaskType.GENERATE_COMPONENT,
                file=component.file_path,
                description="Generate Button",
                metadata={"component": component},
            ),
            blueprint=MagicMock(),
        )

        expected = TaskResult(success=True, output="wrote Button.tsx", files_modified=[component.file_path])
        agent.execute_agentic = AsyncMock(return_value=expected)

        result = await agent.execute(context)

        assert result.success is True
        agent.execute_agentic.assert_awaited_once()

    @pytest.mark.anyio
    async def test_execute_fails_without_component_metadata(
        self, mock_llm, mock_repo_manager
    ):
        agent = self._make_agent(mock_llm, mock_repo_manager)
        context = AgentContext(
            task=Task(
                task_id=2,
                task_type=TaskType.GENERATE_COMPONENT,
                file="src/Unknown.tsx",
                description="Generate unknown",
                metadata={},  # no "component" key
            ),
            blueprint=MagicMock(),
        )
        result = await agent.execute(context)

        assert result.success is False
        assert any("component" in e.lower() for e in result.errors)

    @pytest.mark.anyio
    async def test_execute_wraps_agentic_exception(
        self, mock_llm, mock_repo_manager
    ):
        agent = self._make_agent(mock_llm, mock_repo_manager)
        component = UIComponent(
            name="Broken",
            file_path="src/Broken.tsx",
            component_type="ui",
            description="Will crash",
        )
        context = AgentContext(
            task=Task(
                task_id=3,
                task_type=TaskType.GENERATE_COMPONENT,
                file=component.file_path,
                description="Generate Broken",
                metadata={"component": component},
            ),
            blueprint=MagicMock(),
        )
        agent.execute_agentic = AsyncMock(side_effect=RuntimeError("LLM timeout"))

        result = await agent.execute(context)

        assert result.success is False
        assert "LLM timeout" in result.errors[0]

    def test_build_prompt_includes_component_name_and_type(
        self, mock_llm, mock_repo_manager
    ):
        agent = self._make_agent(mock_llm, mock_repo_manager)
        component = UIComponent(
            name="TaskCard",
            file_path="src/components/feature/TaskCard.tsx",
            component_type="feature",
            description="Displays one task",
            props=["task: Task"],
            state_needs=["taskStore"],
            api_calls=["/api/v1/tasks"],
        )
        context = AgentContext(
            task=Task(
                task_id=4,
                task_type=TaskType.GENERATE_COMPONENT,
                file=component.file_path,
                description="Generate TaskCard",
                metadata={"component": component},
            ),
            blueprint=MagicMock(),
        )
        prompt = agent._build_prompt(context)

        assert "TaskCard" in prompt
        assert "feature" in prompt
        assert "taskStore" in prompt
        assert "/api/v1/tasks" in prompt

    def test_build_prompt_filters_relevant_api_endpoints(
        self, mock_llm, mock_repo_manager, sample_api_contract
    ):
        from core.models import ComponentPlan

        agent = self._make_agent(mock_llm, mock_repo_manager)
        component = UIComponent(
            name="TaskList",
            file_path="src/components/TaskList.tsx",
            component_type="feature",
            description="List tasks",
            api_calls=["/api/v1/tasks"],
        )
        plan = ComponentPlan(framework="nextjs", state_solution="zustand")
        context = AgentContext(
            task=Task(
                task_id=5,
                task_type=TaskType.GENERATE_COMPONENT,
                file=component.file_path,
                description="Generate TaskList",
                metadata={
                    "component": component,
                    "component_plan": plan,
                    "api_contract": sample_api_contract,
                },
            ),
            blueprint=MagicMock(),
        )
        prompt = agent._build_prompt(context)

        # /tasks endpoint should be included; /users/me is not relevant
        assert "/tasks" in prompt

    def test_build_prompt_without_component_returns_empty_body(
        self, mock_llm, mock_repo_manager
    ):
        agent = self._make_agent(mock_llm, mock_repo_manager)
        context = AgentContext(
            task=Task(
                task_id=6,
                task_type=TaskType.GENERATE_COMPONENT,
                file="",
                description="no component",
                metadata={},
            ),
            blueprint=MagicMock(),
        )
        prompt = agent._build_prompt(context)
        # Should not raise; content is empty / framework info only
        assert isinstance(prompt, str)


# ── APIIntegrationAgent tests ─────────────────────────────────────────────────


class TestAPIIntegrationAgent:
    """Tests for the previously-untested APIIntegrationAgent."""

    def _make_agent(self, mock_llm, mock_repo_manager):
        from agents.api_integration_agent import APIIntegrationAgent
        return APIIntegrationAgent(
            llm_client=mock_llm, repo_manager=mock_repo_manager
        )

    @pytest.mark.anyio
    async def test_execute_success(self, mock_llm, mock_repo_manager):
        from core.models import TaskResult

        agent = self._make_agent(mock_llm, mock_repo_manager)
        context = AgentContext(
            task=Task(
                task_id=1,
                task_type=TaskType.INTEGRATE_API,
                file="",
                description="Integrate API",
                metadata={},
            ),
            blueprint=MagicMock(),
        )
        expected = TaskResult(
            success=True,
            output="generated api.ts and hooks",
            files_modified=["src/lib/api.ts", "src/hooks/useTasks.ts"],
        )
        agent.execute_agentic = AsyncMock(return_value=expected)

        result = await agent.execute(context)

        assert result.success is True
        assert len(result.files_modified) == 2

    @pytest.mark.anyio
    async def test_execute_wraps_exception(self, mock_llm, mock_repo_manager):
        agent = self._make_agent(mock_llm, mock_repo_manager)
        context = AgentContext(
            task=Task(
                task_id=2,
                task_type=TaskType.INTEGRATE_API,
                file="",
                description="Integrate API",
                metadata={},
            ),
            blueprint=MagicMock(),
        )
        agent.execute_agentic = AsyncMock(side_effect=ValueError("bad contract"))

        result = await agent.execute(context)

        assert result.success is False
        assert "bad contract" in result.errors[0]

    def test_build_prompt_includes_all_endpoints(
        self, mock_llm, mock_repo_manager, sample_api_contract
    ):
        agent = self._make_agent(mock_llm, mock_repo_manager)
        context = AgentContext(
            task=Task(
                task_id=3,
                task_type=TaskType.INTEGRATE_API,
                file="",
                description="Integrate API",
                metadata={"api_contract": sample_api_contract},
            ),
            blueprint=MagicMock(),
        )
        prompt = agent._build_prompt(context)

        assert "/tasks" in prompt
        assert "/users/me" in prompt
        assert sample_api_contract.base_url in prompt

    def test_build_prompt_marks_auth_required(
        self, mock_llm, mock_repo_manager, sample_api_contract
    ):
        agent = self._make_agent(mock_llm, mock_repo_manager)
        context = AgentContext(
            task=Task(
                task_id=4,
                task_type=TaskType.INTEGRATE_API,
                file="",
                description="Integrate API",
                metadata={"api_contract": sample_api_contract},
            ),
            blueprint=MagicMock(),
        )
        prompt = agent._build_prompt(context)
        # All endpoints are auth_required — all should be marked
        assert "[auth]" in prompt

    def test_build_prompt_without_contract(self, mock_llm, mock_repo_manager):
        agent = self._make_agent(mock_llm, mock_repo_manager)
        context = AgentContext(
            task=Task(
                task_id=5,
                task_type=TaskType.INTEGRATE_API,
                file="",
                description="Integrate API",
                metadata={},
            ),
            blueprint=MagicMock(),
        )
        prompt = agent._build_prompt(context)
        # Should not raise; falls back to minimal prompt
        assert isinstance(prompt, str)
        assert "write_file" in prompt.lower() or "generate" in prompt.lower()

    def test_build_prompt_includes_framework_from_plan(
        self, mock_llm, mock_repo_manager, sample_api_contract, sample_component_plan
    ):
        agent = self._make_agent(mock_llm, mock_repo_manager)
        context = AgentContext(
            task=Task(
                task_id=6,
                task_type=TaskType.INTEGRATE_API,
                file="",
                description="Integrate API",
                metadata={
                    "api_contract": sample_api_contract,
                    "component_plan": sample_component_plan,
                },
            ),
            blueprint=MagicMock(),
        )
        prompt = agent._build_prompt(context)
        assert "nextjs" in prompt.lower()
        assert "zustand" in prompt.lower()


# ── StateManagementAgent tests ────────────────────────────────────────────────


class TestStateManagementAgent:
    """Tests for the previously-untested StateManagementAgent."""

    def _make_agent(self, mock_llm, mock_repo_manager):
        from agents.state_management_agent import StateManagementAgent
        return StateManagementAgent(
            llm_client=mock_llm, repo_manager=mock_repo_manager
        )

    @pytest.mark.anyio
    async def test_execute_success(self, mock_llm, mock_repo_manager):
        from core.models import TaskResult

        agent = self._make_agent(mock_llm, mock_repo_manager)
        context = AgentContext(
            task=Task(
                task_id=1,
                task_type=TaskType.MANAGE_STATE,
                file="",
                description="Manage state",
                metadata={},
            ),
            blueprint=MagicMock(),
        )
        expected = TaskResult(
            success=True,
            output="generated zustand stores",
            files_modified=["src/store/taskStore.ts"],
        )
        agent.execute_agentic = AsyncMock(return_value=expected)

        result = await agent.execute(context)

        assert result.success is True

    @pytest.mark.anyio
    async def test_execute_wraps_exception(self, mock_llm, mock_repo_manager):
        agent = self._make_agent(mock_llm, mock_repo_manager)
        context = AgentContext(
            task=Task(
                task_id=2,
                task_type=TaskType.MANAGE_STATE,
                file="",
                description="Manage state",
                metadata={},
            ),
            blueprint=MagicMock(),
        )
        agent.execute_agentic = AsyncMock(side_effect=RuntimeError("network error"))

        result = await agent.execute(context)

        assert result.success is False
        assert "network error" in result.errors[0]

    def test_build_prompt_collects_unique_state_slices(
        self, mock_llm, mock_repo_manager, sample_component_plan
    ):
        agent = self._make_agent(mock_llm, mock_repo_manager)
        context = AgentContext(
            task=Task(
                task_id=3,
                task_type=TaskType.MANAGE_STATE,
                file="",
                description="Manage state",
                metadata={"component_plan": sample_component_plan},
            ),
            blueprint=MagicMock(),
        )
        prompt = agent._build_prompt(context)

        # Both TaskCard and TaskList need "taskStore" — should appear once
        assert "taskStore" in prompt
        assert prompt.count("taskStore") == 1

    def test_build_prompt_without_plan_returns_fallback(
        self, mock_llm, mock_repo_manager
    ):
        agent = self._make_agent(mock_llm, mock_repo_manager)
        context = AgentContext(
            task=Task(
                task_id=4,
                task_type=TaskType.MANAGE_STATE,
                file="",
                description="Manage state",
                metadata={},
            ),
            blueprint=MagicMock(),
        )
        prompt = agent._build_prompt(context)

        assert isinstance(prompt, str)
        # The fallback message is non-empty
        assert len(prompt.strip()) > 0

    def test_build_prompt_includes_solution_and_framework(
        self, mock_llm, mock_repo_manager, sample_component_plan
    ):
        agent = self._make_agent(mock_llm, mock_repo_manager)
        context = AgentContext(
            task=Task(
                task_id=5,
                task_type=TaskType.MANAGE_STATE,
                file="",
                description="Manage state",
                metadata={"component_plan": sample_component_plan},
            ),
            blueprint=MagicMock(),
        )
        prompt = agent._build_prompt(context)

        assert "zustand" in prompt.lower()
        assert "nextjs" in prompt.lower()

    def test_build_prompt_deduplicates_slices_across_components(
        self, mock_llm, mock_repo_manager
    ):
        from core.models import ComponentPlan

        agent = self._make_agent(mock_llm, mock_repo_manager)
        plan = ComponentPlan(
            framework="nextjs",
            state_solution="redux",
            components=[
                UIComponent(
                    name=f"Comp{i}",
                    file_path=f"src/Comp{i}.tsx",
                    component_type="feature",
                    description="",
                    state_needs=["authStore", "uiStore"],
                )
                for i in range(5)
            ],
        )
        context = AgentContext(
            task=Task(
                task_id=6,
                task_type=TaskType.MANAGE_STATE,
                file="",
                description="Manage state",
                metadata={"component_plan": plan},
            ),
            blueprint=MagicMock(),
        )
        prompt = agent._build_prompt(context)

        # Each slice name should appear exactly once even with 5 components
        assert prompt.count("authStore") == 1
        assert prompt.count("uiStore") == 1


# ── TaskDispatcher tests ──────────────────────────────────────────────────────


class TestTaskDispatcher:
    """Tests for the newly-extracted TaskDispatcher class."""

    def _make_dispatcher(self):
        from core.task_dispatcher import TaskDispatcher

        am = MagicMock()
        am.settings.max_concurrent_agents = 4
        am._metrics = {
            "tasks_completed": 0, "tasks_failed": 0,
            "total_time": 0.0, "agent_metrics": {},
        }
        am._live = None
        am._event_bus = None
        am._embedding_store = None
        am._file_locks = MagicMock()
        am._file_locks.lock_for.return_value = None
        am.repo = MagicMock()
        am.blueprint = MagicMock()
        am.blueprint.file_blueprints = []
        am._dep_store = None
        return TaskDispatcher(am), am

    @pytest.mark.anyio
    async def test_execute_graph_runs_all_ready_tasks(self):
        from core.task_dispatcher import TaskDispatcher
        from core.models import TaskResult

        dispatcher, am = self._make_dispatcher()

        result_obj = TaskResult(success=True, output="ok")
        agent_mock = MagicMock()
        agent_mock.role = MagicMock(); agent_mock.role.value = "coder"
        agent_mock.get_metrics.return_value = {}
        agent_mock.execute = AsyncMock(return_value=result_obj)
        am._create_agent.return_value = agent_mock

        from core.task_engine import TaskGraph
        graph = TaskGraph()
        task = Task(
            task_id=10,
            task_type=TaskType.GENERATE_FILE,
            file="src/App.ts",
            description="gen",
        )
        graph.add_task(task)

        with patch("core.task_dispatcher.ContextBuilder") as MockCB:
            mock_ctx = MagicMock()
            MockCB.return_value.build.return_value = mock_ctx
            with patch("core.task_dispatcher.asyncio.to_thread", new=AsyncMock(return_value=mock_ctx)):
                result = await dispatcher.execute_graph(graph)

        assert result["stats"]["completed"] == 1
        assert result["stats"].get("failed", 0) == 0

    @pytest.mark.anyio
    async def test_execute_graph_detects_deadlock(self):
        from core.task_dispatcher import TaskDispatcher

        dispatcher, am = self._make_dispatcher()

        from core.task_engine import TaskGraph
        graph = TaskGraph()
        # Add two mutually-dependent tasks to provoke deadlock
        t1 = Task(task_id=1, task_type=TaskType.GENERATE_FILE, file="a.ts",
                  description="a", dependencies=[2])
        t2 = Task(task_id=2, task_type=TaskType.GENERATE_FILE, file="b.ts",
                  description="b", dependencies=[1])
        graph.add_task(t1)
        graph.add_task(t2)

        result = await dispatcher.execute_graph(graph)

        # One task is explicitly FAILED, the other is set to BLOCKED downstream.
        assert result["stats"].get("failed", 0) >= 1
        # Nothing completed
        assert result["stats"].get("completed", 0) == 0


# ── LifecycleOrchestrator tests ───────────────────────────────────────────────


class TestLifecycleOrchestrator:
    """Tests for the newly-extracted LifecycleOrchestrator class."""

    def _make_orchestrator(self):
        from core.lifecycle_orchestrator import LifecycleOrchestrator

        am = MagicMock()
        am.settings.max_concurrent_agents = 4
        am.settings.phase_timeout_seconds = 30
        am._metrics = {
            "tasks_completed": 0, "tasks_failed": 0,
            "total_time": 0.0, "agent_metrics": {},
        }
        am._live = None
        am._event_bus = None
        am._embedding_store = None
        am.repo = MagicMock()
        am.blueprint = MagicMock()
        am.blueprint.file_blueprints = []
        am._dep_store = None
        return LifecycleOrchestrator(am), am

    def test_build_lifecycle_metadata_review_trigger(self):
        from core.lifecycle_orchestrator import LifecycleOrchestrator

        lc = MagicMock()
        lc.fix_trigger = "review"
        lc.review_findings = ["unused var"]
        lc.review_output = "Review output text"
        meta = LifecycleOrchestrator._build_lifecycle_metadata(lc)

        assert meta["fix_trigger"] == "review"
        assert meta["review_errors"] == ["unused var"]
        assert meta["review_output"] == "Review output text"

    def test_build_lifecycle_metadata_test_trigger(self):
        from core.lifecycle_orchestrator import LifecycleOrchestrator

        lc = MagicMock()
        lc.fix_trigger = "test"
        lc.test_errors = "assertion failed"
        lc.test_fix_target = "test_foo.py"
        meta = LifecycleOrchestrator._build_lifecycle_metadata(lc)

        assert meta["fix_trigger"] == "test"
        assert meta["test_errors"] == "assertion failed"
        assert meta["test_fix_target"] == "test_foo.py"

    def test_build_lifecycle_metadata_build_trigger(self):
        from core.lifecycle_orchestrator import LifecycleOrchestrator

        lc = MagicMock()
        lc.fix_trigger = "build"
        lc.build_errors = "cannot find symbol"
        meta = LifecycleOrchestrator._build_lifecycle_metadata(lc)

        assert meta["fix_trigger"] == "build"
        assert meta["build_errors"] == "cannot find symbol"

    def test_extract_event_data_test_failure(self):
        from core.lifecycle_orchestrator import LifecycleOrchestrator
        from core.models import TaskResult

        result = TaskResult(success=False, errors=["assert 1 == 2"])
        data = LifecycleOrchestrator._extract_event_data(result, TaskType.GENERATE_TEST)

        assert "assert 1 == 2" in data["errors"]

    def test_extract_event_data_test_success_is_empty(self):
        from core.lifecycle_orchestrator import LifecycleOrchestrator
        from core.models import TaskResult

        result = TaskResult(success=True, output="tests pass")
        data = LifecycleOrchestrator._extract_event_data(result, TaskType.GENERATE_TEST)

        assert data == {}

    def test_extract_event_data_build_failure(self):
        from core.lifecycle_orchestrator import LifecycleOrchestrator
        from core.models import TaskResult

        result = TaskResult(success=False, errors=["undefined reference"])
        data = LifecycleOrchestrator._extract_event_data(result, TaskType.VERIFY_BUILD)

        assert "undefined reference" in data["errors"]

    def test_agent_manager_static_methods_forward_to_orchestrator(self):
        """AgentManager._build_lifecycle_metadata / _extract_event_data must
        delegate to LifecycleOrchestrator (backward-compat forwarding aliases)."""
        from core.agent_manager import AgentManager
        from core.lifecycle_orchestrator import LifecycleOrchestrator
        from core.models import TaskResult

        lc = MagicMock()
        lc.fix_trigger = "build"
        lc.build_errors = "error: missing semicolon"

        am_result = AgentManager._build_lifecycle_metadata(lc)
        orch_result = LifecycleOrchestrator._build_lifecycle_metadata(lc)
        assert am_result == orch_result

        tr = TaskResult(success=False, errors=["build error"])
        am_ev = AgentManager._extract_event_data(tr, TaskType.VERIFY_BUILD)
        orch_ev = LifecycleOrchestrator._extract_event_data(tr, TaskType.VERIFY_BUILD)
        assert am_ev == orch_ev


# ── Figma API integration tests ───────────────────────────────────────────────

class TestDesignParserFigmaIntegration:
    """Unit tests for the real Figma REST API integration path."""

    def test_extract_figma_key_file_url(self):
        from agents.design_parser_agent import DesignParserAgent
        key = DesignParserAgent._extract_figma_key(
            "https://www.figma.com/file/AbC123XyZ/My-Design"
        )
        assert key == "AbC123XyZ"

    def test_extract_figma_key_design_url(self):
        from agents.design_parser_agent import DesignParserAgent
        key = DesignParserAgent._extract_figma_key(
            "https://www.figma.com/design/DefKey456/Dashboard?node-id=0"
        )
        assert key == "DefKey456"

    def test_extract_figma_key_no_match_returns_none(self):
        from agents.design_parser_agent import DesignParserAgent
        assert DesignParserAgent._extract_figma_key("https://example.com/notfigma") is None

    def test_summarise_figma_nodes_extracts_pages_and_styles(self):
        from agents.design_parser_agent import DesignParserAgent

        figma_data = {
            "document": {
                "children": [
                    {
                        "type": "CANVAS",
                        "name": "Home",
                        "children": [
                            {"type": "FRAME", "name": "HeroSection"},
                            {"type": "COMPONENT", "name": "Button"},
                        ],
                    },
                    {"type": "CANVAS", "name": "Dashboard", "children": []},
                ]
            },
            "styles": {
                "s1": {"name": "Brand/Blue", "styleType": "FILL"},
                "s2": {"name": "Heading/XL", "styleType": "TEXT"},
            },
        }

        summary = DesignParserAgent._summarise_figma_nodes(figma_data)
        assert "Home" in summary
        assert "Dashboard" in summary
        assert "FRAME: HeroSection" in summary or "HeroSection" in summary
        assert "Brand/Blue" in summary
        assert "Heading/XL" in summary

    def test_summarise_figma_nodes_empty_document(self):
        from agents.design_parser_agent import DesignParserAgent
        summary = DesignParserAgent._summarise_figma_nodes({})
        assert "(no structured nodes found)" in summary

    @pytest.mark.anyio
    async def test_parse_design_uses_figma_api_when_token_set(self, monkeypatch):
        """When FIGMA_TOKEN is set and URL is valid, _fetch_figma_nodes is called."""
        from unittest.mock import AsyncMock, patch
        from agents.design_parser_agent import DesignParserAgent

        monkeypatch.setenv("FIGMA_TOKEN", "test-token-abc")
        figma_payload = {
            "document": {
                "children": [{"type": "CANVAS", "name": "HomePage", "children": []}]
            },
            "styles": {},
        }
        llm = MagicMock()
        llm.generate_json = AsyncMock(return_value={
            "framework": "nextjs",
            "design_description": "dashboard",
            "figma_url": "",
            "pages": ["HomePage"],
            "global_styles": {},
            "design_tokens": {},
        })
        llm.generate = AsyncMock()
        agent = DesignParserAgent(llm_client=llm, repo_manager=MagicMock())

        with patch.object(agent, "_fetch_figma_nodes", new=AsyncMock(return_value=figma_payload)) as mock_fetch:
            spec = await agent.parse_design(None, figma_url="https://figma.com/file/AbC123/Design")

        mock_fetch.assert_called_once_with("AbC123", "test-token-abc")
        assert "HomePage" in spec.pages

    @pytest.mark.anyio
    async def test_parse_design_falls_back_when_figma_api_raises(self, monkeypatch):
        """If the Figma API raises, parse_design falls back to URL-as-text context."""
        from unittest.mock import AsyncMock, patch
        import httpx
        from agents.design_parser_agent import DesignParserAgent

        monkeypatch.setenv("FIGMA_TOKEN", "bad-token")
        llm = MagicMock()
        llm.generate_json = AsyncMock(return_value={
            "framework": "react",
            "design_description": "app",
            "figma_url": "",
            "pages": ["Home"],
            "global_styles": {},
            "design_tokens": {},
        })
        llm.generate = AsyncMock()
        agent = DesignParserAgent(llm_client=llm, repo_manager=MagicMock())

        with patch.object(
            agent, "_fetch_figma_nodes",
            new=AsyncMock(side_effect=httpx.HTTPStatusError("403", request=MagicMock(), response=MagicMock()))
        ):
            # Should NOT raise — fallback to URL text
            spec = await agent.parse_design(None, figma_url="https://figma.com/file/AbC/D")

        assert spec.framework == "react"

    @pytest.mark.anyio
    async def test_parse_design_skips_api_when_no_token(self, monkeypatch):
        """When FIGMA_TOKEN is absent, _fetch_figma_nodes is never called."""
        from unittest.mock import AsyncMock, patch
        from agents.design_parser_agent import DesignParserAgent

        monkeypatch.delenv("FIGMA_TOKEN", raising=False)
        llm = MagicMock()
        llm.generate_json = AsyncMock(return_value={
            "framework": "nextjs",
            "design_description": "x",
            "figma_url": "",
            "pages": ["Home"],
            "global_styles": {},
            "design_tokens": {},
        })
        llm.generate = AsyncMock()
        agent = DesignParserAgent(llm_client=llm, repo_manager=MagicMock())

        with patch.object(agent, "_fetch_figma_nodes", new=AsyncMock()) as mock_fetch:
            await agent.parse_design(None, figma_url="https://figma.com/file/AbC/D")

        mock_fetch.assert_not_called()


# ── TSX compiler tests ────────────────────────────────────────────────────────

class TestTSXCompiler:
    """Unit tests for core.tsx_compiler.TSXCompiler."""

    def test_parse_output_extracts_errors(self):
        from core.tsx_compiler import TSXCompiler
        from pathlib import Path

        workspace = Path("/workspace")
        raw = (
            "/workspace/src/components/Button.tsx(10,5): error TS2304: "
            "Cannot find name 'x'.\n"
            "/workspace/src/pages/index.tsx(22,3): error TS2322: "
            "Type 'string' is not assignable to type 'number'.\n"
        )
        compiler = TSXCompiler()
        errors = compiler._parse_output(raw, workspace)

        assert len(errors) == 2
        assert errors[0].file == "src/components/Button.tsx"
        assert errors[0].line == 10
        assert errors[0].col == 5
        assert errors[0].code == "TS2304"
        assert "Cannot find name" in errors[0].message
        assert errors[1].file == "src/pages/index.tsx"
        assert errors[1].code == "TS2322"

    def test_parse_output_ignores_non_error_lines(self):
        from core.tsx_compiler import TSXCompiler
        from pathlib import Path

        workspace = Path("/workspace")
        raw = "Found 2 errors.\n\nsome/other/line that is not a tsc error\n"
        compiler = TSXCompiler()
        errors = compiler._parse_output(raw, workspace)
        assert errors == []

    def test_tsx_compile_result_passed_when_no_errors(self):
        from core.tsx_compiler import TSXCompileResult
        result = TSXCompileResult(errors=[], tsc_available=True)
        assert result.passed is True

    def test_tsx_compile_result_not_passed_when_errors(self):
        from core.tsx_compiler import TSXCompileResult, TSXError
        err = TSXError(file="src/a.tsx", line=1, col=1, code="TS2304", message="x")
        result = TSXCompileResult(errors=[err], tsc_available=True)
        assert result.passed is False

    def test_tsx_compile_result_not_passed_when_tsc_unavailable(self):
        from core.tsx_compiler import TSXCompileResult
        result = TSXCompileResult(tsc_available=False)
        assert result.passed is False

    def test_errors_by_file_groups_correctly(self):
        from core.tsx_compiler import TSXCompileResult, TSXError
        e1 = TSXError(file="src/a.tsx", line=1, col=1, code="TS1", message="m1")
        e2 = TSXError(file="src/b.tsx", line=2, col=2, code="TS2", message="m2")
        e3 = TSXError(file="src/a.tsx", line=3, col=3, code="TS3", message="m3")
        result = TSXCompileResult(errors=[e1, e2, e3])
        grouped = result.errors_by_file()
        assert len(grouped["src/a.tsx"]) == 2
        assert len(grouped["src/b.tsx"]) == 1

    @pytest.mark.anyio
    async def test_check_returns_unavailable_when_tsc_not_found(self, tmp_path):
        """When tsc is not on PATH, check() returns tsc_available=False gracefully."""
        import asyncio
        from unittest.mock import patch, AsyncMock
        from core.tsx_compiler import TSXCompiler

        compiler = TSXCompiler()
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("tsc not found"),
        ):
            result = await compiler.check(tmp_path)

        assert result.tsc_available is False
        assert result.errors == []

    @pytest.mark.anyio
    async def test_check_creates_tsconfig_when_missing(self, tmp_path):
        """check() writes a default tsconfig.json if none exists."""
        import asyncio
        from unittest.mock import patch, MagicMock, AsyncMock
        from core.tsx_compiler import TSXCompiler

        compiler = TSXCompiler()
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", new=AsyncMock(return_value=(b"", b""))):
                await compiler.check(tmp_path)

        assert (tmp_path / "tsconfig.json").exists()


# ── TypeScript import validation tests ────────────────────────────────────────

class TestImportValidatorTypeScript:
    """Tests for ImportValidator._validate_typescript."""

    def _make_known_files(self, *paths: str) -> set[str]:
        return set(paths)

    def test_valid_relative_import_with_ts_extension(self):
        from core.import_validator import ImportValidator
        validator = ImportValidator()
        content = "import Button from './Button';\n"
        known = self._make_known_files("src/components/Button.tsx")
        broken = validator._validate_typescript(content, "src/components/Card.tsx", known)
        assert broken == []

    def test_broken_relative_import_reported(self):
        from core.import_validator import ImportValidator
        validator = ImportValidator()
        content = "import Missing from './Missing';\n"
        known = self._make_known_files("src/components/Button.tsx")
        broken = validator._validate_typescript(content, "src/components/Card.tsx", known)
        assert "./Missing" in broken

    def test_parent_directory_import_resolved(self):
        from core.import_validator import ImportValidator
        validator = ImportValidator()
        content = "import utils from '../lib/utils';\n"
        known = self._make_known_files("src/lib/utils.ts")
        broken = validator._validate_typescript(content, "src/components/Card.tsx", known)
        assert broken == []

    def test_third_party_import_skipped(self):
        from core.import_validator import ImportValidator
        validator = ImportValidator()
        content = "import React from 'react';\nimport { classnames } from 'clsx';\n"
        broken = validator._validate_typescript(content, "src/App.tsx", set())
        assert broken == []

    def test_alias_import_skipped(self):
        """@-prefixed path aliases (Next.js / tsconfig paths) are not checked."""
        from core.import_validator import ImportValidator
        validator = ImportValidator()
        content = "import Foo from '@/components/Foo';\n"
        broken = validator._validate_typescript(content, "src/pages/index.tsx", set())
        assert broken == []

    def test_index_file_resolution(self):
        """Import of a directory with an index.ts is valid."""
        from core.import_validator import ImportValidator
        validator = ImportValidator()
        content = "import { theme } from './theme';\n"
        known = self._make_known_files("src/styles/theme/index.ts")
        broken = validator._validate_typescript(content, "src/styles/App.tsx", known)
        assert broken == []

    def test_type_import_checked_same_as_value_import(self):
        from core.import_validator import ImportValidator
        validator = ImportValidator()
        content = "import type { Foo } from './types';\n"
        known: set[str] = set()
        broken = validator._validate_typescript(content, "src/components/Card.tsx", known)
        assert "./types" in broken

    def test_validate_dispatches_to_typescript_for_ts_lang(self):
        """ImportValidator.validate() routes typescript LanguageProfile correctly."""
        from core.import_validator import ImportValidator
        from core.language import TYPESCRIPT
        validator = ImportValidator()
        content = "import Nope from './Nope';\n"
        # known_files must be non-empty (guard) but must not contain the imported path
        known: set[str] = {"src/Other.tsx"}
        broken = validator.validate("src/a.tsx", content, known, TYPESCRIPT)
        assert "./Nope" in broken


# ── Pipeline cross-component import validation tests ─────────────────────────

class TestFrontendPipelineImportValidation:
    """Tests for FrontendPipeline._validate_component_imports."""

    def _make_component(self, file_path: str):
        comp = MagicMock()
        comp.file_path = file_path
        return comp

    def test_no_errors_when_all_imports_resolve(self, tmp_path):
        from core.pipeline_frontend import FrontendPipeline

        # Write two component files that import each other using relative paths
        (tmp_path / "src").mkdir(parents=True)
        button = tmp_path / "src" / "Button.tsx"
        button.write_text("export const Button = () => <button />;", encoding="utf-8")
        card = tmp_path / "src" / "Card.tsx"
        card.write_text("import Button from './Button';\nexport const Card = () => <div />;", encoding="utf-8")

        comps = [
            self._make_component("src/Card.tsx"),
        ]
        errors = FrontendPipeline._validate_component_imports(tmp_path, comps)
        assert errors == []

    def test_broken_import_reported(self, tmp_path):
        from core.pipeline_frontend import FrontendPipeline

        (tmp_path / "src").mkdir(parents=True)
        card = tmp_path / "src" / "Card.tsx"
        card.write_text("import Ghost from './Ghost';\nexport const Card = () => <div />;", encoding="utf-8")

        comps = [self._make_component("src/Card.tsx")]
        errors = FrontendPipeline._validate_component_imports(tmp_path, comps)
        assert len(errors) == 1
        assert "Ghost" in errors[0]
        assert "src/Card.tsx" in errors[0]

    def test_missing_file_is_skipped_gracefully(self, tmp_path):
        """A component whose file was never written is silently skipped."""
        from core.pipeline_frontend import FrontendPipeline

        comps = [self._make_component("src/Nonexistent.tsx")]
        errors = FrontendPipeline._validate_component_imports(tmp_path, comps)
        assert errors == []

    def test_third_party_imports_not_flagged(self, tmp_path):
        from core.pipeline_frontend import FrontendPipeline

        (tmp_path / "src").mkdir(parents=True)
        f = tmp_path / "src" / "App.tsx"
        f.write_text("import React from 'react';\nimport { useState } from 'react';\n", encoding="utf-8")

        comps = [self._make_component("src/App.tsx")]
        errors = FrontendPipeline._validate_component_imports(tmp_path, comps)
        assert errors == []

