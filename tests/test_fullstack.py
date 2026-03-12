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
