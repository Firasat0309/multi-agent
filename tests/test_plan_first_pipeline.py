"""Tests for the plan-first pipeline architecture.

Covers:
- PlanGeneratorAgent: generate_plan, revise_plan, phase validation
- ProductPlannerAgent.parse_from_plan: extraction from plan.md
- ArchitectAgent.design_from_plan: blueprint from PHASE 2
- APIContractAgent.extract_from_plan: contract from PHASE 1
- ArchitectureApprover.approve_plan_md: interactive and non-interactive modes
- FullstackPipeline: plan-first flow integration tests
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from core.models import (
    AgentContext,
    AgentRole,
    APIContract,
    APIEndpoint,
    FileBlueprint,
    ProductRequirements,
    RepositoryBlueprint,
    Task,
    TaskType,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    response = MagicMock()
    response.content = "mock content"
    response.stop_reason = "end_turn"
    response.usage = {"input_tokens": 100, "output_tokens": 200}
    llm.generate = AsyncMock(return_value=response)
    llm.generate_json = AsyncMock(return_value={})
    return llm


@pytest.fixture
def mock_repo_manager(tmp_path):
    rm = MagicMock()
    rm.workspace = tmp_path
    rm.write_file = MagicMock()
    return rm


@pytest.fixture
def sample_plan_md():
    return """\
# Task Manager — Implementation Plan
**Tech Stack:** React 18 + TypeScript | Spring Boot 3.x | H2 | Maven

---

# PHASE 0 — REQUIREMENTS ANALYSIS

## Screens / Routes
| Screen | Route | Auth Required |
| ------ | ----- | ------------- |
| Login | /login | No |
| Dashboard | / | Yes |
| Task List | /tasks | Yes |

## Components per Screen
### Login Page
- Email input (placeholder: "Enter email")
- Password input (placeholder: "Enter password")
- Login button ("Sign In")

### Dashboard
- Summary cards showing task counts
- Navigation sidebar with links to Tasks

### Task List
- Table with columns: Title, Status, Created At
- "Add Task" button
- Delete button per row

## Data Entities
| Entity | Fields (name: type) | Notes |
| ------ | ------------------- | ----- |
| User | id: Long, email: String, password: String | Auth entity |
| Task | id: Long, title: String, status: String, createdAt: LocalDateTime | Core entity |

## API Operations
| Method | Path | Trigger | Auth |
| ------ | ---- | ------- | ---- |
| POST | /api/auth/login | Login button | No |
| GET | /api/tasks | Task list load | Yes |
| POST | /api/tasks | Add Task button | Yes |
| DELETE | /api/tasks/{id} | Delete button | Yes |

## Navigation Map
Sidebar links: Dashboard (/) and Tasks (/tasks).
After login, redirect to Dashboard.

## Auth Model
JWT authentication required. Token stored in localStorage.
JwtAuthFilter validates Bearer token on protected routes.

---

# PHASE 1 — API CONTRACT

POST /api/auth/login
Request body: { email: string, password: string }
Response body: { token: string, userId: long }
HTTP codes: 200 | 400 | 401

GET /api/tasks
Request body: {}
Response body: [{ id: long, title: string, status: string, createdAt: string }]
HTTP codes: 200 | 401

POST /api/tasks
Request body: { title: string, status: string }
Response body: { id: long, title: string, status: string, createdAt: string }
HTTP codes: 201 | 400 | 401

DELETE /api/tasks/{id}
Request body: {}
Response body: {}
HTTP codes: 204 | 401 | 404

---

# PHASE 2 — BACKEND PLAN

## Technology Stack
Java 17, Spring Boot 3.x, H2 in-memory, Maven, Spring Security + JWT.

## File Tree
```
backend/
  src/main/java/com/app/
    config/
      SecurityConfig.java       — CORS + Spring Security filter chain
      CorsConfig.java           — Global CORS bean (allows localhost:5173)
      JwtConfig.java            — JWT settings (secret, expiration)
    controller/
      AuthController.java       — POST /api/auth/login, /api/auth/register
      TaskController.java       — CRUD endpoints for Task entity
    service/
      AuthService.java          — Login and token generation logic
      TaskService.java          — Task business logic
    repository/
      TaskRepository.java       — JpaRepository<Task, Long>
      UserRepository.java       — JpaRepository<User, Long>
    model/
      Task.java                 — Task JPA entity
      User.java                 — User JPA entity
    dto/
      AuthRequestDto.java       — Login request DTO
      AuthResponseDto.java      — Login response DTO with token
      TaskRequestDto.java       — Task create/update DTO
      TaskResponseDto.java      — Task response DTO
    security/
      JwtTokenProvider.java     — JWT generation and validation
      JwtAuthFilter.java        — OncePerRequestFilter reads Bearer token
    exception/
      GlobalExceptionHandler.java  — Maps exceptions to HTTP responses
      ResourceNotFoundException.java — 404 base exception
  src/main/resources/
    application.properties      — Server port, H2 datasource, JWT config
  pom.xml                       — Maven build with Spring Boot parent
```

## Layer Responsibilities
- model: JPA entities (@Entity, @Id, no-arg constructor)
- repository: Spring Data interfaces (JpaRepository)
- service: Business logic, calls repositories, throws domain exceptions
- controller: REST endpoints, call services, return ResponseEntity<DTO>
- config: Security filter chain, CORS bean, JWT settings
- dto: Request/response POJOs, never expose entity directly
- security: JWT generation, validation, and filter
- exception: GlobalExceptionHandler with @RestControllerAdvice

## Public vs Protected Endpoints
PUBLIC: POST /api/auth/login
PROTECTED (JWT required): GET /api/tasks, POST /api/tasks, DELETE /api/tasks/{id}

## CORS Configuration
CorsConfig.java allows origins http://localhost:5173 and http://localhost:3000,
methods GET/POST/PUT/DELETE/PATCH/OPTIONS, all headers, exposes Authorization.
SecurityConfig wires the CorsConfig bean before any other filter.

---

# PHASE 3 — FRONTEND PLAN

## Technology Stack
React 18, TypeScript strict, Vite, React Router v6, Axios, Tailwind CSS,
React Hook Form + Zod.

## File Tree
```
frontend/
  src/
    api/
      client.ts             — Axios instance, JWT interceptor, 401 redirect
      auth.api.ts           — login() function
      tasks.api.ts          — getTasks(), createTask(), deleteTask() functions
    components/
      common/
        Button.tsx          — Reusable button with loading state
        InputField.tsx      — Labeled input with error display
        Table.tsx           — Generic table component
      layout/
        Sidebar.tsx         — Navigation sidebar with route links
        Layout.tsx          — Shell wrapping authenticated pages
    pages/
      LoginPage.tsx         — Login form page
      DashboardPage.tsx     — Summary cards dashboard
      TaskListPage.tsx      — Task table with add/delete actions
    hooks/
      useTasks.ts           — Data fetching hook for tasks
    context/
      AuthContext.tsx       — JWT token state, login/logout actions
    types/
      auth.types.ts         — AuthRequest, AuthResponse interfaces
      task.types.ts         — Task interface matching BE TaskResponseDto
    routes/
      AppRoutes.tsx         — React Router config, ProtectedRoute wrapper
    utils/
      validation.ts         — Zod schemas for forms
  .env                      — VITE_API_BASE_URL=http://localhost:8080
  .env.example              — Template for .env
  vite.config.ts            — Vite config with /api proxy to :8080
  tailwind.config.js        — Tailwind setup
  tsconfig.json             — TypeScript strict config
```

## Component Responsibilities
- LoginPage: email + password form, calls auth.api.ts login(), stores JWT
- DashboardPage: shows task count summary cards, uses useTasks hook
- TaskListPage: renders Task[] in Table, handles create (modal) and delete
- Sidebar: links to / and /tasks, highlights active route

## Routing Configuration
/ → DashboardPage (protected)
/login → LoginPage (public)
/tasks → TaskListPage (protected)
ProtectedRoute: redirects to /login if no JWT in localStorage.

## API Client Setup
client.ts reads VITE_API_BASE_URL from env (default http://localhost:8080).
Request interceptor attaches Authorization: Bearer <token> from localStorage.
Response interceptor on 401: removes token, redirects to /login.

---

# PHASE 4 — VALIDATION CHECKLIST

## Backend
- [ ] All @Entity classes have @Id and no-arg constructor
- [ ] All repositories extend JpaRepository<Entity, Long>
- [ ] CORS bean registered and wired in SecurityConfig
- [ ] Public endpoints explicitly .permitAll() in SecurityConfig
- [ ] GlobalExceptionHandler covers 400/401/403/404/500

## Frontend
- [ ] client.ts base URL reads from VITE_API_BASE_URL
- [ ] Every API function path exactly matches BE controller path
- [ ] Every form field name exactly matches BE DTO field name
- [ ] All TypeScript interfaces match BE DTO shapes
- [ ] No implicit any types

## Integration
- [ ] CORS headers present on BE responses
- [ ] JWT token flow: FE sends Authorization: Bearer <token>
- [ ] All API paths identical between FE and BE

---

# PHASE 5 — AUTO-FIX RULES

If any validation check fails:
1. Identify the exact violation (file name, line number, check that failed)
2. Fix it in the affected file
3. Re-run the relevant validation checks for that file
4. Repeat until all checks pass

Do NOT output code with known errors.

---

# PHASE 6 — OUTPUT FORMAT

Output files in this exact format, one block per file:

=== FILE: backend/src/main/java/com/app/config/SecurityConfig.java ===
<full file content>

=== FILE: frontend/src/api/client.ts ===
<full file content>

Output ALL files. Never truncate. Never output a stub or TODO.

---

# PHASE 7 — DOCUMENTATION

=== FILE: README.md ===
1. Prerequisites: Java 17, Node 18+, Maven
2. Running the Backend: cd backend && mvn spring-boot:run (port 8080)
3. Running the Frontend: cd frontend && npm install && npm run dev (port 5173)
4. API Reference: POST /api/auth/login (no auth), GET /api/tasks (JWT required)
5. Environment Variables: VITE_API_BASE_URL
6. CORS Notes: CorsConfig allows localhost:5173 and localhost:3000
"""


@pytest.fixture
def sample_requirements():
    return ProductRequirements(
        title="Task Manager",
        description="A task management application with JWT auth.",
        user_stories=["As a user, I want to create tasks"],
        features=["Task CRUD", "User auth", "Dashboard"],
        tech_preferences={
            "frontend": "react",
            "backend": "spring boot",
            "db": "h2",
            "styling": "tailwind",
        },
        has_frontend=True,
        has_backend=True,
    )


@pytest.fixture
def sample_blueprint():
    return RepositoryBlueprint(
        name="task-manager",
        description="Task Manager Spring Boot backend",
        architecture_style="REST",
        tech_stack={"language": "java", "framework": "spring boot", "db": "h2", "build_tool": "maven"},
        file_blueprints=[
            FileBlueprint(
                path="backend/src/main/java/com/app/controller/TaskController.java",
                purpose="CRUD endpoints for Task entity",
                depends_on=["backend/src/main/java/com/app/service/TaskService.java"],
                exports=["TaskController"],
                language="java",
                layer="controller",
            ),
            FileBlueprint(
                path="backend/src/main/java/com/app/model/Task.java",
                purpose="Task JPA entity",
                depends_on=[],
                exports=["Task"],
                language="java",
                layer="model",
            ),
            FileBlueprint(
                path="backend/pom.xml",
                purpose="Maven build configuration",
                depends_on=[],
                exports=[],
                language="java",
                layer="infrastructure",
            ),
        ],
    )


# ── PlanGeneratorAgent tests ───────────────────────────────────────────────────


class TestPlanGeneratorAgent:
    @pytest.mark.anyio
    async def test_generate_plan_returns_string(self, mock_llm, mock_repo_manager, sample_plan_md):
        from agents.plan_generator_agent import PlanGeneratorAgent

        mock_llm.generate.return_value.content = sample_plan_md

        agent = PlanGeneratorAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        result = await agent.generate_plan("Build a task manager with login and task CRUD")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "PHASE 0" in result

    @pytest.mark.anyio
    async def test_generate_plan_writes_to_workspace(self, mock_llm, mock_repo_manager, sample_plan_md):
        from agents.plan_generator_agent import PlanGeneratorAgent

        mock_llm.generate.return_value.content = sample_plan_md

        agent = PlanGeneratorAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        await agent.generate_plan("Build a task manager")

        mock_repo_manager.write_file.assert_called_once_with("plan.md", sample_plan_md)

    @pytest.mark.anyio
    async def test_generate_plan_has_all_phases(self, mock_llm, mock_repo_manager, sample_plan_md):
        from agents.plan_generator_agent import PlanGeneratorAgent

        mock_llm.generate.return_value.content = sample_plan_md

        agent = PlanGeneratorAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        result = await agent.generate_plan("Build a task manager")

        for phase in ["PHASE 0", "PHASE 1", "PHASE 2", "PHASE 3", "PHASE 4", "PHASE 5", "PHASE 6", "PHASE 7"]:
            assert phase in result, f"Expected {phase} in plan.md output"

    @pytest.mark.anyio
    async def test_revise_plan_incorporates_feedback(self, mock_llm, mock_repo_manager, sample_plan_md):
        from agents.plan_generator_agent import PlanGeneratorAgent

        revised = sample_plan_md.replace("H2", "PostgreSQL")
        mock_llm.generate.return_value.content = revised

        agent = PlanGeneratorAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        result = await agent.revise_plan(
            user_prompt="Build a task manager",
            current_plan=sample_plan_md,
            feedback="Use PostgreSQL instead of H2",
        )

        assert "PostgreSQL" in result
        # The LLM was called with revision prompt — check any kwarg/arg contains "revis"
        call_args = mock_llm.generate.call_args
        all_text = " ".join(
            str(v).lower() for v in list(call_args.args) + list(call_args.kwargs.values())
        )
        assert "revis" in all_text

    @pytest.mark.anyio
    async def test_revise_plan_writes_to_workspace(self, mock_llm, mock_repo_manager, sample_plan_md):
        from agents.plan_generator_agent import PlanGeneratorAgent

        revised = sample_plan_md + "\n\n(revised)"
        mock_llm.generate.return_value.content = revised

        agent = PlanGeneratorAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        await agent.revise_plan("prompt", sample_plan_md, "feedback")

        mock_repo_manager.write_file.assert_called_with("plan.md", revised)

    @pytest.mark.anyio
    async def test_generate_plan_retries_on_incomplete_output(self, mock_llm, mock_repo_manager, sample_plan_md):
        """If the first response is missing phases, generate_plan should retry."""
        from agents.plan_generator_agent import PlanGeneratorAgent

        # First response is incomplete (missing PHASE 4+)
        incomplete = "\n".join(
            line for line in sample_plan_md.splitlines()
            if "PHASE 4" not in line and "PHASE 5" not in line
            and "PHASE 6" not in line and "PHASE 7" not in line
        )
        # Strip the phase bodies too — simulate LLM cutting off early
        incomplete = incomplete.split("# PHASE 4")[0] if "# PHASE 4" in incomplete else incomplete

        full_response = MagicMock()
        full_response.content = sample_plan_md
        full_response.stop_reason = "end_turn"
        full_response.usage = {"input_tokens": 100, "output_tokens": 200}

        incomplete_response = MagicMock()
        incomplete_response.content = incomplete
        incomplete_response.stop_reason = "end_turn"
        incomplete_response.usage = {"input_tokens": 100, "output_tokens": 200}

        mock_llm.generate = AsyncMock(side_effect=[incomplete_response, full_response])

        agent = PlanGeneratorAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        result = await agent.generate_plan("Build a task manager")

        # Should have made 2 LLM calls (initial + retry)
        assert mock_llm.generate.call_count == 2
        # Final result should be the full plan
        assert result == sample_plan_md

    @pytest.mark.anyio
    async def test_generate_plan_role(self, mock_llm, mock_repo_manager):
        from agents.plan_generator_agent import PlanGeneratorAgent

        agent = PlanGeneratorAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        assert agent.role == AgentRole.PLAN_GENERATOR

    @pytest.mark.anyio
    async def test_generate_plan_system_prompt_has_all_phase_headers(self, mock_llm, mock_repo_manager):
        from agents.plan_generator_agent import PlanGeneratorAgent

        agent = PlanGeneratorAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        sp = agent.system_prompt

        for phase in ["PHASE 0", "PHASE 1", "PHASE 2", "PHASE 3", "PHASE 4", "PHASE 5", "PHASE 6", "PHASE 7"]:
            assert phase in sp, f"System prompt missing {phase}"


# ── ProductPlannerAgent.parse_from_plan tests ─────────────────────────────────


class TestProductPlannerParseFromPlan:
    @pytest.mark.anyio
    async def test_parse_from_plan_extracts_title(self, mock_llm, mock_repo_manager, sample_plan_md):
        from agents.product_planner_agent import ProductPlannerAgent

        mock_llm.generate_json.return_value = {
            "title": "Task Manager",
            "description": "A task management application.",
            "user_stories": ["As a user, I want tasks"],
            "features": ["Task CRUD", "Auth"],
            "tech_preferences": {"frontend": "react", "backend": "spring boot", "db": "h2"},
            "has_frontend": True,
            "has_backend": True,
        }

        agent = ProductPlannerAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        req = await agent.parse_from_plan(sample_plan_md)

        assert req.title == "Task Manager"

    @pytest.mark.anyio
    async def test_parse_from_plan_extracts_tech_preferences(self, mock_llm, mock_repo_manager, sample_plan_md):
        from agents.product_planner_agent import ProductPlannerAgent

        mock_llm.generate_json.return_value = {
            "title": "Task Manager",
            "description": "An app.",
            "user_stories": [],
            "features": [],
            "tech_preferences": {
                "frontend": "react",
                "backend": "spring boot",
                "db": "h2",
                "styling": "tailwind",
            },
            "has_frontend": True,
            "has_backend": True,
        }

        agent = ProductPlannerAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        req = await agent.parse_from_plan(sample_plan_md)

        assert req.tech_preferences.get("frontend") == "react"
        assert req.tech_preferences.get("backend") == "spring boot"
        assert req.tech_preferences.get("db") == "h2"

    @pytest.mark.anyio
    async def test_parse_from_plan_detects_has_frontend_has_backend(self, mock_llm, mock_repo_manager, sample_plan_md):
        from agents.product_planner_agent import ProductPlannerAgent

        mock_llm.generate_json.return_value = {
            "title": "Task Manager",
            "description": "An app.",
            "user_stories": [],
            "features": [],
            "tech_preferences": {},
            "has_frontend": True,
            "has_backend": True,
        }

        agent = ProductPlannerAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        req = await agent.parse_from_plan(sample_plan_md)

        assert req.has_frontend is True
        assert req.has_backend is True

    @pytest.mark.anyio
    async def test_parse_from_plan_with_auth_project(self, mock_llm, mock_repo_manager, sample_plan_md):
        from agents.product_planner_agent import ProductPlannerAgent

        mock_llm.generate_json.return_value = {
            "title": "Task Manager",
            "description": "App with JWT auth.",
            "user_stories": ["As a user, I want to log in"],
            "features": ["Auth", "Task CRUD"],
            "tech_preferences": {"backend": "spring boot"},
            "has_frontend": True,
            "has_backend": True,
        }

        agent = ProductPlannerAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        req = await agent.parse_from_plan(sample_plan_md)

        assert "Auth" in req.features
        assert any("log in" in story.lower() or "login" in story.lower() for story in req.user_stories)

    @pytest.mark.anyio
    async def test_parse_from_plan_passes_plan_to_llm(self, mock_llm, mock_repo_manager, sample_plan_md):
        from agents.product_planner_agent import ProductPlannerAgent

        mock_llm.generate_json.return_value = {
            "title": "T",
            "description": "D",
            "user_stories": [],
            "features": [],
            "tech_preferences": {},
            "has_frontend": True,
            "has_backend": True,
        }

        agent = ProductPlannerAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        await agent.parse_from_plan(sample_plan_md)

        # generate_json should have been called with the plan.md content
        call_args = mock_llm.generate_json.call_args
        user_prompt = call_args.kwargs.get("user_prompt", "") or (
            call_args.args[1] if len(call_args.args) > 1 else ""
        )
        assert "PHASE" in user_prompt or "plan.md" in user_prompt.lower() or "extract" in user_prompt.lower()

    @pytest.mark.anyio
    async def test_parse_from_plan_retries_on_json_error(self, mock_llm, mock_repo_manager, sample_plan_md):
        from agents.product_planner_agent import ProductPlannerAgent

        good_result = {
            "title": "Task Manager",
            "description": "An app.",
            "user_stories": [],
            "features": [],
            "tech_preferences": {},
            "has_frontend": True,
            "has_backend": True,
        }
        mock_llm.generate_json = AsyncMock(side_effect=[RuntimeError("JSON parse error"), good_result])

        agent = ProductPlannerAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        req = await agent.parse_from_plan(sample_plan_md)

        assert req.title == "Task Manager"
        assert mock_llm.generate_json.call_count == 2


# ── ArchitectAgent.design_from_plan tests ─────────────────────────────────────


class TestArchitectDesignFromPlan:
    def _make_blueprint_response(self) -> dict:
        return {
            "name": "task-manager",
            "description": "Task Manager backend",
            "architecture_style": "REST",
            "tech_stack": {
                "language": "java",
                "framework": "spring boot",
                "db": "h2",
                "build_tool": "maven",
            },
            "folder_structure": [
                "backend/src/main/java/com/app/controller",
                "backend/src/main/java/com/app/model",
            ],
            "file_blueprints": [
                {
                    "path": "backend/pom.xml",
                    "purpose": "Maven build configuration",
                    "depends_on": [],
                    "exports": [],
                    "language": "java",
                    "layer": "infrastructure",
                },
                {
                    "path": "backend/src/main/java/com/app/model/Task.java",
                    "purpose": "Task JPA entity",
                    "depends_on": [],
                    "exports": ["Task"],
                    "language": "java",
                    "layer": "model",
                },
                {
                    "path": "backend/src/main/java/com/app/controller/TaskController.java",
                    "purpose": "CRUD endpoints for Task entity",
                    "depends_on": ["backend/src/main/java/com/app/model/Task.java"],
                    "exports": ["TaskController"],
                    "language": "java",
                    "layer": "controller",
                },
                {
                    "path": "backend/src/main/java/com/app/TaskManagerApplication.java",
                    "purpose": "@SpringBootApplication entry point",
                    "depends_on": [],
                    "exports": ["TaskManagerApplication"],
                    "language": "java",
                    "layer": "infrastructure",
                },
                {
                    "path": "backend/src/main/resources/application.properties",
                    "purpose": "Spring Boot config",
                    "depends_on": [],
                    "exports": [],
                    "language": "java",
                    "layer": "infrastructure",
                },
            ],
        }

    @pytest.mark.anyio
    async def test_design_from_plan_extracts_file_blueprints(
        self, mock_llm, mock_repo_manager, sample_plan_md, sample_requirements
    ):
        from agents.architect_agent import ArchitectAgent

        bp_data = self._make_blueprint_response()
        mock_llm.generate.return_value.content = str(bp_data).replace("'", '"')

        # Patch _parse_json_response to return the dict directly
        with patch.object(
            ArchitectAgent, "_parse_json_response", return_value=bp_data
        ), patch.object(
            ArchitectAgent, "_fetch_architecture_doc", new=AsyncMock(side_effect=lambda bp, _: bp)
        ):
            agent = ArchitectAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
            blueprint = await agent.design_from_plan(sample_plan_md, sample_requirements)

        assert blueprint.name == "task-manager"
        assert len(blueprint.file_blueprints) > 0

    @pytest.mark.anyio
    async def test_design_from_plan_applies_mandatory_spring_boot_files(
        self, mock_llm, mock_repo_manager, sample_plan_md, sample_requirements
    ):
        from agents.architect_agent import ArchitectAgent

        # Blueprint without Application.java and application.properties
        bp_data = self._make_blueprint_response()
        bp_data["file_blueprints"] = [
            fb for fb in bp_data["file_blueprints"]
            if "Application.java" not in fb["path"]
            and "application.properties" not in fb["path"]
        ]

        with patch.object(
            ArchitectAgent, "_parse_json_response", return_value=bp_data
        ), patch.object(
            ArchitectAgent, "_fetch_architecture_doc", new=AsyncMock(side_effect=lambda bp, _: bp)
        ):
            agent = ArchitectAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
            blueprint = await agent.design_from_plan(sample_plan_md, sample_requirements)

        paths = [fb.path for fb in blueprint.file_blueprints]
        # Mandatory injection should add Application.java and application.properties
        assert any("Application.java" in p or "App.java" in p for p in paths), \
            "Application.java should be injected"
        assert any("application.properties" in p for p in paths), \
            "application.properties should be injected"

    @pytest.mark.anyio
    async def test_design_from_plan_deduplicates_paths(
        self, mock_llm, mock_repo_manager, sample_plan_md, sample_requirements
    ):
        from agents.architect_agent import ArchitectAgent

        bp_data = self._make_blueprint_response()
        # Add a duplicate
        duplicate = dict(bp_data["file_blueprints"][1])  # Task.java
        bp_data["file_blueprints"].append(duplicate)

        with patch.object(
            ArchitectAgent, "_parse_json_response", return_value=bp_data
        ), patch.object(
            ArchitectAgent, "_fetch_architecture_doc", new=AsyncMock(side_effect=lambda bp, _: bp)
        ):
            agent = ArchitectAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
            blueprint = await agent.design_from_plan(sample_plan_md, sample_requirements)

        paths = [fb.path for fb in blueprint.file_blueprints]
        # All paths must be unique
        assert len(paths) == len(set(paths)), "Duplicate paths should be removed"

    @pytest.mark.anyio
    async def test_design_from_plan_strips_frontend_files(
        self, mock_llm, mock_repo_manager, sample_plan_md, sample_requirements
    ):
        from agents.architect_agent import ArchitectAgent

        bp_data = self._make_blueprint_response()
        # Add a frontend file that should be stripped
        bp_data["file_blueprints"].append({
            "path": "frontend/src/pages/TaskListPage.tsx",
            "purpose": "Task list page",
            "depends_on": [],
            "exports": ["TaskListPage"],
            "language": "typescript",
            "layer": "page",
        })

        with patch.object(
            ArchitectAgent, "_parse_json_response", return_value=bp_data
        ), patch.object(
            ArchitectAgent, "_fetch_architecture_doc", new=AsyncMock(side_effect=lambda bp, _: bp)
        ):
            agent = ArchitectAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
            blueprint = await agent.design_from_plan(sample_plan_md, sample_requirements)

        paths = [fb.path for fb in blueprint.file_blueprints]
        assert not any("frontend/" in p for p in paths), \
            "Frontend files should be stripped from backend blueprint"

    @pytest.mark.anyio
    async def test_design_from_plan_extracts_phase2_section(
        self, mock_llm, mock_repo_manager, sample_plan_md, sample_requirements
    ):
        from agents.architect_agent import ArchitectAgent

        bp_data = self._make_blueprint_response()
        captured_prompts = []

        original_llm_with_heartbeat = ArchitectAgent._llm_with_heartbeat

        async def capturing_llm(self_inner, system_prompt, user_prompt, max_tokens, label=""):
            captured_prompts.append(user_prompt)
            return mock_llm.generate.return_value

        with patch.object(ArchitectAgent, "_llm_with_heartbeat", capturing_llm), \
             patch.object(ArchitectAgent, "_parse_json_response", return_value=bp_data), \
             patch.object(
                 ArchitectAgent, "_fetch_architecture_doc",
                 new=AsyncMock(side_effect=lambda bp, _: bp)
             ):
            agent = ArchitectAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
            await agent.design_from_plan(sample_plan_md, sample_requirements)

        # The architecture prompt should reference the PHASE 2 plan content
        assert len(captured_prompts) > 0
        combined = " ".join(captured_prompts)
        assert "PHASE 2" in combined or "BACKEND PLAN" in combined or "SecurityConfig" in combined


# ── APIContractAgent.extract_from_plan tests ──────────────────────────────────


class TestAPIContractExtractFromPlan:
    def _make_contract_response(self) -> dict:
        return {
            "title": "Task Manager API",
            "version": "1.0.0",
            "base_url": "/api",
            "contract_format": "openapi",
            "endpoints": [
                {
                    "path": "/api/auth/login",
                    "method": "POST",
                    "description": "Authenticate user",
                    "request_schema": {"email": "string", "password": "string"},
                    "response_schema": {"token": "string", "userId": "long"},
                    "auth_required": False,
                    "tags": ["auth"],
                },
                {
                    "path": "/api/tasks",
                    "method": "GET",
                    "description": "List all tasks",
                    "request_schema": {},
                    "response_schema": {"type": "array"},
                    "auth_required": True,
                    "tags": ["tasks"],
                },
                {
                    "path": "/api/tasks",
                    "method": "POST",
                    "description": "Create task",
                    "request_schema": {"title": "string", "status": "string"},
                    "response_schema": {"id": "long", "title": "string"},
                    "auth_required": True,
                    "tags": ["tasks"],
                },
                {
                    "path": "/api/tasks/{id}",
                    "method": "DELETE",
                    "description": "Delete task",
                    "request_schema": {},
                    "response_schema": {},
                    "auth_required": True,
                    "tags": ["tasks"],
                },
            ],
            "schemas": {
                "Task": {"type": "object", "properties": {"id": {"type": "integer"}, "title": {"type": "string"}}},
                "AuthResponse": {"type": "object", "properties": {"token": {"type": "string"}}},
            },
            "openapi_spec": "openapi: 3.0.3\ninfo:\n  title: Task Manager API",
        }

    @pytest.mark.anyio
    async def test_extract_from_plan_returns_endpoints(
        self, mock_llm, mock_repo_manager, sample_plan_md, sample_requirements, sample_blueprint
    ):
        from agents.api_contract_agent import APIContractAgent

        mock_llm.generate_json.return_value = self._make_contract_response()

        agent = APIContractAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        contract = await agent.extract_from_plan(sample_plan_md, sample_requirements, sample_blueprint)

        assert len(contract.endpoints) == 4
        paths = [ep.path for ep in contract.endpoints]
        assert "/api/auth/login" in paths
        assert "/api/tasks" in paths

    @pytest.mark.anyio
    async def test_extract_from_plan_populates_schemas(
        self, mock_llm, mock_repo_manager, sample_plan_md, sample_requirements, sample_blueprint
    ):
        from agents.api_contract_agent import APIContractAgent

        mock_llm.generate_json.return_value = self._make_contract_response()

        agent = APIContractAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        contract = await agent.extract_from_plan(sample_plan_md, sample_requirements, sample_blueprint)

        assert "Task" in contract.schemas
        assert "AuthResponse" in contract.schemas

    @pytest.mark.anyio
    async def test_extract_from_plan_sets_auth_required(
        self, mock_llm, mock_repo_manager, sample_plan_md, sample_requirements, sample_blueprint
    ):
        from agents.api_contract_agent import APIContractAgent

        mock_llm.generate_json.return_value = self._make_contract_response()

        agent = APIContractAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        contract = await agent.extract_from_plan(sample_plan_md, sample_requirements, sample_blueprint)

        login_ep = next(ep for ep in contract.endpoints if ep.path == "/api/auth/login")
        assert login_ep.auth_required is False

        task_get = next(ep for ep in contract.endpoints if ep.path == "/api/tasks" and ep.method == "GET")
        assert task_get.auth_required is True

    @pytest.mark.anyio
    async def test_extract_from_plan_raises_on_empty_endpoints(
        self, mock_llm, mock_repo_manager, sample_plan_md, sample_requirements, sample_blueprint
    ):
        from agents.api_contract_agent import APIContractAgent

        mock_llm.generate_json.return_value = {
            "title": "Empty",
            "version": "1.0.0",
            "base_url": "/api",
            "contract_format": "openapi",
            "endpoints": [],
            "schemas": {},
            "openapi_spec": "",
        }

        agent = APIContractAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
        with pytest.raises(ValueError, match="0 endpoints"):
            await agent.extract_from_plan(sample_plan_md, sample_requirements, sample_blueprint)

    @pytest.mark.anyio
    async def test_extract_from_plan_uses_phase1_content(
        self, mock_llm, mock_repo_manager, sample_plan_md, sample_requirements, sample_blueprint
    ):
        from agents.api_contract_agent import APIContractAgent

        mock_llm.generate_json.return_value = self._make_contract_response()
        captured_prompts = []

        original_call_llm_json = APIContractAgent._call_llm_json

        async def capturing(self_inner, user_prompt, system_override=None):
            captured_prompts.append(user_prompt)
            return mock_llm.generate_json.return_value

        with patch.object(APIContractAgent, "_call_llm_json", capturing):
            agent = APIContractAgent(llm_client=mock_llm, repo_manager=mock_repo_manager)
            await agent.extract_from_plan(sample_plan_md, sample_requirements, sample_blueprint)

        assert len(captured_prompts) > 0
        combined = " ".join(captured_prompts)
        # Should mention PHASE 1 content (endpoints from plan.md)
        assert "PHASE 1" in combined or "/api/auth/login" in combined or "API CONTRACT" in combined


# ── ArchitectureApprover.approve_plan_md tests ────────────────────────────────


class TestArchitectureApproverPlanMd:
    def test_approve_plan_md_interactive_approve(self, tmp_path, sample_plan_md):
        from core.architecture_approver import ArchitectureApprover

        approver = ArchitectureApprover(interactive=True, workspace=tmp_path, live=None)

        with patch("builtins.input", return_value="y"), \
             patch("rich.console.Console.print"):
            result = approver.approve_plan_md(sample_plan_md)

        assert result is True

    def test_approve_plan_md_interactive_reject(self, tmp_path, sample_plan_md):
        from core.architecture_approver import ArchitectureApprover

        approver = ArchitectureApprover(interactive=True, workspace=tmp_path, live=None)

        with patch("builtins.input", return_value="n"), \
             patch("rich.console.Console.print"):
            result = approver.approve_plan_md(sample_plan_md)

        assert result is False

    def test_approve_plan_md_interactive_edit(self, tmp_path, sample_plan_md):
        from core.architecture_approver import ArchitectureApprover

        approver = ArchitectureApprover(interactive=True, workspace=tmp_path, live=None)

        # First input: 'e' to edit, second input: feedback text
        with patch("builtins.input", side_effect=["e", "Use PostgreSQL instead"]), \
             patch("rich.console.Console.print"):
            result = approver.approve_plan_md(sample_plan_md)

        assert result == "Use PostgreSQL instead"

    def test_approve_plan_md_noninteractive_writes_pending_file(self, tmp_path, sample_plan_md):
        from core.architecture_approver import ArchitectureApprover, ArchitecturePendingApprovalError

        approver = ArchitectureApprover(interactive=False, workspace=tmp_path, live=None)

        with pytest.raises(ArchitecturePendingApprovalError):
            approver.approve_plan_md(sample_plan_md)

        pending_file = tmp_path / "pending_plan.md"
        assert pending_file.exists()
        assert "PHASE 0" in pending_file.read_text(encoding="utf-8")

    def test_approve_plan_md_noninteractive_raises_correct_error(self, tmp_path, sample_plan_md):
        from core.architecture_approver import ArchitectureApprover, ArchitecturePendingApprovalError

        approver = ArchitectureApprover(interactive=False, workspace=tmp_path, live=None)

        with pytest.raises(ArchitecturePendingApprovalError) as exc_info:
            approver.approve_plan_md(sample_plan_md)

        assert "plan.md" in str(exc_info.value).lower() or "pending" in str(exc_info.value).lower()

    def test_approve_plan_md_plain_fallback_on_import_error(self, tmp_path, sample_plan_md):
        """Test plain text fallback when rich is not available."""
        from core.architecture_approver import ArchitectureApprover

        approver = ArchitectureApprover(interactive=True, workspace=tmp_path, live=None)

        with patch("builtins.input", return_value="y"), \
             patch.object(approver, "_interactive_plan_md",
                          side_effect=ImportError("no rich")), \
             patch.object(approver, "_plain_plan_md", return_value=True) as mock_plain:
            # _with_live_paused calls _interactive_plan_md; if that fails we fall back
            # Test the plain fallback directly instead
            result = approver._plain_plan_md(sample_plan_md[:100])

        # Just verify the plain method returns something reasonable via input mock
        with patch("builtins.input", return_value="y"):
            result2 = approver._plain_plan_md(sample_plan_md)
        assert result2 is True


# ── Plan-first pipeline integration tests ─────────────────────────────────────


class TestPlanFirstPipelineIntegration:
    def _make_settings(self, tmp_path):
        settings = MagicMock(spec=[])  # no attributes → hasattr checks return False
        settings.workspace_dir = tmp_path
        settings.require_architecture_approval = False
        return settings

    def _make_llm(self, plan_content, requirements_data, blueprint_data, contract_data):
        """Create a mock LLM that returns appropriate responses for each pipeline phase."""
        llm = MagicMock()

        # generate() is called by PlanGeneratorAgent and ArchitectAgent
        plan_response = MagicMock()
        plan_response.content = plan_content
        plan_response.stop_reason = "end_turn"
        plan_response.usage = {"input_tokens": 100, "output_tokens": 500}

        arch_response = MagicMock()
        arch_response.content = '{"name": "task-manager"}'
        arch_response.stop_reason = "end_turn"
        arch_response.usage = {"input_tokens": 100, "output_tokens": 500}

        llm.generate = AsyncMock(side_effect=[plan_response, arch_response, arch_response])

        # generate_json() is called by ProductPlannerAgent and APIContractAgent
        llm.generate_json = AsyncMock(side_effect=[requirements_data, contract_data])

        return llm

    @pytest.mark.anyio
    async def test_full_pipeline_calls_plan_generator_first(
        self, tmp_path, sample_plan_md, sample_requirements
    ):
        from core.pipeline_fullstack import FullstackPipeline

        settings = self._make_settings(tmp_path)
        call_order = []

        with patch("agents.plan_generator_agent.PlanGeneratorAgent.generate_plan",
                   new=AsyncMock(side_effect=lambda p: call_order.append("plan") or sample_plan_md)), \
             patch("agents.product_planner_agent.ProductPlannerAgent.parse_from_plan",
                   new=AsyncMock(return_value=sample_requirements)), \
             patch("agents.architect_agent.ArchitectAgent.design_from_plan",
                   new=AsyncMock(side_effect=lambda *a, **kw: (_ for _ in ()).throw(
                       RuntimeError("stop after phase 3")))):

            llm = MagicMock()
            pipeline = FullstackPipeline(settings=settings, llm=llm, interactive=False)
            import time
            result = await pipeline.execute("Build a task manager", time.monotonic())

        assert "plan" in call_order
        assert call_order[0] == "plan"

    @pytest.mark.anyio
    async def test_approval_gate_shows_plan_md(self, tmp_path, sample_plan_md, sample_requirements):
        from core.pipeline_fullstack import FullstackPipeline

        settings = self._make_settings(tmp_path)
        settings.require_architecture_approval = True

        approval_calls = []

        with patch("agents.plan_generator_agent.PlanGeneratorAgent.generate_plan",
                   new=AsyncMock(return_value=sample_plan_md)), \
             patch("core.architecture_approver.ArchitectureApprover.approve_plan_md",
                   side_effect=lambda content: approval_calls.append(content) or True), \
             patch("agents.product_planner_agent.ProductPlannerAgent.parse_from_plan",
                   new=AsyncMock(side_effect=RuntimeError("stop after approval"))):

            llm = MagicMock()
            pipeline = FullstackPipeline(settings=settings, llm=llm, interactive=True)
            import time
            await pipeline.execute("Build a task manager", time.monotonic())

        assert len(approval_calls) == 1
        assert approval_calls[0] == sample_plan_md

    @pytest.mark.anyio
    async def test_revision_loop_updates_plan_md(self, tmp_path, sample_plan_md, sample_requirements):
        from core.pipeline_fullstack import FullstackPipeline

        settings = self._make_settings(tmp_path)
        settings.require_architecture_approval = True

        revised_plan = sample_plan_md.replace("H2", "PostgreSQL")
        approval_side_effects = [
            "Use PostgreSQL",  # first call: edit with feedback
            True,              # second call: approve
        ]

        with patch("agents.plan_generator_agent.PlanGeneratorAgent.generate_plan",
                   new=AsyncMock(return_value=sample_plan_md)), \
             patch("agents.plan_generator_agent.PlanGeneratorAgent.revise_plan",
                   new=AsyncMock(return_value=revised_plan)) as mock_revise, \
             patch("core.architecture_approver.ArchitectureApprover.approve_plan_md",
                   side_effect=approval_side_effects), \
             patch("agents.product_planner_agent.ProductPlannerAgent.parse_from_plan",
                   new=AsyncMock(side_effect=RuntimeError("stop after approval"))):

            llm = MagicMock()
            pipeline = FullstackPipeline(settings=settings, llm=llm, interactive=True)
            import time
            await pipeline.execute("Build a task manager", time.monotonic())

        mock_revise.assert_called_once()
        call_kwargs = mock_revise.call_args
        assert "PostgreSQL" in str(call_kwargs)

    @pytest.mark.anyio
    async def test_requirements_derived_from_plan(self, tmp_path, sample_plan_md, sample_requirements):
        from core.pipeline_fullstack import FullstackPipeline

        settings = self._make_settings(tmp_path)
        parse_calls = []

        with patch("agents.plan_generator_agent.PlanGeneratorAgent.generate_plan",
                   new=AsyncMock(return_value=sample_plan_md)), \
             patch("agents.product_planner_agent.ProductPlannerAgent.parse_from_plan",
                   side_effect=lambda plan: parse_calls.append(plan) or sample_requirements), \
             patch("agents.architect_agent.ArchitectAgent.design_from_plan",
                   new=AsyncMock(side_effect=RuntimeError("stop"))):

            llm = MagicMock()
            pipeline = FullstackPipeline(settings=settings, llm=llm, interactive=False)
            import time
            await pipeline.execute("Build a task manager", time.monotonic())

        assert len(parse_calls) == 1
        assert parse_calls[0] == sample_plan_md

    @pytest.mark.anyio
    async def test_blueprint_derived_from_plan(self, tmp_path, sample_plan_md, sample_requirements, sample_blueprint):
        from core.pipeline_fullstack import FullstackPipeline

        settings = self._make_settings(tmp_path)
        design_calls = []

        with patch("agents.plan_generator_agent.PlanGeneratorAgent.generate_plan",
                   new=AsyncMock(return_value=sample_plan_md)), \
             patch("agents.product_planner_agent.ProductPlannerAgent.parse_from_plan",
                   new=AsyncMock(return_value=sample_requirements)), \
             patch("agents.architect_agent.ArchitectAgent.design_from_plan",
                   side_effect=lambda plan, req: design_calls.append((plan, req)) or sample_blueprint), \
             patch("agents.api_contract_agent.APIContractAgent.extract_from_plan",
                   new=AsyncMock(side_effect=RuntimeError("stop"))):

            llm = MagicMock()
            pipeline = FullstackPipeline(settings=settings, llm=llm, interactive=False)
            import time
            await pipeline.execute("Build a task manager", time.monotonic())

        assert len(design_calls) == 1
        plan_arg, req_arg = design_calls[0]
        assert plan_arg == sample_plan_md
        assert req_arg.title == sample_requirements.title

    @pytest.mark.anyio
    async def test_contract_derived_from_plan(
        self, tmp_path, sample_plan_md, sample_requirements, sample_blueprint
    ):
        from core.pipeline_fullstack import FullstackPipeline

        settings = self._make_settings(tmp_path)
        sample_contract = APIContract(
            title="Task Manager API",
            version="1.0.0",
            base_url="/api",
            endpoints=[
                APIEndpoint(path="/api/tasks", method="GET", description="List tasks",
                            auth_required=True, tags=["tasks"]),
            ],
            schemas={"Task": {"type": "object"}},
        )
        extract_calls = []

        with patch("agents.plan_generator_agent.PlanGeneratorAgent.generate_plan",
                   new=AsyncMock(return_value=sample_plan_md)), \
             patch("agents.product_planner_agent.ProductPlannerAgent.parse_from_plan",
                   new=AsyncMock(return_value=sample_requirements)), \
             patch("agents.architect_agent.ArchitectAgent.design_from_plan",
                   new=AsyncMock(return_value=sample_blueprint)), \
             patch("agents.api_contract_agent.APIContractAgent.extract_from_plan",
                   side_effect=lambda plan, req, bp: extract_calls.append((plan, req, bp)) or sample_contract), \
             patch("core.pipeline_fullstack.FullstackPipeline._validate_contract_fields"), \
             patch("core.pipeline_fullstack.FullstackPipeline._validate_contract_blueprint"), \
             patch("core.pipeline_fullstack.FullstackPipeline._verify_fe_contract_coverage",
                   new=AsyncMock()), \
             patch("core.pipeline_run.RunPipeline.execute", new=AsyncMock(return_value=MagicMock(
                 success=True, errors=[], task_stats={}, metrics={}))), \
             patch("core.pipeline_frontend.FrontendPipeline.execute", new=AsyncMock(return_value=MagicMock(
                 success=True, errors=[], metrics={}))):

            llm = MagicMock()
            pipeline = FullstackPipeline(settings=settings, llm=llm, interactive=False)
            import time
            await pipeline.execute("Build a task manager", time.monotonic())

        assert len(extract_calls) == 1
        plan_arg, req_arg, bp_arg = extract_calls[0]
        assert plan_arg == sample_plan_md

    @pytest.mark.anyio
    async def test_pipeline_abort_on_plan_rejection(self, tmp_path, sample_plan_md):
        from core.pipeline_fullstack import FullstackPipeline
        from core.pipeline import PipelineResult

        settings = self._make_settings(tmp_path)
        settings.require_architecture_approval = True

        with patch("agents.plan_generator_agent.PlanGeneratorAgent.generate_plan",
                   new=AsyncMock(return_value=sample_plan_md)), \
             patch("core.architecture_approver.ArchitectureApprover.approve_plan_md",
                   return_value=False):

            llm = MagicMock()
            pipeline = FullstackPipeline(settings=settings, llm=llm, interactive=True)
            import time
            result = await pipeline.execute("Build a task manager", time.monotonic())

        assert result.success is False
        assert any("rejected" in e.lower() for e in result.errors)
