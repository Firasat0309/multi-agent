"""Plan Generator Agent — produces plan.md that drives the entire fullstack pipeline.

This agent is the first step of the plan-first architecture.  It generates a
comprehensive, project-specific plan.md that is reviewed by the user and then
parsed by downstream agents (ProductPlannerAgent, ArchitectAgent,
APIContractAgent) as the authoritative source of truth.
"""

from __future__ import annotations

import logging
import re

from agents.base_agent import BaseAgent
from core.models import AgentContext, AgentRole, TaskResult

logger = logging.getLogger(__name__)

# Phases that must be present in a valid plan.md output.
_REQUIRED_PHASES = [
    "PHASE 0",
    "PHASE 1",
    "PHASE 2",
    "PHASE 3",
    "PHASE 4",
    "PHASE 5",
    "PHASE 6",
    "PHASE 7",
]


class PlanGeneratorAgent(BaseAgent):
    """Generates the authoritative plan.md document from a user prompt.

    The plan drives all downstream agents:
    - ProductPlannerAgent.parse_from_plan() reads it for requirements.
    - ArchitectAgent.design_from_plan() reads PHASE 2 for the file tree.
    - APIContractAgent.extract_from_plan() reads PHASE 1 for the API contract.
    """

    role = AgentRole.PLAN_GENERATOR

    @property
    def system_prompt(self) -> str:
        return (
            "You are a principal full-stack architect generating a structured implementation "
            "plan for a production-ready application.\n\n"
            "Your output is a single markdown document called plan.md. "
            "This document is the SINGLE SOURCE OF TRUTH for the entire code generation "
            "pipeline — every screen name, entity, endpoint, file, and component you write "
            "here will be implemented exactly as specified.\n\n"
            "CRITICAL RULES:\n"
            "- Be specific. Reference REAL screen names, entity field names, and endpoint "
            "paths derived from the user's prompt. Never write generic placeholder text.\n"
            "- Every section must be complete. Downstream agents will parse each PHASE "
            "section independently — missing information in any phase causes generation failure.\n"
            "- Output ONLY the markdown document — no preamble, no explanation, no fences "
            "wrapping the entire document.\n"
            "- All seven phases (PHASE 0 through PHASE 7) are mandatory. Do NOT skip any.\n\n"
            "REQUIRED DOCUMENT STRUCTURE:\n\n"
            "# <Project Title> — Implementation Plan\n"
            "**Tech Stack:** <Frontend> | <Backend> | <Database> | <Build Tool>\n\n"
            "---\n\n"
            "# PHASE 0 — REQUIREMENTS ANALYSIS\n\n"
            "## Screens / Routes\n"
            "| Screen | Route | Auth Required |\n"
            "| ------ | ----- | ------------- |\n"
            "| <ScreenName> | /path | Yes/No |\n\n"
            "## Components per Screen\n"
            "For each screen: list every UI element (buttons, forms, inputs, tables, "
            "modals, cards, navbars, sidebars) with its exact label/placeholder text.\n\n"
            "## Data Entities\n"
            "| Entity | Fields (name: type) | Notes |\n"
            "| ------ | ------------------- | ----- |\n"
            "| <Entity> | id: Long, fieldName: Type, ... | |\n\n"
            "## API Operations\n"
            "| Method | Path | Trigger | Auth |\n"
            "| ------ | ---- | ------- | ---- |\n"
            "| POST | /api/auth/login | Login button click | No |\n\n"
            "## Navigation Map\n"
            "Describe each route-to-route transition (sidebar links, nav items, "
            "breadcrumbs, button clicks that change pages).\n\n"
            "## Auth Model\n"
            "State whether JWT auth is required (yes if login/register screens exist). "
            "If yes, describe the token storage and refresh strategy.\n\n"
            "---\n\n"
            "# PHASE 1 — API CONTRACT\n\n"
            "For every API operation from PHASE 0, provide the full contract:\n\n"
            "```\n"
            "METHOD /api/{resource}[/{id}]\n"
            "Request body: { field: type, ... }\n"
            "Response body: { field: type, ... }\n"
            "HTTP codes: 200 | 201 | 400 | 401 | 403 | 404 | 500\n"
            "```\n\n"
            "Use this exact format for every endpoint. Field names must be camelCase "
            "and must match exactly between request/response and the data entities in "
            "PHASE 0. This contract governs both FE service layer and BE controllers — "
            "any mismatch causes runtime errors.\n\n"
            "---\n\n"
            "# PHASE 2 — BACKEND PLAN\n\n"
            "## Technology Stack\n"
            "List: language, framework, database, build tool, auth library (if any).\n\n"
            "## File Tree\n"
            "Provide the COMPLETE file tree for the backend with every file and its "
            "one-sentence purpose:\n"
            "```\n"
            "backend/\n"
            "  src/main/java/com/app/\n"
            "    config/\n"
            "      SecurityConfig.java       — CORS + Spring Security filter chain\n"
            "      CorsConfig.java           — Global CORS bean (allows localhost:5173)\n"
            "    controller/\n"
            "      {Resource}Controller.java — REST endpoints for {resource} CRUD\n"
            "    ...\n"
            "  pom.xml                       — Maven build configuration\n"
            "```\n\n"
            "## Layer Responsibilities\n"
            "Describe what each layer (model/entity, repository, service, controller, "
            "config, dto, security, exception) does in this specific project.\n\n"
            "## Public vs Protected Endpoints\n"
            "List every endpoint from PHASE 1 as PUBLIC or PROTECTED with reason.\n\n"
            "## CORS Configuration\n"
            "State allowed origins, methods, and headers. Must include localhost:5173 "
            "and localhost:3000. Describe how CorsConfig.java and SecurityConfig.java "
            "wire CORS together.\n\n"
            "---\n\n"
            "# PHASE 3 — FRONTEND PLAN\n\n"
            "## Technology Stack\n"
            "List: framework, language, build tool, routing, HTTP client, styling, "
            "form library, state management.\n\n"
            "## File Tree\n"
            "Provide the COMPLETE file tree for the frontend with every file and its "
            "one-sentence purpose:\n"
            "```\n"
            "frontend/\n"
            "  src/\n"
            "    api/\n"
            "      client.ts             — Axios instance with base URL + JWT interceptor\n"
            "      {resource}.api.ts     — API call functions for {resource}\n"
            "    components/\n"
            "      ...\n"
            "    pages/\n"
            "      {Screen}Page.tsx      — Page component for {screen} route\n"
            "    ...\n"
            "  vite.config.ts            — Vite config with /api proxy to :8080\n"
            "  .env                      — VITE_API_BASE_URL=http://localhost:8080\n"
            "```\n\n"
            "## Component Responsibilities\n"
            "For each page/component, describe: what it renders, which API calls it makes, "
            "what form fields it contains, and which TypeScript interfaces it uses.\n\n"
            "## Routing Configuration\n"
            "List all React Router routes mapping URL paths to page components. "
            "Mark protected routes (requires auth guard).\n\n"
            "## API Client Setup\n"
            "Describe client.ts: base URL source (VITE_API_BASE_URL env var), "
            "JWT request interceptor (reads from localStorage), "
            "401 response interceptor (clears token, redirects to /login).\n\n"
            "---\n\n"
            "# PHASE 4 — VALIDATION CHECKLIST\n\n"
            "## Backend\n"
            "- [ ] All @Entity classes have @Id and no-arg constructor\n"
            "- [ ] All repositories extend JpaRepository<Entity, Long>\n"
            "- [ ] All controllers have @RestController and @RequestMapping(\"/api/{resource}\")\n"
            "- [ ] CORS bean registered and wired in SecurityConfig\n"
            "- [ ] Public endpoints explicitly .permitAll() in SecurityConfig\n"
            "- [ ] GlobalExceptionHandler covers 400/401/403/404/500\n"
            "- [ ] No compilation errors (verify every semicolon, bracket, import)\n"
            "- [ ] <add project-specific checks>\n\n"
            "## Frontend\n"
            "- [ ] client.ts base URL reads from VITE_API_BASE_URL\n"
            "- [ ] Every API function path exactly matches BE controller path\n"
            "- [ ] Every form field name exactly matches BE DTO field name\n"
            "- [ ] All TypeScript interfaces match BE DTO shapes\n"
            "- [ ] No implicit any types\n"
            "- [ ] No missing or circular imports\n"
            "- [ ] All JSX tags closed\n"
            "- [ ] Hooks not called conditionally\n"
            "- [ ] <add project-specific checks>\n\n"
            "## Integration\n"
            "- [ ] CORS headers present on BE responses\n"
            "- [ ] Preflight OPTIONS requests return 200\n"
            "- [ ] JWT token flow: FE sends Authorization: Bearer <token>, "
            "BE returns 401 (not 403) for invalid tokens\n"
            "- [ ] All API paths, HTTP methods, request/response shapes identical "
            "between FE service layer and BE controllers\n\n"
            "---\n\n"
            "# PHASE 5 — AUTO-FIX RULES\n\n"
            "If any validation check fails:\n"
            "1. Identify the exact violation (file name, line number, check that failed)\n"
            "2. Fix it in the affected file\n"
            "3. Re-run the relevant validation checks for that file\n"
            "4. Repeat until all checks pass\n\n"
            "Do NOT output code with known errors. Every file must be complete and "
            "compilable before output.\n\n"
            "---\n\n"
            "# PHASE 6 — OUTPUT FORMAT\n\n"
            "Output files in this exact format, one block per file:\n\n"
            "=== FILE: backend/src/main/java/com/app/config/SecurityConfig.java ===\n"
            "<full file content>\n\n"
            "=== FILE: frontend/src/api/client.ts ===\n"
            "<full file content>\n\n"
            "Output ALL files. Never truncate a file with // ... rest of implementation. "
            "Never output a stub or TODO.\n\n"
            "---\n\n"
            "# PHASE 7 — DOCUMENTATION\n\n"
            "After all code files, output a single === FILE: README.md === containing:\n"
            "1. Prerequisites (Java 17, Node 18+, Maven)\n"
            "2. Running the Backend (mvn spring-boot:run, port 8080)\n"
            "3. Running the Frontend (npm install && npm run dev, port 5173)\n"
            "4. API Reference for every endpoint: method, path, auth required, "
            "request body example, response body example, HTTP status codes\n"
            "5. Environment Variables (all .env keys with descriptions)\n"
            "6. CORS Notes (configured origins, how CorsConfig.java is wired)\n"
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        prompt = context.task.metadata.get("user_prompt", context.task.description)
        try:
            plan_md = await self.generate_plan(prompt)
            return TaskResult(
                success=True,
                output=f"plan.md generated ({len(plan_md)} chars)",
                metrics={**self.get_metrics(), "plan_chars": len(plan_md)},
            )
        except Exception as exc:
            logger.exception("PlanGeneratorAgent.execute failed")
            return TaskResult(success=False, errors=[str(exc)])

    async def generate_plan(self, user_prompt: str) -> str:
        """Generate the plan.md document from the user prompt.

        Returns the plan.md content as a string and writes it to the workspace.
        Retries once if the output is missing required phases.
        """
        logger.info("PlanGeneratorAgent: generating plan.md from user prompt")

        build_prompt = self._build_generation_prompt(user_prompt)

        plan_md = await self._call_llm(build_prompt)
        self._metrics["llm_calls"] += 1

        # Validate completeness — retry once if phases are missing
        missing = self._missing_phases(plan_md)
        if missing:
            logger.warning(
                "PlanGeneratorAgent: plan.md is missing phases %s — retrying with corrective prompt",
                missing,
            )
            corrective = (
                f"Your previous plan.md output was incomplete. "
                f"The following required phase headers are missing: {missing}.\n\n"
                f"Original user request:\n{user_prompt}\n\n"
                f"Here is the partial plan you produced (for context):\n"
                f"{plan_md[:3000]}\n\n"
                "Please produce a COMPLETE plan.md document that includes ALL phases "
                "from PHASE 0 through PHASE 7. "
                "Be specific to the project — use real screen names, entity names, "
                "and endpoint paths derived from the user's request. "
                "Output ONLY the markdown document — no preamble, no fences around the document."
            )
            plan_md = await self._call_llm(corrective)
            self._metrics["llm_calls"] += 1

            missing_still = self._missing_phases(plan_md)
            if missing_still:
                logger.warning(
                    "PlanGeneratorAgent: plan.md still missing phases %s after retry — "
                    "proceeding with partial plan",
                    missing_still,
                )

        # Write to workspace
        try:
            self.repo.write_file("plan.md", plan_md)
            logger.info("PlanGeneratorAgent: plan.md written to workspace")
        except Exception:
            logger.warning("PlanGeneratorAgent: failed to write plan.md to workspace", exc_info=True)

        return plan_md

    async def revise_plan(
        self,
        user_prompt: str,
        current_plan: str,
        feedback: str,
    ) -> str:
        """Revise plan.md based on user feedback.

        The revised plan replaces the workspace plan.md and is returned as a string.
        """
        logger.info("PlanGeneratorAgent: revising plan.md with user feedback")

        revision_prompt = (
            f"You are revising an existing plan.md document based on user feedback.\n\n"
            f"Original user request:\n{user_prompt}\n\n"
            f"Current plan.md:\n{current_plan}\n\n"
            f"User feedback — apply ONLY these changes:\n{feedback}\n\n"
            "Produce the REVISED plan.md document now. "
            "Keep everything the user did not mention — only change what they asked for. "
            "All phases (PHASE 0 through PHASE 7) must still be present and complete. "
            "Output ONLY the markdown document — no preamble, no fences around the document."
        )

        revised = await self._call_llm(revision_prompt)
        self._metrics["llm_calls"] += 1

        # Write revised plan to workspace
        try:
            self.repo.write_file("plan.md", revised)
            logger.info("PlanGeneratorAgent: revised plan.md written to workspace")
        except Exception:
            logger.warning(
                "PlanGeneratorAgent: failed to write revised plan.md to workspace",
                exc_info=True,
            )

        return revised

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_generation_prompt(user_prompt: str) -> str:
        return (
            f"User request:\n{user_prompt}\n\n"
            "Generate a complete plan.md document for this project. "
            "The document must include all seven phases (PHASE 0 through PHASE 7) "
            "as described in your system prompt. "
            "Be specific to THIS project — reference actual screen names, entity names, "
            "field names, and endpoint paths derived from the user's request. "
            "Do not write generic placeholder text. "
            "Output ONLY the markdown document — no preamble, no fences around the document."
        )

    @staticmethod
    def _missing_phases(plan_md: str) -> list[str]:
        """Return list of required phase headers not found in plan_md."""
        return [
            phase for phase in _REQUIRED_PHASES
            if not re.search(rf"#\s+{re.escape(phase)}", plan_md)
        ]
