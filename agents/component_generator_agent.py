"""Component Generator Agent — generates and fixes source code for a single UI component."""

from __future__ import annotations

import logging
import re as _re

from agents.base_agent import BaseAgent
from core.agent_tools import ToolDefinition
from core.models import (
    AgentContext,
    AgentRole,
    APIContract,
    ComponentPlan,
    ProductRequirements,
    TaskResult,
    TaskType,
    UIComponent,
    UIDesignSpec,
)

logger = logging.getLogger(__name__)

_WRITE_TOOL = ToolDefinition(
    name="write_file",
    description="Write the generated component source code to disk.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to workspace"},
            "content": {"type": "string", "description": "Complete file content"},
        },
        "required": ["path", "content"],
    },
)

_READ_TOOL = ToolDefinition(
    name="read_file",
    description="Read an existing file for context.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        },
        "required": ["path"],
    },
)

_LIST_TOOL = ToolDefinition(
    name="list_files",
    description="List files in a directory to discover store/lib file names.",
    input_schema={
        "type": "object",
        "properties": {
            "directory": {"type": "string", "description": "Directory path relative to workspace"},
            "pattern": {"type": "string", "description": "Glob pattern (default: **)"},
        },
        "required": [],
    },
)


class ComponentGeneratorAgent(BaseAgent):
    """Generates production-quality source code for a single UI component.

    Uses the agentic tool-use loop (read → write) so the LLM can inspect
    adjacent already-generated files before writing its output.
    """

    role = AgentRole.COMPONENT_GENERATOR
    max_iterations: int = 20

    @property
    def tools(self) -> list[ToolDefinition]:
        return [_READ_TOOL, _WRITE_TOOL, _LIST_TOOL]

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior frontend engineer agent specialising in React/Next.js,\n"
            "Vue 3 (Composition API), and TypeScript.\n\n"
            "Your task is to generate a single, production-quality UI component file.\n\n"
            "Rules:\n"
            "- Use TypeScript throughout (.tsx for React, <script setup lang='ts'> for Vue).\n"
            "- If the component has a `figma_node_id`, use your MCP tools to fetch its code skeleton FIRST.\n"
            "- Keep components focused on a single responsibility.\n"
            "- Use named exports for reusable components and types.\n"
            "- Import child components by relative path from the same src/ tree.\n"
            "- Use the state store (Zustand/Redux/Pinia) ONLY for cross-cutting state;\n"
            "  use local useState/ref for component-local state.\n"
            "- Wrap async data fetching in a custom hook or SWR/React Query.\n"
            "- Include JSDoc comment block at the top of the component.\n"
            "- Write the full file to disk using the write_file tool.\n"
            "- Do NOT include test code in the component file.\n\n"

            "NEXT.JS APP ROUTER — CRITICAL RULES:\n"
            "- Files in src/app/**/page.tsx MUST use 'export default function'. Next.js\n"
            "  pages REQUIRE a default export. Named exports like 'export function DashboardPage'\n"
            "  will cause a build error. Use: export default function DashboardPage() { ... }\n"
            "- Similarly, src/app/**/layout.tsx MUST use 'export default function'.\n"
            "- Any component that uses React hooks (useState, useEffect, useContext, useRef,\n"
            "  useCallback, useMemo, useReducer), event handlers (onClick, onChange, onSubmit),\n"
            "  useRouter, usePathname, useSearchParams, or any browser-only API MUST have\n"
            '  "use client"; as the VERY FIRST LINE of the file (before all imports).\n'
            "- Page components in src/app/ that only render other components and do NOT use\n"
            "  hooks or event handlers directly should be Server Components (no 'use client').\n"
            "- If in doubt, add 'use client'; — a client component that could be a server\n"
            "  component is acceptable; a server component that uses hooks is a build error.\n\n"

            "IMPORT RULES:\n"
            "- Use RELATIVE imports (e.g. '../ui/Button') for importing from other components\n"
            "  within the same src/ tree. Do NOT use @/ path aliases.\n"
            "- For API client imports, use relative paths (e.g. '../../lib/api').\n"
            "- This ensures all imports resolve correctly without tsconfig path aliases.\n\n"

            "STORE / LIB FILE IMPORTS — MANDATORY STEPS (DO NOT SKIP):\n"
            "If the component has state_needs or api_calls, you MUST do the following\n"
            "BEFORE calling write_file:\n"
            "1. Call list_files on the relevant directories (e.g. 'src/store', 'src/lib',\n"
            "   'src/hooks') to discover the EXACT file names that exist on disk.\n"
            "2. Call read_file on each discovered store/hook/lib file to see its ACTUAL\n"
            "   exported types, functions, and properties.\n"
            "3. In your generated code, import ONLY the exact module path found in step 1\n"
            "   (e.g. if the file is 'useAuthStore.ts', import from '../../store/useAuthStore'\n"
            "    — NOT '../../store/authStore').\n"
            "4. Use ONLY the property/method names that actually exist in the file's exports\n"
            "   as found in step 2.  NEVER invent or guess property names like 'checkAuth'\n"
            "   or 'isLoggedIn' — if the store exports 'isAuthenticated', use that exact name.\n"
            "5. If no store files exist yet, create a minimal inline implementation or skip\n"
            "   the store import entirely — do NOT guess file names.\n\n"

            "FUNCTION CALL RULES — MATCH ARGUMENT COUNT EXACTLY:\n"
            "- Before calling any imported function, check its EXACT signature (parameter count and types).\n"
            "- If a store function is defined as `login(token: string)` (1 parameter), you MUST call\n"
            "  it with exactly 1 argument: `login(token)`. Do NOT call `login(user, token)`.\n"
            "- If a function takes `(data: { user: User; token: string })`, pass a single object.\n"
            "- 'Expected N arguments, but got M' errors are caused by mismatched argument counts.\n"
            "- When STORE FUNCTION SIGNATURES are listed in the prompt, those are the authoritative\n"
            "  source of truth for argument counts. Match them exactly."
        )

    def _build_prompt(self, context: AgentContext) -> str:
        component: UIComponent | None = context.task.metadata.get("component")
        plan: ComponentPlan | None = context.task.metadata.get("component_plan")
        contract: APIContract | None = context.task.metadata.get("api_contract")
        design_spec: UIDesignSpec | None = context.task.metadata.get("design_spec")
        requirements: ProductRequirements | None = context.task.metadata.get("requirements")

        comp_text = ""
        if component:
            comp_text = (
                f"Component: {component.name}\n"
                f"File path: {component.file_path}\n"
                f"Type: {component.component_type}\n"
                f"Description: {component.description}\n"
                f"Figma Node ID: {getattr(component, 'figma_node_id', 'None')}\n"
                f"Props: {component.props}\n"
                f"State needs: {component.state_needs}\n"
                f"API calls: {component.api_calls}\n"
                f"Depends on: {component.depends_on}\n"
                f"Children: {component.children}\n"
            )

        plan_text = ""
        if plan:
            plan_text = (
                f"Framework: {plan.framework}\n"
                f"State solution: {plan.state_solution}\n"
                f"API base URL: {plan.api_base_url}\n"
                f"Routing: {plan.routing_solution}\n"
            )

        # Design tokens for consistent styling
        design_text = ""
        if design_spec:
            if design_spec.global_styles:
                styles = ", ".join(f"{k}: {v}" for k, v in design_spec.global_styles.items())
                design_text += f"Global styles: {styles}\n"
            if design_spec.design_tokens:
                import json as _json
                tokens_str = _json.dumps(design_spec.design_tokens, indent=2)
                if len(tokens_str) > 1000:
                    tokens_str = tokens_str[:1000] + "..."
                design_text += f"Design tokens:\n{tokens_str}\n"

        # Business context from product requirements
        req_text = ""
        if requirements:
            req_text = f"Product: {requirements.title}\n"
            if requirements.features:
                req_text += f"Features: {', '.join(requirements.features[:5])}\n"

        contract_text = ""
        if contract and component and component.api_calls:
            relevant = [
                ep for ep in contract.endpoints
                if any(call in ep.path for call in component.api_calls)
            ]
            if relevant:
                endpoint_lines = "\n".join(
                    f"  {ep.method} {ep.path}: {ep.description}"
                    for ep in relevant
                )
                contract_text = f"Relevant API endpoints:\n{endpoint_lines}\n"

        # Build dependency reading instructions
        dep_instructions = ""
        if component and component.depends_on:
            dep_instructions = (
                "\nBEFORE generating code:\n"
                f"1. Use read_file to read each dependency: {', '.join(component.depends_on)}\n"
                "2. Check what each dependency ACTUALLY exports (exact names)\n"
                "3. Only import exports that actually exist in the file\n"
                "4. Do NOT assume a component exports sub-components (e.g. Card does NOT\n"
                "   export CardHeader/CardTitle/CardContent unless you verify it does)\n\n"
            )

        # Force store/lib discovery when the component uses shared state or APIs
        store_discovery = ""
        if component and (component.state_needs or component.api_calls):
            # Check if store/lib files were pre-loaded into related_files
            preloaded = [
                p for p in (context.related_files or {})
                if any(seg in p for seg in ("store", "lib", "hooks"))
            ]
            if preloaded:
                store_discovery = (
                    "\n⚠️  IMPORTANT — The following store/lib/hook files have been pre-loaded\n"
                    "into the Related Files section below. You MUST:\n"
                    "1. Check the EXACT file names and export names from these files.\n"
                    "2. Use the EXACT file name in your import path (e.g. if the file is\n"
                    "   'src/store/useAuthStore.ts', import from '../../store/useAuthStore').\n"
                    "3. Use ONLY properties/methods that actually exist in the exports.\n"
                    "   NEVER invent property names.\n"
                    f"Pre-loaded files: {', '.join(preloaded)}\n\n"
                )
            else:
                dirs_to_scan = []
                if component.state_needs:
                    dirs_to_scan.append("src/store")
                if component.api_calls:
                    dirs_to_scan.extend(["src/lib", "src/hooks"])
                store_discovery = (
                    "\n⚠️  MANDATORY — Do these steps FIRST, before writing any code:\n"
                    f"1. Call list_files for each directory: {', '.join(dirs_to_scan)}\n"
                    "2. Call read_file on every store/hook/lib file found to learn the\n"
                    "   exact export names and types.\n"
                    "3. Use ONLY the real file names and export names in your imports.\n"
                    "   Do NOT guess — if a store file is named useAuthStore.ts, import\n"
                    "   from '../../store/useAuthStore', not '../../store/authStore'.\n\n"
                )

        # Extract and prominently display function signatures from pre-loaded
        # store/lib/hook files so the LLM sees exact parameter counts.
        store_sigs = self._extract_store_signatures(context.related_files or {})

        return (
            f"{req_text}{comp_text}\n{plan_text}\n{design_text}\n{contract_text}\n"
            f"{store_sigs}"
            f"{store_discovery}{dep_instructions}"
            "If a Figma Node ID is provided, use your tools to fetch the structural code skeleton FIRST. "
            "Hydrate the skeleton with the described API and state handlers, then generate "
            "the complete component source code and write it to disk using write_file."
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        # ── Fix path: reuse component context for targeted fixes ──────────
        if context.task.task_type in (TaskType.FIX_CODE, TaskType.FIX_COMPONENT):
            return await self._fix_component(context)

        component: UIComponent | None = context.task.metadata.get("component")
        if component is None:
            return TaskResult(success=False, errors=["No component in task metadata"])

        # Inject the target file path into file_blueprint so the base agent's
        # agentic loop can detect when the file has been written and exit early
        # instead of looping until max_iterations rewriting the same file.
        if context.file_blueprint is None and component.file_path:
            from core.models import FileBlueprint
            context = AgentContext(
                task=context.task,
                blueprint=context.blueprint,
                file_blueprint=FileBlueprint(
                    path=component.file_path,
                    purpose=component.description or component.name,
                    depends_on=component.depends_on or [],
                ),
                related_files=context.related_files,
                architecture_summary=context.architecture_summary,
                dependency_info=context.dependency_info,
            )

        # Pre-read store/lib/hook files so the LLM has real exports in the
        # prompt, preventing it from guessing wrong file names or properties.
        context = await self._inject_store_context(context, component)

        try:
            result = await self.execute_agentic(context)
            if result.success:
                logger.info("ComponentGeneratorAgent: generated %s", component.file_path)
            return result
        except Exception as exc:
            logger.exception("ComponentGeneratorAgent.execute failed for %s", component.name)
            return TaskResult(success=False, errors=[str(exc)])

    # ── Fix path ──────────────────────────────────────────────────────────────

    # Maximum allowed content growth factor for fix rewrites.
    _MAX_CONTENT_GROWTH = 1.35

    async def _fix_component(self, context: AgentContext) -> TaskResult:
        """Fix a component file using component-aware context.

        Unlike ``CoderAgent._fix_code`` this method pre-loads store/lib
        files and injects their function signatures so the LLM has the
        exact parameter counts, exports, and type names needed to fix
        cross-file type errors correctly.
        """
        file_path = context.task.file
        build_errors: str = context.task.metadata.get("build_errors", "")
        fix_trigger: str = context.task.metadata.get("fix_trigger", "build")
        component: UIComponent | None = context.task.metadata.get("component")

        current_content = await self.repo.async_read_file(file_path) or ""
        if not current_content:
            logger.warning("Cannot fix %s — file has no content", file_path)
            return TaskResult(success=False, errors=[f"File {file_path} is empty"])

        # ── Pre-load store/lib context ────────────────────────────────────
        if component:
            context = await self._inject_store_context(context, component)

        store_sigs = self._extract_store_signatures(context.related_files or {})

        # ── Build related-file stubs section ──────────────────────────────
        related_section = ""
        if context.related_files:
            parts: list[str] = []
            for dep_path, dep_content in context.related_files.items():
                if dep_path == file_path:
                    continue
                trimmed = dep_content[:2000]
                parts.append(f"### {dep_path}\n```typescript\n{trimmed}\n```")
            if parts:
                related_section = "## Related Files (imports / dependencies)\n" + "\n".join(parts) + "\n\n"

        # ── Cap error text ────────────────────────────────────────────────
        max_error_chars = 3000
        if len(build_errors) > max_error_chars:
            build_errors = build_errors[:max_error_chars] + "\n... (truncated)"

        # ── Component metadata ────────────────────────────────────────────
        comp_info = ""
        if component:
            comp_info = (
                f"Component: {component.name}\n"
                f"Type: {component.component_type}\n"
                f"Props: {component.props}\n"
                f"State needs: {component.state_needs}\n"
                f"API calls: {component.api_calls}\n\n"
            )

        plan: ComponentPlan | None = context.task.metadata.get("component_plan")
        plan_info = ""
        if plan:
            plan_info = (
                f"Framework: {plan.framework}\n"
                f"State solution: {plan.state_solution}\n\n"
            )

        prompt = (
            f"{comp_info}{plan_info}"
            f"{store_sigs}"
            f"{related_section}"
            f"FIX TASK for: {file_path}\n\n"
            f"Current content:\n```typescript\n{current_content}\n```\n\n"
            f"Build errors to fix ({fix_trigger}):\n{build_errors}\n\n"
            "INSTRUCTIONS:\n"
            "1. Fix ONLY the specific errors listed above.\n"
            "2. Do NOT remove, rename, or reorganise existing functions or components.\n"
            "3. Do NOT add new functionality beyond what is needed to fix errors.\n"
            "4. Do NOT change the component signature (props) unless the error requires it.\n"
            "5. Check the STORE / LIB FUNCTION SIGNATURES above for exact function names,\n"
            "   parameter counts, and types. Match them EXACTLY.\n"
            "6. If the error says 'Expected N arguments, but got M', check the signatures\n"
            "   above and fix the call to pass exactly the right number of arguments.\n"
            "7. Import paths must be RELATIVE (e.g. '../../store/useAuthStore').\n"
            "8. Preserve 'use client'; directive at the top if present.\n"
            "9. Output the COMPLETE corrected file — every line, no markdown fences.\n"
        )

        fixed_code = await self._call_llm(prompt, system_override=self.system_prompt)
        fixed_code = self._strip_fences(fixed_code)

        # ── Validate ──────────────────────────────────────────────────────
        original_size = len(current_content)
        if original_size > 0 and len(fixed_code) > original_size * self._MAX_CONTENT_GROWTH:
            logger.warning(
                "Component fix rejected for %s: new size (%d) exceeds %.0f%% of original (%d)",
                file_path, len(fixed_code), self._MAX_CONTENT_GROWTH * 100, original_size,
            )
            return TaskResult(
                success=True,
                output=f"Fix for {file_path} skipped — rewrite too large (likely duplicated content)",
                files_modified=[],
                metrics={"rewrite_rejected": True},
            )

        # Detect duplicate top-level definitions
        _TS_DEFN = _re.compile(
            r"^(?:export\s+)?(?:class|interface|enum|function|const|type)\s+(\w+)",
            _re.MULTILINE,
        )
        names = _TS_DEFN.findall(fixed_code)
        if len(names) != len(set(names)):
            logger.warning(
                "Component fix rejected for %s: duplicate definitions detected", file_path,
            )
            return TaskResult(
                success=True,
                output=f"Fix for {file_path} skipped — duplicate definitions in LLM output",
                files_modified=[],
                metrics={"rewrite_rejected": True},
            )

        if fixed_code.rstrip() == current_content.rstrip():
            logger.info("Component fix for %s produced identical content — skipping", file_path)
            return TaskResult(
                success=True,
                output=f"No changes needed for {file_path}",
                files_modified=[],
            )

        await self.repo.async_write_file(file_path, fixed_code)
        logger.info("ComponentGeneratorAgent: fixed %s", file_path)

        return TaskResult(
            success=True,
            output=f"Fixed {file_path}",
            files_modified=[file_path],
            metrics=self.get_metrics(),
        )

    @staticmethod
    def _strip_fences(content: str) -> str:
        """Remove markdown code fences the LLM may have wrapped the output in."""
        content = content.strip()
        for fence in ("```typescript", "```tsx", "```"):
            if content.startswith(fence):
                content = content[len(fence):]
                break
        if content.endswith("```"):
            content = content[:-3]
        return content.strip() + "\n"

    @staticmethod
    def _extract_store_signatures(related_files: dict[str, str]) -> str:
        """Extract exported function/const signatures from pre-loaded store/lib/hook files.

        Returns a prominent block listing each function with its exact parameter
        list so the LLM can match argument counts precisely.
        """
        import re as _re

        # Match exported functions, arrow consts, and zustand store actions
        _EXPORT_FN = _re.compile(
            r"^\s*(?:export\s+)"
            r"(?:(?:async\s+)?function\s+(\w+)\s*(\([^)]*\))"  # export function name(params)
            r"|(?:const|let)\s+(\w+)\s*=\s*(?:async\s+)?\(?([^)=]*?)\)?)"  # export const name = (...) =>
            , _re.MULTILINE,
        )
        # Match zustand/pinia store action methods:  actionName: (params) => ...
        # or  actionName(params) { ... }
        _STORE_ACTION = _re.compile(
            r"^\s+(\w+)\s*[:=]\s*(?:async\s+)?\(?([^)]*?)\)?\s*(?:=>|\{)"
            , _re.MULTILINE,
        )
        # Match interface/type members:  methodName(params): ReturnType
        _IFACE_MEMBER = _re.compile(
            r"^\s+(\w+)\s*(\([^)]*\))\s*:\s*([^;]+)"
            , _re.MULTILINE,
        )

        store_lib_files = {
            p: c for p, c in related_files.items()
            if any(seg in p for seg in ("store", "lib", "hooks", "hook"))
        }
        if not store_lib_files:
            return ""

        sections: list[str] = []
        for fpath, content in store_lib_files.items():
            sigs: list[str] = []

            # Exported functions / consts
            for m in _EXPORT_FN.finditer(content):
                fn_name = m.group(1) or m.group(3)
                params = m.group(2) or m.group(4) or ""
                params = params.strip()
                if fn_name:
                    sigs.append(f"  {fn_name}({params})")

            # Interface / type members (for typed stores)
            for m in _IFACE_MEMBER.finditer(content):
                name, params, ret = m.group(1), m.group(2), m.group(3).strip()
                sigs.append(f"  {name}{params}: {ret}")

            # Store actions (zustand `set =>` pattern)
            for m in _STORE_ACTION.finditer(content):
                name, params = m.group(1), m.group(2).strip()
                # Skip if already captured as export or interface member
                if not any(name in s for s in sigs):
                    sigs.append(f"  {name}({params})")

            if sigs:
                sections.append(f"  {fpath}:\n" + "\n".join(sigs))

        if not sections:
            return ""

        return (
            "STORE / LIB FUNCTION SIGNATURES — use these EXACT names and argument counts:\n"
            + "\n".join(sections)
            + "\n\n"
        )

    async def _inject_store_context(
        self, context: AgentContext, component: UIComponent
    ) -> AgentContext:
        """Pre-read store/lib/hook files into ``related_files`` so the LLM
        can see actual exports without needing extra tool-call iterations."""
        dirs_to_scan: list[str] = []
        if component.state_needs:
            dirs_to_scan.append("store")
        if component.api_calls:
            dirs_to_scan.extend(["lib", "hooks"])
        if not dirs_to_scan:
            return context

        from pathlib import Path as _Path

        extra_files: dict[str, str] = {}
        workspace = self.repo.workspace.resolve()

        for sub in dirs_to_scan:
            # Try common frontend source layouts: src/<sub> and <sub>
            for prefix in ("src", ""):
                scan_dir = workspace / prefix / sub if prefix else workspace / sub
                if not scan_dir.is_dir():
                    continue
                for ext in (".ts", ".tsx", ".js", ".jsx"):
                    for fpath in scan_dir.rglob(f"*{ext}"):
                        if not fpath.is_file():
                            continue
                        rel = str(fpath.relative_to(workspace)).replace("\\", "/")
                        if rel in extra_files:
                            continue
                        try:
                            content = await self.repo.async_read_file(rel)
                            if content is not None:
                                extra_files[rel] = content
                        except Exception:
                            pass  # non-critical — LLM still has tool fallback

        if not extra_files:
            return context

        merged = dict(context.related_files) if context.related_files else {}
        merged.update(extra_files)
        logger.info(
            "ComponentGeneratorAgent: pre-loaded %d store/lib/hook file(s) for %s",
            len(extra_files), component.name,
        )
        return AgentContext(
            task=context.task,
            blueprint=context.blueprint,
            file_blueprint=context.file_blueprint,
            related_files=merged,
            architecture_summary=context.architecture_summary,
            dependency_info=context.dependency_info,
        )
