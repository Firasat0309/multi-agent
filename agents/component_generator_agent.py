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


# ── Shared type-extraction helper ────────────────────────────────────────────

# Matches interface/type/enum/class definitions (TS, Java, Go, Rust, Python)
_TYPE_DEF = _re.compile(
    r"^\s*(?:export\s+)?(?:declare\s+)?"
    r"(?:interface|type|class|enum)\s+(\w+)"
)

# Matches type fields: name: Type; / name?: Type; / readonly name: Type
_TYPE_FIELD = _re.compile(
    r"^\s+(?:readonly\s+)?(\w+)\??\s*:\s*(.+?)\s*;?\s*$"
)


def _extract_type_definitions(content: str) -> list[str]:
    """Extract interface/type definitions with their fields from TS/JS source.

    Returns indented lines like:
        interface ButtonProps
            variant: "primary" | "secondary"
            size?: "sm" | "md" | "lg"
    """
    lines: list[str] = []
    in_type = False
    brace_depth = 0

    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        # Start of a new type definition
        m = _TYPE_DEF.match(stripped)
        if m and not in_type:
            in_type = True
            brace_depth = 0
            # Show the full definition line (e.g. "interface ButtonProps")
            header = stripped.rstrip("{").rstrip()
            lines.append(f"  {header}")
            brace_depth += stripped.count("{") - stripped.count("}")
            continue

        if in_type:
            brace_depth += stripped.count("{") - stripped.count("}")
            # Capture field definitions
            fm = _TYPE_FIELD.match(raw_line)
            if fm:
                lines.append(f"    {fm.group(1)}: {fm.group(2).rstrip(';').strip()}")
            # Type definition ended
            if brace_depth <= 0:
                in_type = False

    return lines


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



class ComponentGeneratorAgent(BaseAgent):
    """Generates production-quality source code for a single UI component.

    Uses the agentic tool-use loop (read → write) so the LLM can inspect
    adjacent already-generated files before writing its output.
    """

    role = AgentRole.COMPONENT_GENERATOR
    max_iterations: int = 10

    @property
    def tools(self) -> list[ToolDefinition]:
        return [_READ_TOOL, _WRITE_TOOL]

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior frontend engineer agent specialising in React/Next.js,\n"
            "Vue 3 (Composition API), and TypeScript.\n\n"
            "Your task is to generate a single, production-quality UI component file.\n\n"
            "WORKFLOW — WRITE IMMEDIATELY:\n"
            "The prompt already contains ALL dependency signatures, store exports,\n"
            "API contract schemas, and related file contents you need.\n"
            "Call write_file NOW with the complete component code.\n"
            "Do NOT call read_file — everything you need is already in this prompt.\n"
            "Every extra tool call wastes budget and degrades output quality.\n\n"

            "Rules:\n"
            "- Keep components focused on a single responsibility.\n"
            "- Import child components by relative path from the same src/ tree.\n"
            "- Write the full file to disk using the write_file tool.\n"
            "- Do NOT include test code in the component file.\n\n"

            "IMPORT RULES:\n"
            "- Use RELATIVE imports (e.g. '../ui/Button') — do NOT use @/ path aliases.\n"
            "- Import ONLY exports that exist in the SIGNATURES sections of the prompt.\n"
            "- NEVER invent property names — use the exact names from the signatures.\n\n"

            "TYPE SAFETY:\n"
            "- If API CONTRACT SCHEMAS are listed in the prompt, your TypeScript types\n"
            "  MUST match them exactly — same field names, same types.\n"
            "- If COMPONENT DEPENDENCY SIGNATURES list prop types (e.g. variant: \"primary\" |\n"
            "  \"secondary\"), use ONLY those values — do NOT invent new variants.\n"
            "- Match function argument counts exactly as shown in STORE SIGNATURES."
        )

    @staticmethod
    def _framework_rules(framework: str) -> str:
        """Return framework-specific generation rules for the user prompt."""
        fw = (framework or "").lower()
        if "vue" in fw:
            return (
                "FRAMEWORK: Vue 3 (Composition API)\n"
                "- Use <script setup lang='ts'> for all components (.vue SFC).\n"
                "- Use defineProps<T>() and defineEmits<T>() for typed props / events.\n"
                "- Use ref()/reactive() for local state, computed() for derived state.\n"
                "- Use Pinia stores for cross-cutting state (import from '../../store/...').\n"
                "- Use Vue Router composables (useRouter, useRoute) for navigation.\n"
                "- Use named exports for reusable composable functions.\n"
                "- File extension: .vue for components, .ts for composables/stores.\n\n"
            )
        if "angular" in fw:
            return (
                "FRAMEWORK: Angular\n"
                "- Use standalone components with @Component decorator.\n"
                "- Use TypeScript strict mode throughout.\n"
                "- Use Angular signals or RxJS for state management.\n"
                "- File extension: .component.ts for components.\n\n"
            )
        # React / Next.js (default)
        rules = (
            "FRAMEWORK: React + TypeScript\n"
            "- Use TypeScript throughout (.tsx files).\n"
            "- Use named exports for reusable components and types.\n"
            "- Use Zustand/Redux for cross-cutting state; useState for local state.\n"
        )
        if "next" in fw or fw == "":
            rules += (
                "\nNEXT.JS APP ROUTER:\n"
                "- Files in src/app/ or app/ (page.tsx, layout.tsx) MUST use 'export default function'.\n"
                "- Components using React hooks, event handlers, useRouter, usePathname, or\n"
                '  browser APIs MUST have "use client"; as the VERY FIRST LINE (before imports).\n'
                "- If in doubt, add 'use client'; — it's acceptable; missing it is a build error.\n"
            )
        return rules + "\n"

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
        if contract and component:
            import json as _json

            # ── Endpoint details for components with API calls ────────────
            if component.api_calls and contract.endpoints:
                # Match endpoints where the call path is a prefix of the endpoint
                # path or vice-versa, handling parameterised segments correctly.
                def _paths_match(call: str, ep_path: str) -> bool:
                    call_stripped = call.strip("/").split("?")[0]
                    ep_stripped = ep_path.strip("/")
                    return (
                        call_stripped == ep_stripped
                        or ep_stripped.startswith(call_stripped + "/")
                        or call_stripped.startswith(ep_stripped + "/")
                    )

                relevant = [
                    ep for ep in contract.endpoints
                    if any(_paths_match(call, ep.path) for call in component.api_calls)
                ]
                if relevant:
                    endpoint_lines: list[str] = []
                    for ep in relevant:
                        line = f"  {ep.method} {ep.path}: {ep.description}"
                        if ep.request_schema:
                            try:
                                line += f"\n    Request: {_json.dumps(ep.request_schema)}"
                            except Exception:
                                pass
                        if ep.response_schema:
                            try:
                                line += f"\n    Response: {_json.dumps(ep.response_schema)}"
                            except Exception:
                                pass
                        endpoint_lines.append(line)
                    contract_text = "Relevant API endpoints:\n" + "\n".join(endpoint_lines) + "\n\n"

            # ── Schema definitions — always inject when available ──────────
            # Store files and components that consume API data must define
            # TypeScript types that EXACTLY match the contract schemas.
            # Without this, the LLM invents fields (e.g. AuthResponse.user)
            # that don't exist in the actual API response.
            if contract.schemas:
                # Filter schemas relevant to this component's API calls or
                # state needs; fall back to all schemas for store/lib files.
                is_store = any(
                    seg in (component.file_path or "")
                    for seg in ("store", "lib", "hooks")
                )
                relevant_schemas = contract.schemas
                if not is_store and component.api_calls:
                    # Try to narrow to schemas referenced by matched endpoints
                    ep_text = contract_text.lower()
                    filtered = {
                        k: v for k, v in contract.schemas.items()
                        if k.lower() in ep_text
                    }
                    if filtered:
                        relevant_schemas = filtered

                schema_lines: list[str] = [
                    "API CONTRACT SCHEMAS — your TypeScript types MUST match these EXACTLY.",
                    "Do NOT add, remove, or rename fields. If you define an interface or type",
                    "for an API response, it must have ONLY the fields listed here:\n",
                ]
                for name, definition in relevant_schemas.items():
                    schema_lines.append(f"  {name}:")
                    try:
                        schema_lines.append(f"    {_json.dumps(definition, indent=4)}")
                    except Exception:
                        schema_lines.append(f"    {definition}")
                schema_lines.append("")
                contract_text += "\n".join(schema_lines) + "\n"

        # Build dependency import guidance
        dep_instructions = ""
        if component and component.depends_on:
            # Check which deps are already pre-loaded in related_files.
            # Match on filename stem to avoid false positives (e.g. "Card"
            # matching "useDiscardStore.ts" because "card" is in "discard").
            preloaded_deps = set()
            missing_deps = []
            related_stems = {
                p.rsplit("/", 1)[-1].split(".")[0].lower(): p
                for p in (context.related_files or {})
            }
            for dep_name in component.depends_on:
                found = (
                    dep_name.lower() in related_stems
                    or any(dep_name.lower() == stem for stem in related_stems)
                    or any(dep_name.lower() in p.lower().rsplit("/", 1)[-1]
                           for p in (context.related_files or {}))
                )
                if found:
                    preloaded_deps.add(dep_name)
                else:
                    missing_deps.append(dep_name)

            dep_instructions = (
                "\nDEPENDENCY RULES:\n"
                "- Only import exports that exist in the SIGNATURES sections above.\n"
                "- Do NOT assume a component exports sub-components (e.g. Card does NOT\n"
                "  export CardHeader/CardTitle/CardContent unless shown in signatures).\n"
            )
            if preloaded_deps:
                dep_instructions += (
                    f"- Dependencies already in prompt (do NOT read_file): "
                    f"{', '.join(sorted(preloaded_deps))}\n"
                )
            if missing_deps:
                dep_instructions += (
                    f"- Use read_file ONLY for these missing deps: {', '.join(missing_deps)}\n"
                )
            dep_instructions += "\n"

        # Store/lib usage guidance — files are pre-loaded by _inject_store_context
        store_discovery = ""
        if component and (component.state_needs or component.api_calls):
            preloaded = [
                p for p in (context.related_files or {})
                if any(seg in p for seg in ("store", "lib", "hooks"))
            ]
            if preloaded:
                store_discovery = (
                    "\nSTORE/LIB FILES (already pre-loaded — do NOT read_file for these):\n"
                    f"Files: {', '.join(preloaded)}\n"
                    "- Use EXACT file names in import paths (e.g. '../../store/useAuthStore').\n"
                    "- Use ONLY properties/methods shown in STORE SIGNATURES above.\n\n"
                )
            # No else branch — _inject_store_context always pre-loads these.
            # If somehow missing, the LLM still has signatures from the prompt.

        # Extract and prominently display function signatures from pre-loaded
        # store/lib/hook files so the LLM sees exact parameter counts.
        store_sigs = self._extract_store_signatures(context.related_files or {})
        # Extract prop types from component dependencies (Button, Card, etc.)
        dep_sigs = self._extract_dep_signatures(context.related_files or {})

        # Figma instruction — only when a node ID is actually present
        figma_hint = ""
        if component and getattr(component, "figma_node_id", None):
            figma_hint = (
                "A Figma Node ID is provided. Use your tools to fetch the structural "
                "code skeleton, then hydrate it with the described API and state handlers. "
            )

        # Framework-specific rules injected into the user prompt
        fw_rules = self._framework_rules(plan.framework if plan else "")

        return (
            f"{req_text}{comp_text}\n{plan_text}\n{design_text}\n{contract_text}\n"
            f"{fw_rules}"
            f"{store_sigs}{dep_sigs}"
            f"{store_discovery}{dep_instructions}"
            f"{figma_hint}"
            "Generate the complete component source code and write it to disk using write_file."
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
        dep_sigs = self._extract_dep_signatures(context.related_files or {})

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
        contract: APIContract | None = context.task.metadata.get("api_contract")
        plan_info = ""
        if plan:
            plan_info = (
                f"Framework: {plan.framework}\n"
                f"State solution: {plan.state_solution}\n\n"
            )

        # Inject API contract schemas so the fix agent knows the exact
        # fields each API response type should have.
        schema_text = ""
        if contract and contract.schemas:
            import json as _json
            schema_lines = [
                "API CONTRACT SCHEMAS — your TypeScript types MUST match these EXACTLY.",
                "Do NOT add, remove, or rename fields:\n",
            ]
            for name, definition in contract.schemas.items():
                schema_lines.append(f"  {name}:")
                try:
                    schema_lines.append(f"    {_json.dumps(definition, indent=4)}")
                except Exception:
                    schema_lines.append(f"    {definition}")
            schema_lines.append("")
            schema_text = "\n".join(schema_lines) + "\n"

        prompt = (
            f"{comp_info}{plan_info}"
            f"{store_sigs}{dep_sigs}"
            f"{schema_text}"
            f"{related_section}"
            f"FIX TASK for: {file_path}\n\n"
            f"Current content:\n```typescript\n{current_content}\n```\n\n"
            f"Build errors to fix ({fix_trigger}):\n{build_errors}\n\n"
            "INSTRUCTIONS:\n"
            "1. Fix ONLY the specific errors listed above.\n"
            "2. Do NOT remove, rename, or reorganise existing functions or components.\n"
            "3. Do NOT add new functionality beyond what is needed to fix errors.\n"
            "4. Do NOT change the component signature (props) unless the error requires it.\n"
            "5. Check the STORE / LIB SIGNATURES and COMPONENT DEPENDENCY SIGNATURES above\n"
            "   for exact function names, parameter counts, prop types, and allowed values.\n"
            "   Match them EXACTLY — do not invent prop values that are not listed.\n"
            "6. If the error says 'Expected N arguments, but got M', check the signatures\n"
            "   above and fix the call to pass exactly the right number of arguments.\n"
            "7. Import paths must be RELATIVE (e.g. '../../store/useAuthStore').\n"
            "8. Preserve 'use client'; directive at the top if present.\n"
            "9. Output the COMPLETE corrected file — every line, no markdown fences.\n"
        )

        fixed_code = await self._call_llm(prompt, system_override=self.system_prompt)
        fixed_code = self._strip_fences(fixed_code)

        # ── Validate ──────────────────────────────────────────────────────
        # Reject empty/minimal output — the LLM may have returned only an
        # explanation or an incomplete fragment instead of the full file.
        _min_size = min(50, len(current_content) // 2) if current_content else 50
        if len(fixed_code.strip()) < _min_size:
            logger.warning(
                "Component fix rejected for %s: output too small (%d chars, "
                "min %d) — LLM likely returned explanation instead of code",
                file_path, len(fixed_code.strip()), _min_size,
            )
            return TaskResult(
                success=True,
                output=f"Fix for {file_path} skipped — LLM output too small",
                files_modified=[],
                metrics={"rewrite_rejected": True},
            )

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
        """Extract exported function/const signatures AND type definitions
        from pre-loaded store/lib/hook files.

        Returns a prominent block listing each function with its exact
        parameter list AND each type/interface with its fields, so the LLM
        can match argument counts and field names precisely.
        """
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

            # ── Type / interface definitions with fields ──────────────────
            type_lines = _extract_type_definitions(content)
            if type_lines:
                sigs.extend(type_lines)

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
            "STORE / LIB SIGNATURES — use these EXACT names, argument counts, and types:\n"
            + "\n".join(sections)
            + "\n\n"
        )

    @staticmethod
    def _extract_dep_signatures(related_files: dict[str, str]) -> str:
        """Extract type/interface definitions AND prop types from component
        dependencies (Button, Card, etc.) so the LLM sees the EXACT allowed
        prop values.

        This prevents type mismatches like using variant="outline" when
        ButtonProps only allows "primary" | "secondary".
        """
        # Only process component files (not store/lib which are handled separately)
        comp_files = {
            p: c for p, c in related_files.items()
            if not any(seg in p for seg in ("store", "lib", "hooks", "hook"))
        }
        if not comp_files:
            return ""

        sections: list[str] = []
        for fpath, content in comp_files.items():
            lines: list[str] = []

            # Type/interface definitions with fields
            type_lines = _extract_type_definitions(content)
            if type_lines:
                lines.extend(type_lines)

            # Exported function/component signatures
            for raw_line in content.splitlines():
                stripped = raw_line.strip()
                if not stripped or stripped in ("{", "}", "};"):
                    continue
                # export function ComponentName(props: Props)
                # export default function ComponentName(...)
                # export const ComponentName = ...
                if _re.match(
                    r"^\s*(?:export\s+(?:default\s+)?)"
                    r"(?:(?:async\s+)?function\s+\w+\s*\(|const\s+\w+)",
                    stripped,
                ):
                    sig = stripped.rstrip("{").rstrip().rstrip(";").strip()
                    lines.append(f"  {sig}")

            if lines:
                sections.append(f"  {fpath}:\n" + "\n".join(lines))

        if not sections:
            return ""

        return (
            "COMPONENT DEPENDENCY SIGNATURES — use ONLY these exact prop types and values:\n"
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
                for ext in (".ts", ".tsx", ".js", ".jsx", ".vue"):
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
