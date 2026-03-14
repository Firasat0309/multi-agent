"""Repository Analyzer agent - scans existing repositories to build understanding."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from core.language import detect_language_from_extensions, LanguageProfile
from core.models import (
    AgentContext,
    AgentRole,
    FileIndex,
    ModuleInfo,
    RepoAnalysis,
    TaskResult,
)

logger = logging.getLogger(__name__)

# Common entry-point patterns by language
_ENTRY_POINT_PATTERNS: dict[str, list[str]] = {
    "python": [r"if\s+__name__\s*==\s*['\"]__main__['\"]", r"app\s*=\s*Flask|FastAPI|Django"],
    "java": [r"public\s+static\s+void\s+main\s*\(", r"@SpringBootApplication"],
    "go": [r"func\s+main\s*\(\s*\)"],
    "typescript": [r"app\.listen\(", r"createServer\(", r"express\(\)"],
    "rust": [r"fn\s+main\s*\(\s*\)"],
    "csharp": [r"static\s+void\s+Main\s*\(", r"WebApplication\.CreateBuilder"],
}

# Heuristics for detecting layer from file path and content
_LAYER_HINTS: list[tuple[str, str]] = [
    (r"controller|handler|endpoint|route|view", "controller"),
    (r"service|usecase|interactor|business", "service"),
    (r"repositor|dao|store|persistence|gateway", "repository"),
    (r"model|entity|schema|domain|dto", "model"),
    (r"middleware|interceptor|filter", "middleware"),
    (r"config|setting|constant|env", "config"),
    (r"test|spec|_test|test_", "test"),
    (r"util|helper|common|shared|lib", "util"),
    (r"migration|seed|fixture", "migration"),
]


def _detect_layer(file_path: str, content: str) -> str:
    """Detect the architectural layer of a file from its path and content."""
    path_lower = file_path.lower()
    first_300 = content[:300].lower() if content else ""
    combined = f"{path_lower} {first_300}"
    for pattern, layer in _LAYER_HINTS:
        if re.search(pattern, combined):
            return layer
    return "unknown"


class RepositoryAnalyzerAgent(BaseAgent):
    """Scans an existing repository to build a comprehensive understanding.

    Responsibilities:
    - Scan all source files and build repo index
    - Build the dependency graph
    - Detect tech stack, architecture style, entry points
    - Summarize modules and their responsibilities
    """

    role = AgentRole.REPO_ANALYZER

    @property
    def system_prompt(self) -> str:
        return (
            "You are a repository analysis agent. You examine existing codebases "
            "and produce structured summaries of their architecture, modules, "
            "and dependencies.\n\n"
            "When given a list of files with their contents, you must:\n"
            "1. Identify the tech stack (language, framework, database)\n"
            "2. Determine the architecture style\n"
            "3. Summarize the system's purpose and structure\n"
            "4. Identify entry points and main workflows\n\n"
            "Respond with valid JSON only. No markdown fences or explanations.\n\n"
            "VALUE CONSTRAINTS:\n"
            "- language MUST be one of: python, java, go, typescript, rust, csharp (lowercase)\n"
            "- framework must be a specific lowercase name (e.g., 'spring-boot', 'fastapi', "
            "'gin', 'express', 'actix-web', 'asp.net')\n"
            "- architecture_style MUST be one of: REST, GraphQL, gRPC, monolith, "
            "microservices, event-driven\n"
            "- If you cannot determine a value, use an empty string — do NOT guess"
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        """Analyse the workspace bound to this agent's repo_manager."""
        try:
            analysis = await self.analyze_repository(self.repo.workspace)
            return TaskResult(
                success=True,
                output=(
                    f"Repository analysed: {len(analysis.modules)} module(s), "
                    f"style={analysis.architecture_style!r}, "
                    f"stack={analysis.tech_stack}"
                ),
                metrics=self.get_metrics(),
            )
        except Exception as exc:
            logger.exception("RepositoryAnalyzerAgent.execute failed")
            return TaskResult(success=False, errors=[str(exc)])

    async def analyze_repository(self, workspace_dir: Path) -> RepoAnalysis:
        """Scan an existing repository and produce a complete analysis.

        This performs a two-pass analysis:
        1. Local pass: scan files, extract symbols, build index (no LLM needed)
        2. LLM pass: summarize architecture and module purposes
        """
        logger.info("Analyzing existing repository at %s", workspace_dir)

        # ── Pass 1: Local file scanning ───────────────────────────────
        lang_profile = self._detect_repo_language(workspace_dir)
        file_index_list: list[FileIndex] = []
        file_contents: dict[str, str] = {}

        source_files = self._discover_source_files(workspace_dir, lang_profile)
        logger.info("Discovered %d source files (%s)", len(source_files), lang_profile.display_name)

        for file_path in source_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                rel_path = str(file_path.relative_to(workspace_dir)).replace("\\", "/")
                file_contents[rel_path] = content

                # Index the file using RepositoryManager's indexing logic
                file_idx = self._index_file(rel_path, content, lang_profile)
                file_index_list.append(file_idx)
            except Exception as e:
                logger.warning("Failed to read %s: %s", file_path, e)

        # Build module info from local analysis
        modules = self._build_module_info(file_index_list, file_contents)

        # Detect entry points
        entry_points = self._find_entry_points(file_contents, lang_profile.name)

        # ── Pass 2: LLM-assisted summary ──────────────────────────────
        tech_stack, arch_style, summary = await self._llm_summarize(
            modules, entry_points, lang_profile, file_contents,
        )

        analysis = RepoAnalysis(
            modules=modules,
            tech_stack=tech_stack,
            architecture_style=arch_style,
            entry_points=entry_points,
            summary=summary,
        )

        logger.info(
            "Analysis complete: %d modules, tech=%s, arch=%s",
            len(modules), tech_stack.get("language", "?"), arch_style,
        )
        return analysis

    def _detect_repo_language(self, workspace_dir: Path) -> LanguageProfile:
        """Detect the primary programming language from file extensions."""
        ext_counts: dict[str, int] = {}
        for f in workspace_dir.rglob("*"):
            if f.is_file() and not any(
                part.startswith(".") or part in ("node_modules", "__pycache__", "venv", ".venv", "target", "build", "dist")
                for part in f.parts
            ):
                ext_counts[f.suffix.lower()] = ext_counts.get(f.suffix.lower(), 0) + 1

        return detect_language_from_extensions(ext_counts)

    def _discover_source_files(
        self, workspace_dir: Path, lang_profile: LanguageProfile,
    ) -> list[Path]:
        """Find all source files in the repository, excluding common junk directories."""
        skip_dirs = {
            ".git", ".svn", "node_modules", "__pycache__", ".venv", "venv",
            "target", "build", "dist", ".idea", ".vscode", ".gradle",
            ".mypy_cache", ".pytest_cache", ".tox", "egg-info",
        }
        results: list[Path] = []
        for ext in lang_profile.file_extensions:
            for f in workspace_dir.rglob(f"*{ext}"):
                if f.is_file() and not any(p in skip_dirs for p in f.parts):
                    results.append(f)
        return sorted(results)

    def _index_file(
        self, rel_path: str, content: str, lang_profile: LanguageProfile,
    ) -> FileIndex:
        """Index a single file — extract exports, imports, classes, functions."""
        import hashlib

        exports: list[str] = []
        imports: list[str] = []
        classes: list[str] = []
        functions: list[str] = []

        # Try AST extraction first
        from core.ast_extractor import ASTExtractor
        checksum = hashlib.md5(content.encode()).hexdigest()
        extractor = ASTExtractor()
        sig = extractor.extract(rel_path, content, lang_profile.name, checksum=checksum)

        if sig is not None:
            imports = list(sig.imports)
            for t in sig.types:
                exports.append(t.name)
                if t.kind in ("class", "interface", "enum", "record"):
                    classes.append(t.name)
                for m in t.methods:
                    if not m.is_constructor:
                        functions.append(m.name)
                        exports.append(m.name)
        else:
            # Regex fallback
            for line in content.splitlines():
                stripped = line.strip()
                for pattern in lang_profile.import_patterns:
                    if re.match(pattern, stripped):
                        imports.append(stripped)
                        break
                for pattern in lang_profile.definition_patterns:
                    m = re.match(pattern, stripped)
                    if m:
                        name = m.group(1)
                        if name[0].isupper():
                            classes.append(name)
                        else:
                            functions.append(name)
                        exports.append(name)
                        break

        return FileIndex(
            path=rel_path,
            exports=exports,
            imports=imports,
            classes=classes,
            functions=functions,
            checksum=checksum,
        )

    def _build_module_info(
        self,
        file_indices: list[FileIndex],
        file_contents: dict[str, str],
    ) -> list[ModuleInfo]:
        """Convert file indices into ModuleInfo objects with layer detection."""
        modules: list[ModuleInfo] = []
        for fi in file_indices:
            content = file_contents.get(fi.path, "")
            layer = _detect_layer(fi.path, content)
            # Derive module name from file path (e.g. "services/user_service.py" -> "user_service")
            name = Path(fi.path).stem
            modules.append(ModuleInfo(
                name=name,
                file=fi.path,
                classes=list(fi.classes),
                functions=list(fi.functions),
                imports=list(fi.imports),
                layer=layer,
            ))
        return modules

    def _find_entry_points(
        self, file_contents: dict[str, str], language: str,
    ) -> list[str]:
        """Detect entry-point files (main functions, app bootstraps, etc.)."""
        patterns = _ENTRY_POINT_PATTERNS.get(language, [])
        entry_points: list[str] = []
        for path, content in file_contents.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    entry_points.append(path)
                    break
        return entry_points

    async def _llm_summarize(
        self,
        modules: list[ModuleInfo],
        entry_points: list[str],
        lang_profile: LanguageProfile,
        file_contents: dict[str, str],
    ) -> tuple[dict[str, str], str, str]:
        """Use the LLM to summarize the repository architecture.

        Returns (tech_stack, architecture_style, summary).
        """
        # Build a compact representation of modules for the LLM
        module_summary = []
        for m in modules[:50]:  # Cap to avoid context overflow
            module_summary.append({
                "name": m.name,
                "file": m.file,
                "layer": m.layer,
                "classes": m.classes[:10],
                "functions": m.functions[:15],
                "import_count": len(m.imports),
            })

        # Include a sample of file content for richer analysis
        sample_files: dict[str, str] = {}
        for path in entry_points[:3]:
            content = file_contents.get(path, "")
            sample_files[path] = content[:2000]

        # Also sample a few other key files
        for m in modules[:5]:
            if m.file not in sample_files:
                content = file_contents.get(m.file, "")
                sample_files[m.file] = content[:1500]

        prompt = (
            f"Analyze this existing {lang_profile.display_name} repository and produce "
            f"a JSON summary.\n\n"
            f"Modules ({len(modules)} total):\n"
            f"{json.dumps(module_summary, indent=2)}\n\n"
            f"Entry points: {entry_points}\n\n"
            f"Sample file contents:\n"
        )
        for path, content in sample_files.items():
            prompt += f"\n--- {path} ---\n{content}\n"

        prompt += (
            "\n\nRespond with ONLY this JSON (no fences, no prose):\n"
            "{\n"
            '  "tech_stack": {\n'
            '    "language": "<lowercase: python|java|go|typescript|rust|csharp>",\n'
            '    "framework": "<lowercase-framework-name or empty string>",\n'
            '    "db": "<lowercase-db-name or empty string>"\n'
            "  },\n"
            '  "architecture_style": "<one of: REST|GraphQL|gRPC|monolith|microservices|event-driven>",\n'
            '  "summary": "2-3 sentences describing what this system does and how it is structured"\n'
            "}\n\n"
            "RULES:\n"
            "- language MUST be one of: python, java, go, typescript, rust, csharp\n"
            "- framework must be the specific framework name (e.g., 'spring-boot', "
            "'fastapi', 'gin', 'express')\n"
            "- db must be the specific database name (e.g., 'postgresql', 'mysql', "
            "'mongodb', 'redis', 'h2', 'sqlite')\n"
            "- architecture_style must be EXACTLY one of the options listed\n"
            "- If you cannot determine a value, use an empty string — do NOT guess"
        )

        try:
            response = await self.llm.generate(
                system_prompt=self.system_prompt,
                user_prompt=prompt,
                max_tokens=2048,
            )
            self._metrics["llm_calls"] += 1
            self._metrics["tokens_used"] += sum(response.usage.values())

            data = self._parse_json(response.content)
            tech_stack = data.get("tech_stack", {"language": lang_profile.name})
            arch_style = data.get("architecture_style", "unknown")
            summary = data.get("summary", "")
            return tech_stack, arch_style, summary

        except Exception as e:
            logger.warning("LLM summary failed, using heuristic analysis: %s", e)
            return (
                {"language": lang_profile.name},
                "unknown",
                f"Repository with {len(modules)} modules in {lang_profile.display_name}",
            )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse JSON from LLM output, handling fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            return {}
