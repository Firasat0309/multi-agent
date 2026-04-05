"""Deterministic plan.md parser — extracts structured objects in <1 second.

Eliminates 4-5 sequential LLM calls (~3-4 minutes) by parsing the structured
markdown that PlanGeneratorAgent already outputs:

  - PHASE 0  →  ProductRequirements
  - PHASE 1  →  APIContract
  - PHASE 2  →  RepositoryBlueprint (backend)
  - PHASE 3  →  UIDesignSpec + ComponentPlan (frontend)

Each parser returns ``None`` on failure — the pipeline falls back to
LLM-based agents only for sections that could not be reliably parsed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from core.models import (
    APIContract,
    APIEndpoint,
    ComponentPlan,
    FileBlueprint,
    ProductRequirements,
    RepositoryBlueprint,
    UIComponent,
    UIDesignSpec,
)

logger = logging.getLogger(__name__)

__all__ = ["ParsedPlan", "parse_plan_md"]


# ── Result container ──────────────────────────────────────────────────────────


@dataclass
class ParsedPlan:
    """Result of deterministic plan.md parsing."""

    title: str = ""
    requirements: ProductRequirements | None = None
    backend_blueprint: RepositoryBlueprint | None = None
    api_contract: APIContract | None = None
    design_spec: UIDesignSpec | None = None
    component_plan: ComponentPlan | None = None
    raw_phases: dict[str, str] = field(default_factory=dict)


# ── Master parser ─────────────────────────────────────────────────────────────


def parse_plan_md(plan_md: str) -> ParsedPlan:
    """Parse an entire plan.md into structured objects (<1 second).

    Each section is parsed independently — a failure in one section
    does not block others.
    """
    result = ParsedPlan()
    result.raw_phases = _extract_phases(plan_md)
    result.title = _parse_title(plan_md)
    tech = _parse_tech_stack_header(plan_md)

    has_be = "PHASE 2" in result.raw_phases
    has_fe = "PHASE 3" in result.raw_phases

    # ── Requirements from PHASE 0 ────────────────────────────────────────
    try:
        result.requirements = _parse_requirements(
            result.raw_phases.get("PHASE 0", ""),
            result.title,
            tech,
            has_backend=has_be,
            has_frontend=has_fe,
        )
        logger.info("plan_parser: requirements extracted — %s", result.title)
    except Exception as exc:
        logger.warning("plan_parser: requirements parse failed: %s", exc)

    # ── Backend blueprint from PHASE 2 ───────────────────────────────────
    try:
        result.backend_blueprint = _parse_backend_blueprint(
            result.raw_phases.get("PHASE 2", ""),
            result.title,
            tech,
        )
        if result.backend_blueprint:
            logger.info(
                "plan_parser: backend blueprint — %d files",
                len(result.backend_blueprint.file_blueprints),
            )
    except Exception as exc:
        logger.warning("plan_parser: backend blueprint parse failed: %s", exc)

    # ── API contract from PHASE 1 ────────────────────────────────────────
    try:
        result.api_contract = _parse_api_contract(
            result.raw_phases.get("PHASE 1", ""),
            result.title,
        )
        if result.api_contract:
            logger.info(
                "plan_parser: API contract — %d endpoints",
                len(result.api_contract.endpoints),
            )
    except Exception as exc:
        logger.warning("plan_parser: API contract parse failed: %s", exc)

    # ── Frontend plan from PHASE 3 ───────────────────────────────────────
    try:
        spec, plan = _parse_frontend_plan(
            result.raw_phases.get("PHASE 3", ""),
            result.raw_phases.get("PHASE 0", ""),
            tech,
            result.api_contract,
        )
        result.design_spec = spec
        result.component_plan = plan
        if plan:
            logger.info(
                "plan_parser: component plan — %d components, framework=%s",
                len(plan.components),
                plan.framework,
            )
    except Exception as exc:
        logger.warning("plan_parser: frontend plan parse failed: %s", exc)

    return result


# ── Phase / section extraction ────────────────────────────────────────────────


def _extract_phases(plan_md: str) -> dict[str, str]:
    """Split plan.md into {phase_name: text} sections."""
    phases: dict[str, str] = {}
    pattern = re.compile(r"^#\s+(PHASE\s+\d+)\b[^\n]*", re.MULTILINE | re.IGNORECASE)
    matches = list(pattern.finditer(plan_md))
    for i, m in enumerate(matches):
        key = re.sub(r"\s+", " ", m.group(1)).upper().strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(plan_md)
        phases[key] = plan_md[start:end].strip()
    return phases


def _parse_title(plan_md: str) -> str:
    """Extract project title from first heading."""
    m = re.search(r"^#\s+(.+?)(?:\s*[—\-–]+\s*.*)?$", plan_md, re.MULTILINE)
    if m:
        title = m.group(1).strip()
        # Remove trailing "Implementation Plan" etc.
        title = re.sub(r"\s*[—\-–]\s*(implementation|plan).*", "", title, flags=re.IGNORECASE)
        return title.strip()
    return "project"


def _parse_tech_stack_header(plan_md: str) -> dict[str, str]:
    """Parse the ``**Tech Stack:** X | Y | Z`` header line."""
    m = re.search(r"\*\*Tech\s+Stack[:\s]*\*\*\s*(.+)", plan_md, re.IGNORECASE)
    if not m:
        return {}
    parts = [p.strip() for p in m.group(1).split("|")]
    result: dict[str, str] = {}

    for part in parts:
        lower = part.lower().strip()
        if any(kw in lower for kw in ("spring", "django", "flask", "fastapi", "express",
                                       "nestjs", "gin", "echo", "actix", "axum", ".net", "asp")):
            result["framework"] = part.strip()
        elif any(kw in lower for kw in ("vue", "react", "next", "angular", "svelte")):
            result["frontend"] = part.strip()
        elif any(kw in lower for kw in ("postgres", "mysql", "mongodb", "sqlite", "h2",
                                         "redis", "mariadb", "dynamodb")):
            result["db"] = part.strip()
        elif any(kw in lower for kw in ("maven", "gradle", "npm", "yarn", "pnpm", "cargo")):
            result["build_tool"] = part.strip()
        elif any(kw in lower for kw in ("java", "python", "typescript", "golang",
                                         "rust", "c#", "kotlin")):
            result["language"] = part.strip()

    # Infer language from framework
    if "language" not in result and "framework" in result:
        fw = result["framework"].lower()
        if any(k in fw for k in ("spring", "quarkus")):
            result["language"] = "Java"
        elif any(k in fw for k in ("django", "flask", "fastapi")):
            result["language"] = "Python"
        elif any(k in fw for k in ("express", "nestjs", "koa")):
            result["language"] = "TypeScript"
        elif any(k in fw for k in ("gin", "echo", "fiber")):
            result["language"] = "Go"
        elif any(k in fw for k in ("actix", "axum", "rocket")):
            result["language"] = "Rust"
        elif any(k in fw for k in (".net", "asp")):
            result["language"] = "C#"

    return result


# ── File tree parser ──────────────────────────────────────────────────────────

# Separators between filename and purpose — tried in order.
_PURPOSE_SEPARATORS = (" — ", " – ", " -- ", " - ")

# Extensions that identify source files.
_EXT_TO_LANG: dict[str, str] = {
    ".java": "java", ".py": "python", ".go": "go",
    ".ts": "typescript", ".tsx": "typescript", ".js": "javascript",
    ".jsx": "javascript", ".rs": "rust", ".cs": "csharp",
    ".kt": "java", ".scala": "java", ".vue": "typescript",
}


def _find_tree_text(section: str) -> str:
    """Locate the file tree block inside a phase section."""
    # Strategy 1: code fence
    m = re.search(r"```(?:\w*)\n(.*?)```", section, re.DOTALL)
    if m:
        return m.group(1)
    # Strategy 2: everything between "File Tree" header and next "##" header
    m = re.search(
        r"##\s+File\s+Tree\b[^\n]*\n(.*?)(?=\n##\s|\Z)",
        section,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1)
    return section


def _parse_file_tree(section_text: str) -> list[tuple[str, str, str]]:
    """Parse an indented file tree → [(path, purpose, root_prefix)].

    Returns tuples of (full_path, purpose_text, detected_root_prefix).
    The root_prefix (e.g. ``backend/``) is stripped from paths automatically.
    """
    raw = _find_tree_text(section_text)
    if not raw.strip():
        return []

    lines = raw.split("\n")
    path_stack: list[tuple[int, str]] = []  # (indent, dirname)
    results: list[tuple[str, str]] = []
    root_prefix = ""

    for line in lines:
        if not line.strip():
            continue
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Split name from purpose
        name = stripped
        purpose = ""
        for sep in _PURPOSE_SEPARATORS:
            if sep in stripped:
                name, purpose = stripped.split(sep, 1)
                name = name.strip()
                purpose = purpose.strip()
                break

        # Pop stack entries at same or deeper indentation
        while path_stack and path_stack[-1][0] >= indent:
            path_stack.pop()

        is_dir = name.endswith("/")
        if is_dir:
            path_stack.append((indent, name))
        else:
            # Build full path from stack
            dir_parts = "".join(p[1] for p in path_stack)
            full_path = dir_parts + name
            results.append((full_path, purpose))

    # Detect and strip root prefix (e.g. "backend/", "frontend/")
    if results:
        prefixes = ("backend/", "frontend/", "server/", "client/", "api/")
        for p in prefixes:
            if all(fp.startswith(p) for fp, _ in results):
                root_prefix = p
                results = [(fp[len(p):], purp) for fp, purp in results]
                break

    return [(fp, purp, root_prefix) for fp, purp in results]


# ── Layer / dependency inference ──────────────────────────────────────────────

_LAYER_DIR_MAP: dict[str, str] = {
    "model": "model", "models": "model", "entity": "model", "entities": "model",
    "repository": "repository", "repositories": "repository", "repo": "repository",
    "service": "service", "services": "service",
    "controller": "controller", "controllers": "controller", "rest": "controller",
    "config": "config", "configuration": "config",
    "dto": "dto", "dtos": "dto",
    "security": "security", "auth": "security",
    "exception": "exception", "exceptions": "exception",
    "middleware": "middleware", "filter": "middleware", "interceptor": "middleware",
    "util": "util", "utils": "util", "helper": "util", "helpers": "util",
    "mapper": "util", "mappers": "util",
    "infrastructure": "infrastructure",
}

# Layer dependency order: layer → list of layers it depends on.
_LAYER_DEPS: dict[str, list[str]] = {
    "model": [],
    "dto": ["model"],
    "repository": ["model"],
    "service": ["repository", "model", "dto"],
    "controller": ["service", "dto", "model"],
    "security": ["service", "model"],
    "config": [],
    "exception": [],
    "middleware": [],
    "util": [],
    "infrastructure": [],
}

_CONFIG_FILES = frozenset({
    "pom.xml", "build.gradle", "build.gradle.kts", "package.json",
    "cargo.toml", "go.mod", "requirements.txt", "pyproject.toml",
    "tsconfig.json", "vite.config.ts", "vite.config.js",
    "tailwind.config.js", "tailwind.config.ts",
    ".env", ".env.local",
})


def _infer_layer(path: str) -> str:
    """Infer architectural layer from file path."""
    parts = PurePosixPath(path).parts
    fname = parts[-1].lower() if parts else ""

    # Config / build files
    if fname in _CONFIG_FILES or fname.startswith("application."):
        return "config"
    if "test" in fname or "/test/" in path.lower() or "/tests/" in path.lower():
        return "test"

    # Check directory names
    for part in reversed(parts[:-1]):
        layer = _LAYER_DIR_MAP.get(part.lower())
        if layer:
            return layer

    # Heuristic from filename
    lower = fname
    if "application" in lower or "main" in lower or "app" in lower:
        return "infrastructure"

    return "service"  # safe default


def _infer_exports(path: str) -> list[str]:
    """Derive export names from filename."""
    stem = PurePosixPath(path).stem
    if not stem or stem.startswith("."):
        return []
    # Java/C#: PascalCase class name = stem
    ext = PurePosixPath(path).suffix.lower()
    if ext in (".java", ".cs", ".kt"):
        return [stem]
    # Python: module name
    if ext == ".py":
        return [stem]
    # TypeScript/JavaScript: PascalCase or camelCase
    if ext in (".ts", ".tsx", ".js", ".jsx", ".vue"):
        return [stem]
    return []


def _detect_file_language(path: str, project_lang: str) -> str:
    """Detect programming language from file extension."""
    ext = PurePosixPath(path).suffix.lower()
    return _EXT_TO_LANG.get(ext, project_lang)


def _build_depends_on(
    layer: str,
    all_blueprints: list[FileBlueprint],
) -> list[str]:
    """Infer depends_on from layer hierarchy."""
    dep_layers = _LAYER_DEPS.get(layer, [])
    if not dep_layers:
        return []
    return [
        fb.path for fb in all_blueprints
        if fb.layer in dep_layers
    ]


# ── PHASE 0 → ProductRequirements ────────────────────────────────────────────


def _parse_requirements(
    phase0: str,
    title: str,
    tech: dict[str, str],
    has_backend: bool,
    has_frontend: bool,
) -> ProductRequirements:
    """Extract ProductRequirements from PHASE 0 + plan header."""
    # Extract screen/page names from markdown table
    screens: list[str] = []
    _skip = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS",
                        "METHOD", "SCREEN", "---", "ROUTE"})
    for m in re.finditer(r"\|\s*(\w[\w\s]+?)\s*\|\s*/", phase0):
        name = m.group(1).strip()
        if name.upper() not in _skip:
            screens.append(name)

    # Extract features from entity table or features list
    features: list[str] = []
    if screens:
        features = screens[:]
    # Also grab data entities as feature domains
    for m in re.finditer(r"\|\s*(\w+)\s*\|\s*\w+:", phase0):
        entity = m.group(1).strip()
        if entity.lower() not in ("entity", "field", "---", "name"):
            features.append(f"{entity} management")

    # Deduplicate
    seen: set[str] = set()
    unique_features: list[str] = []
    for f in features:
        if f.lower() not in seen:
            seen.add(f.lower())
            unique_features.append(f)

    # Build tech_preferences from parsed header
    tech_prefs: dict[str, str] = {}
    if tech.get("frontend"):
        tech_prefs["frontend"] = tech["frontend"]
    if tech.get("framework"):
        tech_prefs["backend"] = tech["framework"]
    if tech.get("db"):
        tech_prefs["db"] = tech["db"]
    if tech.get("build_tool"):
        tech_prefs["build_tool"] = tech["build_tool"]

    # Extract user stories from API operations table
    stories: list[str] = []
    for m in re.finditer(
        r"\|\s*(GET|POST|PUT|PATCH|DELETE)\s*\|\s*(/\S+)\s*\|\s*([^|]+)",
        phase0,
    ):
        method, path, trigger = m.group(1), m.group(2), m.group(3).strip()
        stories.append(f"User can {trigger.lower().rstrip('|').strip()} ({method} {path})")

    return ProductRequirements(
        title=title,
        description=f"{title} — full-stack application",
        features=unique_features,
        user_stories=stories,
        tech_preferences=tech_prefs,
        has_frontend=has_frontend,
        has_backend=has_backend,
    )


# ── PHASE 2 → RepositoryBlueprint ────────────────────────────────────────────

# Known layer directory names to strip from common-prefix detection.
_LAYER_DIRS = frozenset({
    "controller", "controllers", "service", "services",
    "repository", "repositories", "model", "models",
    "entity", "entities", "config", "configuration",
    "util", "utils", "helper", "helpers", "dto", "dtos",
    "exception", "exceptions", "filter", "filters",
    "security", "middleware", "interceptor", "mapper",
})


def _parse_backend_blueprint(
    phase2: str,
    title: str,
    tech: dict[str, str],
) -> RepositoryBlueprint | None:
    """Build RepositoryBlueprint from PHASE 2 file tree."""
    if not phase2.strip():
        return None

    # Parse tech stack from PHASE 2 body (more detailed than header)
    ts_section = re.search(
        r"##\s+Technology\s+Stack\b[^\n]*\n(.*?)(?=\n##|\Z)",
        phase2,
        re.DOTALL | re.IGNORECASE,
    )
    lang = tech.get("language", "").lower().strip()
    framework = tech.get("framework", "").lower().strip()
    db = tech.get("db", "").lower().strip()
    build_tool = tech.get("build_tool", "").lower().strip()

    # Refine from PHASE 2 Technology Stack section when available
    if ts_section:
        ts_text = ts_section.group(1).lower()
        if not lang:
            for pattern, l in [("java", "java"), ("python", "python"),
                                ("typescript", "typescript"), ("go", "go"),
                                ("rust", "rust"), ("c#", "csharp")]:
                if pattern in ts_text:
                    lang = l
                    break
        if not framework:
            for pattern, fw in [("spring", "spring-boot"), ("django", "django"),
                                 ("fastapi", "fastapi"), ("express", "express"),
                                 ("nestjs", "nestjs"), ("gin", "gin")]:
                if pattern in ts_text:
                    framework = fw
                    break
        if not db:
            for pattern, d in [("h2", "h2"), ("postgres", "postgresql"),
                                ("mysql", "mysql"), ("sqlite", "sqlite"),
                                ("mongodb", "mongodb")]:
                if pattern in ts_text:
                    db = d
                    break
        if not build_tool:
            for pattern, bt in [("maven", "maven"), ("gradle", "gradle"),
                                 ("npm", "npm"), ("cargo", "cargo")]:
                if pattern in ts_text:
                    build_tool = bt
                    break

    if not lang:
        lang = "python"  # fallback

    tech_stack = {
        "language": lang,
        "framework": framework or lang,
        "db": db or "sqlite",
        "build_tool": build_tool or _default_build_tool(lang),
    }

    # Parse file tree
    tree = _parse_file_tree(phase2)
    if not tree:
        return None

    # Minimum viable blueprint (reject too-small parses)
    if len(tree) < 3:
        logger.warning("plan_parser: only %d files in PHASE 2 tree — rejecting", len(tree))
        return None

    # Build FileBlueprint list
    file_blueprints: list[FileBlueprint] = []
    folder_set: set[str] = set()

    for path, purpose, _ in tree:
        layer = _infer_layer(path)
        file_lang = _detect_file_language(path, lang)
        exports = _infer_exports(path)

        file_blueprints.append(FileBlueprint(
            path=path,
            purpose=purpose or f"{PurePosixPath(path).stem} implementation",
            depends_on=[],  # filled in below
            exports=exports,
            language=file_lang,
            layer=layer,
        ))

        # Collect directory structure
        parts = PurePosixPath(path).parts
        for i in range(1, len(parts)):
            folder_set.add("/".join(parts[:i]))

    # Resolve depends_on from layer hierarchy
    for fb in file_blueprints:
        fb.depends_on = _build_depends_on(fb.layer, file_blueprints)

    # Inject mandatory Spring Boot files if missing
    if lang == "java":
        _inject_java_mandatory(file_blueprints, title, tech_stack)

    # Convert project name to kebab-case
    project_name = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-").lower()

    return RepositoryBlueprint(
        name=project_name or "project",
        description=title,
        architecture_style="REST",
        tech_stack=tech_stack,
        folder_structure=sorted(folder_set),
        file_blueprints=file_blueprints,
        architecture_doc=f"# Architecture (from plan.md)\n\n{phase2[:4000]}",
    )


def _default_build_tool(lang: str) -> str:
    return {
        "java": "maven", "python": "pip", "typescript": "npm",
        "go": "go", "rust": "cargo", "csharp": "dotnet",
    }.get(lang, "")


def _inject_java_mandatory(
    blueprints: list[FileBlueprint],
    title: str,
    tech_stack: dict[str, str],
) -> None:
    """Inject Application.java and application.properties if missing."""
    paths = {fb.path for fb in blueprints}

    # Find common package prefix
    java_dirs = sorted(set(
        "/".join(PurePosixPath(p).parts[:-1])
        for p in paths if p.endswith(".java") and "src/main/java/" in p
    ))
    pkg_prefix = "src/main/java"
    if java_dirs:
        split = [d.split("/") for d in java_dirs]
        min_len = min(len(s) for s in split)
        common: list[str] = []
        for i in range(min_len):
            vals = {s[i] for s in split}
            if len(vals) == 1:
                common.append(vals.pop())
            else:
                break
        if common and common[-1].lower() in _LAYER_DIRS:
            common = common[:-1]
        if common:
            pkg_prefix = "/".join(common)

    # Application.java
    if not any(p.endswith(("Application.java", "App.java", "Main.java")) for p in paths):
        raw = re.sub(r"[^a-zA-Z0-9]+", " ", title)
        cls = "".join(w.capitalize() for w in raw.split()) + "Application"
        if cls[0].isdigit():
            cls = "App" + cls
        app_path = f"{pkg_prefix}/{cls}.java"
        blueprints.append(FileBlueprint(
            path=app_path,
            purpose=f"@SpringBootApplication entry point — {cls}",
            depends_on=[], exports=[cls], language="java", layer="infrastructure",
        ))
        logger.info("plan_parser: injected %s", app_path)

    # application.properties
    if not any(p.endswith(("application.properties", "application.yml")) for p in paths):
        blueprints.append(FileBlueprint(
            path="src/main/resources/application.properties",
            purpose="Spring Boot configuration — server port, datasource, JPA, "
                    "JWT properties (jwt.secret, jwt.expiration, app.jwtSecret, "
                    "app.jwtExpirationMs — include ALL variants to prevent startup crash)",
            depends_on=[], exports=[], language="java", layer="config",
        ))
        logger.info("plan_parser: injected application.properties")


# ── PHASE 1 → APIContract ────────────────────────────────────────────────────

# Matches lines like: "POST /api/auth/register" or "**GET /api/users**"
_ENDPOINT_RE = re.compile(
    r"^\s*\*{0,2}(GET|POST|PUT|PATCH|DELETE)\s+(/\S+)\*{0,2}"
    r"(?:\s*[—\-–→]\s*(.+))?$",
    re.MULTILINE,
)

# Matches request/response bodies: "Request body: { ... }"
_REQ_BODY_RE = re.compile(
    r"[Rr]equest(?:\s+body)?:\s*\{([^}]+)\}", re.DOTALL,
)
_RES_BODY_RE = re.compile(
    r"[Rr]esponse(?:\s+body)?:\s*\{([^}]+)\}", re.DOTALL,
)
_AUTH_RE = re.compile(
    r"[Aa]uth(?:entication)?(?:\s+[Rr]equired)?:\s*(Yes|No|Required|true|false|None)",
    re.IGNORECASE,
)


def _parse_api_contract(
    phase1: str,
    title: str,
) -> APIContract | None:
    """Build APIContract from PHASE 1 endpoint blocks."""
    if not phase1.strip():
        return None

    endpoints: list[APIEndpoint] = []

    # Split phase1 into blocks by endpoint lines
    matches = list(_ENDPOINT_RE.finditer(phase1))
    if not matches:
        return None

    for i, m in enumerate(matches):
        method = m.group(1).upper()
        path = m.group(2).rstrip("*").rstrip(",").rstrip(".")
        desc = (m.group(3) or "").strip()

        # Get the text block following this endpoint (until next endpoint)
        block_start = m.end()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(phase1)
        block = phase1[block_start:block_end]

        # Parse request/response schemas
        req_schema = _parse_inline_schema(_REQ_BODY_RE.search(block))
        res_schema = _parse_inline_schema(_RES_BODY_RE.search(block))

        # Auth
        auth_match = _AUTH_RE.search(block)
        auth_required = False
        if auth_match:
            auth_required = auth_match.group(1).lower() in ("yes", "required", "true")
        elif "/auth/" not in path.lower():
            # Default: non-auth endpoints require auth
            auth_required = True

        # Tags from path
        segments = [s for s in path.strip("/").split("/") if s and not s.startswith("{")]
        tag = segments[-1] if segments else "general"
        # Skip version/api segments
        for seg in segments:
            if seg not in ("api", "v1", "v2") and not seg.startswith("v"):
                tag = seg
                break

        if not desc:
            desc = f"{method} {path}"

        endpoints.append(APIEndpoint(
            path=path,
            method=method,
            description=desc,
            request_schema=req_schema,
            response_schema=res_schema,
            auth_required=auth_required,
            tags=[tag],
        ))

    if not endpoints:
        return None

    # Detect common base URL
    paths = [ep.path for ep in endpoints]
    base = _detect_base_url(paths)

    return APIContract(
        title=f"{title} API",
        version="1.0.0",
        base_url=base,
        endpoints=endpoints,
        contract_format="openapi",
    )


def _parse_inline_schema(match: re.Match | None) -> dict[str, Any]:
    """Parse ``{ field: type, field2: type }`` into a JSON-Schema-like dict."""
    if not match:
        return {}
    inner = match.group(1).strip()
    props: dict[str, Any] = {}
    for pair in re.split(r",\s*", inner):
        pair = pair.strip()
        if ":" in pair:
            name, tstr = pair.split(":", 1)
            name = name.strip().strip('"').strip("'")
            tstr = tstr.strip().strip('"').strip("'")
            if name and not name.startswith("//"):
                props[name] = {"type": _map_ts_type(tstr)}
    if not props:
        return {}
    return {"type": "object", "properties": props}


_TS_TYPE_MAP = {
    "string": "string", "number": "integer", "integer": "integer",
    "boolean": "boolean", "long": "integer", "int": "integer",
    "float": "number", "double": "number", "date": "string",
}


def _map_ts_type(ts: str) -> str:
    ts_lower = ts.lower().strip()
    if ts_lower.endswith("[]"):
        return "array"
    return _TS_TYPE_MAP.get(ts_lower, "string")


def _detect_base_url(paths: list[str]) -> str:
    if not paths:
        return "/api"
    split = [p.strip("/").split("/") for p in paths]
    min_len = min(len(s) for s in split)
    common: list[str] = []
    for i in range(min_len):
        vals = {s[i] for s in split}
        if len(vals) == 1 and not vals.pop().startswith("{"):
            common.append(split[0][i])
        else:
            break
    return "/" + "/".join(common) if common else "/api"


# ── PHASE 3 → UIDesignSpec + ComponentPlan ────────────────────────────────────

_FE_TYPE_MAP: dict[str, str] = {
    "pages": "pages", "page": "pages", "app": "pages", "views": "pages",
    "layout": "layout", "layouts": "layout",
    "feature": "feature", "features": "feature",
    "ui": "ui", "common": "ui", "shared": "shared",
    "api": "shared", "store": "shared", "stores": "shared",
    "hooks": "shared", "composables": "shared",
    "types": "shared", "lib": "shared", "utils": "shared",
}


def _infer_component_type(path: str, name: str) -> str:
    """Infer component_type from path segments."""
    parts = PurePosixPath(path).parts
    for part in parts:
        ct = _FE_TYPE_MAP.get(part.lower())
        if ct:
            return ct
    # Heuristic from name
    lower = name.lower()
    if "page" in lower or "view" in lower:
        return "pages"
    if lower in ("header", "footer", "sidebar", "navbar", "layout", "nav"):
        return "layout"
    if lower in ("button", "input", "card", "modal", "badge", "avatar", "spinner"):
        return "ui"
    if lower.startswith("use") or lower.endswith("store") or lower.endswith("api"):
        return "shared"
    return "feature"


def _infer_framework(tech: dict[str, str]) -> str:
    """Infer FE framework from tech header."""
    fe = tech.get("frontend", "").lower()
    if "vue" in fe:
        return "vue"
    if "next" in fe:
        return "nextjs"
    if "angular" in fe:
        return "angular"
    if "svelte" in fe:
        return "svelte"
    if "react" in fe:
        return "react"
    return "react"


def _parse_frontend_plan(
    phase3: str,
    phase0: str,
    tech: dict[str, str],
    api_contract: APIContract | None,
) -> tuple[UIDesignSpec | None, ComponentPlan | None]:
    """Build UIDesignSpec + ComponentPlan from PHASE 3 file tree."""
    if not phase3.strip():
        return None, None

    framework = _infer_framework(tech)

    # ── UIDesignSpec ──
    # Extract page names from PHASE 0 screens table
    pages: list[str] = []
    _skip_pages = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD",
                              "OPTIONS", "METHOD", "SCREEN", "---", "ROUTE"})
    for m in re.finditer(r"\|\s*(\w[\w\s]+?)\s*\|\s*/", phase0):
        name = m.group(1).strip()
        if name.upper() not in _skip_pages:
            pages.append(name)
    if not pages:
        pages = ["Home", "Dashboard"]

    design_spec = UIDesignSpec(
        framework=framework,
        design_description=f"UI design for {tech.get('frontend', framework)} application",
        pages=pages,
        global_styles={"primary_color": "#3B82F6", "font_family": "Inter, sans-serif"},
    )

    # ── ComponentPlan ──
    tree = _parse_file_tree(phase3)
    if not tree or len(tree) < 2:
        return design_spec, None

    # Filter to source files only (skip config files at root)
    source_exts = (".tsx", ".jsx", ".ts", ".js", ".vue", ".svelte")
    source_files = [
        (p, purp) for p, purp, _ in tree
        if PurePosixPath(p).suffix.lower() in source_exts
    ]

    if not source_files:
        return design_spec, None

    # Build components
    components: list[UIComponent] = []
    for path, purpose in source_files:
        stem = PurePosixPath(path).stem
        comp_type = _infer_component_type(path, stem)
        components.append(UIComponent(
            name=stem,
            file_path=path,
            component_type=comp_type,
            description=purpose or f"{stem} component",
        ))

    # Wire depends_on by type hierarchy:
    # shared → [] ; ui → [] ; layout → [ui names] ; feature → [shared names] ; pages → [feature+layout names]
    shared_names = [c.name for c in components if c.component_type == "shared"]
    ui_names = [c.name for c in components if c.component_type == "ui"]
    layout_names = [c.name for c in components if c.component_type == "layout"]
    feature_names = [c.name for c in components if c.component_type == "feature"]

    for c in components:
        if c.component_type == "layout":
            c.depends_on = ui_names[:3]  # don't over-connect
        elif c.component_type == "feature":
            c.depends_on = shared_names[:5]
        elif c.component_type == "pages":
            c.depends_on = (layout_names[:2] + feature_names[:5])

    # Wire api_calls from contract
    if api_contract:
        all_paths = [ep.path for ep in api_contract.endpoints]
        for c in components:
            if c.component_type in ("feature", "pages"):
                # Match by name overlap with endpoint paths
                name_lower = c.name.lower().replace("page", "").replace("list", "").replace("form", "")
                c.api_calls = [
                    p for p in all_paths
                    if name_lower and len(name_lower) > 2 and name_lower in p.lower()
                ]

    # Infer state/routing solutions
    state_map = {"vue": "pinia", "angular": "ngrx", "react": "zustand", "nextjs": "zustand", "svelte": "svelte/store"}
    routing_map = {"vue": "vue-router", "angular": "angular", "react": "react-router", "nextjs": "nextjs", "svelte": "svelte-kit"}

    plan = ComponentPlan(
        components=components,
        framework=framework,
        state_solution=state_map.get(framework, "zustand"),
        api_base_url=api_contract.base_url if api_contract else "/api",
        routing_solution=routing_map.get(framework, "react-router"),
    )

    return design_spec, plan
