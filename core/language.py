"""Language profiles defining language-specific toolchains, patterns, and conventions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LanguageProfile:
    """Everything the system needs to know about a target language."""

    name: str                           # "python", "java", "go", "typescript", etc.
    display_name: str                   # "Python", "Java", "Go", "TypeScript"
    file_extensions: list[str]          # [".py"], [".java"], [".go"], [".ts"]
    glob_pattern: str                   # "**/*.py", "**/*.java"
    docker_image: str                   # "python:3.11-slim", "eclipse-temurin:21-jdk"
    test_command: str                   # "pytest -v --tb=short", "mvn test", "go test ./..."
    lint_command: str                   # "ruff check", "checkstyle", "golangci-lint run"
    type_check_command: str             # "mypy", "", "go vet ./..."
    security_scan_command: str          # "bandit -r . -f json", "spotbugs", "gosec ./..."
    build_command: str                  # "", "mvn package", "go build ./..."
    allowed_commands: list[str] = field(default_factory=list)
    package_init_file: str = ""         # "__init__.py" for Python, "" for others
    import_patterns: list[str] = field(default_factory=list)  # regex patterns to detect imports
    definition_patterns: list[str] = field(default_factory=list)  # regex for class/func defs
    module_separator: str = "."         # "." for Python/Java, "/" for Go
    code_fence_name: str = ""           # "python", "java", "go", "typescript"
    source_root: str = ""               # subdirectory under workspace where source files live
    test_root: str = ""                 # subdirectory under workspace where test files live
    cache_paths: list[str] = field(default_factory=list)  # container paths for dependency caches

    def matches_extension(self, path: str) -> bool:
        return any(path.endswith(ext) for ext in self.file_extensions)

    def source_glob(self, directory: str = "") -> str:
        if directory:
            return f"{directory}/{self.glob_pattern.lstrip('**/')}"
        return self.glob_pattern

    def to_module_path(self, file_path: str) -> str:
        """Convert a file path to a module/package path."""
        result = file_path
        for ext in self.file_extensions:
            result = result.removesuffix(ext)
        return result.replace("/", self.module_separator)

    def resolve_import_to_path(
        self, import_str: str, known_files: set[str]
    ) -> str | None:
        """Convert an import statement string to a workspace-relative file path.

        Resolves the mixed-node-type problem where the dependency graph was
        populated with raw import strings (e.g. ``"services.user"``) instead of
        resolved file paths (e.g. ``"services/user.py"``).  Only returns a path
        that actually exists in ``known_files`` so the graph stays clean.

        Each language section extracts the module identifier from the import
        statement and maps it to the most likely file path.  Relative imports
        and wildcard imports are skipped (return ``None``) because they cannot
        be resolved without a full module-resolution pass.
        """
        cleaned = import_str.strip().rstrip(";")

        if self.name == "python":
            # Skip relative imports — too ambiguous without package context
            if re.match(r"^from\s+\.", cleaned):
                return None
            m = re.match(r"^from\s+([\w.]+)\s+import", cleaned)
            if m:
                module = m.group(1)
            else:
                m = re.match(r"^import\s+([\w.]+)", cleaned)
                if not m:
                    return None
                # "import os.path" → use "os.path", not "os"
                module = m.group(1).split(",")[0].strip()
            candidate = module.replace(".", "/") + ".py"
            if candidate in known_files:
                return candidate
            # Package import: "from services import user" → services/__init__.py
            init_candidate = module.replace(".", "/") + "/__init__.py"
            if init_candidate in known_files:
                return init_candidate
            return None

        elif self.name == "java":
            # "import com.example.services.UserService;" → "com/example/.../UserService.java"
            m = re.match(r"^import\s+(?:static\s+)?([\w.]+?)(?:\.\*)?$", cleaned)
            if not m:
                return None
            fqn = m.group(1)
            candidate = fqn.replace(".", "/") + ".java"
            if candidate in known_files:
                return candidate
            # Try common Java source roots
            for prefix in ("src/main/java/", "src/"):
                if (prefix + candidate) in known_files:
                    return prefix + candidate
            # Fallback: match by class name only (last segment)
            class_name = fqn.split(".")[-1]
            for path in known_files:
                if path.endswith(f"/{class_name}.java") or path == f"{class_name}.java":
                    return path
            return None

        elif self.name == "go":
            # `"github.com/app/services"` → services/services.go or services/*.go
            m = re.match(r'^["\s]*([\w./\-]+)["\s]*$', cleaned)
            if not m:
                return None
            import_path = m.group(1).strip('"').strip("'")
            last_seg = import_path.split("/")[-1]
            for candidate in (f"{last_seg}/{last_seg}.go", f"{last_seg}/main.go"):
                if candidate in known_files:
                    return candidate
            # Any .go file in that directory
            for path in known_files:
                if f"/{last_seg}/" in f"/{path}" or path.startswith(f"{last_seg}/"):
                    return path
            return None

        elif self.name == "typescript":
            # `import { X } from './services/user'` → services/user.ts
            m = re.match(r"""^import.*from\s+['"]([^'"]+)['"]""", cleaned)
            if not m:
                m = re.match(r"""^import\s+['"]([^'"]+)['"]""", cleaned)
            if not m:
                return None
            module_path = m.group(1).lstrip("./").lstrip("../")
            for ext in (".ts", ".tsx", ".js", ".jsx"):
                if (module_path + ext) in known_files:
                    return module_path + ext
            if (module_path + "/index.ts") in known_files:
                return module_path + "/index.ts"
            return None

        elif self.name == "rust":
            # "use crate::services::user;" → services/user.rs
            m = re.match(r"^use\s+(?:crate|super|self)::([\w:]+?)(?:::\{.*\})?$", cleaned)
            if not m:
                m = re.match(r"^use\s+([\w:]+?)(?:::\{.*\})?$", cleaned)
            if not m:
                return None
            path = m.group(1).replace("::", "/")
            if (path + ".rs") in known_files:
                return path + ".rs"
            # Last segment may be a type, not a module — try without it
            parts = path.rsplit("/", 1)
            if len(parts) > 1 and (parts[0] + ".rs") in known_files:
                return parts[0] + ".rs"
            return None

        elif self.name == "csharp":
            # "using Company.App.Services;" → Services.cs or Services/*.cs
            m = re.match(r"^using\s+(?:static\s+)?([A-Za-z][\w.]+)$", cleaned)
            if not m:
                return None
            last_seg = m.group(1).split(".")[-1]
            for path in known_files:
                if path.endswith(f"/{last_seg}.cs") or path == f"{last_seg}.cs":
                    return path
            return None

        return None


# ── Built-in profiles ────────────────────────────────────────────────────────

PYTHON = LanguageProfile(
    name="python",
    display_name="Python",
    file_extensions=[".py"],
    glob_pattern="**/*.py",
    docker_image="python:3.11-slim",
    test_command="pytest -v --tb=short",
    lint_command="ruff check",
    type_check_command="mypy",
    security_scan_command="bandit -r . -f json",
    build_command="",
    allowed_commands=["python", "pytest", "pip", "ruff", "mypy", "bandit",
                      "ls", "cat", "head", "tail", "grep", "find", "wc", "echo", "pwd", "env"],
    package_init_file="__init__.py",
    import_patterns=[r"^import\s+", r"^from\s+\S+\s+import\s+"],
    definition_patterns=[r"^class\s+(\w+)", r"^def\s+(\w+)"],
    module_separator=".",
    code_fence_name="python",
    source_root="src",
    test_root="tests",
    cache_paths=["/root/.cache/pip"],
)

JAVA = LanguageProfile(
    name="java",
    display_name="Java",
    file_extensions=[".java"],
    glob_pattern="**/*.java",
    docker_image="maven:3.9-eclipse-temurin-21-alpine",
    test_command="mvn test",
    lint_command="mvn checkstyle:check",
    type_check_command="",  # Java is compiled
    security_scan_command="mvn spotbugs:check",
    build_command="mvn package -DskipTests",
    allowed_commands=["java", "javac", "mvn", "gradle",
                      "ls", "cat", "head", "tail", "grep", "find", "wc", "echo", "pwd", "env"],
    package_init_file="",
    import_patterns=[r"^import\s+"],
    definition_patterns=[r"(?:public|private|protected)?\s*(?:static\s+)?(?:class|interface|enum|record)\s+(\w+)",
                         r"(?:public|private|protected)\s+(?:static\s+)?\S+\s+(\w+)\s*\("],
    module_separator=".",
    code_fence_name="java",
    source_root="src/main",
    test_root="src/test",
    cache_paths=["/root/.m2"],
)

GO = LanguageProfile(
    name="go",
    display_name="Go",
    file_extensions=[".go"],
    glob_pattern="**/*.go",
    docker_image="golang:1.22-alpine",
    test_command="go test ./... -v",
    lint_command="golangci-lint run",
    type_check_command="go vet ./...",
    security_scan_command="gosec ./...",
    build_command="go build ./...",
    allowed_commands=["go", "golangci-lint", "gosec",
                      "ls", "cat", "head", "tail", "grep", "find", "wc", "echo", "pwd", "env"],
    package_init_file="",
    import_patterns=[r'^import\s+[\("]+', r'^\s+"[^"]+"'],
    definition_patterns=[r"^func\s+(?:\([^)]+\)\s+)?(\w+)", r"^type\s+(\w+)\s+struct",
                         r"^type\s+(\w+)\s+interface"],
    module_separator="/",
    code_fence_name="go",
    cache_paths=["/go/pkg/mod", "/root/.cache/go-build"],
)

TYPESCRIPT = LanguageProfile(
    name="typescript",
    display_name="TypeScript",
    file_extensions=[".ts"],
    glob_pattern="**/*.ts",
    docker_image="node:20-alpine",
    test_command="npx jest --verbose",
    lint_command="npx eslint .",
    type_check_command="npx tsc --noEmit",
    security_scan_command="npx audit-ci --config audit-ci.json",
    build_command="npx tsc",
    allowed_commands=["node", "npm", "npx", "jest", "tsc",
                      "ls", "cat", "head", "tail", "grep", "find", "wc", "echo", "pwd", "env"],
    package_init_file="",
    import_patterns=[r'^import\s+', r'^import\s*\{', r"^import\s+.*\s+from\s+"],
    definition_patterns=[r"(?:export\s+)?(?:class|interface|enum|type)\s+(\w+)",
                         r"(?:export\s+)?(?:function|const|let)\s+(\w+)"],
    module_separator="/",
    code_fence_name="typescript",
    cache_paths=["/root/.npm"],
)

RUST = LanguageProfile(
    name="rust",
    display_name="Rust",
    file_extensions=[".rs"],
    glob_pattern="**/*.rs",
    docker_image="rust:1.77-alpine",
    test_command="cargo test",
    lint_command="cargo clippy",
    type_check_command="cargo check",
    security_scan_command="cargo audit",
    build_command="cargo build --release",
    allowed_commands=["cargo", "rustc",
                      "ls", "cat", "head", "tail", "grep", "find", "wc", "echo", "pwd", "env"],
    package_init_file="",
    import_patterns=[r"^use\s+"],
    definition_patterns=[r"^pub\s+(?:fn|struct|enum|trait|type)\s+(\w+)",
                         r"^fn\s+(\w+)", r"^struct\s+(\w+)", r"^enum\s+(\w+)"],
    module_separator="::",
    code_fence_name="rust",
    cache_paths=["/usr/local/cargo/registry"],
)

CSHARP = LanguageProfile(
    name="csharp",
    display_name="C#",
    file_extensions=[".cs"],
    glob_pattern="**/*.cs",
    docker_image="mcr.microsoft.com/dotnet/sdk:8.0-alpine",
    test_command="dotnet test",
    lint_command="dotnet format --verify-no-changes",
    type_check_command="dotnet build --no-restore",
    security_scan_command="dotnet tool run security-scan",
    build_command="dotnet build",
    allowed_commands=["dotnet",
                      "ls", "cat", "head", "tail", "grep", "find", "wc", "echo", "pwd", "env"],
    package_init_file="",
    import_patterns=[r"^using\s+"],
    definition_patterns=[r"(?:public|private|internal|protected)\s+(?:static\s+)?(?:class|interface|enum|struct|record)\s+(\w+)",
                         r"(?:public|private|internal|protected)\s+(?:static\s+)?\S+\s+(\w+)\s*\("],
    module_separator=".",
    code_fence_name="csharp",
    cache_paths=["/root/.nuget"],
)


# ── Registry ─────────────────────────────────────────────────────────────────

LANGUAGE_PROFILES: dict[str, LanguageProfile] = {
    "python": PYTHON,
    "java": JAVA,
    "go": GO,
    "golang": GO,
    "typescript": TYPESCRIPT,
    "ts": TYPESCRIPT,
    "rust": RUST,
    "rs": RUST,
    "csharp": CSHARP,
    "c#": CSHARP,
}


def get_language_profile(name: str) -> LanguageProfile:
    """Get a language profile by name. Defaults to Python if unknown."""
    profile = LANGUAGE_PROFILES.get(name.lower().strip())
    if profile is None:
        # Try to detect from common framework names
        name_lower = name.lower()
        if any(kw in name_lower for kw in ("spring", "maven", "gradle", "jdk")):
            return JAVA
        if any(kw in name_lower for kw in ("express", "nest", "node")):
            return TYPESCRIPT
        if any(kw in name_lower for kw in ("gin", "echo", "fiber")):
            return GO
        if any(kw in name_lower for kw in ("actix", "rocket", "tokio")):
            return RUST
        if any(kw in name_lower for kw in ("asp.net", "dotnet", ".net")):
            return CSHARP
        return PYTHON
    return profile


def detect_language_from_blueprint(tech_stack: dict[str, str]) -> LanguageProfile:
    """Detect language from a blueprint's tech_stack."""
    lang = tech_stack.get("language", "")
    if lang:
        return get_language_profile(lang)

    framework = tech_stack.get("framework", "")
    if framework:
        return get_language_profile(framework)

    return PYTHON


def detect_language_from_extensions(ext_counts: dict[str, int]) -> LanguageProfile:
    """Detect the primary language from a map of file-extension counts.

    Picks the language whose extensions have the highest total file count.
    Falls back to Python when no recognisable extensions are found.
    """
    # Map each profile's extensions to accumulated counts
    lang_scores: dict[str, int] = {}
    for profile in (PYTHON, JAVA, GO, TYPESCRIPT, RUST, CSHARP):
        score = sum(ext_counts.get(ext, 0) for ext in profile.file_extensions)
        if score > 0:
            lang_scores[profile.name] = score

    if not lang_scores:
        return PYTHON

    best = max(lang_scores, key=lambda k: lang_scores[k])
    return LANGUAGE_PROFILES[best]
