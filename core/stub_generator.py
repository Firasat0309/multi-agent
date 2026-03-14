"""Stub generator for forward references in tier-based compilation.

When Tier 0 files reference types from Tier 1 (e.g., a model importing an
interface that hasn't been generated yet), the build checkpoint fails with
"type not found" errors.  The stub generator creates minimal placeholder
files for ungenerated tiers so the current tier can compile.

Stubs are deleted before the actual file is generated.

Supported languages:
  - Java: empty class/interface with correct package declaration
  - Kotlin: empty class/interface with correct package declaration
  - Go: package declaration + empty struct
  - TypeScript: empty exported interface/class
  - Rust: empty pub struct
  - C#: empty class in correct namespace
  - Python: importable module with placeholder class/function exports

Usage::

    generator = StubGenerator(lang_profile, workspace)
    stubs = generator.generate_stubs(
        stub_files=["services/UserService.java"],
        blueprints=file_blueprints_map,
    )
    # ... run build checkpoint ...
    generator.cleanup_stubs(stubs)
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class StubGenerator:
    """Creates minimal type stubs for files that haven't been generated yet."""

    def __init__(
        self,
        language_name: str,
        workspace: Path,
    ) -> None:
        self._lang = language_name
        self._workspace = workspace

    def generate_stubs(
        self,
        stub_files: list[str],
        blueprints: dict[str, object] | None = None,
    ) -> list[str]:
        """Generate stub files on disk for the given paths.

        Only creates stubs for files that don't already exist.
        Returns list of paths that were actually created (for cleanup).

        Args:
            stub_files: Workspace-relative paths to create stubs for.
            blueprints: Optional mapping of path → FileBlueprint for richer stubs.
        """
        created: list[str] = []

        for file_path in stub_files:
            abs_path = self._workspace / file_path
            if abs_path.exists():
                continue  # file already generated, no stub needed

            content = self._generate_stub_content(file_path, blueprints)
            if content is None:
                continue

            # Ensure parent dirs exist
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content, encoding="utf-8")
            created.append(file_path)
            logger.debug("Created stub: %s", file_path)

        if created:
            logger.info("Generated %d stubs for forward references: %s", len(created), created)
        return created

    def cleanup_stubs(self, stub_files: list[str]) -> None:
        """Delete previously generated stub files.

        Called before the actual file generation begins for a tier.
        """
        for file_path in stub_files:
            abs_path = self._workspace / file_path
            if abs_path.exists():
                abs_path.unlink()
                logger.debug("Cleaned up stub: %s", file_path)

    def _generate_stub_content(
        self,
        file_path: str,
        blueprints: dict[str, object] | None,
    ) -> str | None:
        """Generate language-appropriate stub content."""
        if self._lang == "java":
            return self._java_stub(file_path, blueprints)
        elif self._lang == "kotlin":
            return self._kotlin_stub(file_path, blueprints)
        elif self._lang == "go":
            return self._go_stub(file_path)
        elif self._lang == "typescript":
            return self._typescript_stub(file_path, blueprints)
        elif self._lang == "rust":
            return self._rust_stub(file_path, blueprints)
        elif self._lang == "csharp":
            return self._csharp_stub(file_path, blueprints)
        elif self._lang == "python":
            return self._python_stub(file_path, blueprints)
        return None

    @staticmethod
    def _java_stub(file_path: str, blueprints: dict[str, object] | None) -> str:
        """Generate a minimal Java class stub."""
        # Extract class name from file path
        parts = file_path.replace("\\", "/")
        class_name = parts.split("/")[-1].replace(".java", "")

        # Determine package from path
        # e.g., src/main/java/com/example/services/UserService.java
        # → package com.example.services;
        package_path = parts
        for prefix in ("src/main/java/", "src/"):
            if package_path.startswith(prefix):
                package_path = package_path[len(prefix):]
                break
        package_parts = package_path.rsplit("/", 1)
        package = package_parts[0].replace("/", ".") if len(package_parts) > 1 else ""

        lines = []
        if package:
            lines.append(f"package {package};")
            lines.append("")

        # Check if it's an interface based on naming convention
        is_interface = class_name.endswith("Repository") or class_name.startswith("I")

        exports = []
        if blueprints and file_path in blueprints:
            bp = blueprints[file_path]
            if hasattr(bp, "exports"):
                exports = bp.exports  # type: ignore[attr-defined]

        if is_interface:
            lines.append(f"public interface {class_name} {{")
        else:
            lines.append(f"public class {class_name} {{")

        # Add stub methods from exports if available
        for export in exports[:5]:  # limit to 5 stubs
            if export[0].isupper() and "(" not in export:
                continue  # Skip class names
            lines.append(f"    // stub: {export}")

        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _go_stub(file_path: str) -> str:
        """Generate a minimal Go package stub."""
        parts = file_path.replace("\\", "/").split("/")
        if len(parts) > 1:
            package = parts[-2]  # directory name is package name
        else:
            package = "main"
        return f"package {package}\n"

    @staticmethod
    def _typescript_stub(
        file_path: str, blueprints: dict[str, object] | None
    ) -> str:
        """Generate a minimal TypeScript stub."""
        parts = file_path.replace("\\", "/")
        name = parts.split("/")[-1].replace(".ts", "").replace(".tsx", "")
        # PascalCase the name for the class
        class_name = name[0].upper() + name[1:] if name else "Stub"

        lines = [f"export class {class_name} {{}}"]
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _rust_stub(file_path: str, blueprints: dict[str, object] | None) -> str:
        """Generate a minimal Rust stub."""
        name = file_path.split("/")[-1].replace(".rs", "")
        struct_name = "".join(w.capitalize() for w in name.split("_"))
        return f"pub struct {struct_name};\n"

    @staticmethod
    def _csharp_stub(
        file_path: str, blueprints: dict[str, object] | None
    ) -> str:
        """Generate a minimal C# stub."""
        class_name = file_path.split("/")[-1].replace(".cs", "")
        # Guess namespace from path
        parts = file_path.replace("\\", "/").rsplit("/", 1)
        namespace = parts[0].replace("/", ".") if len(parts) > 1 else "App"
        return (
            f"namespace {namespace}\n"
            f"{{\n"
            f"    public class {class_name} {{ }}\n"
            f"}}\n"
        )

    @staticmethod
    def _kotlin_stub(
        file_path: str, blueprints: dict[str, object] | None
    ) -> str:
        """Generate a minimal Kotlin class/interface stub.

        Mirrors the Java stub logic but uses Kotlin syntax.  Interfaces are
        detected by the same naming convention (suffix ``Repository`` or
        prefix ``I``).
        """
        parts = file_path.replace("\\", "/")
        class_name = parts.split("/")[-1].replace(".kt", "")

        # Determine package from path
        package_path = parts
        for prefix in ("src/main/kotlin/", "src/main/java/", "src/"):
            if package_path.startswith(prefix):
                package_path = package_path[len(prefix):]
                break
        package_parts = package_path.rsplit("/", 1)
        package = package_parts[0].replace("/", ".") if len(package_parts) > 1 else ""

        lines: list[str] = []
        if package:
            lines.append(f"package {package}")
            lines.append("")

        is_interface = class_name.endswith("Repository") or class_name.startswith("I")
        if is_interface:
            lines.append(f"interface {class_name}")
        else:
            lines.append(f"class {class_name}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _python_stub(
        file_path: str, blueprints: dict[str, object] | None
    ) -> str:
        """Generate a minimal Python module stub.

        Creates an importable module with placeholder exports.  For files
        that declare exports in the blueprint, each export is stubbed as
        either a class (PascalCase) or a function (snake_case) so that
        downstream ``import`` statements resolve successfully.
        """
        exports: list[str] = []
        if blueprints and file_path in blueprints:
            bp = blueprints[file_path]
            if hasattr(bp, "exports"):
                exports = bp.exports  # type: ignore[attr-defined]

        lines: list[str] = ['"""Auto-generated stub for forward references."""', ""]

        if not exports:
            # Minimal stub — just enough to make the module importable
            lines.append("")
            return "\n".join(lines)

        for export in exports[:10]:
            # PascalCase → class stub; otherwise → function stub
            if export and export[0].isupper() and "_" not in export:
                lines.append(f"class {export}:")
                lines.append("    pass")
            else:
                lines.append(f"def {export}(*args, **kwargs):")
                lines.append("    raise NotImplementedError")
            lines.append("")

        return "\n".join(lines)
