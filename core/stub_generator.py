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


def _infer_method_signature_java(
    name: str, layer: str, is_interface: bool,
) -> str | None:
    """Infer a compilable Java method stub from an export name and layer.

    Returns a method declaration string, or None if the export looks like a
    class/type name (PascalCase with no parens) rather than a method.
    """
    # Skip class/type names (PascalCase, no parens)
    if name and name[0].isupper() and "(" not in name and "_" not in name:
        return None

    # Strip any trailing parens the architect may have included
    name = name.split("(")[0].strip()
    if not name:
        return None

    body = ";" if is_interface else " { return null; }"
    void_body = ";" if is_interface else " {}"
    lower = name.lower()

    # Repository layer: standard CRUD signatures
    if layer in ("repository", "dao"):
        if lower.startswith("findall") or lower.startswith("getall"):
            return f"    public java.util.List<Object> {name}(){body}"
        if lower.startswith("find") or lower.startswith("get"):
            return f"    public Object {name}(Long id){body}"
        if lower in ("save", "create", "insert", "update"):
            return f"    public Object {name}(Object entity){body}"
        if lower.startswith("delete") or lower.startswith("remove"):
            return f"    public void {name}(Long id){void_body}"
        if lower.startswith("exists") or lower.startswith("count"):
            return f"    public boolean {name}(Long id){body}"
        return f"    public Object {name}(Object arg){body}"

    # Controller layer — Spring REST annotations for better compilation
    if layer == "controller":
        if lower.startswith(("register", "create", "add", "signup")):
            return f"    public org.springframework.http.ResponseEntity<Object> {name}(Object request){body}"
        if lower.startswith(("get", "find", "fetch", "load")):
            if "all" in lower or "list" in lower:
                return f"    public org.springframework.http.ResponseEntity<java.util.List<Object>> {name}(){body}"
            return f"    public org.springframework.http.ResponseEntity<Object> {name}(Long id){body}"
        if lower.startswith(("update", "edit", "modify", "patch")):
            return f"    public org.springframework.http.ResponseEntity<Object> {name}(Long id, Object request){body}"
        if lower.startswith(("delete", "remove")):
            return f"    public org.springframework.http.ResponseEntity<Void> {name}(Long id){body}"
        if lower in ("login", "authenticate", "signin"):
            return f"    public org.springframework.http.ResponseEntity<Object> {name}(Object credentials){body}"
        if lower in ("logout", "signout"):
            return f"    public org.springframework.http.ResponseEntity<Void> {name}(){body}"
        return f"    public org.springframework.http.ResponseEntity<Object> {name}(Object arg){body}"

    # Service / Handler layers
    if layer in ("service", "handler", "router", "resource", "middleware"):
        if lower.startswith(("register", "create", "add", "signup")):
            return f"    public Object {name}(Object request){body}"
        if lower.startswith(("get", "find", "fetch", "load")):
            return f"    public Object {name}(Long id){body}"
        if lower.startswith(("update", "edit", "modify", "patch")):
            return f"    public Object {name}(Long id, Object request){body}"
        if lower.startswith(("delete", "remove")):
            return f"    public void {name}(Long id){void_body}"
        if lower in ("login", "authenticate", "signin"):
            return f"    public Object {name}(Object credentials){body}"
        if lower in ("logout", "signout"):
            return f"    public void {name}(){void_body}"
        return f"    public Object {name}(Object arg){body}"

    # Default: generic method stub
    return f"    public Object {name}(Object arg){body}"


def _infer_method_signature_go(name: str, layer: str) -> str | None:
    """Infer a compilable Go function stub from an export name."""
    if not name or not name[0].isupper():
        return None  # unexported
    return f"func {name}() interface{{}} {{ return nil }}"


def _infer_method_signature_ts(name: str, layer: str) -> str | None:
    """Infer a TypeScript function/method export stub."""
    if not name:
        return None
    # PascalCase → class export; lowercase/camelCase → function
    if name[0].isupper() and "_" not in name:
        return None  # class/type, handled separately
    return f"export function {name}(...args: any[]): any {{ return null as any; }}"


def _infer_method_signature_csharp(
    name: str, layer: str, is_interface: bool,
) -> str | None:
    """Infer a C# method stub."""
    if name and name[0].isupper() and "_" not in name and "(" not in name:
        return None  # class/type name
    name = name.split("(")[0].strip()
    if not name:
        return None
    body = ";" if is_interface else " => default!;"
    return f"        public object {name}(object arg) {body}"


class StubGenerator:
    """Creates minimal type stubs for files that haven't been generated yet.

    Stubs are kept in an in-memory overlay and only flushed to disk just
    before a build checkpoint needs them.  This avoids 2× disk I/O per stub
    (write + delete) on every checkpoint cycle.  Call ``flush()`` to write
    all pending stubs to disk, and ``cleanup_stubs()`` to remove them after
    the build.
    """

    def __init__(
        self,
        language_name: str,
        workspace: Path,
    ) -> None:
        self._lang = language_name
        self._workspace = workspace
        # Virtual overlay: path → stub content (not yet on disk)
        self._overlay: dict[str, str] = {}
        # Paths that were actually written to disk (need cleanup)
        self._on_disk: set[str] = set()

    def generate_stubs(
        self,
        stub_files: list[str],
        blueprints: dict[str, object] | None = None,
    ) -> list[str]:
        """Add stubs to the in-memory overlay for the given paths.

        Only generates stubs for files that don't already exist on disk.
        Returns list of paths that were added (for tracking).
        Call ``flush()`` to write them to disk before a build.

        Args:
            stub_files: Workspace-relative paths to create stubs for.
            blueprints: Optional mapping of path → FileBlueprint for richer stubs.
        """
        created: list[str] = []

        for file_path in stub_files:
            abs_path = self._workspace / file_path
            if abs_path.exists():
                continue  # file already generated, no stub needed
            if file_path in self._overlay:
                continue  # already in overlay

            content = self._generate_stub_content(file_path, blueprints)
            if content is None:
                continue

            self._overlay[file_path] = content
            created.append(file_path)
            logger.debug("Queued stub (in-memory): %s", file_path)

        if created:
            logger.info("Queued %d stubs for forward references: %s", len(created), created)
        return created

    def flush(self) -> int:
        """Write all in-memory stubs to disk.

        Called just before a build checkpoint so the compiler can see them.
        Returns the number of stubs written.
        """
        written = 0
        for file_path, content in self._overlay.items():
            abs_path = self._workspace / file_path
            if abs_path.exists():
                continue  # real file appeared in the meantime
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content, encoding="utf-8")
            self._on_disk.add(file_path)
            written += 1
            logger.debug("Flushed stub to disk: %s", file_path)
        if written:
            logger.info("Flushed %d stubs to disk for build checkpoint", written)
        return written

    def cleanup_stubs(self, stub_files: list[str]) -> None:
        """Delete previously generated stub files from disk and overlay.

        Called before the actual file generation begins for a tier.
        """
        for file_path in stub_files:
            self._overlay.pop(file_path, None)
            if file_path in self._on_disk:
                abs_path = self._workspace / file_path
                if abs_path.exists():
                    abs_path.unlink()
                    logger.debug("Cleaned up stub: %s", file_path)
                self._on_disk.discard(file_path)

    def cleanup_all(self) -> None:
        """Remove all stubs from disk and clear the overlay."""
        for file_path in list(self._on_disk):
            abs_path = self._workspace / file_path
            if abs_path.exists():
                abs_path.unlink()
        self._on_disk.clear()
        self._overlay.clear()

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
            return self._go_stub(file_path, blueprints)
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
        """Generate a Java class/interface stub with compilable method signatures."""
        parts = file_path.replace("\\", "/")
        class_name = parts.split("/")[-1].replace(".java", "")

        # Determine package from path
        package_path = parts
        for prefix in ("src/main/java/", "src/"):
            if package_path.startswith(prefix):
                package_path = package_path[len(prefix):]
                break
        package_parts = package_path.rsplit("/", 1)
        package = package_parts[0].replace("/", ".") if len(package_parts) > 1 else ""

        lines: list[str] = []
        if package:
            lines.append(f"package {package};")
            lines.append("")

        is_interface = class_name.endswith("Repository") or class_name.startswith("I")

        exports: list[str] = []
        layer = ""
        if blueprints and file_path in blueprints:
            bp = blueprints[file_path]
            if hasattr(bp, "exports"):
                exports = bp.exports  # type: ignore[attr-defined]
            if hasattr(bp, "layer"):
                layer = bp.layer or ""  # type: ignore[attr-defined]

        # Add Spring annotations for controllers
        if layer == "controller" and not is_interface:
            lines.append("import org.springframework.web.bind.annotation.RestController;")
            lines.append("import org.springframework.http.ResponseEntity;")
            lines.append("")
            lines.append("@RestController")

        if is_interface:
            lines.append(f"public interface {class_name} {{")
        else:
            lines.append(f"public class {class_name} {{")
            lines.append(f"    public {class_name}() {{}}")

        # Generate compilable method stubs from exports
        for export in exports[:10]:
            sig = _infer_method_signature_java(export, layer, is_interface)
            if sig:
                lines.append(sig)

        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _go_stub(file_path: str, blueprints: dict[str, object] | None = None) -> str:
        """Generate a Go package stub with exported function/struct stubs."""
        parts = file_path.replace("\\", "/").split("/")
        if len(parts) > 1:
            package = parts[-2]
        else:
            package = "main"

        lines = [f"package {package}", ""]

        exports: list[str] = []
        layer = ""
        if blueprints and file_path in blueprints:
            bp = blueprints[file_path]
            if hasattr(bp, "exports"):
                exports = bp.exports  # type: ignore[attr-defined]
            if hasattr(bp, "layer"):
                layer = bp.layer or ""  # type: ignore[attr-defined]

        for export in exports[:10]:
            if not export:
                continue
            # PascalCase names → exported struct or function
            if export[0].isupper() and "_" not in export:
                # Could be a struct type
                lines.append(f"type {export} struct{{}}")
            else:
                sig = _infer_method_signature_go(export, layer)
                if sig:
                    lines.append(sig)
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _typescript_stub(
        file_path: str, blueprints: dict[str, object] | None
    ) -> str:
        """Generate a TypeScript stub with exported class/function stubs."""
        parts = file_path.replace("\\", "/")
        name = parts.split("/")[-1].replace(".ts", "").replace(".tsx", "")
        class_name = name[0].upper() + name[1:] if name else "Stub"

        exports: list[str] = []
        layer = ""
        if blueprints and file_path in blueprints:
            bp = blueprints[file_path]
            if hasattr(bp, "exports"):
                exports = bp.exports  # type: ignore[attr-defined]
            if hasattr(bp, "layer"):
                layer = bp.layer or ""  # type: ignore[attr-defined]

        lines: list[str] = []
        class_methods: list[str] = []
        standalone_funcs: list[str] = []

        for export in exports[:10]:
            if not export:
                continue
            sig = _infer_method_signature_ts(export, layer)
            if sig:
                standalone_funcs.append(sig)
            elif export[0].isupper():
                # Class/type export — will be added as a class
                pass  # handled by default class below

        # Always emit the primary class
        lines.append(f"export class {class_name} {{")
        # Add camelCase exports as methods on the class
        for export in exports[:10]:
            if export and not export[0].isupper():
                lines.append(f"    {export}(...args: any[]): any {{ return null as any; }}")
        lines.append("}")

        # Standalone function exports
        for func in standalone_funcs:
            lines.append(func)

        # Additional type exports
        for export in exports[:10]:
            if export and export[0].isupper() and export != class_name:
                lines.append(f"export interface {export} {{ [key: string]: any; }}")

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _rust_stub(file_path: str, blueprints: dict[str, object] | None) -> str:
        """Generate a Rust stub with struct and impl block."""
        name = file_path.split("/")[-1].replace(".rs", "")
        struct_name = "".join(w.capitalize() for w in name.split("_"))

        exports: list[str] = []
        if blueprints and file_path in blueprints:
            bp = blueprints[file_path]
            if hasattr(bp, "exports"):
                exports = bp.exports  # type: ignore[attr-defined]

        lines = [f"pub struct {struct_name};", ""]

        # Add impl block with method stubs for non-type exports
        methods: list[str] = []
        for export in exports[:10]:
            if not export:
                continue
            if export[0].isupper() and "_" not in export:
                if export != struct_name:
                    lines.append(f"pub struct {export};")
            else:
                methods.append(f"    pub fn {export}(&self) -> Option<()> {{ None }}")

        if methods:
            lines.append(f"impl {struct_name} {{")
            lines.extend(methods)
            lines.append("}")

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _csharp_stub(
        file_path: str, blueprints: dict[str, object] | None
    ) -> str:
        """Generate a C# stub with method signatures."""
        class_name = file_path.split("/")[-1].replace(".cs", "")
        parts = file_path.replace("\\", "/").rsplit("/", 1)
        namespace = parts[0].replace("/", ".") if len(parts) > 1 else "App"

        exports: list[str] = []
        layer = ""
        is_interface = class_name.startswith("I") and len(class_name) > 1 and class_name[1].isupper()
        if blueprints and file_path in blueprints:
            bp = blueprints[file_path]
            if hasattr(bp, "exports"):
                exports = bp.exports  # type: ignore[attr-defined]
            if hasattr(bp, "layer"):
                layer = bp.layer or ""  # type: ignore[attr-defined]

        keyword = "interface" if is_interface else "class"
        lines = [
            f"namespace {namespace}",
            "{",
            f"    public {keyword} {class_name}",
            "    {",
        ]

        for export in exports[:10]:
            sig = _infer_method_signature_csharp(export, layer, is_interface)
            if sig:
                lines.append(sig)

        lines.append("    }")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _kotlin_stub(
        file_path: str, blueprints: dict[str, object] | None
    ) -> str:
        """Generate a Kotlin class/interface stub with method signatures."""
        parts = file_path.replace("\\", "/")
        class_name = parts.split("/")[-1].replace(".kt", "")

        package_path = parts
        for prefix in ("src/main/kotlin/", "src/main/java/", "src/"):
            if package_path.startswith(prefix):
                package_path = package_path[len(prefix):]
                break
        package_parts = package_path.rsplit("/", 1)
        package = package_parts[0].replace("/", ".") if len(package_parts) > 1 else ""

        exports: list[str] = []
        layer = ""
        if blueprints and file_path in blueprints:
            bp = blueprints[file_path]
            if hasattr(bp, "exports"):
                exports = bp.exports  # type: ignore[attr-defined]
            if hasattr(bp, "layer"):
                layer = bp.layer or ""  # type: ignore[attr-defined]

        lines: list[str] = []
        if package:
            lines.append(f"package {package}")
            lines.append("")

        is_interface = class_name.endswith("Repository") or class_name.startswith("I")
        if is_interface:
            lines.append(f"interface {class_name} {{")
        else:
            lines.append(f"open class {class_name} {{")

        for export in exports[:10]:
            if not export or (export[0].isupper() and "_" not in export):
                continue
            name = export.split("(")[0].strip()
            if is_interface:
                lines.append(f"    fun {name}(vararg args: Any?): Any?")
            else:
                lines.append(f"    fun {name}(vararg args: Any?): Any? = null")

        lines.append("}")
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
