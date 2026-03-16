"""AST-based signature extraction using tree-sitter.

Instead of feeding raw source code to agents, this module extracts only the
public API surface (class signatures, method signatures, interface contracts,
field types) so the LLM context stays small and focused.

Architecture:
  - ``ASTExtractor`` is the public interface — language-agnostic
  - Each language has a dedicated ``_extract_<lang>()`` implementation
  - Languages without a tree-sitter grammar fall back to raw truncation
  - Results are cached per (path, checksum) to avoid re-parsing unchanged files

Currently supported: **Java**, **Python**
Planned: Go, TypeScript, Rust, C#
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Lazy-loaded tree-sitter grammars ────────────────────────────────────
# We load grammars on first use to avoid import-time failures if a grammar
# package isn't installed.
_PARSERS: dict[str, Any] = {}


def _get_parser(language: str) -> Any | None:
    """Return a tree-sitter Parser for *language*, or None if unavailable."""
    if language in _PARSERS:
        return _PARSERS[language]

    try:
        import tree_sitter  # noqa: F811

        if language == "java":
            import tree_sitter_java as ts_java
            lang = tree_sitter.Language(ts_java.language())
        elif language == "python":
            import tree_sitter_python as ts_python
            lang = tree_sitter.Language(ts_python.language())
        else:
            _PARSERS[language] = None
            return None

        parser = tree_sitter.Parser(lang)
        _PARSERS[language] = parser
        return parser
    except ImportError:
        logger.debug("tree-sitter grammar not available for %s", language)
        _PARSERS[language] = None
        return None


# ── Data models ─────────────────────────────────────────────────────────

@dataclass
class FieldSignature:
    """A class/record field."""
    name: str
    type_name: str
    modifiers: str = ""           # e.g. "private final"
    annotations: list[str] = field(default_factory=list)


@dataclass
class MethodSignature:
    """A method or constructor signature (no body)."""
    name: str
    return_type: str              # "" for constructors
    parameters: str               # e.g. "(Long id, String name)"
    modifiers: str = ""           # e.g. "public static"
    annotations: list[str] = field(default_factory=list)
    is_constructor: bool = False


@dataclass
class TypeSignature:
    """A class, interface, enum, or record."""
    name: str
    kind: str                     # "class" | "interface" | "enum" | "record" | "module"
    modifiers: str = ""
    annotations: list[str] = field(default_factory=list)
    extends: str = ""             # superclass or extends clause
    implements: list[str] = field(default_factory=list)
    fields: list[FieldSignature] = field(default_factory=list)
    methods: list[MethodSignature] = field(default_factory=list)
    enum_constants: list[str] = field(default_factory=list)
    record_components: str = ""   # e.g. "(String name, String email)"
    docstring: str = ""           # Python: class-level docstring


@dataclass
class FileSignature:
    """Extracted public API surface of an entire source file."""
    path: str
    language: str
    package: str = ""
    imports: list[str] = field(default_factory=list)
    types: list[TypeSignature] = field(default_factory=list)

    def to_stub(self) -> str:
        """Render a compact text stub suitable for LLM context injection.

        The stub contains only the information an LLM needs to write code that
        depends on this file: package, imports, type signatures, public method
        signatures, and field types.  Implementation bodies are omitted.
        """
        if self.language == "python":
            return self._to_python_stub()
        return self._to_java_stub()

    def _to_python_stub(self) -> str:
        """Python-style stub: ``def func(...) -> ret: ...`` syntax."""
        lines: list[str] = []

        if self.imports:
            for imp in self.imports:
                lines.append(imp)
            lines.append("")

        for t in self.types:
            if t.kind == "module":
                # Module-level functions — no enclosing class
                for m in t.methods:
                    for ann in m.annotations:
                        lines.append(ann)
                    sig = f"def {m.name}{m.parameters}"
                    if m.return_type:
                        sig += f" -> {m.return_type}"
                    lines.append(sig + ": ...")
                    lines.append("")
                continue

            # Class definition header
            if t.extends or t.implements:
                bases = ([t.extends] if t.extends else []) + list(t.implements)
                header = f"class {t.name}({', '.join(bases)}):"
            else:
                header = f"class {t.name}:"
            lines.append(header)

            # Class docstring
            if t.docstring:
                # Render as triple-quoted docstring, indented
                first_line = t.docstring.split("\n")[0].strip().strip("\"'")
                lines.append(f'    """{first_line}"""')
                lines.append("")

            if not t.methods:
                lines.append("    ...")
            else:
                for m in t.methods:
                    for ann in m.annotations:
                        lines.append(f"    {ann}")
                    sig = f"    def {m.name}{m.parameters}"
                    if m.return_type:
                        sig += f" -> {m.return_type}"
                    lines.append(sig + ": ...")
                    lines.append("")

            lines.append("")

        if not lines:
            return None
        return "\n".join(lines).strip() + "\n"

    def _to_java_stub(self) -> str:
        """Java-style stub (original implementation)."""
        lines: list[str] = []

        if self.package:
            lines.append(f"package {self.package};")
            lines.append("")

        if self.imports:
            for imp in self.imports:
                lines.append(imp)
            lines.append("")

        for t in self.types:
            # Annotations
            for ann in t.annotations:
                lines.append(ann)

            # Type header
            header = f"{t.modifiers} {t.kind} {t.name}".strip()
            if t.extends:
                header += f" {t.extends}"
            if t.implements:
                header += f" implements {', '.join(t.implements)}"
            if t.kind == "record" and t.record_components:
                # Insert components before body
                header += t.record_components

            lines.append(header + " {")

            # Enum constants
            if t.enum_constants:
                lines.append(f"    {', '.join(t.enum_constants)};")

            # Fields
            for f in t.fields:
                ann_str = " ".join(f.annotations)
                if ann_str:
                    lines.append(f"    {ann_str}")
                lines.append(f"    {f.modifiers} {f.type_name} {f.name};".strip() + ";")

            # Methods
            for m in t.methods:
                ann_str = " ".join(m.annotations)
                if ann_str:
                    lines.append(f"    {ann_str}")
                if m.is_constructor:
                    sig = f"    {m.modifiers} {m.name}{m.parameters}".strip()
                else:
                    sig = f"    {m.modifiers} {m.return_type} {m.name}{m.parameters}".strip()
                lines.append(sig + ";")

            lines.append("}")
            lines.append("")

        return "\n".join(lines).strip() + "\n"


# ── Signature cache ─────────────────────────────────────────────────────
# Keyed by (file_path, checksum) so we don't re-parse unchanged files.
_SIGNATURE_CACHE: dict[tuple[str, str], FileSignature] = {}


# ── Public API ──────────────────────────────────────────────────────────

class ASTExtractor:
    """Extracts public API signatures from source files using tree-sitter.

    Usage::

        extractor = ASTExtractor()
        sig = extractor.extract("com/example/UserService.java", source_code, "java")
        stub = sig.to_stub()   # compact text for LLM context
    """

    def extract(
        self,
        file_path: str,
        source_code: str,
        language: str,
        checksum: str = "",
    ) -> FileSignature | None:
        """Extract signatures from *source_code*.

        Returns ``None`` if the language is not supported by tree-sitter,
        in which case the caller should fall back to raw truncation.
        """
        # Check cache
        if checksum:
            cache_key = (file_path, checksum)
            if cache_key in _SIGNATURE_CACHE:
                return _SIGNATURE_CACHE[cache_key]

        lang = language.lower()
        parser = _get_parser(lang)
        if parser is None:
            return None

        if lang == "java":
            sig = _extract_java(parser, file_path, source_code)
        elif lang == "python":
            sig = _extract_python(parser, file_path, source_code)
        else:
            return None

        # Cache result
        if checksum:
            _SIGNATURE_CACHE[(file_path, checksum)] = sig

        return sig

    def extract_stub(
        self,
        file_path: str,
        source_code: str,
        language: str,
        checksum: str = "",
    ) -> str | None:
        """Extract and render a compact stub, or None if unsupported.

        Tries tree-sitter first, then falls back to regex-based extraction
        for languages without a tree-sitter grammar (TypeScript, Go, Rust, C#).
        Returns None only if both methods fail.
        """
        sig = self.extract(file_path, source_code, language, checksum)
        if sig:
            return sig.to_stub()

        # Regex fallback for unsupported languages
        stub = _extract_regex_stub(file_path, source_code, language)
        if stub:
            logger.debug(
                "Regex fallback stub for %s (%d→%d chars, %.0f%% reduction)",
                file_path, len(source_code), len(stub),
                (1 - len(stub) / len(source_code)) * 100 if source_code else 0,
            )
        return stub

    @staticmethod
    def is_supported(language: str) -> bool:
        """Check if AST extraction is available for *language*."""
        return _get_parser(language.lower()) is not None


# ── Java extraction ─────────────────────────────────────────────────────

def _node_text(node: Any, source: bytes) -> str:
    """Get the text content of a tree-sitter node."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _extract_modifiers(node: Any, source: bytes) -> tuple[str, list[str]]:
    """Extract modifier keywords and annotations from a 'modifiers' node.

    Returns (modifier_string, annotations_list).
    """
    modifiers: list[str] = []
    annotations: list[str] = []

    for child in node.children:
        if child.type in ("marker_annotation", "annotation"):
            annotations.append(_node_text(child, source))
        else:
            modifiers.append(_node_text(child, source))

    return " ".join(modifiers), annotations


def _extract_java_method(node: Any, source: bytes) -> MethodSignature:
    """Extract a method signature from a method_declaration node."""
    modifiers = ""
    annotations: list[str] = []
    return_type = ""
    name = ""
    parameters = ""

    for child in node.children:
        if child.type == "modifiers":
            modifiers, annotations = _extract_modifiers(child, source)
        elif child.type in (
            "type_identifier", "generic_type", "void_type",
            "integral_type", "floating_point_type", "boolean_type",
            "scoped_type_identifier", "array_type",
        ):
            return_type = _node_text(child, source)
        elif child.type == "identifier":
            name = _node_text(child, source)
        elif child.type == "formal_parameters":
            parameters = _node_text(child, source)

    return MethodSignature(
        name=name,
        return_type=return_type,
        parameters=parameters,
        modifiers=modifiers,
        annotations=annotations,
    )


def _extract_java_constructor(node: Any, source: bytes) -> MethodSignature:
    """Extract a constructor signature from a constructor_declaration node."""
    modifiers = ""
    annotations: list[str] = []
    name = ""
    parameters = ""

    for child in node.children:
        if child.type == "modifiers":
            modifiers, annotations = _extract_modifiers(child, source)
        elif child.type == "identifier":
            name = _node_text(child, source)
        elif child.type == "formal_parameters":
            parameters = _node_text(child, source)

    return MethodSignature(
        name=name,
        return_type="",
        parameters=parameters,
        modifiers=modifiers,
        annotations=annotations,
        is_constructor=True,
    )


def _extract_java_field(node: Any, source: bytes) -> FieldSignature | None:
    """Extract a field signature from a field_declaration node."""
    modifiers = ""
    annotations: list[str] = []
    type_name = ""
    field_name = ""

    for child in node.children:
        if child.type == "modifiers":
            modifiers, annotations = _extract_modifiers(child, source)
        elif child.type in (
            "type_identifier", "generic_type", "integral_type",
            "floating_point_type", "boolean_type", "scoped_type_identifier",
            "array_type",
        ):
            type_name = _node_text(child, source)
        elif child.type == "variable_declarator":
            # First identifier inside variable_declarator is the field name
            for vc in child.children:
                if vc.type == "identifier":
                    field_name = _node_text(vc, source)
                    break

    if not field_name:
        return None

    return FieldSignature(
        name=field_name,
        type_name=type_name,
        modifiers=modifiers,
        annotations=annotations,
    )


def _extract_java_type(node: Any, source: bytes) -> TypeSignature:
    """Extract a class/interface/enum/record from its declaration node."""
    kind_map = {
        "class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "record_declaration": "record",
    }
    kind = kind_map.get(node.type, "class")

    modifiers = ""
    annotations: list[str] = []
    name = ""
    extends = ""
    implements: list[str] = []
    fields: list[FieldSignature] = []
    methods: list[MethodSignature] = []
    enum_constants: list[str] = []
    record_components = ""

    for child in node.children:
        if child.type == "modifiers":
            modifiers, annotations = _extract_modifiers(child, source)
        elif child.type == "identifier":
            name = _node_text(child, source)
        elif child.type == "superclass":
            extends = _node_text(child, source)
        elif child.type == "extends_interfaces":
            extends = _node_text(child, source)
        elif child.type == "super_interfaces":
            impl_text = _node_text(child, source)
            # "implements Foo, Bar" → ["Foo", "Bar"]
            if impl_text.startswith("implements "):
                implements = [s.strip() for s in impl_text[11:].split(",")]
        elif child.type == "formal_parameters":
            # Record components
            record_components = _node_text(child, source)
        elif child.type in ("class_body", "interface_body", "enum_body"):
            # Collect all member nodes — enum bodies nest methods inside
            # an 'enum_body_declarations' wrapper, so we flatten one level.
            members = []
            for member in child.children:
                if member.type == "enum_body_declarations":
                    members.extend(member.children)
                else:
                    members.append(member)

            for member in members:
                if member.type == "method_declaration":
                    methods.append(_extract_java_method(member, source))
                elif member.type == "constructor_declaration":
                    methods.append(_extract_java_constructor(member, source))
                elif member.type == "field_declaration":
                    f = _extract_java_field(member, source)
                    if f:
                        fields.append(f)
                elif member.type == "enum_constant":
                    enum_constants.append(_node_text(member, source).split("(")[0].strip())

    return TypeSignature(
        name=name,
        kind=kind,
        modifiers=modifiers,
        annotations=annotations,
        extends=extends,
        implements=implements,
        fields=fields,
        methods=methods,
        enum_constants=enum_constants,
        record_components=record_components,
    )


def _extract_java(parser: Any, file_path: str, source_code: str) -> FileSignature:
    """Extract all public signatures from a Java source file."""
    source = source_code.encode("utf-8")
    tree = parser.parse(source)
    root = tree.root_node

    package = ""
    imports: list[str] = []
    types: list[TypeSignature] = []

    for node in root.children:
        if node.type == "package_declaration":
            # Extract package name (skip 'package' keyword and ';')
            for child in node.children:
                if child.type == "scoped_identifier" or child.type == "identifier":
                    package = _node_text(child, source)
        elif node.type == "import_declaration":
            imports.append(_node_text(node, source).rstrip(";").strip())
        elif node.type in (
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
        ):
            types.append(_extract_java_type(node, source))

    return FileSignature(
        path=file_path,
        language="java",
        package=package,
        imports=imports,
        types=types,
    )


# ── Python extraction ────────────────────────────────────────────────────────

def _extract_python(parser: Any, file_path: str, source_code: str) -> FileSignature:
    """Extract public API signatures from a Python source file.

    Captures:
    - Module-level imports (import / from … import)
    - Class definitions with bases and class-level docstring
    - Method signatures (name, full parameter list with annotations, return type)
    - Module-level function signatures
    - Decorators (@property, @staticmethod, @classmethod, @abstractmethod, …)

    Does NOT capture:
    - Function/method bodies
    - Private methods (prefix ``__``) except ``__init__``
    - Module-level variables (only imports)
    """
    source = source_code.encode("utf-8")
    tree = parser.parse(source)
    root = tree.root_node

    imports: list[str] = []
    types: list[TypeSignature] = []
    module_functions: list[MethodSignature] = []

    for node in root.children:
        if node.type in ("import_statement", "import_from_statement"):
            imports.append(_node_text(node, source).strip())
        elif node.type == "class_definition":
            t = _extract_python_class(node, source)
            if t is not None:
                types.append(t)
        elif node.type == "decorated_definition":
            decorators, inner = _split_decorated(node, source)
            if inner is None:
                continue
            if inner.type == "class_definition":
                t = _extract_python_class(inner, source, decorators)
                if t is not None:
                    types.append(t)
            elif inner.type == "function_definition":
                m = _extract_python_function(inner, source, decorators)
                if m is not None and not _is_private_method(m.name):
                    module_functions.append(m)
        elif node.type == "function_definition":
            m = _extract_python_function(node, source)
            if m is not None and not _is_private_method(m.name):
                module_functions.append(m)

    # Wrap module-level functions in a synthetic TypeSignature
    if module_functions:
        types.insert(
            0,
            TypeSignature(name="", kind="module", methods=module_functions),
        )

    return FileSignature(
        path=file_path,
        language="python",
        imports=imports,
        types=types,
    )


def _is_private_method(name: str) -> bool:
    """Return True for dunder methods other than ``__init__``."""
    return name.startswith("__") and name != "__init__"


def _split_decorated(
    node: Any, source: bytes
) -> tuple[list[str], Any | None]:
    """Split a ``decorated_definition`` into (decorators, inner_definition)."""
    decorators: list[str] = []
    inner = None
    for child in node.children:
        if child.type == "decorator":
            decorators.append(_node_text(child, source).strip())
        elif child.type in ("class_definition", "function_definition"):
            inner = child
    return decorators, inner


def _extract_python_class(
    node: Any,
    source: bytes,
    class_decorators: list[str] | None = None,
) -> TypeSignature | None:
    """Extract a Python class definition into a ``TypeSignature``."""
    name = ""
    bases: list[str] = []
    docstring = ""
    methods: list[MethodSignature] = []

    for child in node.children:
        if child.type == "identifier":
            name = _node_text(child, source)
        elif child.type == "argument_list":
            # Base classes — filter out punctuation tokens
            for arg in child.children:
                if arg.type not in (",", "(", ")"):
                    bases.append(_node_text(arg, source).strip())
        elif child.type == "block":
            first_checked = False
            for stmt in child.children:
                # First expression_statement may be the class docstring
                if not first_checked and stmt.type == "expression_statement":
                    first_checked = True
                    for sub in stmt.children:
                        if sub.type in ("string", "concatenated_string"):
                            raw = _node_text(sub, source).strip()
                            # Unwrap triple quotes
                            for q in ('"""', "'''", '"', "'"):
                                if raw.startswith(q) and raw.endswith(q) and len(raw) > 2 * len(q):
                                    raw = raw[len(q):-len(q)]
                                    break
                            docstring = raw.strip().split("\n")[0]
                            break
                    continue

                if stmt.type == "function_definition":
                    m = _extract_python_function(stmt, source)
                    if m is not None and not _is_private_method(m.name):
                        methods.append(m)
                elif stmt.type == "decorated_definition":
                    decs, inner = _split_decorated(stmt, source)
                    if inner and inner.type == "function_definition":
                        m = _extract_python_function(inner, source, decs)
                        if m is not None and not _is_private_method(m.name):
                            methods.append(m)

    if not name:
        return None

    return TypeSignature(
        name=name,
        kind="class",
        annotations=class_decorators or [],
        extends=bases[0] if bases else "",
        implements=bases[1:] if len(bases) > 1 else [],
        methods=methods,
        docstring=docstring,
    )


def _extract_python_function(
    node: Any,
    source: bytes,
    decorators: list[str] | None = None,
) -> MethodSignature | None:
    """Extract a Python function/method definition into a ``MethodSignature``."""
    name = ""
    parameters = ""
    return_type = ""

    for child in node.children:
        t = child.type
        if t == "identifier":
            name = _node_text(child, source)
        elif t == "parameters":
            parameters = _node_text(child, source)
        elif t == "type":
            # The ``->`` return annotation lives here
            return_type = _node_text(child, source)

    if not name:
        return None

    return MethodSignature(
        name=name,
        return_type=return_type,
        parameters=parameters,
        annotations=decorators or [],
    )


# ── Regex-based fallback extraction ──────────────────────────────────────
#
# For languages without tree-sitter support (TypeScript, Go, Rust, C#),
# this extracts imports + declaration signatures via regex.  It skips
# function/method bodies by tracking brace depth, producing a compact
# stub (~2KB) instead of blind char-truncation (~8-20KB).

# Patterns that match import / use / require lines across languages.
_IMPORT_RE = re.compile(
    r"^(?:"
    r"import\s+.+|"                        # Java, Python, TS, Go
    r"from\s+\S+\s+import\s+.+|"          # Python
    r"(?:const|let|var)\s+.+=\s*require\(.+|"  # Node CJS
    r"use\s+.+|"                           # Rust
    r"using\s+.+"                          # C#
    r")$",
    re.MULTILINE,
)

# Patterns that match type/class/interface/struct/enum declaration headers.
_DECL_HEADER_RE = re.compile(
    r"^[ \t]*"
    r"(?:export\s+)?(?:default\s+)?(?:abstract\s+)?"
    r"(?:pub(?:\(crate\))?\s+)?"
    r"(?:async\s+)?"
    r"(?:"
    r"(?:class|interface|struct|enum|type|trait|impl|namespace|module)\s+"
    r"|func\s+"                             # Go top-level func
    r"|fn\s+"                               # Rust fn
    r"|function\s+"                         # TS/JS function
    r"|(?:public|private|protected|internal|static)\s+"  # C# member prefix
    r")"
    r".+",
    re.MULTILINE,
)

# Decorator / annotation lines that precede declarations.
_DECORATOR_RE = re.compile(r"^[ \t]*(?:@\w+|#\[.+\]).*$", re.MULTILINE)


def _extract_regex_stub(file_path: str, source_code: str, language: str) -> str | None:
    """Extract a compact signature stub using regex heuristics.

    Works for any language — extracts imports, type headers, and
    function/method signatures while skipping implementation bodies.
    Returns None only if the file has no recognizable declarations.
    """
    lines = source_code.split("\n")
    out: list[str] = []
    brace_depth = 0
    in_body = False
    body_start_depth = 0

    # Phase 1: collect imports (always at top)
    in_import_block = False  # Go: import ( ... )
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("#"):
            continue
        # Handle Go multi-line import blocks: import ( ... )
        if in_import_block:
            out.append(line.rstrip())
            if ")" in stripped:
                in_import_block = False
            continue
        if _IMPORT_RE.match(stripped):
            out.append(line.rstrip())
            # Check for Go-style `import (` without closing paren
            if stripped.startswith("import") and "(" in stripped and ")" not in stripped:
                in_import_block = True
        elif stripped.startswith("package ") or stripped.startswith("namespace "):
            out.append(line.rstrip())
        else:
            # Stop collecting imports once we hit non-import code
            break

    if out:
        out.append("")

    # Phase 2: extract declarations, skip bodies
    i = 0
    pending_decorators: list[str] = []
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines and comments when not tracking body
        if not in_body:
            if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
                i += 1
                continue

            # Collect decorators/annotations
            if _DECORATOR_RE.match(stripped):
                pending_decorators.append(line.rstrip())
                i += 1
                continue

            # Check if this is a declaration header
            if _DECL_HEADER_RE.match(stripped):
                # Emit pending decorators
                out.extend(pending_decorators)
                pending_decorators = []

                # Extract just the signature (up to opening brace)
                sig_line = line.rstrip()

                # Count braces on this line
                open_b = stripped.count("{") - stripped.count("}")

                if _is_type_declaration(stripped):
                    # Type/class/struct: emit header, enter body-skipping at depth 1
                    # but still extract nested method signatures
                    out.append(sig_line)
                    if open_b > 0:
                        brace_depth = open_b
                        in_body = True
                        body_start_depth = 0  # we want to extract members at depth 1
                elif "{" in stripped:
                    # Function with body on same line — emit sig only
                    sig_only = sig_line.split("{")[0].rstrip()
                    out.append(sig_only + ";")
                    # Skip the body
                    if open_b > 0:
                        brace_depth = open_b
                        in_body = True
                        body_start_depth = brace_depth
                else:
                    # No brace yet (e.g., Go func signature spans lines)
                    out.append(sig_line)
            else:
                pending_decorators = []

        else:
            # We're inside a body — track braces
            open_b = stripped.count("{") - stripped.count("}")
            prev_depth = brace_depth
            brace_depth += open_b

            # At depth 1 inside a type body, extract member signatures
            if prev_depth == 1 and body_start_depth == 0:
                if _DECORATOR_RE.match(stripped):
                    out.append(line.rstrip())
                elif _is_member_signature(stripped):
                    sig_only = line.rstrip()
                    if "{" in sig_only:
                        sig_only = sig_only.split("{")[0].rstrip() + ";"
                    out.append(sig_only)

            if brace_depth <= 0:
                if body_start_depth == 0:
                    out.append("}")  # close the type
                    out.append("")
                in_body = False
                brace_depth = 0

        i += 1

    # Filter out duplicate/empty and check we got something useful
    result = "\n".join(out).strip()
    if not result or len(result) < 20:
        return None

    return result + "\n"


def _is_type_declaration(line: str) -> bool:
    """Check if a line declares a type (class/interface/struct/enum/trait/impl)."""
    # Remove common prefixes
    stripped = re.sub(
        r"^(?:export\s+)?(?:default\s+)?(?:abstract\s+)?(?:pub(?:\(crate\))?\s+)?",
        "", line,
    ).strip()
    return bool(re.match(
        r"(?:"
        r"(?:class|interface|struct|enum|trait|impl|namespace|module)\s"
        r"|type\s+\w+\s*=\s*\{"       # TS: type X = {
        r"|type\s+\w+\s+(?:struct|interface)\s"  # Go: type X struct {
        r")",
        stripped,
    ))


def _is_member_signature(line: str) -> bool:
    """Check if a line looks like a method/field declaration inside a type body."""
    return bool(re.match(
        r"(?:"
        r"(?:public|private|protected|internal|static|abstract|async|override|readonly|final|virtual)\s+.+"
        r"|(?:fn|func|function|def|get|set)\s+\w+"
        r"|(?:const|let|var|val)\s+\w+"
        r"|(?:pub(?:\(crate\))?\s+)?fn\s+\w+"  # Rust methods
        r"|\w+\s*\(.*\)\s*(?::\s*\w+)?"  # TS method shorthand: name(...): Type
        r"|[a-zA-Z_]\w*\s+[*&]?[a-zA-Z_][\w.<>\[\]*&]*"  # Go struct fields: name Type
        r"|(?:pub\s+)?[a-zA-Z_]\w*\s*:\s*\S+"  # Rust struct fields: name: Type
        r")",
        line,
    ))
