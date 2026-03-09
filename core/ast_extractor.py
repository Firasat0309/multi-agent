"""AST-based signature extraction using tree-sitter.

Instead of feeding raw source code to agents, this module extracts only the
public API surface (class signatures, method signatures, interface contracts,
field types) so the LLM context stays small and focused.

Architecture:
  - ``ASTExtractor`` is the public interface — language-agnostic
  - Each language has a dedicated ``_extract_<lang>()`` implementation
  - Languages without a tree-sitter grammar fall back to raw truncation
  - Results are cached per (path, checksum) to avoid re-parsing unchanged files

Currently supported: **Java**
Planned: Python, Go, TypeScript, Rust, C#
"""

from __future__ import annotations

import logging
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
        # Future languages go here:
        # elif language == "python":
        #     import tree_sitter_python as ts_python
        #     lang = tree_sitter.Language(ts_python.language())
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
    kind: str                     # "class" | "interface" | "enum" | "record"
    modifiers: str = ""
    annotations: list[str] = field(default_factory=list)
    extends: str = ""             # superclass or extends clause
    implements: list[str] = field(default_factory=list)
    fields: list[FieldSignature] = field(default_factory=list)
    methods: list[MethodSignature] = field(default_factory=list)
    enum_constants: list[str] = field(default_factory=list)
    record_components: str = ""   # e.g. "(String name, String email)"


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
        """Extract and render a compact stub, or None if unsupported."""
        sig = self.extract(file_path, source_code, language, checksum)
        return sig.to_stub() if sig else None

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
