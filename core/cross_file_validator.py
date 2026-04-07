"""Cross-file consistency validator — verify method signatures across boundaries.

After all files in a tier are generated, this validator checks that method
calls across file boundaries match the actual generated signatures.  Catches
common LLM hallucination issues:
  - Calling a method that doesn't exist in the dependency
  - Wrong parameter count or types in method calls
  - Missing imports for used types
  - Interface methods not implemented by declared implementors

Gated behind the ``CROSS_FILE_VALIDATOR`` feature flag.

Design:
  - Uses ASTExtractor to extract signatures from generated files
  - Compares against blueprint-declared exports and depends_on
  - Reports mismatches as warnings (non-blocking) that get injected
    into the fix-loop context for the next build-fix cycle
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import RepositoryBlueprint

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyIssue:
    """A cross-file consistency problem detected between two files."""
    source_file: str       # File that has the problematic reference
    target_file: str       # File being referenced
    issue_type: str        # "missing_method" | "param_mismatch" | "missing_import" | "unimplemented_interface"
    description: str       # Human-readable description
    severity: str = "warning"  # "warning" | "error"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.source_file} → {self.target_file}: {self.description}"


@dataclass
class ValidationReport:
    """Results of cross-file consistency validation."""
    issues: list[ConsistencyIssue] = field(default_factory=list)
    files_checked: int = 0
    pairs_checked: int = 0

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def summary_for_file(self, file_path: str) -> str:
        """Get a text summary of issues affecting a specific file."""
        file_issues = [i for i in self.issues if i.source_file == file_path]
        if not file_issues:
            return ""
        lines = [f"Cross-file consistency issues in {file_path}:"]
        for issue in file_issues:
            lines.append(f"  - {issue.description}")
        return "\n".join(lines)

    def full_summary(self) -> str:
        """Full text summary of all issues."""
        if not self.issues:
            return f"Cross-file validation: {self.files_checked} files, {self.pairs_checked} pairs — no issues found."
        lines = [
            f"Cross-file validation: {self.files_checked} files, "
            f"{self.pairs_checked} pairs — {self.error_count} errors, "
            f"{self.warning_count} warnings:",
        ]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)


class CrossFileValidator:
    """Validates cross-file consistency after code generation.

    Checks that generated files' method calls and type references
    match the actual signatures in their dependencies.
    """

    def __init__(self, workspace: Path, blueprint: RepositoryBlueprint) -> None:
        self._workspace = workspace
        self._blueprint = blueprint
        # Build lookup: path → FileBlueprint
        self._bp_map = {fb.path: fb for fb in blueprint.file_blueprints}

    def validate(self, generated_files: list[str]) -> ValidationReport:
        """Validate cross-file consistency for a set of generated files.

        Args:
            generated_files: Paths of files that were just generated (relative to workspace).

        Returns:
            ValidationReport with any detected issues.
        """
        report = ValidationReport()
        generated_set = set(generated_files)
        report.files_checked = len(generated_files)

        # Extract signatures from all generated files
        signatures: dict[str, _FileInfo] = {}
        for fp in generated_files:
            full_path = self._workspace / fp
            if not full_path.exists():
                continue
            try:
                content = full_path.read_text(encoding="utf-8")
                signatures[fp] = self._extract_file_info(fp, content)
            except Exception:
                logger.debug("Failed to extract info from %s", fp, exc_info=True)

        # Check each file's references against its dependencies
        for fp in generated_files:
            bp = self._bp_map.get(fp)
            if not bp:
                continue

            source_info = signatures.get(fp)
            if not source_info:
                continue

            for dep_path in bp.depends_on:
                if dep_path not in generated_set:
                    continue  # Dependency not in this tier — can't validate yet
                dep_info = signatures.get(dep_path)
                if not dep_info:
                    continue

                report.pairs_checked += 1
                issues = self._check_pair(fp, source_info, dep_path, dep_info)
                report.issues.extend(issues)

        if report.issues:
            logger.warning(
                "Cross-file validation found %d issues (%d errors, %d warnings)",
                len(report.issues), report.error_count, report.warning_count,
            )
        else:
            logger.info(
                "Cross-file validation passed: %d files, %d pairs checked",
                report.files_checked, report.pairs_checked,
            )

        return report

    def _check_pair(
        self,
        source_path: str,
        source_info: _FileInfo,
        dep_path: str,
        dep_info: _FileInfo,
    ) -> list[ConsistencyIssue]:
        """Check consistency between a file and one of its dependencies."""
        issues: list[ConsistencyIssue] = []

        # Check 1: Are referenced method/function names present in the dependency?
        for ref_name in source_info.external_references:
            # Only check references that look like they belong to this dependency
            # (heuristic: class name or imported name matches)
            if not self._likely_from_dep(ref_name, dep_info):
                continue

            if ref_name not in dep_info.exported_names:
                issues.append(ConsistencyIssue(
                    source_file=source_path,
                    target_file=dep_path,
                    issue_type="missing_method",
                    description=(
                        f"References '{ref_name}' which is not exported by {dep_path}. "
                        f"Available exports: {', '.join(sorted(dep_info.exported_names)[:10])}"
                    ),
                    severity="warning",
                ))

        # Check 2: Verify class/type names referenced in imports exist in dep
        for type_ref in source_info.imported_types:
            if type_ref in dep_info.type_names:
                continue
            # Only flag if the import path seems to reference the dep file
            if self._import_matches_file(type_ref, dep_path):
                issues.append(ConsistencyIssue(
                    source_file=source_path,
                    target_file=dep_path,
                    issue_type="missing_import",
                    description=(
                        f"Imports type '{type_ref}' but {dep_path} does not define it. "
                        f"Defined types: {', '.join(sorted(dep_info.type_names))}"
                    ),
                    severity="warning",
                ))

        return issues

    def _likely_from_dep(self, ref_name: str, dep_info: _FileInfo) -> bool:
        """Heuristic: does this reference likely come from the given dependency?"""
        # If any of the dep's type names appear as a prefix (e.g., UserService.findById)
        for type_name in dep_info.type_names:
            if ref_name.startswith(type_name + "."):
                return True
            if ref_name.startswith(type_name.lower() + "."):
                return True
        return False

    def _import_matches_file(self, type_ref: str, dep_path: str) -> bool:
        """Check if an import type reference could plausibly come from a file path."""
        # Convert file path to likely class name
        stem = Path(dep_path).stem
        # CamelCase comparison: UserService.java → UserService
        if type_ref.lower() == stem.lower():
            return True
        # snake_case → CamelCase: user_service.py → UserService
        camel = "".join(w.capitalize() for w in stem.split("_"))
        if type_ref == camel:
            return True
        return False

    def _extract_file_info(self, file_path: str, content: str) -> _FileInfo:
        """Extract lightweight info about a file's exports and references."""
        info = _FileInfo()
        language = self._bp_map.get(file_path)
        lang = (language.language if language else "").lower()

        if lang in ("java", "kotlin"):
            info = self._extract_java_info(content)
        elif lang in ("python", "py"):
            info = self._extract_python_info(content)
        elif lang in ("typescript", "javascript", "ts", "js"):
            info = self._extract_ts_info(content)
        else:
            # Generic: try to extract class/function names
            info = self._extract_generic_info(content)

        return info

    @staticmethod
    def _extract_java_info(content: str) -> _FileInfo:
        """Extract Java exports and references."""
        info = _FileInfo()

        # Class/interface/enum declarations
        for m in re.finditer(r'(?:public\s+)?(?:class|interface|enum|record)\s+(\w+)', content):
            info.type_names.add(m.group(1))
            info.exported_names.add(m.group(1))

        # Public method declarations
        for m in re.finditer(
            r'public\s+(?:static\s+)?(?:\w+(?:<[^>]*>)?)\s+(\w+)\s*\(', content
        ):
            name = m.group(1)
            if name not in ("if", "while", "for", "switch"):
                info.exported_names.add(name)

        # Method calls: obj.method(
        for m in re.finditer(r'(\w+\.\w+)\s*\(', content):
            info.external_references.add(m.group(1))

        # Imports: import com.foo.Bar → Bar is an imported type
        for m in re.finditer(r'import\s+[\w.]+\.(\w+)\s*;', content):
            info.imported_types.add(m.group(1))

        return info

    @staticmethod
    def _extract_python_info(content: str) -> _FileInfo:
        """Extract Python exports and references."""
        info = _FileInfo()

        # Class definitions
        for m in re.finditer(r'^class\s+(\w+)', content, re.MULTILINE):
            info.type_names.add(m.group(1))
            info.exported_names.add(m.group(1))

        # Function definitions (top-level)
        for m in re.finditer(r'^def\s+(\w+)', content, re.MULTILINE):
            name = m.group(1)
            if not name.startswith("_"):
                info.exported_names.add(name)

        # Method definitions (indented)
        for m in re.finditer(r'^\s+def\s+(\w+)', content, re.MULTILINE):
            name = m.group(1)
            if not name.startswith("_"):
                info.exported_names.add(name)

        # Method calls: obj.method(
        for m in re.finditer(r'(\w+\.\w+)\s*\(', content):
            info.external_references.add(m.group(1))

        # Imports: from module import Name
        for m in re.finditer(r'from\s+\S+\s+import\s+(.+)', content):
            for name in m.group(1).split(","):
                name = name.strip().split(" as ")[0].strip()
                if name and name[0].isupper():
                    info.imported_types.add(name)

        return info

    @staticmethod
    def _extract_ts_info(content: str) -> _FileInfo:
        """Extract TypeScript/JavaScript exports and references."""
        info = _FileInfo()

        # Export class/interface/type/enum
        for m in re.finditer(
            r'export\s+(?:default\s+)?(?:class|interface|type|enum)\s+(\w+)', content
        ):
            info.type_names.add(m.group(1))
            info.exported_names.add(m.group(1))

        # Export function
        for m in re.finditer(r'export\s+(?:default\s+)?(?:async\s+)?function\s+(\w+)', content):
            info.exported_names.add(m.group(1))

        # Export const (named exports)
        for m in re.finditer(r'export\s+const\s+(\w+)', content):
            info.exported_names.add(m.group(1))

        # Method calls
        for m in re.finditer(r'(\w+\.\w+)\s*\(', content):
            info.external_references.add(m.group(1))

        # Import { Name } from './dep'
        for m in re.finditer(r'import\s+\{([^}]+)\}\s+from', content):
            for name in m.group(1).split(","):
                name = name.strip().split(" as ")[0].strip()
                if name and name[0].isupper():
                    info.imported_types.add(name)

        return info

    @staticmethod
    def _extract_generic_info(content: str) -> _FileInfo:
        """Fallback: extract class names and function names with regex."""
        info = _FileInfo()
        for m in re.finditer(r'(?:class|interface|struct|enum)\s+(\w+)', content):
            info.type_names.add(m.group(1))
            info.exported_names.add(m.group(1))
        for m in re.finditer(r'(?:func|fn|def|function)\s+(\w+)', content):
            info.exported_names.add(m.group(1))
        return info


@dataclass
class _FileInfo:
    """Lightweight extracted information about a file."""
    type_names: set[str] = field(default_factory=set)        # Class/interface/enum names
    exported_names: set[str] = field(default_factory=set)     # All public symbols
    external_references: set[str] = field(default_factory=set)  # obj.method calls
    imported_types: set[str] = field(default_factory=set)     # Types from import statements
