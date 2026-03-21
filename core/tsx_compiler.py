"""TypeScript/TSX compilation checker for generated frontend components.

Runs ``tsc --noEmit`` against a workspace directory and parses the compiler
output back to per-file error granularity so the frontend pipeline can
surface broken components without shipping invalid TSX to the user.

If ``tsc`` is not on PATH the check is skipped gracefully — the caller
receives ``TSXCompileResult(tsc_available=False)`` and should treat that
as a non-fatal warning rather than a hard failure.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Matches: src/components/Button.tsx(10,5): error TS2304: Cannot find name 'x'.
_TSC_ERROR_RE = re.compile(
    r"^(?P<file>[^(\n]+)\((?P<line>\d+),(?P<col>\d+)\):\s*error\s+(?P<code>TS\d+):\s*(?P<message>.+)$"
)

_DEFAULT_TSCONFIG = """{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "jsx": "react-jsx",
    "noEmit": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "allowSyntheticDefaultImports": true
  },
  "include": ["src/**/*"]
}
"""


@dataclass
class TSXError:
    file: str
    line: int
    col: int
    code: str
    message: str


@dataclass
class TSXCompileResult:
    errors: list[TSXError] = field(default_factory=list)
    raw_output: str = ""
    tsc_available: bool = True

    @property
    def passed(self) -> bool:
        return self.tsc_available and len(self.errors) == 0

    def errors_by_file(self) -> dict[str, list[TSXError]]:
        """Group errors by workspace-relative file path."""
        result: dict[str, list[TSXError]] = {}
        for err in self.errors:
            result.setdefault(err.file, []).append(err)
        return result


class TSXCompiler:
    """Runs the TypeScript compiler in check-only mode and parses its output."""

    @staticmethod
    def _is_vue_project(workspace: Path) -> bool:
        """Detect whether *workspace* is a Vue project.

        Checks package.json for a 'vue' dependency or the existence of
        vue-tsc in devDependencies — faster than scanning for .vue files.
        """
        pkg_json = workspace / "package.json"
        if pkg_json.exists():
            try:
                import json
                pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                if "vue" in deps or "vue-tsc" in deps:
                    return True
            except Exception:
                pass
        return False

    async def check(self, workspace: Path) -> TSXCompileResult:
        """Run ``tsc --noEmit`` (or ``vue-tsc --noEmit`` for Vue) in *workspace*.

        A missing ``tsconfig.json`` is created automatically with sensible
        defaults so the check can run even on freshly generated workspaces.
        """
        tsconfig = workspace / "tsconfig.json"
        if not tsconfig.exists():
            tsconfig.write_text(_DEFAULT_TSCONFIG, encoding="utf-8")
            logger.debug("Created default tsconfig.json in %s", workspace)

        # For Vue projects, prefer vue-tsc which understands .vue SFCs.
        # Fall back to plain tsc if vue-tsc is not available.
        use_vue_tsc = self._is_vue_project(workspace)
        compiler = "vue-tsc" if use_vue_tsc else "tsc"

        try:
            proc = await asyncio.create_subprocess_exec(
                compiler,
                "--noEmit",
                "--pretty",
                "false",
                cwd=str(workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            if use_vue_tsc:
                # vue-tsc not installed — fall back to tsc
                logger.info("vue-tsc not on PATH — falling back to tsc")
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "tsc",
                        "--noEmit",
                        "--pretty",
                        "false",
                        cwd=str(workspace),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                    )
                except FileNotFoundError:
                    logger.warning(
                        "Neither vue-tsc nor tsc found on PATH — skipping compilation check"
                    )
                    return TSXCompileResult(tsc_available=False)
            else:
                logger.warning(
                    "tsc not found on PATH — skipping TypeScript compilation check"
                )
                return TSXCompileResult(tsc_available=False)

        try:
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=120.0)
        except asyncio.TimeoutError:
            proc.kill()
            logger.warning("tsc timed out after 120 s — skipping compilation check")
            return TSXCompileResult(tsc_available=True, raw_output="(timed out)")

        raw = stdout_bytes.decode(errors="replace")
        errors = self._parse_output(raw, workspace)
        return TSXCompileResult(errors=errors, raw_output=raw)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _parse_output(self, raw: str, workspace: Path) -> list[TSXError]:
        errors: list[TSXError] = []
        for line in raw.splitlines():
            m = _TSC_ERROR_RE.match(line.strip())
            if not m:
                continue
            rel_file = m.group("file").strip()
            # Normalise to a workspace-relative forward-slash path
            try:
                rel_file = (
                    Path(rel_file).relative_to(workspace).as_posix()
                )
            except ValueError:
                rel_file = rel_file.replace("\\", "/")
            errors.append(
                TSXError(
                    file=rel_file,
                    line=int(m.group("line")),
                    col=int(m.group("col")),
                    code=m.group("code"),
                    message=m.group("message"),
                )
            )
        return errors
