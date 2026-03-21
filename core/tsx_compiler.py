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

# tsc --pretty false format:
#   src/components/Button.tsx(10,5): error TS2304: Cannot find name 'x'.
# vue-tsc format (2.x+, even with --pretty false):
#   src/components/Button.vue:10:5 - error TS2304: Cannot find name 'x'.
_TSC_ERROR_RE = re.compile(
    r"^(?P<file>[^(\n]+)\((?P<line>\d+),(?P<col>\d+)\):\s*error\s+(?P<code>TS\d+):\s*(?P<message>.+)$"
)
_VUE_TSC_ERROR_RE = re.compile(
    r"^(?P<file>[^:\n]+):(?P<line>\d+):(?P<col>\d+)\s+-\s+error\s+(?P<code>TS\d+):\s*(?P<message>.+)$"
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

    @staticmethod
    def _resolve_compiler(workspace: Path, use_vue_tsc: bool) -> list[str]:
        """Resolve the compiler command, returning args list for subprocess.

        Resolution order:
        1. Local ``node_modules/.bin/<compiler>`` (most reliable after npm install)
        2. ``npx <compiler>`` (resolves from node_modules automatically)
        3. Bare ``<compiler>`` command (requires global install on PATH)
        """
        import platform
        is_win = platform.system() == "Windows"
        ext = ".cmd" if is_win else ""

        preferred = "vue-tsc" if use_vue_tsc else "tsc"
        fallback = "tsc" if use_vue_tsc else None

        for name in (preferred, fallback):
            if name is None:
                continue
            local_bin = workspace / "node_modules" / ".bin" / f"{name}{ext}"
            if local_bin.exists():
                logger.debug("Using local compiler: %s", local_bin)
                return [str(local_bin)]

        # No local binary — use npx which resolves from node_modules
        # automatically and works even when .bin isn't on PATH.
        npx = "npx.cmd" if is_win else "npx"
        return [npx, preferred]

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

        # Resolve the compiler command — prefers local node_modules/.bin/,
        # falls back to npx which resolves from node_modules automatically.
        compiler_cmd = self._resolve_compiler(workspace, use_vue_tsc)
        logger.debug("TypeScript check command: %s", compiler_cmd)

        try:
            proc = await asyncio.create_subprocess_exec(
                *compiler_cmd,
                "--noEmit",
                "--pretty",
                "false",
                cwd=str(workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            if use_vue_tsc:
                # vue-tsc not found even via npx — fall back to tsc
                fallback_cmd = self._resolve_compiler(workspace, use_vue_tsc=False)
                logger.info(
                    "%s not found — falling back to %s",
                    " ".join(compiler_cmd), " ".join(fallback_cmd),
                )
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *fallback_cmd,
                        "--noEmit",
                        "--pretty",
                        "false",
                        cwd=str(workspace),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                    )
                except FileNotFoundError:
                    logger.warning(
                        "Neither vue-tsc nor tsc found (local, npx, or PATH) — "
                        "skipping compilation check"
                    )
                    return TSXCompileResult(tsc_available=False)
            else:
                logger.warning(
                    "tsc not found (local, npx, or PATH) — "
                    "skipping TypeScript compilation check"
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
            stripped = line.strip()
            m = _TSC_ERROR_RE.match(stripped) or _VUE_TSC_ERROR_RE.match(stripped)
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
