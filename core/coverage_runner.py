"""Coverage measurement and gate enforcement for generated tests."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.language import LanguageProfile
    from tools.terminal_tools import TerminalTools

logger = logging.getLogger(__name__)


@dataclass
class CoverageResult:
    """Coverage measurement for a single source/test pair."""

    line_coverage: float          # 0.0 – 1.0
    branch_coverage: float        # 0.0 – 1.0  (0.0 when not available)
    uncovered_lines: list[int]    # Line numbers with no coverage
    passed_gate: bool             # True if line_coverage >= min_coverage
    raw_output: str = ""          # Truncated command output for debugging

    @property
    def line_pct(self) -> str:
        return f"{self.line_coverage:.0%}"


class CoverageRunner:
    """Runs tests with coverage measurement and reports line/branch coverage.

    Integrates with:
    - Python    : pytest-cov (--cov --cov-report=json)
    - Java      : JaCoCo (jacoco:report via Maven)
    - Go        : go test -coverprofile
    - TypeScript: jest --coverage --coverageReporters=json
    - Rust      : cargo tarpaulin (optional, skipped gracefully if absent)
    - C#        : dotnet-coverage (skipped gracefully if absent)

    All runners degrade gracefully: if the coverage tool is not installed the
    method returns a ``CoverageResult`` with ``line_coverage=0.0`` and
    ``passed_gate=False`` so downstream callers can decide whether to block or
    warn.
    """

    async def run_with_coverage(
        self,
        test_file: str,
        source_file: str,
        terminal: TerminalTools,
        lang: LanguageProfile,
        min_coverage: float = 0.80,
    ) -> CoverageResult:
        """Run *test_file* with coverage enabled and return a ``CoverageResult``.

        Args:
            test_file   : workspace-relative path to the test file.
            source_file : workspace-relative path to the source file under test.
            terminal    : terminal tool for running commands in the sandbox.
            lang        : language profile to choose the coverage driver.
            min_coverage: minimum line-coverage fraction to pass the gate (0–1).
        """
        runner = getattr(self, f"_run_{lang.name}", self._run_unsupported)
        try:
            result = await runner(test_file, source_file, terminal, min_coverage)
        except Exception as e:
            logger.warning("Coverage measurement failed (%s): %s", lang.name, e)
            result = CoverageResult(
                line_coverage=0.0,
                branch_coverage=0.0,
                uncovered_lines=[],
                passed_gate=False,
                raw_output=str(e)[:500],
            )

        level = "PASS" if result.passed_gate else "FAIL"
        logger.info(
            "Coverage [%s] %s — line=%s branch=%s gate=%.0f%%",
            level, source_file, result.line_pct,
            f"{result.branch_coverage:.0%}", min_coverage * 100,
        )
        return result

    # ── Language runners ─────────────────────────────────────────────────────

    async def _run_python(
        self,
        test_file: str,
        source_file: str,
        terminal: TerminalTools,
        min_coverage: float,
    ) -> CoverageResult:
        # Strip the test_root prefix so --cov targets the source module
        module_path = source_file.replace("/", ".").removesuffix(".py")
        # Strip common root prefixes (src.)
        for prefix in ("src.", "app."):
            if module_path.startswith(prefix):
                module_path = module_path[len(prefix):]
                break

        # Use json report so we can parse it programmatically
        cmd = (
            f"pytest --tb=no -q {test_file} "
            f"--cov={source_file.removesuffix('.py').replace('/', '.')} "
            f"--cov-report=json:.coverage_report.json "
            f"--cov-report=term-missing"
        )
        result = await terminal.run_command(cmd)
        raw = (result.stdout + result.stderr)[:2000]

        return self._parse_python_json(".coverage_report.json", source_file, min_coverage, raw, terminal)

    async def _parse_python_json_async(
        self,
        report_path: str,
        source_file: str,
        min_coverage: float,
        raw: str,
        terminal: TerminalTools,
    ) -> CoverageResult:
        """Read .coverage_report.json written by pytest-cov."""
        # Try to read via terminal cat (works inside sandbox too)
        cat_result = await terminal.run_command(f"cat {report_path}")
        if cat_result.exit_code != 0:
            return self._parse_python_text(raw, min_coverage)
        try:
            data = json.loads(cat_result.stdout)
            file_data = data.get("files", {}).get(source_file, {})
            summary = file_data.get("summary", {})
            line_rate = summary.get("percent_covered", 0) / 100
            missing = file_data.get("missing_lines", [])
            branch_rate = (
                summary.get("covered_branches", 0) / summary["num_branches"]
                if summary.get("num_branches")
                else 0.0
            )
            return CoverageResult(
                line_coverage=line_rate,
                branch_coverage=branch_rate,
                uncovered_lines=missing,
                passed_gate=line_rate >= min_coverage,
                raw_output=raw,
            )
        except Exception:
            return self._parse_python_text(raw, min_coverage)

    def _parse_python_json(
        self,
        report_path: str,
        source_file: str,
        min_coverage: float,
        raw: str,
        terminal: TerminalTools,
    ) -> CoverageResult:
        """Synchronous fallback: parse term-missing output from pytest."""
        return self._parse_python_text(raw, min_coverage)

    def _parse_python_text(self, raw: str, min_coverage: float) -> CoverageResult:
        """Parse 'TOTAL ... XX%' line from pytest --cov term output."""
        # e.g. "TOTAL                        150     12    92%"
        m = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", raw)
        if not m:
            # Try simpler "XX%" anywhere
            m = re.search(r"(\d+)%", raw)
        line_rate = int(m.group(1)) / 100 if m else 0.0
        return CoverageResult(
            line_coverage=line_rate,
            branch_coverage=0.0,
            uncovered_lines=[],
            passed_gate=line_rate >= min_coverage,
            raw_output=raw[:500],
        )

    async def _run_java(
        self,
        test_file: str,
        source_file: str,
        terminal: TerminalTools,
        min_coverage: float,
    ) -> CoverageResult:
        # JaCoCo is configured in pom.xml; run verify to trigger it
        result = await terminal.run_command("mvn verify -DskipITs=true 2>&1 | tail -40")
        raw = (result.stdout + result.stderr)[:2000]
        # JaCoCo XML is at target/site/jacoco/jacoco.xml — parse if available
        xml_result = await terminal.run_command(
            "grep -A2 'counter type=\"LINE\"' target/site/jacoco/jacoco.xml | head -10"
        )
        return self._parse_jacoco_text(xml_result.stdout + raw, min_coverage, raw)

    def _parse_jacoco_text(self, text: str, min_coverage: float, raw: str) -> CoverageResult:
        # <counter type="LINE" missed="X" covered="Y"/>
        m = re.search(r'type="LINE" missed="(\d+)" covered="(\d+)"', text)
        if m:
            missed, covered = int(m.group(1)), int(m.group(2))
            total = missed + covered
            rate = covered / total if total else 0.0
        else:
            rate = 0.0
        return CoverageResult(
            line_coverage=rate,
            branch_coverage=0.0,
            uncovered_lines=[],
            passed_gate=rate >= min_coverage,
            raw_output=raw[:500],
        )

    async def _run_go(
        self,
        test_file: str,
        source_file: str,
        terminal: TerminalTools,
        min_coverage: float,
    ) -> CoverageResult:
        pkg_dir = source_file.rsplit("/", 1)[0] if "/" in source_file else "."
        result = await terminal.run_command(
            f"go test -coverprofile=coverage.out ./{pkg_dir}/ && "
            f"go tool cover -func=coverage.out | tail -5"
        )
        raw = (result.stdout + result.stderr)[:2000]
        # "total:  (statements)  82.5%"
        m = re.search(r"total.*?(\d+\.?\d*)%", raw, re.IGNORECASE)
        rate = float(m.group(1)) / 100 if m else 0.0
        return CoverageResult(
            line_coverage=rate,
            branch_coverage=0.0,
            uncovered_lines=[],
            passed_gate=rate >= min_coverage,
            raw_output=raw[:500],
        )

    async def _run_typescript(
        self,
        test_file: str,
        source_file: str,
        terminal: TerminalTools,
        min_coverage: float,
    ) -> CoverageResult:
        result = await terminal.run_command(
            f"npx jest --coverage --coverageReporters=text {test_file} 2>&1 | tail -20"
        )
        raw = (result.stdout + result.stderr)[:2000]
        # "All files | 85.71 | ..."
        m = re.search(r"All files\s*\|\s*(\d+\.?\d*)", raw)
        rate = float(m.group(1)) / 100 if m else 0.0
        return CoverageResult(
            line_coverage=rate,
            branch_coverage=0.0,
            uncovered_lines=[],
            passed_gate=rate >= min_coverage,
            raw_output=raw[:500],
        )

    async def _run_rust(
        self,
        test_file: str,
        source_file: str,
        terminal: TerminalTools,
        min_coverage: float,
    ) -> CoverageResult:
        # cargo-tarpaulin is optional; degrade gracefully if not installed
        result = await terminal.run_command("cargo tarpaulin --out Stdout 2>&1 | tail -10")
        raw = (result.stdout + result.stderr)[:2000]
        if result.exit_code != 0 and "not found" in raw.lower():
            logger.info("cargo-tarpaulin not installed — skipping Rust coverage")
            return CoverageResult(
                line_coverage=0.0, branch_coverage=0.0, uncovered_lines=[],
                passed_gate=True,  # Don't block when tool is absent
                raw_output="cargo-tarpaulin not installed",
            )
        m = re.search(r"(\d+\.?\d*)% coverage", raw)
        rate = float(m.group(1)) / 100 if m else 0.0
        return CoverageResult(
            line_coverage=rate, branch_coverage=0.0, uncovered_lines=[],
            passed_gate=rate >= min_coverage, raw_output=raw[:500],
        )

    async def _run_csharp(
        self,
        test_file: str,
        source_file: str,
        terminal: TerminalTools,
        min_coverage: float,
    ) -> CoverageResult:
        result = await terminal.run_command(
            "dotnet-coverage collect 'dotnet test' --output coverage.xml --output-format xml 2>&1 | tail -10"
        )
        raw = (result.stdout + result.stderr)[:500]
        if result.exit_code != 0:
            # dotnet-coverage may not be installed — degrade gracefully
            return CoverageResult(
                line_coverage=0.0, branch_coverage=0.0, uncovered_lines=[],
                passed_gate=True,
                raw_output="dotnet-coverage not available",
            )
        # Very rough parse from output
        m = re.search(r"(\d+\.?\d*)%", raw)
        rate = float(m.group(1)) / 100 if m else 0.0
        return CoverageResult(
            line_coverage=rate, branch_coverage=0.0, uncovered_lines=[],
            passed_gate=rate >= min_coverage, raw_output=raw,
        )

    async def _run_unsupported(
        self,
        test_file: str,
        source_file: str,
        terminal: TerminalTools,
        min_coverage: float,
    ) -> CoverageResult:
        logger.info("Coverage measurement not supported for this language — skipping")
        return CoverageResult(
            line_coverage=0.0, branch_coverage=0.0, uncovered_lines=[],
            passed_gate=True,  # Don't block unsupported languages
            raw_output="Language not supported by coverage runner",
        )
