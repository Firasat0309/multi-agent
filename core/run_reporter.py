"""Run reporters — write structured JSON reports for generation and modification runs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.task_engine import TaskGraph

logger = logging.getLogger(__name__)


class RunReporter:
    """Writes structured JSON run reports to the workspace directory."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    # ── Generation run ────────────────────────────────────────────────────────

    def write_run_report(
        self,
        *,
        prompt: str,
        blueprint: Any,
        task_graph: TaskGraph,
        stats: dict[str, int],
        elapsed: float,
        success: bool,
        code_success: bool = True,
        tests_passed: bool = True,
        token_cost: Any | None = None,
    ) -> Path:
        """Write workspace/run_report.json and return the path."""
        task_entries = self._task_entries(task_graph, include_metrics=True)

        report: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "success": success,
            # Granular quality signals — lets callers distinguish between
            # "code generation failed" and "tests didn't fully pass".
            "code_success": code_success,
            "tests_passed": tests_passed,
            "elapsed_seconds": round(elapsed, 2),
            "prompt": prompt,
            "project": blueprint.name if blueprint else "",
            "language": blueprint.tech_stack.get("language", "") if blueprint else "",
            "architecture_style": blueprint.architecture_style if blueprint else "",
            "task_stats": stats,
            "tasks": task_entries,
        }
        if token_cost is not None:
            report["token_cost"] = {
                "input_tokens": token_cost.input_tokens,
                "output_tokens": token_cost.output_tokens,
                "model": token_cost.model,
                "cost_usd": round(token_cost.cost_usd, 6),
            }

        path = self._workspace / "run_report.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.info("Run report written to %s", path)
        return path

    # ── Modification run ──────────────────────────────────────────────────────

    def write_modify_report(
        self,
        *,
        prompt: str,
        change_plan: Any,
        task_graph: TaskGraph,
        stats: dict[str, int],
        elapsed: float,
        success: bool,
        changed_files: list[str] | None = None,
        diff_stats: dict[str, int] | None = None,
        token_cost: Any | None = None,
    ) -> Path:
        """Write workspace/modify_report.json and return the path."""
        task_entries = self._task_entries(task_graph, include_metrics=False)
        change_entries = [
            {
                "type": c.type.value,
                "file": c.file,
                "description": c.description,
                "function": c.function,
                "class_name": c.class_name,
            }
            for c in change_plan.changes
        ]

        report: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "enhance",
            "success": success,
            "elapsed_seconds": round(elapsed, 2),
            "prompt": prompt,
            "change_plan": {
                "summary": change_plan.summary,
                "changes": change_entries,
                "new_files": [nf.path for nf in change_plan.new_files],
                "risk_notes": change_plan.risk_notes,
            },
            "diff_summary": {
                "files_modified": changed_files or [],
                "lines_added": (diff_stats or {}).get("lines_added", 0),
                "lines_removed": (diff_stats or {}).get("lines_removed", 0),
            },
            "task_stats": stats,
            "tasks": task_entries,
        }
        if token_cost is not None:
            report["token_cost"] = {
                "input_tokens": token_cost.input_tokens,
                "output_tokens": token_cost.output_tokens,
                "model": token_cost.model,
                "cost_usd": round(token_cost.cost_usd, 6),
            }

        path = self._workspace / "modify_report.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.info("Modify report written to %s", path)
        return path

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _task_entries(graph: TaskGraph, *, include_metrics: bool) -> list[dict[str, Any]]:
        entries = []
        for task in graph.tasks.values():
            entry: dict[str, Any] = {
                "id": task.task_id,
                "type": task.task_type.value,
                "file": task.file,
                "description": task.description,
                "status": task.status.value,
                "retries": task.retry_count,
            }
            if task.result:
                entry["output"] = task.result.output
                entry["errors"] = task.result.errors
                entry["files_modified"] = task.result.files_modified
                if include_metrics:
                    entry["metrics"] = task.result.metrics
            entries.append(entry)
        return entries
