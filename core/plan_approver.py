"""Human-in-the-loop approval gate for change plans."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from core.models import ChangePlan

if TYPE_CHECKING:
    from core.live_console import LiveConsole

logger = logging.getLogger(__name__)


class PlanPendingApprovalError(Exception):
    """Raised in non-interactive mode when a plan needs human approval.

    The pending plan is written to ``workspace/pending_plan.json`` for the
    caller (e.g. an API layer) to surface to the user.
    """


class PlanApprover:
    """Presents the change plan to the user and waits for approval.

    Two modes:
    - interactive=True  : Rich console table + y/n prompt (CLI usage).
    - interactive=False : Writes plan to ``pending_plan.json`` and raises
                          ``PlanPendingApprovalError`` (API / programmatic usage).

    When a ``LiveConsole`` is provided, the live display is temporarily
    paused while waiting for user input so that the prompt is visible.
    """

    def __init__(
        self,
        interactive: bool = True,
        workspace: Path | None = None,
        live: LiveConsole | None = None,
    ) -> None:
        self._interactive = interactive
        self._workspace = workspace
        self._live = live

    # ── Public API ────────────────────────────────────────────────────────────

    def display_and_approve(self, plan: ChangePlan) -> bool:
        """Display *plan* and return True if approved, False if rejected.

        Raises:
            PlanPendingApprovalError: when interactive=False.
        """
        if self._interactive:
            return self._with_live_paused(self._interactive_approve, plan)
        else:
            return self._noninteractive_approve(plan)

    def _with_live_paused(self, fn, *args):
        """Stop the LiveConsole before prompting, restart it after."""
        if self._live:
            self._live.stop()
        try:
            return fn(*args)
        finally:
            if self._live:
                self._live.start()

    # ── Interactive (CLI) path ────────────────────────────────────────────────

    def _interactive_approve(self, plan: ChangePlan) -> bool:
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.prompt import Confirm

            console = Console()
            self._display_plan_rich(plan, console, Table, Panel)
            return Confirm.ask("\n[bold]Proceed with these changes?[/bold]", default=True)
        except ImportError:
            # Rich not installed — fall back to plain text
            return self._display_plan_plain(plan)

    def _display_plan_rich(
        self,
        plan: ChangePlan,
        console: object,
        Table: type,
        Panel: type,
    ) -> None:
        """Render plan with Rich tables and panels."""

        console.print(f"\n[bold cyan]Change Plan:[/bold cyan] {plan.summary}\n")

        if plan.changes:
            table = Table(title="Planned Modifications", show_lines=True)
            table.add_column("Type", style="cyan", no_wrap=True)
            table.add_column("File", style="yellow")
            table.add_column("Description")
            for c in plan.changes:
                table.add_row(c.type.value, c.file, c.description)
            console.print(table)

        if plan.new_files:
            nf_table = Table(title="New Files", show_lines=True)
            nf_table.add_column("Path", style="green")
            nf_table.add_column("Purpose")
            for nf in plan.new_files:
                nf_table.add_row(nf.path, nf.purpose)
            console.print(nf_table)

        if plan.affected_tests:
            console.print(
                Panel(
                    "\n".join(f"• {t}" for t in plan.affected_tests),
                    title="[blue]Affected Tests[/blue]",
                    expand=False,
                )
            )

        if plan.risk_notes:
            console.print(
                Panel(
                    "\n".join(f"⚠  {r}" for r in plan.risk_notes),
                    title="[yellow]Risk Notes[/yellow]",
                    expand=False,
                )
            )

    def _display_plan_plain(self, plan: ChangePlan) -> bool:
        """Fallback plain-text display when Rich is unavailable."""
        print(f"\n=== Change Plan: {plan.summary} ===\n")

        if plan.changes:
            print("Modifications:")
            for c in plan.changes:
                print(f"  [{c.type.value}] {c.file}: {c.description}")

        if plan.new_files:
            print("\nNew files:")
            for nf in plan.new_files:
                print(f"  + {nf.path}: {nf.purpose}")

        if plan.risk_notes:
            print("\nRisk notes:")
            for r in plan.risk_notes:
                print(f"  ⚠  {r}")

        answer = input("\nProceed with these changes? [Y/n] ").strip().lower()
        return answer in ("", "y", "yes")

    # ── Non-interactive (API) path ────────────────────────────────────────────

    def _noninteractive_approve(self, plan: ChangePlan) -> bool:
        """Write plan to disk and raise PlanPendingApprovalError."""
        payload = {
            "summary": plan.summary,
            "changes": [
                {
                    "type": c.type.value,
                    "file": c.file,
                    "description": c.description,
                    "function": c.function,
                    "class_name": c.class_name,
                }
                for c in plan.changes
            ],
            "new_files": [
                {"path": nf.path, "purpose": nf.purpose}
                for nf in plan.new_files
            ],
            "affected_tests": plan.affected_tests,
            "risk_notes": plan.risk_notes,
        }

        if self._workspace:
            pending_path = self._workspace / "pending_plan.json"
            pending_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            logger.info("Pending plan written to %s", pending_path)

        raise PlanPendingApprovalError(
            "Change plan requires human approval before execution. "
            "Review pending_plan.json and re-run with approval."
        )
