"""Human-in-the-loop approval gates for architecture decisions.

Provides approval checkpoints at three stages of the fullstack pipeline:
1. Product plan (requirements, features, tech choices)
2. Backend architecture (tech stack, file blueprints, layers)
3. Frontend architecture (components, framework, state solution)

This prevents wasting tokens on generation when the plan is wrong.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.live_console import LiveConsole
    from core.models import (
        ComponentPlan,
        ProductRequirements,
        RepositoryBlueprint,
    )

logger = logging.getLogger(__name__)


class ArchitectureApprovalRejected(Exception):
    """Raised when the user rejects an architecture plan."""


class ArchitecturePendingApprovalError(Exception):
    """Raised in non-interactive mode when a plan needs human approval."""


class ArchitectureApprover:
    """Displays architecture plans and waits for human approval.

    Two modes:
    - interactive=True  : Rich console display + y/n prompt (CLI usage).
    - interactive=False : Writes plan to JSON and raises
                          ``ArchitecturePendingApprovalError``.

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

    # ── Product Requirements Approval ──────────────────────────────────────

    def approve_product_plan(self, requirements: ProductRequirements) -> bool:
        """Display product requirements and return True if approved."""
        if self._interactive:
            return self._with_live_paused(self._interactive_product, requirements)
        return self._noninteractive_product(requirements)

    def _interactive_product(self, req: ProductRequirements) -> bool:
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.prompt import Confirm

            console = Console()
            console.print("\n[bold cyan]═══ Product Plan ═══[/bold cyan]\n")
            console.print(f"[bold]Title:[/bold] {req.title}")
            console.print(f"[bold]Description:[/bold] {req.description}")
            console.print(f"[bold]Frontend:[/bold] {'Yes' if req.has_frontend else 'No'}")
            console.print(f"[bold]Backend:[/bold] {'Yes' if req.has_backend else 'No'}")

            if req.features:
                console.print("\n[bold]Features:[/bold]")
                for f in req.features:
                    console.print(f"  • {f}")

            if req.tech_preferences:
                table = Table(title="Tech Preferences", show_lines=False)
                table.add_column("Category", style="cyan")
                table.add_column("Choice", style="yellow")
                for k, v in req.tech_preferences.items():
                    table.add_row(k, v)
                console.print()
                console.print(table)

            if req.user_stories:
                console.print("\n[bold]User Stories:[/bold]")
                for s in req.user_stories[:5]:
                    console.print(f"  • {s}")
                if len(req.user_stories) > 5:
                    console.print(f"  … and {len(req.user_stories) - 5} more")

            return Confirm.ask(
                "\n[bold]Approve this product plan?[/bold]", default=True
            )
        except ImportError:
            return self._plain_product(req)

    def _plain_product(self, req: ProductRequirements) -> bool:
        print(f"\n=== Product Plan ===")
        print(f"Title: {req.title}")
        print(f"Description: {req.description}")
        print(f"Frontend: {'Yes' if req.has_frontend else 'No'}")
        print(f"Backend: {'Yes' if req.has_backend else 'No'}")
        if req.features:
            print("Features:")
            for f in req.features:
                print(f"  • {f}")
        if req.tech_preferences:
            print("Tech preferences:")
            for k, v in req.tech_preferences.items():
                print(f"  {k}: {v}")
        answer = input("\nApprove this product plan? [Y/n] ").strip().lower()
        return answer in ("", "y", "yes")

    def _noninteractive_product(self, req: ProductRequirements) -> bool:
        payload = {
            "checkpoint": "product_plan",
            "title": req.title,
            "description": req.description,
            "has_frontend": req.has_frontend,
            "has_backend": req.has_backend,
            "features": req.features,
            "tech_preferences": req.tech_preferences,
            "user_stories": req.user_stories,
        }
        self._write_pending(payload, "pending_product_plan.json")
        raise ArchitecturePendingApprovalError(
            "Product plan requires human approval. "
            "Review pending_product_plan.json and re-run with approval."
        )

    # ── Backend Architecture Approval ──────────────────────────────────────

    def approve_backend_architecture(self, blueprint: RepositoryBlueprint) -> bool:
        """Display backend architecture and return True if approved."""
        if self._interactive:
            return self._with_live_paused(self._interactive_backend, blueprint)
        return self._noninteractive_backend(blueprint)

    def _interactive_backend(self, bp: RepositoryBlueprint) -> bool:
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.prompt import Confirm

            console = Console()
            console.print("\n[bold cyan]═══ Backend Architecture ═══[/bold cyan]\n")
            console.print(f"[bold]Project:[/bold] {bp.name}")
            console.print(f"[bold]Description:[/bold] {bp.description}")
            console.print(f"[bold]Style:[/bold] {bp.architecture_style}")

            if bp.tech_stack:
                table = Table(title="Tech Stack", show_lines=False)
                table.add_column("Key", style="cyan")
                table.add_column("Value", style="yellow")
                for k, v in bp.tech_stack.items():
                    table.add_row(k, str(v))
                console.print()
                console.print(table)

            if bp.file_blueprints:
                ft = Table(title=f"File Blueprints ({len(bp.file_blueprints)} files)", show_lines=False)
                ft.add_column("Path", style="green")
                ft.add_column("Layer", style="cyan")
                ft.add_column("Purpose")
                for fb in bp.file_blueprints:
                    ft.add_row(fb.path, fb.layer or "-", fb.purpose)
                console.print()
                console.print(ft)

            return Confirm.ask(
                "\n[bold]Approve this backend architecture?[/bold]", default=True
            )
        except ImportError:
            return self._plain_backend(bp)

    def _plain_backend(self, bp: RepositoryBlueprint) -> bool:
        print(f"\n=== Backend Architecture ===")
        print(f"Project: {bp.name}")
        print(f"Description: {bp.description}")
        print(f"Style: {bp.architecture_style}")
        if bp.tech_stack:
            print("Tech stack:")
            for k, v in bp.tech_stack.items():
                print(f"  {k}: {v}")
        if bp.file_blueprints:
            print(f"\nFiles ({len(bp.file_blueprints)}):")
            for fb in bp.file_blueprints:
                print(f"  [{fb.layer or '-'}] {fb.path}: {fb.purpose}")
        answer = input("\nApprove this backend architecture? [Y/n] ").strip().lower()
        return answer in ("", "y", "yes")

    def _noninteractive_backend(self, bp: RepositoryBlueprint) -> bool:
        payload = {
            "checkpoint": "backend_architecture",
            "name": bp.name,
            "description": bp.description,
            "architecture_style": bp.architecture_style,
            "tech_stack": bp.tech_stack,
            "files": [
                {"path": fb.path, "layer": fb.layer, "purpose": fb.purpose}
                for fb in bp.file_blueprints
            ],
        }
        self._write_pending(payload, "pending_backend_architecture.json")
        raise ArchitecturePendingApprovalError(
            "Backend architecture requires human approval. "
            "Review pending_backend_architecture.json and re-run with approval."
        )

    # ── Frontend Architecture Approval ─────────────────────────────────────

    def approve_frontend_architecture(self, plan: ComponentPlan) -> bool:
        """Display frontend component plan and return True if approved."""
        if self._interactive:
            return self._with_live_paused(self._interactive_frontend, plan)
        return self._noninteractive_frontend(plan)

    def _interactive_frontend(self, plan: ComponentPlan) -> bool:
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.prompt import Confirm

            console = Console()
            console.print("\n[bold cyan]═══ Frontend Architecture ═══[/bold cyan]\n")
            console.print(f"[bold]Framework:[/bold] {plan.framework}")
            console.print(f"[bold]State Solution:[/bold] {plan.state_solution}")
            console.print(f"[bold]Routing:[/bold] {plan.routing_solution}")
            console.print(f"[bold]API Base URL:[/bold] {plan.api_base_url}")

            if plan.components:
                ct = Table(
                    title=f"Components ({len(plan.components)})",
                    show_lines=False,
                )
                ct.add_column("Name", style="green")
                ct.add_column("Type", style="cyan")
                ct.add_column("File Path", style="yellow")
                ct.add_column("Description")
                for c in plan.components:
                    ct.add_row(
                        c.name,
                        c.component_type or "-",
                        c.file_path or "-",
                        (c.description or "")[:60],
                    )
                console.print()
                console.print(ct)

            return Confirm.ask(
                "\n[bold]Approve this frontend architecture?[/bold]", default=True
            )
        except ImportError:
            return self._plain_frontend(plan)

    def _plain_frontend(self, plan: ComponentPlan) -> bool:
        print(f"\n=== Frontend Architecture ===")
        print(f"Framework: {plan.framework}")
        print(f"State: {plan.state_solution}")
        print(f"Routing: {plan.routing_solution}")
        print(f"API Base: {plan.api_base_url}")
        if plan.components:
            print(f"\nComponents ({len(plan.components)}):")
            for c in plan.components:
                print(f"  [{c.component_type or '-'}] {c.name}: {(c.description or '')[:60]}")
        answer = input("\nApprove this frontend architecture? [Y/n] ").strip().lower()
        return answer in ("", "y", "yes")

    def _noninteractive_frontend(self, plan: ComponentPlan) -> bool:
        payload = {
            "checkpoint": "frontend_architecture",
            "framework": plan.framework,
            "state_solution": plan.state_solution,
            "routing_solution": plan.routing_solution,
            "api_base_url": plan.api_base_url,
            "components": [
                {
                    "name": c.name,
                    "type": c.component_type,
                    "file_path": c.file_path,
                    "description": c.description,
                }
                for c in plan.components
            ],
        }
        self._write_pending(payload, "pending_frontend_architecture.json")
        raise ArchitecturePendingApprovalError(
            "Frontend architecture requires human approval. "
            "Review pending_frontend_architecture.json and re-run with approval."
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _with_live_paused(self, fn, *args):
        """Stop the LiveConsole before prompting, restart it after."""
        if self._live:
            self._live.stop()
        try:
            return fn(*args)
        finally:
            if self._live:
                self._live.start()

    def _write_pending(self, payload: dict, filename: str) -> None:
        if self._workspace:
            pending_path = self._workspace / filename
            pending_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            logger.info("Pending plan written to %s", pending_path)
