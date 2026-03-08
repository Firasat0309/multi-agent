"""Interactive live console for real-time pipeline progress display."""

from __future__ import annotations

import logging
import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich.layout import Layout
from rich.columns import Columns

console = Console()


class LiveConsole:
    """Real-time interactive display of pipeline execution."""

    def __init__(self) -> None:
        self._live: Live | None = None
        self._phases: list[dict[str, Any]] = []
        self._current_phase: str = ""
        self._tasks: dict[int, dict[str, Any]] = {}
        self._agent_log: list[str] = []
        self._start_time: float = 0.0
        self._blueprint_info: dict[str, Any] = {}
        self._max_log_lines = 12

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._live = Live(self._render(), console=console, refresh_per_second=4)
        self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    def set_phase(self, phase: str, status: str = "running") -> None:
        self._current_phase = phase
        # Update or add phase
        for p in self._phases:
            if p["name"] == phase:
                p["status"] = status
                self._refresh()
                return
        self._phases.append({"name": phase, "status": status})
        self._refresh()

    def complete_phase(self, phase: str) -> None:
        for p in self._phases:
            if p["name"] == phase:
                p["status"] = "done"
        self._refresh()

    def fail_phase(self, phase: str, error: str = "") -> None:
        for p in self._phases:
            if p["name"] == phase:
                p["status"] = "failed"
        if error:
            self.log(f"[red]Error: {error}[/red]")
        self._refresh()

    def set_blueprint(self, name: str, language: str, files: int, style: str) -> None:
        self._blueprint_info = {
            "name": name,
            "language": language,
            "files": files,
            "style": style,
        }
        self._refresh()

    def update_task(
        self, task_id: int, description: str, status: str, agent: str = ""
    ) -> None:
        self._tasks[task_id] = {
            "description": description,
            "status": status,
            "agent": agent,
        }
        self._refresh()

    def log(self, message: str) -> None:
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        timestamp = f"[dim]{elapsed:6.1f}s[/dim]"
        self._agent_log.append(f"{timestamp}  {message}")
        if len(self._agent_log) > self._max_log_lines:
            self._agent_log = self._agent_log[-self._max_log_lines:]
        self._refresh()

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._render())

    def _render(self) -> Panel:
        """Build the full live display."""
        parts: list[Any] = []

        # Phase progress
        parts.append(self._render_phases())

        # Blueprint info
        if self._blueprint_info:
            parts.append(self._render_blueprint())

        # Task table
        if self._tasks:
            parts.append(self._render_tasks())

        # Live log
        if self._agent_log:
            parts.append(self._render_log())

        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        from rich.console import Group
        body = Group(*parts)
        return Panel(
            body,
            title=f"[bold blue]Multi-Agent Code Generator[/bold blue] [dim]({elapsed:.0f}s)[/dim]",
            border_style="blue",
        )

    def _render_phases(self) -> Table:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("icon", width=3)
        table.add_column("phase", min_width=30)
        table.add_column("status", min_width=10)

        phase_defs = [
            ("1", "Architecture Design"),
            ("2", "Task Planning"),
            ("3", "Code Generation & Review"),
            ("4", "Finalize"),
        ]

        for idx, (num, label) in enumerate(phase_defs):
            if idx < len(self._phases):
                p = self._phases[idx]
                if p["status"] == "done":
                    icon = "[green]  [/green]"
                    status = "[green]done[/green]"
                elif p["status"] == "failed":
                    icon = "[red]  [/red]"
                    status = "[red]failed[/red]"
                else:
                    icon = "[yellow]  [/yellow]"
                    status = "[yellow]running...[/yellow]"
            else:
                icon = "[dim]  [/dim]"
                status = "[dim]pending[/dim]"

            table.add_row(icon, f"Phase {num}: {label}", status)

        return table

    def _render_blueprint(self) -> Panel:
        info = self._blueprint_info
        text = (
            f"[bold]{info.get('name', '')}[/bold]  |  "
            f"Language: [cyan]{info.get('language', '')}[/cyan]  |  "
            f"Style: {info.get('style', '')}  |  "
            f"Files: {info.get('files', 0)}"
        )
        return Panel(text, title="Blueprint", border_style="dim", height=3)

    def _render_tasks(self) -> Table:
        table = Table(title="Tasks", box=None, padding=(0, 1), show_edge=False)
        table.add_column("ID", style="dim", width=4)
        table.add_column("Task", min_width=35)
        table.add_column("Agent", min_width=10)
        table.add_column("Status", min_width=12)

        status_icons = {
            "pending": "[dim]pending[/dim]",
            "ready": "[dim]ready[/dim]",
            "in_progress": "[yellow]running...[/yellow]",
            "completed": "[green]done[/green]",
            "failed": "[red]failed[/red]",
            "blocked": "[dim]blocked[/dim]",
        }

        # Show at most 15 tasks, prioritizing active ones
        sorted_tasks = sorted(
            self._tasks.items(),
            key=lambda x: (
                0 if x[1]["status"] == "in_progress" else
                1 if x[1]["status"] == "failed" else
                2 if x[1]["status"] == "completed" else 3
            ),
        )
        for task_id, task in sorted_tasks[:15]:
            table.add_row(
                str(task_id),
                task["description"][:40],
                task.get("agent", ""),
                status_icons.get(task["status"], task["status"]),
            )
        if len(sorted_tasks) > 15:
            table.add_row("", f"[dim]... and {len(sorted_tasks) - 15} more[/dim]", "", "")

        return table

    def _render_log(self) -> Panel:
        log_text = "\n".join(self._agent_log)
        return Panel(log_text, title="Activity Log", border_style="dim", height=min(self._max_log_lines + 2, 14))


class LiveConsoleHandler(logging.Handler):
    """Logging handler that routes messages to the LiveConsole."""

    def __init__(self, live_console: LiveConsole) -> None:
        super().__init__()
        self.live_console = live_console

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # Color by log level
            if record.levelno >= logging.ERROR:
                msg = f"[red]{msg}[/red]"
            elif record.levelno >= logging.WARNING:
                msg = f"[yellow]{msg}[/yellow]"
            self.live_console.log(msg)
        except Exception:
            pass
