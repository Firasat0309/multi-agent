"""CLI entry point for the multi-agent code generation platform."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from config.settings import Settings, LLMConfig, LLMProvider, SandboxConfig, SandboxType
from core.pipeline import Pipeline, PipelineResult

console = Console()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """Multi-Agent Backend Code Generator"""
    setup_logging(verbose)


@cli.command()
@click.argument("prompt")
@click.option("--workspace", "-w", default="workspace", help="Output directory")
@click.option("--model", "-m", default="claude-sonnet-4-20250514", help="LLM model")
@click.option("--provider", "-p", default="anthropic", help="LLM provider")
@click.option("--sandbox", "-s", default="local", help="Sandbox type (docker/local)")
@click.option("--max-agents", default=4, help="Max concurrent agents")
def generate(
    prompt: str,
    workspace: str,
    model: str,
    provider: str,
    sandbox: str,
    max_agents: int,
) -> None:
    """Generate a backend project from a natural language prompt."""
    console.print(Panel.fit(
        f"[bold blue]Multi-Agent Code Generator[/bold blue]\n"
        f"Prompt: {prompt[:80]}...\n"
        f"Workspace: {workspace}\n"
        f"Model: {model}",
        title="Starting",
    ))

    settings = Settings(
        workspace_dir=Path(workspace),
        llm=LLMConfig(
            provider=LLMProvider(provider),
            model=model,
        ),
        sandbox=SandboxConfig(sandbox_type=SandboxType(sandbox)),
        max_concurrent_agents=max_agents,
    )

    pipeline = Pipeline(settings)
    result = asyncio.run(pipeline.run(prompt))
    _display_result(result)

    sys.exit(0 if result.success else 1)


@cli.command()
@click.argument("workspace", default="workspace")
def status(workspace: str) -> None:
    """Show status of a workspace."""
    ws = Path(workspace)
    if not ws.exists():
        console.print(f"[red]Workspace not found: {workspace}[/red]")
        return

    import json
    table = Table(title=f"Workspace: {workspace}")
    table.add_column("File", style="cyan")
    table.add_column("Exists", style="green")

    for name in ["architecture.md", "file_blueprints.json", "dependency_graph.json", "repo_index.json"]:
        exists = (ws / name).exists()
        table.add_row(name, "Yes" if exists else "[red]No[/red]")

    src_files = list((ws / "src").rglob("*.py")) if (ws / "src").exists() else []
    test_files = list((ws / "tests").rglob("*.py")) if (ws / "tests").exists() else []
    deploy_files = list((ws / "deploy").rglob("*")) if (ws / "deploy").exists() else []

    table.add_row("Source files", str(len(src_files)))
    table.add_row("Test files", str(len(test_files)))
    table.add_row("Deploy artifacts", str(len(deploy_files)))

    console.print(table)


def _display_result(result: PipelineResult) -> None:
    """Display pipeline results in a formatted table."""
    if result.success:
        console.print(Panel.fit(
            "[bold green]Pipeline completed successfully![/bold green]",
            title="Success",
        ))
    else:
        console.print(Panel.fit(
            "[bold red]Pipeline completed with issues[/bold red]",
            title="Warning",
        ))

    table = Table(title="Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Workspace", str(result.workspace_path))
    table.add_row("Elapsed", f"{result.elapsed_seconds:.1f}s")

    if result.blueprint:
        table.add_row("Project", result.blueprint.name)
        table.add_row("Files", str(len(result.blueprint.file_blueprints)))

    for status_name, count in result.task_stats.items():
        table.add_row(f"Tasks ({status_name})", str(count))

    if result.errors:
        table.add_row("Errors", "\n".join(result.errors[:5]))

    console.print(table)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
