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
from core.llm_client import LLMConfigError

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
@click.option("--sandbox", "-s", default="docker", help="Sandbox type (docker/local)")
@click.option("--max-agents", default=4, help="Max concurrent agents")
@click.option("--no-interactive", is_flag=True, default=False, help="Disable live display (plain logs)")
@click.option(
    "--allow-host-execution",
    is_flag=True,
    default=False,
    help="Allow running without Docker isolation (NOT recommended for untrusted prompts)",
)
@click.option("--skip-tester", is_flag=True, default=False, help="Skip the test generation agent")
@click.option("--skip-reviewer", is_flag=True, default=False, help="Skip the code review agent")
@click.option("--skip-security", is_flag=True, default=False, help="Skip the security hardening checkpoint")
@click.option("--skip-integration", is_flag=True, default=False, help="Skip the integration test checkpoint")
@click.option("--resume", is_flag=True, default=False, help="Resume from last checkpoint — skip files that already PASSED")
@click.option(
    "--contract", "contract_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to an api_contract.json file. Skips LLM contract generation and injects "
         "the provided contract directly into every code-generation agent.",
)
def generate(
    prompt: str,
    workspace: str,
    model: str,
    provider: str,
    sandbox: str,
    max_agents: int,
    no_interactive: bool,
    allow_host_execution: bool,
    skip_tester: bool,
    skip_reviewer: bool,
    skip_security: bool,
    skip_integration: bool,
    resume: bool,
    contract_path: str | None,
) -> None:
    """Generate a backend project from a natural language prompt."""
    try:
        skip = set()
        if skip_tester:
            skip.add("tester")
        if skip_reviewer:
            skip.add("reviewer")
        if skip_security:
            skip.add("security")
        if skip_integration:
            skip.add("integration")

        settings = Settings(
            workspace_dir=Path(workspace).resolve(),
            llm=LLMConfig(
                provider=LLMProvider(provider),
                model=model,
            ),
            sandbox=SandboxConfig(sandbox_type=SandboxType(sandbox)),
            max_concurrent_agents=max_agents,
            allow_host_execution=allow_host_execution or sandbox == "local",
            skip_agents=frozenset(skip),
        )

        pipeline = Pipeline(settings, interactive=not no_interactive)
    except LLMConfigError as e:
        # Display configuration error with better formatting
        console.print(Panel(
            f"[bold red]{str(e)}[/bold red]",
            title="❌ Configuration Error",
            border_style="red",
        ))
        sys.exit(1)
    except ValueError as e:
        # Handle invalid provider or sandbox type
        console.print(Panel(
            f"[bold red]Invalid configuration: {e}[/bold red]\n\n"
            f"Valid providers: anthropic, openai, gemini\n"
            f"Valid sandbox types: docker, local",
            title="❌ Configuration Error",
            border_style="red",
        ))
        sys.exit(1)

    result = asyncio.run(pipeline.run(prompt, resume=resume, api_contract_path=contract_path))
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

    table = Table(title=f"Workspace: {workspace}")
    table.add_column("File", style="cyan")
    table.add_column("Exists", style="green")

    for name in ["architecture.md", "file_blueprints.json", "dependency_graph.json", "repo_index.json"]:
        exists = (ws / name).exists()
        table.add_row(name, "Yes" if exists else "[red]No[/red]")

    src_files = list(p for p in (ws / "src").rglob("*") if p.is_file() and p.suffix in {".py", ".java", ".go", ".ts", ".rs", ".cs"}) if (ws / "src").exists() else []
    test_files = list(p for p in (ws / "tests").rglob("*") if p.is_file() and p.suffix in {".py", ".java", ".go", ".ts", ".rs", ".cs"}) if (ws / "tests").exists() else []
    deploy_files = list((ws / "deploy").rglob("*")) if (ws / "deploy").exists() else []

    table.add_row("Source files", str(len(src_files)))
    table.add_row("Test files", str(len(test_files)))
    table.add_row("Deploy artifacts", str(len(deploy_files)))

    console.print(table)


@cli.command()
@click.argument("prompt")
@click.option("--workspace", "-w", required=True, help="Existing project directory to modify")
@click.option("--model", "-m", default="claude-sonnet-4-20250514", help="LLM model")
@click.option("--provider", "-p", default="anthropic", help="LLM provider")
@click.option("--sandbox", "-s", default="docker", help="Sandbox type (docker/local)")
@click.option("--max-agents", default=4, help="Max concurrent agents")
@click.option("--no-interactive", is_flag=True, default=False, help="Disable live display")
@click.option(
    "--allow-host-execution",
    is_flag=True,
    default=False,
    help="Allow running without Docker isolation",
)
@click.option("--skip-tester", is_flag=True, default=False, help="Skip the test generation agent")
@click.option("--skip-reviewer", is_flag=True, default=False, help="Skip the code review agent")
@click.option("--skip-security", is_flag=True, default=False, help="Skip the security hardening checkpoint")
@click.option("--skip-integration", is_flag=True, default=False, help="Skip the integration test checkpoint")
def enhance(
    prompt: str,
    workspace: str,
    model: str,
    provider: str,
    sandbox: str,
    max_agents: int,
    no_interactive: bool,
    allow_host_execution: bool,
    skip_tester: bool,
    skip_reviewer: bool,
    skip_security: bool,
    skip_integration: bool,
) -> None:
    """Modify an existing project based on a natural language prompt.

    Unlike 'generate' which creates a new project from scratch, 'enhance'
    analyzes an existing codebase and makes targeted modifications:

    \b
    1. Scans the repository to understand structure & dependencies
    2. Plans the minimal set of changes needed
    3. Applies targeted edits (not full file rewrites)
    4. Runs tests and reviews on the changes

    Example:
        python -m core.cli enhance "Add password reset feature" -w ./my-project
    """
    ws = Path(workspace).resolve()
    if not ws.exists():
        console.print(f"[red]Workspace not found: {workspace}[/red]")
        sys.exit(1)

    try:
        skip = set()
        if skip_tester:
            skip.add("tester")
        if skip_reviewer:
            skip.add("reviewer")
        if skip_security:
            skip.add("security")
        if skip_integration:
            skip.add("integration")

        settings = Settings(
            workspace_dir=ws,
            llm=LLMConfig(
                provider=LLMProvider(provider),
                model=model,
            ),
            sandbox=SandboxConfig(sandbox_type=SandboxType(sandbox)),
            max_concurrent_agents=max_agents,
            allow_host_execution=allow_host_execution or sandbox == "local",
            skip_agents=frozenset(skip),
        )

        pipeline = Pipeline(settings, interactive=not no_interactive)
    except LLMConfigError as e:
        console.print(Panel(
            f"[bold red]{str(e)}[/bold red]",
            title="❌ Configuration Error",
            border_style="red",
        ))
        sys.exit(1)
    except ValueError as e:
        console.print(Panel(
            f"[bold red]Invalid configuration: {e}[/bold red]\n\n"
            f"Valid providers: anthropic, openai, gemini\n"
            f"Valid sandbox types: docker, local",
            title="❌ Configuration Error",
            border_style="red",
        ))
        sys.exit(1)

    result = asyncio.run(pipeline.enhance(prompt))
    _display_enhance_result(result)

    sys.exit(0 if result.success else 1)


@cli.command()
@click.argument("prompt")
@click.option("--workspace", "-w", default="workspace", help="Output directory")
@click.option("--model", "-m", default="claude-sonnet-4-20250514", help="LLM model")
@click.option("--provider", "-p", default="anthropic", help="LLM provider")
@click.option("--sandbox", "-s", default="docker", help="Sandbox type (docker/local)")
@click.option("--max-agents", default=4, help="Max concurrent agents")
@click.option("--no-interactive", is_flag=True, default=False, help="Disable live display")
@click.option("--figma-url", default="", help="Figma design URL for the UI design parser")
@click.option(
    "--allow-host-execution",
    is_flag=True,
    default=False,
    help="Allow running without Docker isolation (NOT recommended for untrusted prompts)",
)
@click.option("--skip-tester", is_flag=True, default=False, help="Skip the test generation agent")
@click.option("--skip-reviewer", is_flag=True, default=False, help="Skip the code review agent")
@click.option("--skip-security", is_flag=True, default=False, help="Skip the security hardening checkpoint")
@click.option("--skip-integration", is_flag=True, default=False, help="Skip the integration test checkpoint")
@click.option(
    "--contract", "contract_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to an api_contract.json file. Skips LLM contract generation and uses "
         "the provided contract as the FE/BE handshake.",
)
def fullstack(
    prompt: str,
    workspace: str,
    model: str,
    provider: str,
    sandbox: str,
    max_agents: int,
    no_interactive: bool,
    figma_url: str,
    allow_host_execution: bool,
    skip_tester: bool,
    skip_reviewer: bool,
    skip_security: bool,
    skip_integration: bool,
    contract_path: str | None,
) -> None:
    """Generate a complete fullstack project (backend + frontend) from a prompt.

    Runs product planning, backend architecture, API contract generation,
    and then the backend and frontend pipelines **in parallel**:

    \b
      workspace/backend/   — generated backend API
      workspace/frontend/  — generated React/Next.js/Vue UI

    Optionally supply a Figma design URL via --figma-url for richer UI design.

    Example:
        python -m core.cli fullstack "Build a Task Manager SaaS app" -w ./my-app
    """
    try:
        skip = set()
        if skip_tester:
            skip.add("tester")
        if skip_reviewer:
            skip.add("reviewer")
        if skip_security:
            skip.add("security")
        if skip_integration:
            skip.add("integration")

        settings = Settings(
            workspace_dir=Path(workspace).resolve(),
            llm=LLMConfig(
                provider=LLMProvider(provider),
                model=model,
            ),
            sandbox=SandboxConfig(sandbox_type=SandboxType(sandbox)),
            max_concurrent_agents=max_agents,
            allow_host_execution=allow_host_execution or sandbox == "local",
            skip_agents=frozenset(skip),
        )

        pipeline = Pipeline(settings, interactive=not no_interactive)
    except LLMConfigError as e:
        console.print(Panel(
            f"[bold red]{str(e)}[/bold red]",
            title="❌ Configuration Error",
            border_style="red",
        ))
        sys.exit(1)
    except ValueError as e:
        console.print(Panel(
            f"[bold red]Invalid configuration: {e}[/bold red]\n\n"
            f"Valid providers: anthropic, openai, gemini\n"
            f"Valid sandbox types: docker, local",
            title="❌ Configuration Error",
            border_style="red",
        ))
        sys.exit(1)

    result = asyncio.run(pipeline.run_fullstack(prompt, figma_url=figma_url, api_contract_path=contract_path))
    _display_fullstack_result(result)

    sys.exit(0 if result.success else 1)


def _display_enhance_result(result: PipelineResult) -> None:
    """Display modification pipeline results."""
    if result.success:
        console.print(Panel.fit(
            "[bold green]Modification completed successfully![/bold green]",
            title="✅ Enhanced",
        ))
    else:
        console.print(Panel.fit(
            "[bold red]Modification completed with issues[/bold red]",
            title="⚠️  Warning",
        ))

    table = Table(title="Modification Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Workspace", str(result.workspace_path))
    table.add_row("Elapsed", f"{result.elapsed_seconds:.1f}s")

    if result.change_plan:
        table.add_row("Summary", result.change_plan.summary)
        table.add_row("Files modified", str(len(result.change_plan.changes)))
        table.add_row("New files", str(len(result.change_plan.new_files)))
        if result.change_plan.risk_notes:
            table.add_row("Risks", "\n".join(result.change_plan.risk_notes[:3]))

    if result.repo_analysis:
        table.add_row("Modules analyzed", str(len(result.repo_analysis.modules)))

    for status_name, count in result.task_stats.items():
        table.add_row(f"Tasks ({status_name})", str(count))

    if result.errors:
        error_text = "\n".join(result.errors[:3])
        if len(result.errors) > 3:
            error_text += f"\n... and {len(result.errors) - 3} more"
        table.add_row("Errors", error_text)

    console.print(table)


def _display_fullstack_result(result: PipelineResult) -> None:
    """Display fullstack pipeline results."""
    if result.success:
        console.print(Panel.fit(
            "[bold green]Fullstack project generated successfully![/bold green]",
            title="✅ Fullstack Complete",
        ))
    else:
        console.print(Panel.fit(
            "[bold yellow]Fullstack generation completed with issues[/bold yellow]",
            title="⚠️  Warning",
        ))

    table = Table(title="Fullstack Pipeline Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Workspace", str(result.workspace_path))
    table.add_row("Backend", str(result.workspace_path / "backend"))
    table.add_row("Frontend", str(result.workspace_path / "frontend"))
    table.add_row("Elapsed", f"{result.elapsed_seconds:.1f}s")

    m = result.metrics or {}
    if m.get("api_endpoints"):
        table.add_row("API endpoints", str(m["api_endpoints"]))
    if m.get("components_planned"):
        table.add_row("Components planned", str(m["components_planned"]))
    if m.get("components_generated") is not None:
        table.add_row("Components generated", str(m["components_generated"]))

    for status_name, count in result.task_stats.items():
        table.add_row(f"BE tasks ({status_name})", str(count))

    if result.errors:
        error_text = "\n".join(result.errors[:5])
        if len(result.errors) > 5:
            error_text += f"\n... and {len(result.errors) - 5} more"
        table.add_row("Errors", error_text)

    console.print(table)


def _display_result(result: PipelineResult) -> None:
    """Display pipeline results in a formatted table."""
    if result.success:
        console.print(Panel.fit(
            "[bold green]Pipeline completed successfully![/bold green]",
            title="✅ Success",
        ))
    else:
        console.print(Panel.fit(
            "[bold red]Pipeline completed with issues[/bold red]",
            title="⚠️  Warning",
        ))

    # Check if error is a config error (contains ❌ or specific keywords)
    is_config_error = result.errors and any(
        error.startswith("❌") or "missing api key" in error.lower() or 
        "not found" in error.lower() or "authentication" in error.lower() or 
        "no longer available" in error.lower()
        for error in result.errors
    )

    if is_config_error and result.errors:
        # Display config error with full message (rich formatting preserved)
        error_msg = result.errors[0]
        console.print(Panel(
            error_msg,
            title="Configuration Error",
            border_style="red" if not result.success else "yellow",
        ))
    else:
        # Display standard results table
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
            # Show first few errors
            error_text = "\n".join(result.errors[:3])
            if len(result.errors) > 3:
                error_text += f"\n... and {len(result.errors) - 3} more"
            table.add_row("Errors", error_text)

        console.print(table)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
