from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest

from core.simple_loop_executor import SimpleLoopExecutor


def _make_executor(tmp_path, *, type_check_command: str, build_command: str) -> SimpleLoopExecutor:
    agent_manager = MagicMock()
    agent_manager.repo.workspace = tmp_path
    agent_manager.blueprint.file_blueprints = [
        SimpleNamespace(path="src/main.rs"),
    ]
    agent_manager.build_terminal = MagicMock()
    agent_manager._metrics = {
        "agent_metrics": {},
        "tasks_completed": 0,
        "tasks_failed": 0,
    }

    settings = SimpleNamespace(
        max_concurrent_agents=2,
        skip_agents=frozenset(),
        phase_timeout_seconds=60,
    )
    lang_profile = SimpleNamespace(
        type_check_command=type_check_command,
        build_command=build_command,
    )
    return SimpleLoopExecutor(agent_manager, settings, lang_profile)


@pytest.mark.anyio
async def test_incremental_build_prefers_type_check_command(tmp_path):
    executor = _make_executor(
        tmp_path,
        type_check_command="cargo check",
        build_command="cargo build",
    )
    commands: list[str] = []

    async def run_command(command: str, *, timeout: int = 120):
        commands.append(command)
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    executor._am.build_terminal.run_command = AsyncMock(side_effect=run_command)

    result = await executor._run_incremental_build()

    assert result.passed
    assert commands == ["cargo check"]


@pytest.mark.anyio
async def test_incremental_builds_are_serialized(tmp_path):
    executor = _make_executor(
        tmp_path,
        type_check_command="cargo check",
        build_command="cargo build",
    )
    in_flight = 0
    max_in_flight = 0

    async def run_command(command: str, *, timeout: int = 120):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await anyio.sleep(0.01)
        in_flight -= 1
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    executor._am.build_terminal.run_command = AsyncMock(side_effect=run_command)

    await asyncio.gather(
        executor._run_incremental_build(),
        executor._run_incremental_build(),
    )

    assert max_in_flight == 1


def test_pipeline_executor_is_a_simple_loop_compatibility_wrapper(tmp_path):
    from core.pipeline_executor import PipelineExecutor

    executor = _make_executor(
        tmp_path,
        type_check_command="cargo check",
        build_command="cargo build",
    )
    compat = PipelineExecutor(executor._am, executor._settings, executor._lang)

    assert isinstance(compat, SimpleLoopExecutor)