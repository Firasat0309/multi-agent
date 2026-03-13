"""Tests for BuildVerifierAgent output shaping for FIX_CODE handoff."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.build_verifier_agent import BuildVerifierAgent
from config.settings import LLMConfig, LLMProvider
from core.llm_client import LLMClient
from core.models import AgentContext, RepositoryBlueprint, Task, TaskType
from core.repository_manager import RepositoryManager


@pytest.mark.anyio
async def test_build_failure_output_is_coder_friendly(tmp_path: Path) -> None:
    repo = RepositoryManager(tmp_path)
    llm = LLMClient(LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-20250514", api_key="test"))

    terminal = MagicMock()
    terminal.run_command = AsyncMock(return_value=SimpleNamespace(
        exit_code=1,
        stdout="[INFO] compiling...\\nsrc/main/java/com/example/A.java:12: error: cannot find symbol",
        stderr="[ERROR] symbol: class MissingType\\n[ERROR] location: class A",
    ))

    agent = BuildVerifierAgent(llm_client=llm, repo_manager=repo, terminal=terminal)

    context = AgentContext(
        task=Task(
            task_id=1,
            task_type=TaskType.VERIFY_BUILD,
            file="src/main/java/com/example/A.java",
            description="Verify build",
        ),
        blueprint=RepositoryBlueprint(
            name="demo",
            description="demo",
            architecture_style="REST",
            tech_stack={"language": "java", "framework": "spring", "build_tool": "maven"},
        ),
    )

    result = await agent.execute(context)

    assert result.success is False
    assert "target_mentioned=True" in result.output
    assert result.errors and "compiler_output_tail" in result.errors[0]
    assert result.metrics.get("build_command") == "mvn package -DskipTests"
    assert result.metrics.get("target_path") == "src/main/java/com/example/A.java"
    assert result.metrics.get("exit_code") == 1
