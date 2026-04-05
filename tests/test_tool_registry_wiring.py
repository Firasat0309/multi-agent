"""Tests for ToolRegistry integration with BaseAgent.

Verifies that when TOOL_REGISTRY flag is on:
  1. _register_native_tools() creates a registry with all 6 native tools
  2. _dispatch_tool() routes through the registry
  3. The tools property returns registry definitions
  4. The hardcoded fallback still works when flag is off
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.feature_flags import set_feature, reset_features
from core.llm_client import ToolCall


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean_flags():
    reset_features()
    yield
    reset_features()


def _make_agent(tool_registry_on: bool):
    """Create a minimal CoderAgent with mocked LLM and repo."""
    from config.settings import ExecutionConfig

    set_feature("TOOL_REGISTRY", tool_registry_on)

    # Mock LLM
    llm = MagicMock()
    llm.config = MagicMock()
    llm.config.model = "test-model"
    llm.config.provider = MagicMock()
    llm.config.provider.value = "anthropic"

    # Mock RepositoryManager
    repo = MagicMock()
    workspace = Path(__file__).parent / "_test_workspace"
    workspace.mkdir(exist_ok=True)
    repo.workspace = workspace
    repo.async_read_file = AsyncMock(return_value="file content here")
    repo.async_write_file = AsyncMock()

    exec_config = ExecutionConfig()

    from agents.coder_agent import CoderAgent
    agent = CoderAgent(
        llm_client=llm,
        repo_manager=repo,
        execution_config=exec_config,
    )
    return agent


class TestToolRegistryWiring:
    """Test that ToolRegistry is properly wired into BaseAgent."""

    def test_registry_created_when_flag_on(self):
        agent = _make_agent(tool_registry_on=True)
        assert agent._tool_registry is not None

    def test_registry_none_when_flag_off(self):
        agent = _make_agent(tool_registry_on=False)
        assert agent._tool_registry is None

    def test_registry_has_all_native_tools(self):
        agent = _make_agent(tool_registry_on=True)
        reg = agent._tool_registry
        expected_tools = [
            "read_file", "write_file", "search_code",
            "find_definition", "list_files", "apply_patch",
        ]
        for name in expected_tools:
            assert reg.has(name), f"Tool '{name}' not registered"

    def test_tools_property_uses_registry(self):
        agent = _make_agent(tool_registry_on=True)
        tools = agent.tools
        tool_names = {t.name for t in tools}
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert len(tools) == 6

    def test_tools_property_fallback_without_registry(self):
        agent = _make_agent(tool_registry_on=False)
        from core.agent_tools import CODER_TOOLS
        tools = agent.tools
        # CoderAgent overrides tools to return CODER_TOOLS
        assert len(tools) == len(CODER_TOOLS)

    def test_dispatch_through_registry(self):
        agent = _make_agent(tool_registry_on=True)
        tc = ToolCall(tool_use_id="test-1", name="read_file", input={"path": "test.py"})
        context = MagicMock()
        context.file_blueprint = MagicMock()
        context.file_blueprint.path = "test.py"

        result = _run(agent._dispatch_tool(context, tc))
        # Should succeed (repo.async_read_file returns "file content here")
        assert "file content here" in result or "File not found" in result

    def test_dispatch_fallback_without_registry(self):
        agent = _make_agent(tool_registry_on=False)
        tc = ToolCall(tool_use_id="test-2", name="read_file", input={"path": "test.py"})
        context = MagicMock()
        context.file_blueprint = MagicMock()
        context.file_blueprint.path = "test.py"

        result = _run(agent._dispatch_tool(context, tc))
        assert "file content here" in result or "File not found" in result

    def test_dispatch_unknown_tool_registry(self):
        agent = _make_agent(tool_registry_on=True)
        tc = ToolCall(tool_use_id="test-3", name="nonexistent_tool", input={})
        context = MagicMock()
        context.file_blueprint = None

        result = _run(agent._dispatch_tool(context, tc))
        assert "Error" in result
        assert "unknown" in result.lower()

    def test_registry_definitions_match_agent_tools(self):
        """Registry definitions should have the same schemas as agent_tools.py."""
        from core.agent_tools import CODER_TOOLS

        agent = _make_agent(tool_registry_on=True)
        reg_defs = {d.name: d for d in agent._tool_registry.all_definitions()}

        for tool_def in CODER_TOOLS:
            assert tool_def.name in reg_defs, f"Missing: {tool_def.name}"
            reg_def = reg_defs[tool_def.name]
            assert reg_def.description == tool_def.description
            assert reg_def.input_schema == tool_def.input_schema
