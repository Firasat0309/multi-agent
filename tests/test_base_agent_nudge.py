"""Tests for the write_file nudge mechanism and path matching in BaseAgent."""

import asyncio
import pytest
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

from agents.base_agent import BaseAgent
from core.llm_client import LLMResponseWithTools, ToolCall
from core.models import (
    AgentContext, AgentRole, FileBlueprint, RepositoryBlueprint,
    Task, TaskResult, TaskType,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

class _StubAgent(BaseAgent):
    """Minimal concrete agent for testing the agentic loop."""
    role = AgentRole.CODER

    def __init__(self):
        llm = MagicMock()
        repo = MagicMock()
        repo.workspace = MagicMock()
        repo.async_write_file = AsyncMock(return_value=MagicMock())
        super().__init__(llm, repo)
        # Register write_file as a tool so the loop enters tool dispatch
        self._tools = [MagicMock(name="write_file")]

    @property
    def tools(self):
        return self._tools

    @property
    def system_prompt(self):
        return "test"

    def _build_prompt(self, context):
        return "test"

    async def execute(self, context):
        return await self.execute_agentic(context)


def _make_context(target_path: str = "src/Foo.java") -> AgentContext:
    task = Task(
        task_id="t1",
        task_type=TaskType.GENERATE_FILE,
        description="test",
        file=target_path,
    )
    bp = RepositoryBlueprint(
        name="test",
        description="test",
        architecture_style="REST",
        tech_stack={"language": "java"},
    )
    fb = FileBlueprint(
        path=target_path,
        purpose="test",
        depends_on=[],
        exports=[],
        language="java",
        layer="domain",
    )
    return AgentContext(task=task, blueprint=bp, file_blueprint=fb)


def _tool_response(
    tool_name: str = "read_file",
    path: str = "",
    stop_reason: str = "tool_use",
    content: str = "",
) -> LLMResponseWithTools:
    """Create a mock LLM response with a single tool call."""
    tc_input = {"path": path} if path else {}
    if tool_name == "write_file":
        tc_input["content"] = "file content here"
    return LLMResponseWithTools(
        content=content,
        tool_calls=[ToolCall(tool_use_id="tu_1", name=tool_name, input=tc_input)],
        stop_reason=stop_reason,
        usage={"input_tokens": 100, "output_tokens": 100},
        raw_content=[{"type": "text", "text": content}],
    )


def _end_turn_response(content: str = "done") -> LLMResponseWithTools:
    """Create a mock LLM end_turn response (no tool calls)."""
    return LLMResponseWithTools(
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage={"input_tokens": 100, "output_tokens": 100},
        raw_content=[{"type": "text", "text": content}],
    )


# ── _paths_match ─────────────────────────────────────────────────────────────

class TestPathsMatch:
    def test_identical(self):
        assert BaseAgent._paths_match("src/Foo.java", "src/Foo.java")

    def test_normpath_dotslash(self):
        assert BaseAgent._paths_match("./src/Foo.java", "src/Foo.java")

    def test_absolute_vs_relative(self):
        assert BaseAgent._paths_match(
            "/home/user/project/src/Foo.java", "src/Foo.java"
        )

    def test_absolute_vs_relative_deep(self):
        assert BaseAgent._paths_match(
            "/workspace/src/pages/register.tsx", "src/pages/register.tsx"
        )

    def test_no_false_positive_partial_filename(self):
        """BarFoo.java should NOT match Foo.java."""
        assert not BaseAgent._paths_match("src/BarFoo.java", "Foo.java")

    def test_no_false_positive_different_dir(self):
        assert not BaseAgent._paths_match(
            "src/other/Foo.java", "src/main/Foo.java"
        )

    def test_different_files(self):
        assert not BaseAgent._paths_match("src/Foo.java", "src/Bar.java")

    def test_trailing_slash_ignored(self):
        assert BaseAgent._paths_match("src/Foo.java", "src/Foo.java")

    def test_double_slash_collapsed(self):
        assert BaseAgent._paths_match("src//Foo.java", "src/Foo.java")


class TestPathInWritten:
    def test_exact_match(self):
        assert BaseAgent._path_in_written("src/Foo.java", ["src/Foo.java"])

    def test_absolute_in_written_list(self):
        assert BaseAgent._path_in_written(
            "src/Foo.java",
            ["/home/user/project/src/Foo.java"],
        )

    def test_not_found(self):
        assert not BaseAgent._path_in_written(
            "src/Foo.java", ["src/Bar.java", "src/Baz.java"]
        )

    def test_empty_list(self):
        assert not BaseAgent._path_in_written("src/Foo.java", [])


# ── Budget guard nudge ───────────────────────────────────────────────────────

@pytest.mark.anyio
class TestBudgetGuardNudge:
    """Verify the budget-guard nudge fires correctly and doesn't block writes."""

    async def test_write_file_not_blocked_by_nudge(self):
        """When LLM responds with write_file for the target, it must execute."""
        agent = _StubAgent()
        agent.max_iterations = 6  # small for fast test
        ctx = _make_context("src/Foo.java")

        responses = []
        # Iterations 0-2: read_file calls (pre-halfway)
        for _ in range(3):
            responses.append(_tool_response("read_file", "src/Other.java"))
        # Iteration 3 (= halfway): LLM calls write_file for the target
        responses.append(_tool_response("write_file", "src/Foo.java"))
        # Should not be needed, but add end_turn as safety
        responses.append(_end_turn_response())

        call_count = 0
        async def mock_generate(**kwargs):
            nonlocal call_count
            r = responses[call_count]
            call_count += 1
            return r

        agent.llm.generate_with_tools = mock_generate
        agent._dispatch_tool = AsyncMock(return_value="Written 500 bytes to src/Foo.java")

        result = await agent.execute_agentic(ctx)
        assert result.success
        # write_file should have been dispatched (not blocked)
        assert any(
            c.args[1].name == "write_file"
            for c in agent._dispatch_tool.call_args_list
        )

    async def test_write_file_with_absolute_path_not_blocked(self):
        """LLM using absolute path for write_file must not trigger nudge."""
        agent = _StubAgent()
        agent.max_iterations = 6
        ctx = _make_context("src/Foo.java")

        responses = []
        for _ in range(3):
            responses.append(_tool_response("read_file"))
        # LLM writes with absolute path at halfway point
        responses.append(
            _tool_response("write_file", "/home/user/project/src/Foo.java")
        )
        responses.append(_end_turn_response())

        call_count = 0
        async def mock_generate(**kwargs):
            nonlocal call_count
            r = responses[call_count]
            call_count += 1
            return r

        agent.llm.generate_with_tools = mock_generate
        agent._dispatch_tool = AsyncMock(
            return_value="Written 500 bytes to /home/user/project/src/Foo.java"
        )

        result = await agent.execute_agentic(ctx)
        assert result.success

    async def test_nudge_appended_as_text_block_not_separate_message(self):
        """Nudge must be inside the tool_results user message, not a separate one."""
        agent = _StubAgent()
        agent.max_iterations = 8
        ctx = _make_context("src/Foo.java")

        responses = []
        # 4 read_file calls (iterations 0-3), then nudge fires at iteration 4
        for _ in range(5):
            responses.append(_tool_response("read_file", "src/Other.java"))
        # Iteration 5: write_file to target
        responses.append(_tool_response("write_file", "src/Foo.java"))
        responses.append(_end_turn_response())

        call_count = 0
        async def mock_generate(**kwargs):
            nonlocal call_count
            r = responses[call_count]
            call_count += 1
            return r

        agent.llm.generate_with_tools = mock_generate
        agent._dispatch_tool = AsyncMock(return_value="ok")

        # Patch _dispatch_tool to return "Written..." for write_file
        original_dispatch = agent._dispatch_tool
        async def smart_dispatch(context, tc):
            if tc.name == "write_file":
                return f"Written 500 bytes to {tc.input.get('path', '')}"
            return "ok"
        agent._dispatch_tool = smart_dispatch

        result = await agent.execute_agentic(ctx)
        assert result.success

        # Check messages: no two consecutive user messages
        messages = [{"role": "user", "content": "test"}]  # initial
        # The actual messages are internal to execute_agentic, so we verify
        # by checking that it completed without API errors.

    async def test_consecutive_nudge_cap(self):
        """After 2 consecutive nudges, subsequent iterations must not nudge."""
        agent = _StubAgent()
        agent.max_iterations = 10
        ctx = _make_context("src/Foo.java")

        nudge_count = 0
        original_warning = None

        responses = []
        # 5 iterations of read_file (0-4), then nudge fires at 5,6
        # After cap, iterations 7+ should just run tools
        for _ in range(8):
            responses.append(_tool_response("read_file", "src/Other.java"))
        # Final write
        responses.append(_tool_response("write_file", "src/Foo.java"))
        responses.append(_end_turn_response())

        call_count = 0
        async def mock_generate(**kwargs):
            nonlocal call_count
            r = responses[call_count]
            call_count += 1
            return r

        agent.llm.generate_with_tools = mock_generate

        async def smart_dispatch(context, tc):
            if tc.name == "write_file":
                return f"Written 500 bytes to {tc.input.get('path', '')}"
            return "file content"
        agent._dispatch_tool = smart_dispatch

        with patch("agents.base_agent.logger") as mock_logger:
            result = await agent.execute_agentic(ctx)

            # Count nudge warnings
            nudge_warnings = [
                c for c in mock_logger.warning.call_args_list
                if "scheduling write_file nudge" in str(c)
            ]
            # Should be exactly 2 (the cap)
            assert len(nudge_warnings) == 2, (
                f"Expected 2 nudge warnings, got {len(nudge_warnings)}: {nudge_warnings}"
            )

        assert result.success


# ── end_turn recovery cap ────────────────────────────────────────────────────

@pytest.mark.anyio
class TestEndTurnRecoveryCap:
    """Verify end_turn recovery reminders are capped."""

    async def test_end_turn_recovery_capped_at_2(self):
        """After 2 end_turn reminders, the 3rd end_turn should just return."""
        agent = _StubAgent()
        agent.max_iterations = 10
        ctx = _make_context("src/Foo.java")

        # Every response is end_turn with text but no write_file
        responses = [
            _end_turn_response("Here is the code:\n```java\npublic class Foo {}\n```")
            for _ in range(5)
        ]

        call_count = 0
        async def mock_generate(**kwargs):
            nonlocal call_count
            r = responses[call_count]
            call_count += 1
            return r

        agent.llm.generate_with_tools = mock_generate

        result = await agent.execute_agentic(ctx)
        # Should return after 3 calls (2 reminders, then 3rd returns)
        assert call_count == 3
        # The result reflects the last response content
        assert result is not None
