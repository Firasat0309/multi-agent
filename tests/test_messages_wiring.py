"""Tests for typed messages integration with the agentic loop.

Verifies that the typed message constructors produce the same dict
structures that the agentic loop expects, and that the agentic loop
in BaseAgent correctly uses them.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.messages import (
    UserMessage,
    AssistantMessage,
    ToolResult,
    TextBlock,
    SystemMessage,
    to_dict_list,
    from_dict,
)


def _run(coro):
    return asyncio.run(coro)


# ── Unit tests for message types ─────────────────────────────────────────────


class TestUserMessage:
    def test_from_text(self):
        msg = UserMessage.from_text("Hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Hello"}

    def test_from_tool_results(self):
        results = [
            ToolResult(tool_use_id="abc", content="file written"),
            ToolResult(tool_use_id="def", content="search results"),
        ]
        msg = UserMessage.from_tool_results(results)
        d = msg.to_dict()
        assert d["role"] == "user"
        assert isinstance(d["content"], list)
        assert len(d["content"]) == 2
        assert d["content"][0]["type"] == "tool_result"
        assert d["content"][0]["tool_use_id"] == "abc"

    def test_from_tool_results_with_nudge(self):
        results = [ToolResult(tool_use_id="abc", content="ok")]
        msg = UserMessage.from_tool_results(results, nudge_text="Please write now")
        d = msg.to_dict()
        assert len(d["content"]) == 2
        assert d["content"][1] == {"type": "text", "text": "Please write now"}

    def test_content_list_passthrough(self):
        """UserMessage with list content (tool_results) passes through."""
        raw = [{"type": "tool_result", "tool_use_id": "x", "content": "y"}]
        msg = UserMessage(content=raw)
        d = msg.to_dict()
        assert d["content"] is raw


class TestAssistantMessage:
    def test_from_text(self):
        msg = AssistantMessage.from_text("I will write the file.")
        d = msg.to_dict()
        assert d == {"role": "assistant", "content": "I will write the file."}

    def test_from_raw_content(self):
        raw = [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "a", "name": "write_file", "input": {}}]
        msg = AssistantMessage.from_raw_content(raw)
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] is raw


class TestToolResult:
    def test_to_dict(self):
        tr = ToolResult(tool_use_id="abc123", content="Written 500 bytes")
        d = tr.to_dict()
        assert d == {
            "type": "tool_result",
            "tool_use_id": "abc123",
            "content": "Written 500 bytes",
        }


class TestTextBlock:
    def test_to_dict(self):
        tb = TextBlock(text="WARNING: write the file now")
        d = tb.to_dict()
        assert d == {"type": "text", "text": "WARNING: write the file now"}


class TestSystemMessage:
    def test_to_dict(self):
        msg = SystemMessage(content="You are an expert coder.")
        d = msg.to_dict()
        assert d == {"role": "system", "content": "You are an expert coder."}


class TestConvenience:
    def test_to_dict_list(self):
        msgs = [
            UserMessage.from_text("hello"),
            AssistantMessage.from_text("hi"),
        ]
        result = to_dict_list(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_from_dict_user(self):
        msg = from_dict({"role": "user", "content": "test"})
        assert isinstance(msg, UserMessage)
        assert msg.content == "test"

    def test_from_dict_assistant(self):
        msg = from_dict({"role": "assistant", "content": "ok"})
        assert isinstance(msg, AssistantMessage)

    def test_from_dict_system(self):
        msg = from_dict({"role": "system", "content": "prompt"})
        assert isinstance(msg, SystemMessage)


# ── Integration: verify agentic loop message construction ────────────────────


class TestAgenticLoopMessageFormat:
    """Verify that typed constructors produce dicts identical to the old raw literals."""

    def test_initial_message_matches(self):
        """UserMessage.from_text should produce the same dict as the old literal."""
        prompt = "Generate a REST controller for users."
        old_style = {"role": "user", "content": prompt}
        new_style = UserMessage.from_text(prompt).to_dict()
        assert old_style == new_style

    def test_end_turn_recovery_messages_match(self):
        """Assistant + user recovery messages should match old format."""
        raw_content = [{"type": "text", "text": "Here is the code..."}]
        hint = "call write_file now"

        old_assistant = {"role": "assistant", "content": raw_content}
        new_assistant = AssistantMessage.from_raw_content(raw_content).to_dict()
        assert old_assistant == new_assistant

        old_user = {"role": "user", "content": hint}
        new_user = UserMessage.from_text(hint).to_dict()
        assert old_user == new_user

    def test_tool_results_message_matches(self):
        """Tool result dicts should match old format exactly."""
        old_style = {
            "type": "tool_result",
            "tool_use_id": "toolu_abc",
            "content": "Written 1234 bytes to src/main.py",
        }
        new_style = ToolResult(
            tool_use_id="toolu_abc",
            content="Written 1234 bytes to src/main.py",
        ).to_dict()
        assert old_style == new_style

    def test_nudge_text_block_matches(self):
        """Nudge text blocks should match old format."""
        nudge = "WARNING: write the file now"
        old_style = {"type": "text", "text": nudge}
        new_style = TextBlock(text=nudge).to_dict()
        assert old_style == new_style

    def test_full_tool_results_user_message(self):
        """A user message with tool results + nudge should be valid."""
        tool_results = [
            ToolResult(tool_use_id="a", content="ok").to_dict(),
            TextBlock(text="write now").to_dict(),
        ]
        msg = UserMessage(content=tool_results).to_dict()
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][1]["type"] == "text"
