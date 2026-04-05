"""Typed message schema for the agentic tool-use conversation loop.

Replaces raw ``list[dict]`` message passing with structured dataclasses that
provide type safety, IDE autocompletion, and prevent the class of bugs caused
by typos in dict keys (``"conntent"``, ``"roole"``).

Inspired by Claude Code's typed message blocks (``TextBlock``, ``ToolUseBlock``,
``ToolResultBlock``) that flow through the generator-based query loop.

The classes remain fully dict-compatible: ``to_dict()`` converts to the raw
format expected by the Anthropic/OpenAI APIs, and ``from_dict()`` parses API
responses back into typed objects.

Usage::

    from core.messages import UserMessage, AssistantMessage, ToolResult

    msg = UserMessage(content="Generate a REST controller for users.")
    messages = [msg.to_dict()]

    # After LLM response:
    assistant = AssistantMessage.from_raw_content(response.raw_content)
    messages.append(assistant.to_dict())

    # Tool results:
    results = [ToolResult(tool_use_id="abc", content="Written 500 bytes to ...")]
    user_reply = UserMessage.from_tool_results(results)
    messages.append(user_reply.to_dict())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Content blocks ───────────────────────────────────────────────────────────

@dataclass(slots=True)
class TextBlock:
    """A plain text content block."""
    text: str
    type: str = "text"

    def to_dict(self) -> dict:
        return {"type": self.type, "text": self.text}


@dataclass(slots=True)
class ToolUseBlock:
    """A tool invocation block from the assistant."""
    tool_use_id: str
    name: str
    input: dict
    type: str = "tool_use"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "id": self.tool_use_id,
            "name": self.name,
            "input": self.input,
        }


@dataclass(slots=True)
class ToolResult:
    """A single tool execution result."""
    tool_use_id: str
    content: str
    type: str = "tool_result"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "tool_use_id": self.tool_use_id,
            "content": self.content,
        }


ContentBlock = TextBlock | ToolUseBlock | ToolResult


# ── Message types ────────────────────────────────────────────────────────────

@dataclass(slots=True)
class UserMessage:
    """A user-role message (task prompt, tool results, or nudge)."""
    content: str | list[dict]
    role: str = "user"

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_text(cls, text: str) -> "UserMessage":
        return cls(content=text)

    @classmethod
    def from_tool_results(
        cls,
        results: list[ToolResult],
        nudge_text: str | None = None,
    ) -> "UserMessage":
        """Build a user message containing tool results and an optional nudge."""
        content: list[dict] = [r.to_dict() for r in results]
        if nudge_text:
            content.append({"type": "text", "text": nudge_text})
        return cls(content=content)


@dataclass(slots=True)
class AssistantMessage:
    """An assistant-role message (LLM response)."""
    content: str | list[Any]
    role: str = "assistant"

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_raw_content(cls, raw_content: list[Any]) -> "AssistantMessage":
        """Create from the raw content block list returned by the LLM API."""
        return cls(content=raw_content)

    @classmethod
    def from_text(cls, text: str) -> "AssistantMessage":
        return cls(content=text)


@dataclass(slots=True)
class SystemMessage:
    """A system-role message (typically the first message)."""
    content: str
    role: str = "system"

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


# ── Convenience constructors ─────────────────────────────────────────────────

Message = UserMessage | AssistantMessage | SystemMessage


def to_dict_list(messages: list[Message]) -> list[dict]:
    """Convert a typed message list to the raw dict format for API calls."""
    return [m.to_dict() for m in messages]


def from_dict(raw: dict) -> Message:
    """Parse a raw message dict into a typed message object."""
    role = raw.get("role", "user")
    content = raw.get("content", "")
    if role == "assistant":
        return AssistantMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)
    return UserMessage(content=content)
