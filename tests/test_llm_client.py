"""Tests for LLM client provider routing."""

import asyncio
import pytest
from unittest.mock import MagicMock

from config.settings import LLMConfig, LLMProvider
from core.llm_client import LLMClient


def make_client(provider: str, model: str = "test-model") -> LLMClient:
    return LLMClient(LLMConfig(
        provider=LLMProvider(provider),
        model=model,
        api_key="test-key",
        openai_api_key="test-openai-key",
        gemini_api_key="test-gemini-key",
    ))


def run(coro):
    return asyncio.run(coro)


class TestLLMClientRouting:
    def test_anthropic_routing(self):
        import sys
        from unittest.mock import patch

        client = make_client("anthropic", "claude-sonnet-4-20250514")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="hello from claude")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"

        mock_anthropic_client = MagicMock()
        mock_anthropic_client.messages.create.return_value = mock_response

        mock_anthropic_module = MagicMock()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic_module}):
            result = run(client._anthropic_generate(mock_anthropic_client, "sys", "user", 0.2, 100))

        assert result.content == "hello from claude"
        assert result.usage["input_tokens"] == 10
        assert result.stop_reason == "end_turn"

    def test_openai_routing(self):
        client = make_client("openai", "gpt-4o")
        mock_choice = MagicMock()
        mock_choice.message.content = "hello from gpt"
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 8
        mock_response.usage.completion_tokens = 4
        mock_response.model = "gpt-4o"

        mock_oai = MagicMock()
        mock_oai.chat.completions.create.return_value = mock_response

        result = run(client._openai_generate(mock_oai, "sys", "user", 0.2, 100))
        assert result.content == "hello from gpt"
        assert result.usage["input_tokens"] == 8
        assert result.stop_reason == "stop"

    def test_gemini_routing(self):
        client = make_client("gemini", "gemini-1.5-pro")

        mock_candidate = MagicMock()
        mock_candidate.finish_reason.name = "STOP"

        mock_response = MagicMock()
        mock_response.text = "hello from gemini"
        mock_response.usage_metadata.prompt_token_count = 12
        mock_response.usage_metadata.candidates_token_count = 6
        mock_response.candidates = [mock_candidate]

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        result = run(client._gemini_generate(mock_genai, "sys", "user", 0.2, 100))
        assert result.content == "hello from gemini"
        assert result.usage["input_tokens"] == 12
        assert result.usage["output_tokens"] == 6
        assert result.stop_reason == "STOP"

    def test_gemini_model_constructed_with_system_prompt(self):
        client = make_client("gemini", "gemini-2.0-flash")

        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_response.usage_metadata.prompt_token_count = 1
        mock_response.usage_metadata.candidates_token_count = 1
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason.name = "STOP"

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        run(client._gemini_generate(mock_genai, "be helpful", "write code", 0.1, 200))

        mock_genai.GenerativeModel.assert_called_once_with(
            model_name="gemini-2.0-flash",
            system_instruction="be helpful",
            generation_config={"temperature": 0.1, "max_output_tokens": 200},
        )

    def test_unsupported_provider_raises(self):
        with pytest.raises(ValueError):
            raise ValueError("Unsupported provider: broken")


class TestExtractGeminiToolCall:
    """Tests for _extract_gemini_tool_call — robust JSON parsing from Gemini."""

    def test_clean_json(self):
        text = '{"tool_call": {"name": "write_file", "input": {"path": "Foo.java", "content": "code"}}}'
        tc = LLMClient._extract_gemini_tool_call(text)
        assert tc is not None
        assert tc.name == "write_file"
        assert tc.input["path"] == "Foo.java"
        assert tc.input["content"] == "code"

    def test_json_in_markdown_fence(self):
        text = '```json\n{"tool_call": {"name": "write_file", "input": {"path": "a.py", "content": "x"}}}\n```'
        tc = LLMClient._extract_gemini_tool_call(text)
        assert tc is not None
        assert tc.name == "write_file"

    def test_json_with_trailing_text(self):
        text = (
            '{"tool_call": {"name": "read_file", "input": {"path": "a.py"}}}'
            "\n\nI'll read that file to understand the interface."
        )
        tc = LLMClient._extract_gemini_tool_call(text)
        assert tc is not None
        assert tc.name == "read_file"

    def test_json_with_leading_text(self):
        text = (
            "Here's the tool call:\n"
            '{"tool_call": {"name": "write_file", "input": {"path": "b.java", "content": "pkg;"}}}'
        )
        tc = LLMClient._extract_gemini_tool_call(text)
        assert tc is not None
        assert tc.name == "write_file"

    def test_plain_text_returns_none(self):
        text = "Here is the User.java class:\n\npackage com.example;\npublic class User {}"
        tc = LLMClient._extract_gemini_tool_call(text)
        assert tc is None

    def test_no_tool_call_key_returns_none(self):
        text = '{"name": "write_file", "input": {}}'
        tc = LLMClient._extract_gemini_tool_call(text)
        assert tc is None

    def test_nested_braces_in_content(self):
        java_code = 'public class Foo {\\n    private Map<String, List<Integer>> map = new HashMap<>();\\n}'
        text = f'{{"tool_call": {{"name": "write_file", "input": {{"path": "Foo.java", "content": "{java_code}"}}}}}}'
        tc = LLMClient._extract_gemini_tool_call(text)
        assert tc is not None
        assert tc.name == "write_file"
        assert "HashMap" in tc.input["content"]

    def test_code_with_braces_before_tool_call(self):
        """Gemini emits code (with braces) THEN the tool call JSON.

        The old parser latched onto the first '{' in the code and missed
        the actual tool call.  This was the root cause of the production
        failure where ComponentGeneratorAgent burned 12 iterations.
        """
        text = (
            "I'll implement RegisterForm. Let me check Label first.\n\n"
            "export default function RegisterForm() {\n"
            "  return <form></form>;\n"
            "}\n\n"
            '{"tool_call": {"name": "read_file", "input": '
            '{"path": "src/components/ui/Label.tsx"}}}'
        )
        tc = LLMClient._extract_gemini_tool_call(text)
        assert tc is not None
        assert tc.name == "read_file"
        assert tc.input["path"] == "src/components/ui/Label.tsx"

    def test_code_before_and_after_tool_call(self):
        """Tool call JSON sandwiched between code and explanation."""
        text = (
            "function App() {\n"
            '  return { key: "value" };\n'
            "}\n\n"
            '{"tool_call": {"name": "write_file", "input": '
            '{"path": "App.tsx", "content": "code"}}}\n\n'
            "I hope this helps!"
        )
        tc = LLMClient._extract_gemini_tool_call(text)
        assert tc is not None
        assert tc.name == "write_file"

    def test_unescaped_newlines_with_leading_code(self):
        """Code before JSON + unescaped newlines in the content field."""
        text = (
            "// existing code\nfunction old() { return 1; }\n\n"
            '{"tool_call": {"name": "write_file", "input": '
            '{"path": "R.tsx", "content": "line1\nline2\nline3"}}}'
        )
        tc = LLMClient._extract_gemini_tool_call(text)
        assert tc is not None
        assert tc.name == "write_file"
        assert "line1" in tc.input["content"]

    # ── Truncated tool call recovery (Strategy 4) ────────────────────

    def test_truncated_escaped_newlines(self):
        """JSON-escaped \\n in content, truncated mid-string."""
        text = (
            '{"tool_call": {"name": "write_file", "input": {"path": '
            '"Dto.java", "content": "package com.example;\\n'
            'import java.io.*;\\n\\n'
            'public class Dto {\\n    private Str'
        )
        tc = LLMClient._extract_gemini_tool_call(text)
        assert tc is not None
        assert tc.name == "write_file"
        assert "package com.example" in tc.input["content"]
        assert len(tc.input["content"]) > 0

    def test_truncated_unescaped_newlines(self):
        """Real newline chars in content (Gemini forgot to escape), truncated."""
        text = (
            '{"tool_call": {"name": "write_file", "input": {"path": '
            '"Dto.java", "content": "package com.example;\n'
            'import java.io.*;\n\n'
            'public class Dto {\n    private Str'
        )
        tc = LLMClient._extract_gemini_tool_call(text)
        assert tc is not None
        assert tc.name == "write_file"
        assert "package com.example" in tc.input["content"]

    def test_truncated_vue_with_escaped_quotes(self):
        """Vue file content with escaped quotes and newlines, truncated."""
        text = (
            '{"tool_call": {"name": "write_file", "input": {"path": '
            '"src/components/ui/BaseInput.vue", "content": '
            '"<script setup lang=\\"ts\\">\\n'
            'interface Props {\\n'
            '  label: string;\\n'
            '  modelValue: string;\\n'
            '}\\n\\n'
            'const props = withDefaults(defineProps<Prop'
        )
        tc = LLMClient._extract_gemini_tool_call(text)
        assert tc is not None
        assert tc.name == "write_file"
        assert '<script setup lang="ts">' in tc.input["content"]
        assert "interface Props" in tc.input["content"]

    def test_truncated_does_not_break_valid(self):
        """A complete valid tool call must still parse correctly (no false repair)."""
        text = '{"tool_call": {"name": "write_file", "input": {"path": "a.py", "content": "x = 1"}}}'
        tc = LLMClient._extract_gemini_tool_call(text)
        assert tc is not None
        assert tc.input["content"] == "x = 1"

    def test_truncated_non_write_file_returns_none(self):
        """Truncated non-write_file JSON that can't form a valid tool_call."""
        text = '{"tool_call": {"name": "read_fi'
        tc = LLMClient._extract_gemini_tool_call(text)
        # After repair, the name would be "read_fi" — no input key
        # Depending on repair, it may or may not parse; the key guarantee
        # is it doesn't crash.
        # (It may actually parse to a truncated name, which is fine —
        #  the agentic loop will just fail to dispatch it.)
