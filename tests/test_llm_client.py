"""Tests for LLM client provider routing."""

import asyncio
import pytest
from unittest.mock import MagicMock

from config.settings import LLMConfig, LLMProvider
from core.llm_client import LLMClient, LLMResponse


def make_client(provider: str, model: str = "test-model") -> LLMClient:
    return LLMClient(LLMConfig(
        provider=LLMProvider(provider),
        model=model,
        api_key="test-key",
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
