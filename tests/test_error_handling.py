"""Tests for LLM error handling: missing API keys, invalid models, auth failures."""

from unittest.mock import MagicMock

import pytest

from config.settings import LLMConfig, LLMProvider
from core.llm_client import LLMClient, LLMConfigError


def make_config(**kwargs) -> LLMConfig:
    defaults = dict(
        provider=LLMProvider.ANTHROPIC,
        model="claude-sonnet-4-20250514",
        api_key="",
        openai_api_key="",
        gemini_api_key="",
    )
    defaults.update(kwargs)
    return LLMConfig(**defaults)


class TestMissingApiKeys:
    def test_missing_anthropic_key_raises(self):
        with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
            LLMClient(make_config(provider=LLMProvider.ANTHROPIC, api_key=""))

    def test_missing_openai_key_raises(self):
        with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
            LLMClient(make_config(provider=LLMProvider.OPENAI, openai_api_key=""))

    def test_missing_gemini_key_raises(self):
        with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
            LLMClient(make_config(provider=LLMProvider.GEMINI, gemini_api_key=""))

    def test_whitespace_only_key_raises(self):
        with pytest.raises(LLMConfigError):
            LLMClient(make_config(provider=LLMProvider.ANTHROPIC, api_key="   "))

    def test_valid_anthropic_key_does_not_raise(self):
        client = LLMClient(make_config(provider=LLMProvider.ANTHROPIC, api_key="sk-ant-real"))
        assert client is not None

    def test_valid_openai_key_does_not_raise(self):
        client = LLMClient(make_config(
            provider=LLMProvider.OPENAI,
            openai_api_key="sk-proj-real",
        ))
        assert client is not None

    def test_valid_gemini_key_does_not_raise(self):
        client = LLMClient(make_config(
            provider=LLMProvider.GEMINI,
            gemini_api_key="AIza-real-key",
        ))
        assert client is not None


class TestInvalidModel:
    def test_unknown_model_warns_not_raises(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            client = LLMClient(make_config(
                provider=LLMProvider.GEMINI,
                model="gemini-invalid-99",
                gemini_api_key="real-key",
            ))
        assert client is not None
        assert "not in expected models" in caplog.text

    def test_known_model_no_warning(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            LLMClient(make_config(
                provider=LLMProvider.GEMINI,
                model="gemini-1.5-pro",
                gemini_api_key="real-key",
            ))
        assert "not in expected models" not in caplog.text


class TestRuntimeAuthErrors:
    def test_openai_auth_error_becomes_llm_config_error(self):
        import asyncio
        client = LLMClient(make_config(
            provider=LLMProvider.OPENAI,
            openai_api_key="bad-key",
        ))
        # Patch _get_client so no real SDK call is made
        mock_oai = MagicMock()
        mock_oai.chat.completions.create.side_effect = Exception("401 Unauthorized")
        client._client = mock_oai

        with pytest.raises(LLMConfigError, match="Authentication failed"):
            asyncio.run(client.generate("sys", "user"))

    def test_gemini_not_found_error_becomes_llm_config_error(self):
        import asyncio

        class FakeNotFound(Exception):
            pass
        FakeNotFound.__name__ = "NotFound"

        client = LLMClient(make_config(
            provider=LLMProvider.GEMINI,
            model="gemini-old",
            gemini_api_key="real-key",
        ))
        mock_genai = MagicMock()
        mock_genai.GenerativeModel.side_effect = FakeNotFound("404 not found: model gemini-old")
        client._client = mock_genai

        with pytest.raises(LLMConfigError, match="not found"):
            asyncio.run(client.generate("sys", "user"))
