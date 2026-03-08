"""LLM client abstraction supporting multiple providers."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from config.settings import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    usage: dict[str, int]
    model: str
    stop_reason: str = ""


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass


class LLMConfigError(LLMClientError):
    """Exception for configuration errors."""
    pass


class LLMClient:
    """Unified LLM client supporting Anthropic and OpenAI."""

    # Valid models for each provider
    VALID_MODELS = {
        LLMProvider.ANTHROPIC: [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-sonnet-4-20250514",
            "claude-3-sonnet-20240229",
        ],
        LLMProvider.OPENAI: [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
        ],
        LLMProvider.GEMINI: [
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-pro",
        ],
    }

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Any = None
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate LLM configuration before use."""
        provider = self.config.provider

        # Check API key
        if provider == LLMProvider.ANTHROPIC:
            if not self.config.api_key or self.config.api_key.strip() == "":
                raise LLMConfigError(
                    f"❌ Missing API key for Anthropic\n\n"
                    f"Set your API key with:\n"
                    f"  export ANTHROPIC_API_KEY=\"sk-ant-your-key-here\"\n\n"
                    f"Get your key at: https://console.anthropic.com"
                )
        elif provider == LLMProvider.OPENAI:
            if not self.config.openai_api_key or self.config.openai_api_key.strip() == "":
                raise LLMConfigError(
                    f"❌ Missing API key for OpenAI\n\n"
                    f"Set your API key with:\n"
                    f"  export OPENAI_API_KEY=\"sk-proj-your-key-here\"\n\n"
                    f"Get your key at: https://platform.openai.com/api-keys"
                )
        elif provider == LLMProvider.GEMINI:
            if not self.config.gemini_api_key or self.config.gemini_api_key.strip() == "":
                raise LLMConfigError(
                    f"❌ Missing API key for Google Gemini\n\n"
                    f"Set your API key with:\n"
                    f"  export GEMINI_API_KEY=\"your-api-key-here\"\n\n"
                    f"Get your key at: https://aistudio.google.com/app/apikey"
                )

        # Check model validity
        valid_models = self.VALID_MODELS.get(provider, [])
        if valid_models and self.config.model not in valid_models:
            logger.warning(
                f"⚠️  Model '{self.config.model}' not in expected models for {provider.value}: "
                f"{', '.join(valid_models)}\n"
                f"This may work but could cause unexpected errors."
            )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            if self.config.provider == LLMProvider.ANTHROPIC:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.config.api_key)
            elif self.config.provider == LLMProvider.OPENAI:
                import openai
                self._client = openai.OpenAI(api_key=self.config.openai_api_key)
            elif self.config.provider == LLMProvider.GEMINI:
                import google.generativeai as genai
                genai.configure(api_key=self.config.gemini_api_key)
                self._client = genai
            else:
                raise LLMConfigError(f"Unsupported provider: {self.config.provider}")
        except LLMConfigError:
            raise
        except Exception as e:
            raise LLMConfigError(
                f"Failed to initialize {self.config.provider.value} client: {e}"
            ) from e
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a completion from the LLM."""
        client = self._get_client()
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        try:
            if self.config.provider == LLMProvider.ANTHROPIC:
                return await self._anthropic_generate(client, system_prompt, user_prompt, temp, tokens)
            elif self.config.provider == LLMProvider.GEMINI:
                return await self._gemini_generate(client, system_prompt, user_prompt, temp, tokens)
            else:
                return await self._openai_generate(client, system_prompt, user_prompt, temp, tokens)
        except LLMClientError:
            # Re-raise LLM-specific errors (including LLMConfigError)
            raise
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Handle provider-specific errors
            if "401" in error_msg or "Unauthorized" in error_msg or "Unauthenticated" in error_type:
                raise LLMConfigError(
                    f"❌ Authentication failed for {self.config.provider.value}\n\n"
                    f"Your API key may be invalid or expired.\n"
                    f"Please check your {self.config.provider.value.upper()}_API_KEY environment variable."
                ) from e
            elif "NotFound" in error_type or "404" in error_msg or "not found" in error_msg.lower():
                raise LLMConfigError(
                    f"❌ Model '{self.config.model}' not found or no longer available\n\n"
                    f"Valid models for {self.config.provider.value}:\n"
                    f"  {', '.join(self.VALID_MODELS.get(self.config.provider, []))}\n\n"
                    f"Use: codegen \"prompt\" --provider {self.config.provider.value} --model <valid-model>"
                ) from e
            elif "is not supported" in error_msg.lower():
                raise LLMConfigError(
                    f"❌ Model '{self.config.model}' is not supported for this operation\n\n"
                    f"Try a different model: {', '.join(self.VALID_MODELS.get(self.config.provider, []))}"
                ) from e
            
            # Catch unhandled errors and re-raise as LLMClientError
            raise LLMClientError(
                f"Failed to generate with {self.config.provider.value}: {e}"
            ) from e

    async def _anthropic_generate(
        self, client: Any, system: str, user: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        import anthropic

        def _sync() -> Any:
            try:
                return client.messages.create(
                    model=self.config.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
            except TypeError as e:
                if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                    raise LLMConfigError(
                        f"❌ Anthropic authentication failed\n\n"
                        f"Your API key is missing or invalid.\n"
                        f"Set: export ANTHROPIC_API_KEY=\"sk-ant-your-key\"\n"
                        f"Get key at: https://console.anthropic.com"
                    ) from e
                raise

        response = await asyncio.to_thread(_sync)
        return LLMResponse(
            content=response.content[0].text,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            model=response.model,
            stop_reason=response.stop_reason,
        )

    async def _openai_generate(
        self, client: Any, system: str, user: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        def _sync() -> Any:
            return client.chat.completions.create(
                model=self.config.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )

        response = await asyncio.to_thread(_sync)
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
            model=response.model,
            stop_reason=choice.finish_reason,
        )

    async def _gemini_generate(
        self, client: Any, system: str, user: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        def _sync() -> Any:
            model = client.GenerativeModel(
                model_name=self.config.model,
                system_instruction=system,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            return model.generate_content(user)

        try:
            response = await asyncio.to_thread(_sync)
        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__

            # Check for NotFound errors (model doesn't exist or is deprecated)
            if "notfound" in error_type.lower() or "404" in str(e) or "not found" in error_str:
                if "deprecated" in error_str or "no longer available" in error_str:
                    raise LLMConfigError(
                        f"❌ Gemini model '{self.config.model}' is deprecated or no longer available\n\n"
                        f"Try using a newer model:\n"
                        f"  • gemini-1.5-pro (recommended)\n"
                        f"  • gemini-1.5-flash\n"
                        f"  • gemini-1.5-8b\n\n"
                        f"Example: codegen \"prompt\" --provider gemini --model gemini-1.5-pro"
                    ) from e
                else:
                    raise LLMConfigError(
                        f"❌ Gemini model '{self.config.model}' not found\n\n"
                        f"Valid Gemini models:\n"
                        f"  • gemini-1.5-pro (recommended)\n"
                        f"  • gemini-1.5-flash\n"
                        f"  • gemini-1.5-8b\n\n"
                        f"Use: codegen \"prompt\" --provider gemini --model <valid-model>"
                    ) from e
            elif "unauthenticated" in error_str or "authentication" in error_str or "api_key" in error_str or "401" in str(e):
                raise LLMConfigError(
                    f"❌ Gemini authentication failed\n\n"
                    f"Your API key is missing or invalid.\n"
                    f"Set: export GEMINI_API_KEY=\"your-api-key\"\n"
                    f"Get key at: https://aistudio.google.com/app/apikey"
                ) from e
            elif "google" in error_type.lower() or "grpc" in error_type.lower():
                raise LLMConfigError(
                    f"❌ Google Gemini API error: {str(e)}\n\n"
                    f"Please check:\n"
                    f"  • Your GEMINI_API_KEY is valid\n"
                    f"  • The model name is correct\n"
                    f"  • Your account has access to this model"
                ) from e
            raise
        return LLMResponse(
            content=response.text,
            usage={
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
            },
            model=self.config.model,
            stop_reason=response.candidates[0].finish_reason.name if response.candidates else "",
        )

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate and parse a JSON response."""
        response = await self.generate(
            system_prompt=system_prompt + "\n\nRespond with valid JSON only. No markdown fences.",
            user_prompt=user_prompt,
            temperature=temperature,
        )
        text = response.content.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # Remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return json.loads(text)
