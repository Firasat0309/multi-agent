"""LLM client abstraction supporting multiple providers."""

from __future__ import annotations

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


class LLMClient:
    """Unified LLM client supporting Anthropic and OpenAI."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if self.config.provider == LLMProvider.ANTHROPIC:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.config.api_key)
        elif self.config.provider == LLMProvider.OPENAI:
            import openai
            self._client = openai.OpenAI(api_key=self.config.api_key)
        elif self.config.provider == LLMProvider.GEMINI:
            import google.generativeai as genai
            genai.configure(api_key=self.config.gemini_api_key)
            self._client = genai
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
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

        if self.config.provider == LLMProvider.ANTHROPIC:
            return await self._anthropic_generate(client, system_prompt, user_prompt, temp, tokens)
        elif self.config.provider == LLMProvider.GEMINI:
            return await self._gemini_generate(client, system_prompt, user_prompt, temp, tokens)
        else:
            return await self._openai_generate(client, system_prompt, user_prompt, temp, tokens)

    async def _anthropic_generate(
        self, client: Any, system: str, user: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        import anthropic
        response = client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
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
        response = client.chat.completions.create(
            model=self.config.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
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
        model = client.GenerativeModel(
            model_name=self.config.model,
            system_instruction=system,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        response = model.generate_content(user)
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
