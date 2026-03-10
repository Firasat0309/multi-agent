"""LLM client abstraction supporting multiple providers."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from config.settings import LLMConfig, LLMProvider
from core.circuit_breaker import CircuitBreaker, CircuitOpenError

logger = logging.getLogger(__name__)

# Per-million-token pricing in USD (update as providers publish new rates).
# Keys match model IDs from VALID_MODELS; prefix-matching is used for aliases.
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-sonnet-4-20250514": {"input": 3.00,  "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00,  "output": 15.00},
    "claude-3-5-haiku-20241022":  {"input": 0.80,  "output": 4.00},
    "claude-3-opus-20240229":     {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229":   {"input": 3.00,  "output": 15.00},
    # OpenAI
    "gpt-4o":               {"input": 2.50,  "output": 10.00},
    "gpt-4-turbo":          {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-preview":  {"input": 10.00, "output": 30.00},
    "gpt-4":                {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo":        {"input": 0.50,  "output": 1.50},
    # Google Gemini
    "gemini-2.0-flash":     {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro":       {"input": 3.50,  "output": 10.50},
    "gemini-1.5-flash":     {"input": 0.075, "output": 0.30},
    "gemini-pro":           {"input": 0.50,  "output": 1.50},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a given model and token counts.

    Uses prefix matching so ``claude-sonnet-4-20250514`` and any future
    variants still resolve to a price.  Returns 0.0 for unknown models.
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        # Prefix match — e.g. future "claude-sonnet-4-20260101" hits "claude-sonnet-4-"
        for key, val in MODEL_PRICING.items():
            if model.startswith(key.rsplit("-", 1)[0]):
                pricing = val
                break
    if pricing is None:
        return 0.0
    return (
        input_tokens  * pricing["input"]  / 1_000_000
        + output_tokens * pricing["output"] / 1_000_000
    )


@dataclass
class LLMResponse:
    content: str
    usage: dict[str, int]
    model: str
    stop_reason: str = ""


# ── Tool-use dataclasses ──────────────────────────────────────────────────────

@dataclass
class ToolDefinition:
    """Describes a tool the LLM may call (JSON Schema input)."""
    name: str
    description: str
    input_schema: dict  # JSON Schema object


@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM."""
    tool_use_id: str
    name: str
    input: dict


@dataclass
class LLMResponseWithTools:
    """Response from a tool-use enabled generate call."""
    content: str                  # concatenated text blocks (may be empty)
    tool_calls: list[ToolCall]    # tool_use blocks from the response
    stop_reason: str              # "tool_use" | "end_turn"
    usage: dict
    raw_content: list             # raw content block list for multi-turn messages


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
        self._circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
        # Cumulative token counters for cost tracking across the client's lifetime
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
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
        """Generate a completion from the LLM with exponential backoff retry.

        Retries up to 3 extra times (4 total) on transient errors such as
        rate-limit (429) and server-side overload (500/503).  Non-retryable
        errors (auth, bad model) surface immediately.
        """
        client = self._get_client()
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        _max_attempts = 4
        _base_delay = 2.0  # seconds; doubles each attempt (2 → 4 → 8 → …)

        async def _generate_once() -> LLMResponse:
            if self.config.provider == LLMProvider.ANTHROPIC:
                return await self._anthropic_generate(
                    client, system_prompt, user_prompt, temp, tokens
                )
            elif self.config.provider == LLMProvider.GEMINI:
                return await self._gemini_generate(
                    client, system_prompt, user_prompt, temp, tokens
                )
            else:
                return await self._openai_generate(
                    client, system_prompt, user_prompt, temp, tokens
                )

        for attempt in range(_max_attempts):
            try:
                response = await self._circuit.call(_generate_once)
                # Accumulate lifetime token counts for cost reporting
                self.total_input_tokens += response.usage.get("input_tokens", 0)
                self.total_output_tokens += response.usage.get("output_tokens", 0)
                return response
            except CircuitOpenError as e:
                raise LLMClientError(str(e)) from e
            except LLMClientError:
                # Config / auth errors are never retried — surface immediately.
                raise
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__

                # Classify as retryable (rate-limit / transient server error)
                is_rate_limit = (
                    "429" in error_msg
                    or "rate_limit" in error_msg.lower()
                    or "ratelimit" in error_msg.lower()
                    or "too many requests" in error_msg.lower()
                )
                is_transient = is_rate_limit or any(
                    token in error_msg for token in ("500", "502", "503", "overloaded")
                )

                if is_transient and attempt < _max_attempts - 1:
                    delay = _base_delay * (2 ** attempt)
                    logger.warning(
                        "LLM %s on attempt %d/%d — retrying in %.1fs: %s",
                        "rate-limit" if is_rate_limit else "transient error",
                        attempt + 1, _max_attempts,
                        delay, error_msg[:120],
                    )
                    await asyncio.sleep(delay)
                    continue

                # Permanent or exhausted — translate to domain errors
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

                raise LLMClientError(
                    f"Failed to generate with {self.config.provider.value}: {e}"
                ) from e

        # Unreachable — the loop always returns or raises inside.
        raise LLMClientError("generate: retry loop exhausted without result")

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

    async def generate_streaming(
        self,
        system_prompt: str,
        user_prompt: str,
        on_chunk: Any | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Stream response chunks, calling *on_chunk(text)* for each delta.

        Falls back to a regular ``generate()`` call for providers that do not
        support streaming (Gemini, OpenAI in the current driver set).  Anthropic
        uses the native streaming API.

        Args:
            on_chunk: Optional callable receiving each text delta as a string.
                      Called synchronously from the stream-reading thread.
        """
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        client = self._get_client()

        if self.config.provider == LLMProvider.ANTHROPIC:
            import anthropic

            collected: list[str] = []
            usage: dict[str, int] = {}

            def _stream_sync() -> None:
                with client.messages.stream(
                    model=self.config.model,
                    max_tokens=tokens,
                    temperature=self.config.temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                ) as stream:
                    for text in stream.text_stream:
                        collected.append(text)
                        if on_chunk is not None:
                            on_chunk(text)
                    final = stream.get_final_message()
                    usage["input_tokens"] = final.usage.input_tokens
                    usage["output_tokens"] = final.usage.output_tokens

            await asyncio.to_thread(_stream_sync)
            content = "".join(collected)
            self.total_input_tokens += usage.get("input_tokens", 0)
            self.total_output_tokens += usage.get("output_tokens", 0)
            return LLMResponse(
                content=content,
                usage=usage,
                model=self.config.model,
                stop_reason="end_turn",
            )

        # Non-Anthropic providers: fall back to non-streaming generate
        return await self.generate(system_prompt, user_prompt, max_tokens=max_tokens)

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
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON; extracting first JSON object from response")
            # Try to extract a JSON object from anywhere in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            logger.error(f"Could not parse JSON from LLM response: {text[:200]}")
            return {}

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[ToolDefinition],
        system_prompt: str = "",
        max_tokens: int = 8192,
    ) -> LLMResponseWithTools:
        """Multi-turn tool-use call following the Claude tool_use pattern.

        Only supported for the Anthropic provider.  Raises ``LLMClientError``
        for other providers.

        Args:
            messages: Conversation history in Claude message format.
            tools:    Tool definitions available to the model.
            system_prompt: Optional system prompt.
            max_tokens: Maximum tokens for the response.

        Returns:
            ``LLMResponseWithTools`` with ``stop_reason`` of either
            ``"tool_use"`` (model wants to call tools) or ``"end_turn"``
            (model is finished).
        """
        if self.config.provider != LLMProvider.ANTHROPIC:
            raise LLMClientError(
                f"Tool-use is only supported with Anthropic provider "
                f"(current: {self.config.provider.value})"
            )

        client = self._get_client()
        api_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

        def _sync() -> Any:
            kwargs: dict[str, Any] = dict(
                model=self.config.model,
                max_tokens=max_tokens,
                tools=api_tools,
                messages=messages,
            )
            if system_prompt:
                kwargs["system"] = system_prompt
            return client.messages.create(**kwargs)

        try:
            async def _call_async() -> Any:
                return await asyncio.to_thread(_sync)

            response = await self._circuit.call(_call_async)
        except CircuitOpenError as exc:
            raise LLMClientError(str(exc)) from exc

        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens
        self._update_metrics(response.usage)

        # Collect text blocks and tool_use blocks from the response
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        tool_use_id=block.id,
                        name=block.name,
                        input=block.input,
                    )
                )

        return LLMResponseWithTools(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            raw_content=response.content,
        )

    def _update_metrics(self, usage: Any) -> None:
        """No-op hook for subclasses / tests to observe token usage."""
