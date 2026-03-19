"""LLM client abstraction supporting multiple providers."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from config.settings import LLMConfig, LLMProvider
from core.circuit_breaker import CircuitBreaker, CircuitOpenError

logger = logging.getLogger(__name__)

# Per-request timeout in seconds.  Applies to each individual API call
# (not the full retry loop).  Prevents the pipeline from hanging if a
# provider endpoint stalls.  The value is generous enough for large
# architecture responses (~16k tokens) while still failing in a
# human-reasonable timeframe.
_REQUEST_TIMEOUT = 180  # 3 minutes


def _blk(block: Any, key: str) -> Any:
    """Read a field from a normalized content block (plain dict or SDK object).

    Content blocks stored in message history can be either plain ``dict``
    objects (our normalized internal format) or provider SDK objects that
    expose the same fields as attributes.  This helper abstracts over both.
    """
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)

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


class LLMRetryableError(Exception):
    """Transient LLM error that should be retried (not a subclass of LLMClientError)."""
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
                    "❌ Missing API key for Anthropic\n\n"
                    "Set your API key with:\n"
                    "  export ANTHROPIC_API_KEY=\"sk-ant-your-key-here\"\n\n"
                    "Get your key at: https://console.anthropic.com"
                )
        elif provider == LLMProvider.OPENAI:
            if not self.config.openai_api_key or self.config.openai_api_key.strip() == "":
                raise LLMConfigError(
                    "❌ Missing API key for OpenAI\n\n"
                    "Set your API key with:\n"
                    "  export OPENAI_API_KEY=\"sk-proj-your-key-here\"\n\n"
                    "Get your key at: https://platform.openai.com/api-keys"
                )
        elif provider == LLMProvider.GEMINI:
            if not self.config.gemini_api_key or self.config.gemini_api_key.strip() == "":
                raise LLMConfigError(
                    "❌ Missing API key for Google Gemini\n\n"
                    "Set your API key with:\n"
                    "  export GEMINI_API_KEY=\"your-api-key-here\"\n\n"
                    "Get your key at: https://aistudio.google.com/app/apikey"
                )

        # Check model validity (skip for custom base URLs — in-house models won't match)
        if provider == LLMProvider.OPENAI and self.config.openai_base_url:
            return
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
                self._client = anthropic.Anthropic(
                    api_key=self.config.api_key,
                    timeout=_REQUEST_TIMEOUT,
                )
            elif self.config.provider == LLMProvider.OPENAI:
                import openai
                kwargs: dict[str, Any] = {
                    "api_key": self.config.openai_api_key,
                    "timeout": _REQUEST_TIMEOUT,
                }
                if self.config.openai_base_url:
                    kwargs["base_url"] = self.config.openai_base_url
                self._client = openai.OpenAI(**kwargs)
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
                logger.info(
                    "LLM request (attempt %d/%d, provider=%s, model=%s) …",
                    attempt + 1, _max_attempts,
                    self.config.provider.value, self.config.model,
                )
                response = await asyncio.wait_for(
                    self._circuit.call(_generate_once),
                    timeout=_REQUEST_TIMEOUT + 30,  # generous over SDK timeout
                )
                # Accumulate lifetime token counts for cost reporting
                self.total_input_tokens += response.usage.get("input_tokens", 0)
                self.total_output_tokens += response.usage.get("output_tokens", 0)
                return response
            except asyncio.TimeoutError:
                logger.error(
                    "LLM request timed out after %ds (attempt %d/%d)",
                    _REQUEST_TIMEOUT + 30, attempt + 1, _max_attempts,
                )
                if attempt < _max_attempts - 1:
                    delay = _base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                raise LLMClientError(
                    f"LLM request timed out after {_REQUEST_TIMEOUT + 30}s "
                    f"({_max_attempts} attempts). The {self.config.provider.value} API "
                    f"may be overloaded — try again later or use --no-interactive for "
                    f"more verbose logging."
                )
            except CircuitOpenError as e:
                if attempt < _max_attempts - 1:
                    wait = self._circuit.recovery_timeout + 1
                    logger.warning(
                        "Circuit breaker OPEN on attempt %d/%d — waiting %.0fs for recovery",
                        attempt + 1, _max_attempts, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise LLMClientError(str(e)) from e
            except LLMRetryableError as e:
                # Explicitly retryable (e.g. Gemini blocked/invalid function call)
                if attempt < _max_attempts - 1:
                    delay = _base_delay * (2 ** attempt)
                    logger.warning(
                        "LLM retryable error on attempt %d/%d — retrying in %.1fs: %s",
                        attempt + 1, _max_attempts, delay, str(e)[:120],
                    )
                    await asyncio.sleep(delay)
                    continue
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
                        "❌ Anthropic authentication failed\n\n"
                        "Your API key is missing or invalid.\n"
                        "Set: export ANTHROPIC_API_KEY=\"sk-ant-your-key\"\n"
                        "Get key at: https://console.anthropic.com"
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
            return model.generate_content(
                user,
                request_options={"timeout": _REQUEST_TIMEOUT},
            )

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
                    "❌ Gemini authentication failed\n\n"
                    "Your API key is missing or invalid.\n"
                    "Set: export GEMINI_API_KEY=\"your-api-key\"\n"
                    "Get key at: https://aistudio.google.com/app/apikey"
                ) from e
            elif any(code in str(e) for code in ("500", "502", "503", "529")) or "overloaded" in error_str or "resource_exhausted" in error_str or "429" in str(e):
                # Transient server / rate-limit errors — re-raise the original
                # exception so the caller's retry loop can handle it.
                raise
            elif "google" in error_type.lower() or "grpc" in error_type.lower():
                raise LLMConfigError(
                    f"❌ Google Gemini API error: {str(e)}\n\n"
                    f"Please check:\n"
                    f"  • Your GEMINI_API_KEY is valid\n"
                    f"  • The model name is correct\n"
                    f"  • Your account has access to this model"
                ) from e
            raise
        # response.text raises ValueError if the content was blocked by safety
        # filters or if no candidates were returned.  Handle gracefully.
        try:
            text = response.text
        except ValueError as ve:
            # Blocked response or invalid function call (finish_reason=10).
            # Raise as retryable so the retry loop gets a chance to re-attempt
            # before surfacing the error.
            block_reason = ""
            if hasattr(response, "prompt_feedback"):
                block_reason = str(getattr(response.prompt_feedback, "block_reason", ""))
            raise LLMRetryableError(
                f"Gemini response blocked: {ve}. "
                f"Block reason: {block_reason or 'unknown'}. "
                f"Try rephrasing the prompt or using a different model."
            ) from ve
        return LLMResponse(
            content=text,
            usage={
                "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) or 0,
                "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0) or 0,
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

    @staticmethod
    def _repair_json_text(text: str) -> str:
        """Best-effort repair of common LLM JSON mistakes.

        Handles trailing commas, single-quoted strings, JS comments,
        unquoted property names, and **truncated output** (unclosed
        braces, brackets, and strings).
        """
        # Strip JS-style comments
        text = re.sub(r"//[^\n]*", "", text)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        # Replace single-quoted strings with double-quoted
        text = re.sub(r"(?<=[:,\[\{])\s*'([^']*)'", r' "\1"', text)
        text = re.sub(r"'([^']*)'(?=\s*[:,\]\}])", r'"\1"', text)
        # Remove trailing commas before } or ]
        text = re.sub(r",\s*([}\]])", r"\1", text)
        # Quote unquoted property names
        text = re.sub(r'(?<=[{,])\s*([a-zA-Z_]\w*)\s*:', r' "\1":', text)

        # ── Truncation repair ─────────────────────────────────────────────
        # If Gemini (or any LLM) hits max tokens, the JSON is cut off mid-
        # stream.  Close any unclosed strings, strip trailing partial
        # key-value pairs, and balance braces/brackets so json.loads works.
        # This gives us a *partial* but valid result rather than nothing.

        # Close unclosed string literal
        in_string = False
        escaped = False
        for ch in text:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
        if in_string:
            text += '"'

        # Iteratively strip trailing incomplete entries and balance brackets.
        # Each pass peels off one layer of partial data from the end.
        for _ in range(20):
            stripped = text.rstrip()
            # Strip trailing partial key-value: ,"key": "val  or  ,"key":
            stripped = re.sub(r',\s*"[^"]*"\s*:\s*"[^"]*"\s*$', "", stripped)
            # Strip key with colon but no value (with or without comma):
            #   { "key":   or   , "key":
            stripped = re.sub(r'[,{]\s*"[^"]*"\s*:\s*$', lambda m: "{" if m.group(0).strip().startswith("{") else "", stripped)
            # Strip trailing dangling value: , "something"
            stripped = re.sub(r',\s*"[^"]*"\s*$', "", stripped)
            # Remove dangling comma
            stripped = re.sub(r",\s*$", "", stripped)

            if stripped == text.rstrip():
                break
            text = stripped

        # Balance braces and brackets — close innermost first.
        # Walk the text to track nesting order so we close in the right sequence.
        stack: list[str] = []
        in_str = False
        esc = False
        for ch in text:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack:
                stack.pop()
        # Close in reverse nesting order
        text += "".join(reversed(stack))

        return text

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

        # Attempt 1: strict parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Attempt 2: extract JSON object from surrounding prose
        logger.warning("LLM returned non-JSON; extracting first JSON object from response")
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            snippet = text[start:end]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                # Attempt 3: repair common JSON mistakes then re-parse
                try:
                    repaired = self._repair_json_text(snippet)
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass

        # Attempt 4: repair the full text as a last resort
        try:
            repaired = self._repair_json_text(text)
            r_start = repaired.find("{")
            r_end = repaired.rfind("}") + 1
            if r_start != -1 and r_end > r_start:
                return json.loads(repaired[r_start:r_end])
        except json.JSONDecodeError:
            pass

        # Attempt 5: ask the LLM to complete/fix its own truncated JSON
        logger.warning("All local JSON repair attempts failed — asking LLM to fix its output")
        try:
            fix_response = await self.generate(
                system_prompt="You are a JSON repair tool. Return ONLY valid JSON, no explanation.",
                user_prompt=(
                    "The following JSON is truncated or malformed. "
                    "Complete it and return ONLY the valid JSON object:\n\n"
                    f"{text[:3000]}"
                ),
                temperature=0.0,
            )
            fix_text = fix_response.content.strip()
            if fix_text.startswith("```"):
                fix_lines = fix_text.split("\n")
                fix_lines = fix_lines[1:]
                if fix_lines and fix_lines[-1].strip() == "```":
                    fix_lines = fix_lines[:-1]
                fix_text = "\n".join(fix_lines)
            fix_start = fix_text.find("{")
            fix_end = fix_text.rfind("}") + 1
            if fix_start != -1 and fix_end > fix_start:
                result = json.loads(fix_text[fix_start:fix_end])
                logger.info("LLM successfully repaired truncated JSON")
                return result
        except Exception as repair_exc:
            logger.debug("LLM JSON repair attempt failed: %s", repair_exc)

        logger.error("Could not parse JSON from LLM response: %s", text[:200])
        return {}

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[ToolDefinition],
        system_prompt: str = "",
        max_tokens: int = 16384,
    ) -> LLMResponseWithTools:
        """Multi-turn tool-use call — supported for all three providers.

        Messages use a normalized internal format (plain dicts matching the
        Anthropic tool-use schema) so the same agentic loop in ``BaseAgent``
        works regardless of which LLM is configured:

        * **Anthropic** — native tool_use blocks via the Messages API.
        * **OpenAI** — native function-calling via the Chat Completions API.
          Messages are translated to/from the OpenAI format internally.
        * **Gemini** — structured-prompt simulation: tools are described in
          the system prompt; the model signals a call via a JSON envelope
          ``{"tool_call": {"name": ..., "input": {...}}}``.

        ``raw_content`` in the returned ``LLMResponseWithTools`` is always a
        list of plain normalized dicts so it can be safely re-fed on the next
        iteration without provider-specific objects in the history.
        """
        if self.config.provider == LLMProvider.ANTHROPIC:
            return await self._anthropic_generate_with_tools(
                messages, tools, system_prompt, max_tokens,
            )
        if self.config.provider == LLMProvider.OPENAI:
            return await self._openai_generate_with_tools(
                messages, tools, system_prompt, max_tokens,
            )
        # Gemini — structured-prompt simulation
        return await self._gemini_generate_with_tools(
            messages, tools, system_prompt, max_tokens,
        )

    # ── Per-provider tool-use implementations ────────────────────────────────

    async def _anthropic_generate_with_tools(
        self,
        messages: list[dict],
        tools: list[ToolDefinition],
        system_prompt: str,
        max_tokens: int,
    ) -> LLMResponseWithTools:
        client = self._get_client()
        api_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
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
            response = await self._circuit.call(lambda: asyncio.to_thread(_sync))
        except CircuitOpenError as exc:
            raise LLMClientError(str(exc)) from exc

        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens
        self._update_metrics(response.usage)

        # Normalize SDK objects → plain dicts so history is always serialisable
        # and can be re-fed to any provider without Anthropic SDK objects inside.
        normalized: list[dict] = []
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
                normalized.append({"type": "text", "text": block.text})
            elif btype == "tool_use":
                tool_calls.append(ToolCall(
                    tool_use_id=block.id, name=block.name, input=block.input,
                ))
                normalized.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        return LLMResponseWithTools(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            raw_content=normalized,
        )

    async def _openai_generate_with_tools(
        self,
        messages: list[dict],
        tools: list[ToolDefinition],
        system_prompt: str,
        max_tokens: int,
    ) -> LLMResponseWithTools:
        client = self._get_client()
        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]
        oai_messages = self._translate_messages_to_openai(messages, system_prompt)

        def _sync() -> Any:
            return client.chat.completions.create(
                model=self.config.model,
                max_tokens=max_tokens,
                tools=oai_tools,
                tool_choice="auto",
                messages=oai_messages,
            )

        try:
            response = await self._circuit.call(lambda: asyncio.to_thread(_sync))
        except CircuitOpenError as exc:
            raise LLMClientError(str(exc)) from exc

        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens

        choice = response.choices[0]
        msg = choice.message
        finish = choice.finish_reason  # "tool_calls" | "stop"

        # Build normalized raw_content and ToolCall list
        normalized: list[dict] = []
        tool_calls: list[ToolCall] = []
        text = msg.content or ""
        if text:
            normalized.append({"type": "text", "text": text})
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    inp = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    inp = {}
                tool_calls.append(ToolCall(
                    tool_use_id=tc.id, name=tc.function.name, input=inp,
                ))
                normalized.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": inp,
                })

        stop_reason = "tool_use" if finish == "tool_calls" else "end_turn"
        return LLMResponseWithTools(
            content=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
            raw_content=normalized,
        )

    async def _gemini_generate_with_tools(
        self,
        messages: list[dict],
        tools: list[ToolDefinition],
        system_prompt: str,
        max_tokens: int,
    ) -> LLMResponseWithTools:
        """Simulate tool use for Gemini via structured-prompt conventions.

        The tools are described as a JSON schema block appended to the system
        prompt.  The model is instructed to signal a tool call by responding
        with ONLY ``{"tool_call": {"name": "...", "input": {...}}}``.  Any
        other response is treated as a final text answer (``end_turn``).
        """
        tool_schema = json.dumps(
            [{"name": t.name, "description": t.description, "parameters": t.input_schema}
             for t in tools],
            indent=2,
        )
        tool_instruction = (
            "\n\nYou have access to these tools:\n"
            f"```json\n{tool_schema}\n```\n\n"
            "To call a tool, respond with ONLY this JSON (no other text):\n"
            '{"tool_call": {"name": "<tool_name>", "input": {<arguments>}}}\n'
            "Otherwise respond normally."
        )
        full_system = system_prompt + tool_instruction

        # Flatten normalized messages into a plain conversation string for Gemini
        conversation = self._flatten_messages_for_gemini(messages)

        _max_attempts = 4
        _base_delay = 2.0
        last_exc: Exception | None = None
        response: LLMResponse | None = None
        for _attempt in range(_max_attempts):
            try:
                response = await self._gemini_generate(
                    self._get_client(), full_system, conversation,
                    self.config.temperature, max_tokens,
                )
                last_exc = None
                break
            except LLMRetryableError as _exc:
                # Explicitly retryable (e.g. Gemini blocked/invalid function call)
                last_exc = _exc
                if _attempt < _max_attempts - 1:
                    _delay = _base_delay * (2 ** _attempt)
                    logger.warning(
                        "Gemini retryable error (attempt %d/%d) — "
                        "retrying in %.1fs: %s",
                        _attempt + 1, _max_attempts, _delay, str(_exc)[:120],
                    )
                    await asyncio.sleep(_delay)
                    continue
            except LLMClientError:
                # Config / auth errors — surface immediately, no retry.
                raise
            except Exception as _exc:
                last_exc = _exc
                _msg = str(_exc)
                _msg_lower = _msg.lower()
                _is_transient = (
                    any(code in _msg for code in ("500", "502", "503", "529"))
                    or "overloaded" in _msg_lower
                    or "resource_exhausted" in _msg_lower
                    or "rate_limit" in _msg_lower
                    or "ratelimit" in _msg_lower
                    or "429" in _msg
                    or "too many requests" in _msg_lower
                )
                if _is_transient and _attempt < _max_attempts - 1:
                    _delay = _base_delay * (2 ** _attempt)
                    logger.warning(
                        "Gemini tool-use transient error (attempt %d/%d) — "
                        "retrying in %.1fs: %s",
                        _attempt + 1, _max_attempts, _delay, _msg[:120],
                    )
                    await asyncio.sleep(_delay)
                    continue
                raise
        if last_exc is not None or response is None:
            raise LLMClientError(
                "Gemini generate_with_tools: retry loop exhausted"
            ) from last_exc
        text = response.content.strip()

        # Try to detect a tool-call JSON envelope
        try:
            start = text.find("{")
            if start != -1:
                data = json.loads(text[start:])
                if "tool_call" in data:
                    tc_data = data["tool_call"]
                    tc = ToolCall(
                        tool_use_id=f"gemini-{tc_data['name']}-0",
                        name=tc_data["name"],
                        input=tc_data.get("input", {}),
                    )
                    normalized = [{"type": "tool_use", "id": tc.tool_use_id,
                                   "name": tc.name, "input": tc.input}]
                    return LLMResponseWithTools(
                        content="",
                        tool_calls=[tc],
                        stop_reason="tool_use",
                        usage=response.usage,
                        raw_content=normalized,
                    )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Plain text response — end_turn
        return LLMResponseWithTools(
            content=text,
            tool_calls=[],
            stop_reason="end_turn",
            usage=response.usage,
            raw_content=[{"type": "text", "text": text}],
        )

    # ── Message format translation helpers ───────────────────────────────────

    def _translate_messages_to_openai(
        self,
        messages: list[dict],
        system_prompt: str = "",
    ) -> list[dict]:
        """Convert normalized internal messages to OpenAI Chat Completions format.

        Internal normalized format uses Anthropic-style content blocks
        (plain dicts with ``type`` keys).  Tool results live in
        ``{"role": "user", "content": [{"type": "tool_result", ...}]}`` turns.
        """
        result: list[dict] = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                result.append({"role": role, "content": content})
                continue

            if not isinstance(content, list):
                result.append({"role": role, "content": str(content)})
                continue

            tool_uses = [b for b in content if _blk(b, "type") == "tool_use"]
            tool_results = [b for b in content if _blk(b, "type") == "tool_result"]
            text_blocks = [b for b in content if _blk(b, "type") == "text"]

            if role == "assistant" and tool_uses:
                text = "\n".join(_blk(b, "text") for b in text_blocks) or None
                oai_tool_calls = [
                    {
                        "id": _blk(b, "id"),
                        "type": "function",
                        "function": {
                            "name": _blk(b, "name"),
                            "arguments": json.dumps(_blk(b, "input") or {}),
                        },
                    }
                    for b in tool_uses
                ]
                result.append({
                    "role": "assistant",
                    "content": text,
                    "tool_calls": oai_tool_calls,
                })
            elif role == "user" and tool_results:
                # Each tool result becomes a separate "tool" role message
                for b in tool_results:
                    raw = _blk(b, "content") or ""
                    if isinstance(raw, list):
                        raw = "\n".join(
                            item.get("text", "") if isinstance(item, dict) else str(item)
                            for item in raw
                        )
                    result.append({
                        "role": "tool",
                        "tool_call_id": _blk(b, "tool_use_id"),
                        "content": str(raw),
                    })
            else:
                # Plain user text or assistant text without tool calls
                text = "\n".join(_blk(b, "text") for b in text_blocks)
                result.append({"role": role, "content": text})

        return result

    def _flatten_messages_for_gemini(self, messages: list[dict]) -> str:
        """Flatten normalized messages into a single prompt string for Gemini.

        Gemini's multi-turn tool-use API requires a stateful chat session and
        is not compatible with the stateless message-list pattern used here.
        We flatten the conversation to a string so that the existing agentic
        loop works without a redesign; the model sees the full context each
        turn.
        """
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, str):
                parts.append(f"{role.upper()}: {content}")
            elif isinstance(content, list):
                for block in content:
                    btype = _blk(block, "type")
                    if btype == "text":
                        parts.append(f"{role.upper()}: {_blk(block, 'text')}")
                    elif btype == "tool_use":
                        parts.append(
                            f"TOOL_CALL: {_blk(block, 'name')} "
                            f"args={json.dumps(_blk(block, 'input') or {})}"
                        )
                    elif btype == "tool_result":
                        raw = _blk(block, "content") or ""
                        if isinstance(raw, list):
                            raw = "\n".join(
                                item.get("text", "") if isinstance(item, dict) else str(item)
                                for item in raw
                            )
                        parts.append(f"TOOL_RESULT: {raw}")
        return "\n".join(parts)

    def _update_metrics(self, usage: Any) -> None:
        """No-op hook for subclasses / tests to observe token usage."""
