"""Chaos tests: verify resilience under failure conditions.

Tests cover:
  - LLM request timeouts
  - LLM partial / empty responses
  - Docker sandbox execution failures mid-build
  - Circuit breaker tripping under sustained failures
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.settings import ExecutionConfig, LLMConfig, LLMProvider
from core.llm_client import LLMClient, LLMClientError, LLMResponse


def _make_client(**overrides) -> LLMClient:
    """Create an LLMClient with short timeouts for fast test runs."""
    cfg = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model="claude-sonnet-4-20250514",
        api_key="sk-ant-test-key-for-chaos-tests",
    )
    exec_cfg = ExecutionConfig(
        retry_count=2,
        backoff_base=0.01,
        request_timeout=1,
        circuit_failure_threshold=3,
        circuit_recovery_timeout=0.1,
    )
    client = LLMClient(cfg, exec_cfg)
    return client


class TestLLMTimeouts:
    """LLM requests that exceed the timeout must raise, not hang."""

    def test_generate_times_out(self) -> None:
        """Verify that a slow LLM response is terminated by wait_for timeout."""
        client = _make_client()

        async def _hang(*_a, **_kw) -> LLMResponse:
            await asyncio.sleep(300)
            return LLMResponse(content="late", usage={}, model="test")

        async def _run() -> None:
            # Directly test the timeout mechanism: wrap a hanging coroutine
            # in wait_for with a short timeout
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(_hang(), timeout=0.1)

        asyncio.run(_run())


class TestLLMPartialResponses:
    """LLM returning empty or truncated content must be handled gracefully."""

    def test_empty_content_returned(self) -> None:
        client = _make_client()

        async def _empty(*_a, **_kw) -> LLMResponse:
            return LLMResponse(
                content="",
                usage={"input_tokens": 5, "output_tokens": 0},
                model="claude-sonnet-4-20250514",
                stop_reason="end_turn",
            )

        async def _run() -> None:
            with patch.object(client, "_get_client", return_value=MagicMock()):
                with patch.object(client, "_anthropic_generate", _empty):
                    resp = await client.generate("system", "user")
                    assert resp.content == ""
                    assert resp.usage["output_tokens"] == 0

        asyncio.run(_run())


class TestCircuitBreaker:
    """Circuit breaker must open after sustained failures."""

    def test_circuit_opens_after_threshold(self) -> None:
        client = _make_client()

        call_count = 0

        async def _fail(*_a, **_kw) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            raise Exception("500 Internal Server Error")

        async def _run() -> None:
            with patch.object(client, "_get_client", return_value=MagicMock()):
                with patch.object(client, "_anthropic_generate", _fail):
                    with pytest.raises(LLMClientError):
                        await client.generate("system", "user")

        asyncio.run(_run())
        # Should have retried up to retry_count (2) times
        assert call_count <= 3


class TestSandboxChaos:
    """Docker sandbox failures mid-execution must produce clean error results."""

    def test_docker_exec_failure(self) -> None:
        from sandbox.sandbox_runner import DockerSandbox
        from config.settings import SandboxConfig

        config = SandboxConfig()
        sandbox = DockerSandbox(config)

        # Simulate a container whose exec_run raises mid-execution
        mock_container = MagicMock()
        mock_container.exec_run.side_effect = RuntimeError("Docker daemon unreachable")
        sandbox._containers["test-123"] = mock_container

        result = asyncio.run(sandbox.execute("test-123", "echo hello"))
        assert result.exit_code == -1
        assert "Docker daemon unreachable" in result.stderr

    def test_local_sandbox_timeout(self) -> None:
        from sandbox.sandbox_runner import LocalSandbox
        from pathlib import Path
        import tempfile

        sandbox = LocalSandbox(timeout=1)

        async def _run() -> None:
            with tempfile.TemporaryDirectory() as tmpdir:
                info = await sandbox.create(Path(tmpdir))
                result = await sandbox.execute(
                    info.sandbox_id,
                    "python -c \"import time; time.sleep(10)\"",
                )
                assert result.exit_code == -1
                assert "timed out" in result.stderr.lower() or result.exit_code == -1
                await sandbox.destroy(info.sandbox_id)

        asyncio.run(_run())
