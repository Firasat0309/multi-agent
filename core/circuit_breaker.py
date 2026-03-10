"""Circuit breaker for LLM API calls — prevents cascade failures on sustained outages."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation — all calls pass through
    OPEN = "open"           # Failing — reject calls immediately
    HALF_OPEN = "half_open" # Testing recovery — one probe call allowed


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is OPEN."""


class CircuitBreaker:
    """Async circuit breaker protecting against sustained LLM API failures.

    State transitions:
        CLOSED → OPEN     : after ``failure_threshold`` consecutive failures
        OPEN → HALF_OPEN  : after ``recovery_timeout`` seconds of silence
        HALF_OPEN → CLOSED: on first successful probe call
        HALF_OPEN → OPEN  : on probe failure (resets the recovery timer)

    Usage::

        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
        result = await cb.call(my_async_fn, arg1, kwarg=val)
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        """Current circuit state (read-only; automatically transitions OPEN→HALF_OPEN)."""
        if (
            self._state == CircuitState.OPEN
            and time.monotonic() - self._last_failure_time >= self.recovery_timeout
        ):
            # Optimistically allow one probe without holding the lock —
            # the lock inside ``call()`` prevents concurrent half-open probes.
            self._state = CircuitState.HALF_OPEN
            logger.info(
                "Circuit OPEN → HALF_OPEN after %.0fs", self.recovery_timeout
            )
        return self._state

    async def call(
        self,
        fn: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Invoke *fn* through the circuit breaker.

        Raises:
            CircuitOpenError: when the circuit is OPEN and calls are blocked.
            Exception: any exception raised by *fn* (after recording the failure).
        """
        async with self._lock:
            current = self.state
            if current == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"LLM circuit breaker is OPEN — {self._failure_count} consecutive "
                    f"failures; will retry after {self.recovery_timeout:.0f}s"
                )

        # Execute the call outside the lock so other callers are not serialised.
        try:
            result = await fn(*args, **kwargs)
        except CircuitOpenError:
            # Don't count our own rejection as another failure
            raise
        except Exception:
            async with self._lock:
                self._record_failure()
            raise
        else:
            async with self._lock:
                self._record_success()
            return result

    # ── Internal state management ─────────────────────────────────────────────

    def _record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._state == CircuitState.HALF_OPEN:
            # Probe failed — stay open and reset the recovery timer
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit HALF_OPEN → OPEN (probe failed; %d total failures)",
                self._failure_count,
            )
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.error(
                "Circuit CLOSED → OPEN after %d consecutive failures",
                self._failure_count,
            )

    def _record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            logger.info("Circuit HALF_OPEN → CLOSED (probe succeeded)")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
