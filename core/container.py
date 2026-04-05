"""Lightweight dependency-injection container.

Provides a simple service-locator / composition-root that replaces the manual
wiring spread across ``pipeline_run.py``.  Each service is registered with a
factory callable that is invoked **once** on first resolve (singleton scope).

Usage::

    container = Container()
    container.register(EventBus, lambda c: EventBus())
    container.register(AgentManager, lambda c: AgentManager(
        settings=c.resolve(Settings), ...
    ))

    bus = container.resolve(EventBus)      # created on first call
    bus2 = container.resolve(EventBus)     # same instance returned

For tests, call :meth:`override` to replace a service with a mock/stub before
the code under test resolves it.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

__all__ = ["Container"]

T = TypeVar("T")


class Container:
    """Minimal IoC container with singleton semantics."""

    def __init__(self) -> None:
        self._factories: dict[type, Callable[[Container], Any]] = {}
        self._instances: dict[type, Any] = {}

    # ── registration ────────────────────────────────────────────────────

    def register(self, service_type: type[T], factory: Callable[[Container], T]) -> None:
        """Register a factory for *service_type*.

        The factory receives this container so it can resolve its own
        dependencies.  Overwrites any previous registration.
        """
        self._factories[service_type] = factory
        # Clear any cached instance so next resolve uses the new factory.
        self._instances.pop(service_type, None)

    def instance(self, service_type: type[T], value: T) -> None:
        """Register a pre-built instance directly (no factory needed)."""
        self._instances[service_type] = value

    # ── resolution ──────────────────────────────────────────────────────

    def resolve(self, service_type: type[T]) -> T:
        """Return the singleton instance of *service_type*, creating it if needed."""
        if service_type in self._instances:
            return self._instances[service_type]
        factory = self._factories.get(service_type)
        if factory is None:
            raise KeyError(
                f"{service_type.__name__} is not registered in the container"
            )
        inst = factory(self)
        self._instances[service_type] = inst
        return inst

    def has(self, service_type: type) -> bool:
        """Check if *service_type* is registered (factory or instance)."""
        return service_type in self._instances or service_type in self._factories

    # ── test helpers ────────────────────────────────────────────────────

    def override(self, service_type: type[T], value: T) -> None:
        """Replace a service with *value*.  Intended for tests."""
        self._instances[service_type] = value

    def reset(self) -> None:
        """Clear all registrations and cached instances."""
        self._factories.clear()
        self._instances.clear()
