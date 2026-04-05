"""Plugin discovery and loading for the hook system.

Discovers ``HookPlugin`` subclasses from:
  1. Python modules in a ``plugins/`` directory (file-based discovery).
  2. Installed packages advertising the ``multi_agent.plugins`` entry-point group.

Usage::

    from core.plugin_loader import discover_and_load_plugins
    from core.hooks import HookRegistry

    registry = HookRegistry()
    loaded = discover_and_load_plugins(registry, plugins_dir="plugins")
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.hooks import HookPlugin, HookRegistry

logger = logging.getLogger(__name__)

__all__ = [
    "discover_plugins_from_directory",
    "discover_plugins_from_entrypoint",
    "discover_and_load_plugins",
]


def discover_plugins_from_directory(plugins_dir: str | Path) -> list[HookPlugin]:
    """Import all ``HookPlugin`` subclasses from Python files in *plugins_dir*.

    Each ``.py`` file in the directory (non-recursive) is imported as a module.
    Any top-level class that inherits from ``HookPlugin`` is instantiated
    (zero-arg constructor) and returned.
    """
    from core.hooks import HookPlugin as _HP

    plugins_path = Path(plugins_dir)
    if not plugins_path.is_dir():
        logger.debug("Plugins directory does not exist: %s", plugins_path)
        return []

    found: list[HookPlugin] = []
    for py_file in sorted(plugins_path.glob("*.py")):
        if py_file.name.startswith("_"):
            continue  # skip __init__.py, __pycache__, etc.

        module_name = f"plugins.{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
        except Exception:
            logger.warning("Failed to import plugin module %s", py_file, exc_info=True)
            continue

        # Find all HookPlugin subclasses defined in this module
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, _HP)
                and attr is not _HP
                and attr.__module__ == mod.__name__
            ):
                try:
                    instance = attr()
                    found.append(instance)
                    logger.info("Discovered plugin: %s from %s", instance.name, py_file.name)
                except Exception:
                    logger.warning(
                        "Failed to instantiate plugin %s from %s",
                        attr_name, py_file.name, exc_info=True,
                    )

    return found


def discover_plugins_from_entrypoints(group: str = "multi_agent.plugins") -> list[HookPlugin]:
    """Load plugins registered via Python package entry points.

    Packages can advertise plugins in ``pyproject.toml``::

        [project.entry-points."multi_agent.plugins"]
        my_plugin = "my_package.plugin:MyPlugin"
    """
    from core.hooks import HookPlugin as _HP

    found: list[HookPlugin] = []
    try:
        if hasattr(importlib.metadata, "entry_points"):
            eps = importlib.metadata.entry_points()
            # Python 3.12+ returns a SelectableGroups; older returns dict
            if hasattr(eps, "select"):
                entries = eps.select(group=group)
            elif isinstance(eps, dict):
                entries = eps.get(group, [])
            else:
                entries = []
        else:
            entries = []
    except Exception:
        logger.debug("Entry-point discovery failed", exc_info=True)
        return found

    for ep in entries:
        try:
            cls = ep.load()
            if isinstance(cls, type) and issubclass(cls, _HP):
                instance = cls()
                found.append(instance)
                logger.info("Discovered entry-point plugin: %s (%s)", instance.name, ep.name)
            else:
                logger.warning("Entry point %s does not expose a HookPlugin subclass", ep.name)
        except Exception:
            logger.warning("Failed to load entry-point plugin %s", ep.name, exc_info=True)

    return found


def discover_and_load_plugins(
    registry: HookRegistry,
    *,
    plugins_dir: str | Path | None = None,
    enable_entrypoints: bool = True,
) -> list[str]:
    """Discover and load all plugins into the registry.

    Returns a list of loaded plugin names.
    """
    plugins: list[HookPlugin] = []

    if plugins_dir is not None:
        plugins.extend(discover_plugins_from_directory(plugins_dir))

    if enable_entrypoints:
        plugins.extend(discover_plugins_from_entrypoints())

    loaded_names: list[str] = []
    for plugin in plugins:
        try:
            registry.load_plugin(plugin)
            loaded_names.append(plugin.name)
        except Exception:
            logger.warning("Failed to load plugin: %s", plugin.name, exc_info=True)

    if loaded_names:
        logger.info("Loaded %d plugin(s): %s", len(loaded_names), loaded_names)

    return loaded_names
