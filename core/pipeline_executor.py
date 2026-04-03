"""Backward-compatible import for the simple loop executor.

The legacy tier/checkpoint executor implementation has been removed.
``PipelineExecutor`` remains as the historical public symbol so existing
imports continue to work, but it now delegates entirely to the single
simple generate -> build -> fix executor.
"""

from __future__ import annotations

from core.simple_loop_executor import SimpleLoopExecutor


class PipelineExecutor(SimpleLoopExecutor):
    """Compatibility wrapper for the single supported executor."""


__all__ = ["PipelineExecutor"]
