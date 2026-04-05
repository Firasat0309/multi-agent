"""Structured JSON logging configuration.

Provides a JSON formatter for production/CI environments and a helper to
configure logging with either JSON or Rich (interactive) output.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any

__all__ = ["JSONFormatter", "configure_logging"]


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects.

    Fields emitted:
        ts, level, logger, message, task_id, file_path, agent
    Plus ``exc_info`` / ``stack_info`` when present.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Propagate structured context attached by callers via `extra=`.
        for key in ("task_id", "file_path", "agent", "phase", "tier", "duration_s"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val

        if record.exc_info and record.exc_info[1] is not None:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            payload["exc_message"] = str(record.exc_info[1])
            payload["exc_traceback"] = traceback.format_exception(*record.exc_info)

        if record.stack_info:
            payload["stack_info"] = record.stack_info

        return json.dumps(payload, default=str)


def configure_logging(
    *,
    verbose: bool = False,
    force_json: bool = False,
) -> None:
    """Set up the root logger.

    Behaviour:
    * If ``LOG_FORMAT=json`` env-var is set **or** ``force_json`` is True,
      output newline-delimited JSON to stderr.
    * Otherwise fall back to ``RichHandler`` for interactive / local usage.

    Both paths apply the :class:`SecretRedactionFilter`.
    """
    from core.log_redact import SecretRedactionFilter

    level = logging.DEBUG if verbose else logging.INFO
    use_json = force_json or os.environ.get("LOG_FORMAT", "").lower() == "json"

    if use_json:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JSONFormatter())
    else:
        # Lazy import so Rich is only required for interactive sessions.
        from rich.console import Console
        from rich.logging import RichHandler

        handler = RichHandler(console=Console(), rich_tracebacks=True)

    handler.addFilter(SecretRedactionFilter())

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )
