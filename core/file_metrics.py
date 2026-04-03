"""Per-file metrics tracking for pipeline observability.

Tracks time, tokens, retries, and fix counts per source file so operators
can identify which files are expensive or flaky.

Usage::

    from core.file_metrics import FileMetrics, file_metrics

    fm = file_metrics()
    fm.start_file("src/models.py")
    fm.record_tokens("src/models.py", input_tokens=800, output_tokens=1200)
    fm.record_retry("src/models.py")
    fm.finish_file("src/models.py")

    report = fm.summary()
    # [{"file": "src/models.py", "elapsed_s": 12.3, "tokens": 2000, ...}]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _FileRecord:
    file_path: str
    start_time: float = 0.0
    end_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    fix_count: int = 0
    errors: int = 0

    @property
    def elapsed(self) -> float:
        if self.end_time > 0:
            return self.end_time - self.start_time
        if self.start_time > 0:
            return time.monotonic() - self.start_time
        return 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class FileMetrics:
    """Accumulates per-file metrics during a pipeline run."""

    def __init__(self) -> None:
        self._records: dict[str, _FileRecord] = {}

    def _get(self, file_path: str) -> _FileRecord:
        if file_path not in self._records:
            self._records[file_path] = _FileRecord(file_path=file_path)
        return self._records[file_path]

    def start_file(self, file_path: str) -> None:
        rec = self._get(file_path)
        rec.start_time = time.monotonic()

    def finish_file(self, file_path: str) -> None:
        rec = self._get(file_path)
        rec.end_time = time.monotonic()

    def record_tokens(self, file_path: str, input_tokens: int = 0, output_tokens: int = 0) -> None:
        rec = self._get(file_path)
        rec.input_tokens += input_tokens
        rec.output_tokens += output_tokens

    def record_retry(self, file_path: str) -> None:
        self._get(file_path).retries += 1

    def record_fix(self, file_path: str) -> None:
        self._get(file_path).fix_count += 1

    def record_error(self, file_path: str) -> None:
        self._get(file_path).errors += 1

    def get(self, file_path: str) -> _FileRecord | None:
        return self._records.get(file_path)

    def summary(self) -> list[dict[str, Any]]:
        """Return a list of per-file metric dicts, sorted by total tokens desc."""
        rows = []
        for rec in self._records.values():
            rows.append({
                "file": rec.file_path,
                "elapsed_s": round(rec.elapsed, 2),
                "input_tokens": rec.input_tokens,
                "output_tokens": rec.output_tokens,
                "total_tokens": rec.total_tokens,
                "retries": rec.retries,
                "fix_count": rec.fix_count,
                "errors": rec.errors,
            })
        rows.sort(key=lambda r: r["total_tokens"], reverse=True)
        return rows


# Module-level singleton — one per process, reset between pipeline runs.
_instance: FileMetrics | None = None


def file_metrics() -> FileMetrics:
    """Return the global FileMetrics singleton (lazily created)."""
    global _instance
    if _instance is None:
        _instance = FileMetrics()
    return _instance


def reset_file_metrics() -> None:
    """Reset the singleton.  Called between pipeline runs / in tests."""
    global _instance
    _instance = None
