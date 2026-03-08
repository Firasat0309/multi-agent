"""Terminal/command execution tools for agents."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class TerminalTools:
    """Provides sandboxed command execution capabilities."""

    def __init__(self, working_dir: Path, timeout: int = 120) -> None:
        self.working_dir = working_dir
        self.timeout = timeout
        self._allowed_commands = {
            "python", "pytest", "pip", "ruff", "mypy", "bandit",
            "ls", "cat", "head", "tail", "grep", "find", "wc",
            "echo", "pwd", "env",
        }

    async def run_command(self, command: str) -> CommandResult:
        """Execute a command in the working directory."""
        # Basic command validation
        base_cmd = command.split()[0] if command.split() else ""
        if base_cmd not in self._allowed_commands:
            logger.warning(f"Blocked command: {base_cmd}")
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"Command not allowed: {base_cmd}",
            )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self.working_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONPATH": str(self.working_dir)},
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
                return CommandResult(
                    exit_code=proc.returncode or 0,
                    stdout=stdout.decode("utf-8", errors="replace"),
                    stderr=stderr.decode("utf-8", errors="replace"),
                )
            except asyncio.TimeoutError:
                proc.kill()
                return CommandResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"Command timed out after {self.timeout}s",
                    timed_out=True,
                )
        except Exception as e:
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
            )

    async def run_tests(self, test_path: str = "") -> CommandResult:
        """Run pytest on the workspace."""
        cmd = "pytest -v --tb=short"
        if test_path:
            cmd += f" {test_path}"
        return await self.run_command(cmd)

    async def run_linter(self, file_path: str = ".") -> CommandResult:
        """Run ruff linter."""
        return await self.run_command(f"ruff check {file_path}")

    async def run_type_check(self, file_path: str = ".") -> CommandResult:
        """Run mypy type checker."""
        return await self.run_command(f"mypy {file_path}")

    async def run_security_scan(self, file_path: str = ".") -> CommandResult:
        """Run bandit security scanner."""
        return await self.run_command(f"bandit -r {file_path} -f json")
