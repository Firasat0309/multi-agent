"""Terminal/command execution tools for agents."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from core.language import LanguageProfile, PYTHON

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class TerminalTools:
    """Provides sandboxed command execution capabilities."""

    def __init__(
        self, working_dir: Path, timeout: int = 120,
        language: LanguageProfile | None = None,
    ) -> None:
        self.working_dir = working_dir
        self.timeout = timeout
        self._lang = language or PYTHON
        self._allowed_commands = set(self._lang.allowed_commands)

    async def run_command(self, command: str) -> CommandResult:
        """Execute a command in the working directory.

        Uses execvp-style execution (no shell) to prevent command injection.
        The command string is split via shlex so options and quoted args work,
        but shell metacharacters (;, &&, |, >, ` etc.) are never interpreted.
        """
        try:
            parts = shlex.split(command)
        except ValueError as e:
            return CommandResult(exit_code=1, stdout="", stderr=f"Invalid command: {e}")

        if not parts:
            return CommandResult(exit_code=1, stdout="", stderr="Empty command")

        base_cmd = parts[0]
        if base_cmd not in self._allowed_commands:
            logger.warning(f"Blocked command: {base_cmd}")
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"Command not allowed: {base_cmd}",
            )

        try:
            proc = await asyncio.create_subprocess_exec(
                *parts,
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
            return CommandResult(exit_code=-1, stdout="", stderr=str(e))

    async def run_tests(self, test_path: str = "") -> CommandResult:
        """Run tests using the language-appropriate test runner."""
        cmd = self._lang.test_command
        if test_path:
            cmd += f" {test_path}"
        return await self.run_command(cmd)

    async def run_linter(self, file_path: str = ".") -> CommandResult:
        """Run linter for the configured language."""
        if not self._lang.lint_command:
            return CommandResult(exit_code=0, stdout="No linter configured", stderr="")
        return await self.run_command(f"{self._lang.lint_command} {file_path}")

    async def run_type_check(self, file_path: str = ".") -> CommandResult:
        """Run type checker for the configured language."""
        if not self._lang.type_check_command:
            return CommandResult(exit_code=0, stdout="No type checker configured", stderr="")
        return await self.run_command(f"{self._lang.type_check_command} {file_path}")

    async def run_security_scan(self, file_path: str = ".") -> CommandResult:
        """Run security scanner for the configured language."""
        if not self._lang.security_scan_command:
            return CommandResult(exit_code=0, stdout="No security scanner configured", stderr="")
        return await self.run_command(f"{self._lang.security_scan_command}")

    async def run_build(self) -> CommandResult:
        """Run build command for the configured language."""
        if not self._lang.build_command:
            return CommandResult(exit_code=0, stdout="No build step required", stderr="")
        return await self.run_command(self._lang.build_command)
