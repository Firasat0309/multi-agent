"""Runtime smoke test — verify the generated application actually starts.

After all files pass compilation and the global build succeeds, this module
starts the application and checks that:
  1. The process starts without crashing within a timeout
  2. A health endpoint (if configured) returns HTTP 200
  3. The process can be stopped cleanly

This catches runtime issues that pass the compiler but fail at startup:
  - Missing bean definitions (Spring)
  - Circular dependency injection
  - Invalid configuration values
  - Port conflicts or missing environment variables

Gated behind the ``RUNTIME_SMOKE_TEST`` feature flag.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Common health endpoint paths by framework
_HEALTH_ENDPOINTS = [
    "/actuator/health",    # Spring Boot
    "/health",             # Generic / Express / FastAPI
    "/healthz",            # Kubernetes convention
    "/api/health",         # Common REST pattern
    "/_health",            # Some frameworks
]

# Common startup commands by tech stack
_START_COMMANDS: dict[str, list[str]] = {
    "spring": ["mvn", "spring-boot:run", "-q"],
    "spring-gradle": ["./gradlew", "bootRun", "-q"],
    "express": ["node", "src/index.js"],
    "fastapi": ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"],
    "flask": ["python", "-m", "flask", "run", "--host", "0.0.0.0", "--port", "8080"],
    "django": ["python", "manage.py", "runserver", "0.0.0.0:8080"],
    "go": ["go", "run", "."],
    "dotnet": ["dotnet", "run"],
}

# Common startup log patterns that indicate the app is ready
_READY_PATTERNS = [
    "Started .* in .* seconds",     # Spring Boot
    "Tomcat started on port",       # Spring Boot / Tomcat
    "Application startup complete", # FastAPI / Uvicorn
    "Listening on port",            # Express / Node
    "Running on http",              # Flask
    "Starting development server",  # Django
    "server started",               # Generic
    "ready to accept connections",  # Generic
]


@dataclass
class SmokeTestResult:
    """Result of a runtime smoke test."""
    success: bool
    startup_ok: bool = False
    health_check_ok: bool = False
    startup_time_seconds: float = 0.0
    health_endpoint: str = ""
    errors: list[str] = field(default_factory=list)
    stdout_tail: str = ""   # Last N lines of stdout for debugging

    def summary(self) -> str:
        if self.success:
            return (
                f"Smoke test PASSED: startup={self.startup_time_seconds:.1f}s, "
                f"health={self.health_endpoint} OK"
            )
        issues = "; ".join(self.errors) if self.errors else "unknown failure"
        return f"Smoke test FAILED: {issues}"


class RuntimeSmokeTest:
    """Runs a quick startup + health check on the generated application."""

    def __init__(
        self,
        workspace: Path,
        tech_stack: dict[str, str],
        *,
        startup_timeout: float = 30.0,
        health_timeout: float = 10.0,
        port: int = 8080,
    ) -> None:
        self._workspace = workspace
        self._tech_stack = tech_stack
        self._startup_timeout = startup_timeout
        self._health_timeout = health_timeout
        self._port = port

    async def run(self) -> SmokeTestResult:
        """Run the smoke test: start app → check health → stop."""
        result = SmokeTestResult(success=False)

        # Determine start command
        start_cmd = self._detect_start_command()
        if not start_cmd:
            result.errors.append(
                "Cannot determine start command for tech stack: "
                + str(self._tech_stack)
            )
            return result

        process = None
        try:
            # Start the application
            logger.info(
                "Smoke test: starting app with %s (timeout=%ds)",
                " ".join(start_cmd), self._startup_timeout,
            )
            process = await asyncio.create_subprocess_exec(
                *start_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self._workspace),
            )

            # Wait for startup (watch for ready pattern in stdout)
            startup_ok = await self._wait_for_startup(process, result)
            result.startup_ok = startup_ok

            if not startup_ok:
                result.errors.append(
                    f"App did not start within {self._startup_timeout}s"
                )
                return result

            # Health check
            health_ok, endpoint = await self._check_health()
            result.health_check_ok = health_ok
            result.health_endpoint = endpoint

            if health_ok:
                result.success = True
            else:
                result.errors.append(
                    f"Health check failed on port {self._port} "
                    f"(tried: {', '.join(_HEALTH_ENDPOINTS)})"
                )

        except Exception as e:
            result.errors.append(f"Smoke test exception: {e}")
            logger.exception("Smoke test failed with exception")
        finally:
            # Clean up: terminate the process
            if process and process.returncode is None:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass

        logger.info("Smoke test result: %s", result.summary())
        return result

    async def _wait_for_startup(
        self,
        process: asyncio.subprocess.Process,
        result: SmokeTestResult,
    ) -> bool:
        """Watch stdout for a ready pattern or timeout."""
        import re
        import time

        start_time = time.monotonic()
        output_lines: list[str] = []
        ready_pattern = re.compile("|".join(_READY_PATTERNS), re.IGNORECASE)

        try:
            while time.monotonic() - start_time < self._startup_timeout:
                if process.returncode is not None:
                    # Process exited early — startup failed
                    result.errors.append(
                        f"Process exited with code {process.returncode} during startup"
                    )
                    break

                try:
                    line_bytes = await asyncio.wait_for(
                        process.stdout.readline(), timeout=2.0,
                    )
                except asyncio.TimeoutError:
                    continue

                if not line_bytes:
                    if process.returncode is not None:
                        break
                    continue

                line = line_bytes.decode("utf-8", errors="replace").rstrip()
                output_lines.append(line)

                if ready_pattern.search(line):
                    result.startup_time_seconds = time.monotonic() - start_time
                    result.stdout_tail = "\n".join(output_lines[-20:])
                    return True

            # Timeout — check if the process is still running (might be ready without logging)
            if process.returncode is None:
                # Process is running but didn't log a ready pattern.
                # Try health check anyway — some apps don't log a ready message.
                result.startup_time_seconds = time.monotonic() - start_time
                result.stdout_tail = "\n".join(output_lines[-20:])
                return True  # Optimistically assume ready

        except Exception as e:
            result.errors.append(f"Startup monitoring error: {e}")

        result.stdout_tail = "\n".join(output_lines[-20:])
        return False

    async def _check_health(self) -> tuple[bool, str]:
        """Try common health endpoints until one returns 200."""
        try:
            import aiohttp
        except ImportError:
            # Fallback: use urllib (synchronous)
            return await self._check_health_urllib()

        async with aiohttp.ClientSession() as session:
            for endpoint in _HEALTH_ENDPOINTS:
                url = f"http://localhost:{self._port}{endpoint}"
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=self._health_timeout)) as resp:
                        if resp.status == 200:
                            return True, endpoint
                except Exception:
                    continue

        return False, ""

    async def _check_health_urllib(self) -> tuple[bool, str]:
        """Fallback health check using urllib (no aiohttp dependency)."""
        import urllib.request
        import urllib.error

        for endpoint in _HEALTH_ENDPOINTS:
            url = f"http://localhost:{self._port}{endpoint}"
            try:
                req = urllib.request.Request(url, method="GET")
                resp = await asyncio.to_thread(
                    urllib.request.urlopen, req, timeout=self._health_timeout,
                )
                if resp.status == 200:
                    return True, endpoint
            except (urllib.error.URLError, urllib.error.HTTPError, Exception):
                continue

        return False, ""

    def _detect_start_command(self) -> list[str] | None:
        """Detect the appropriate start command from the tech stack and workspace files."""
        framework = self._tech_stack.get("framework", "").lower()
        language = self._tech_stack.get("language", "").lower()

        # Check for Spring Boot
        if "spring" in framework or "spring" in language:
            if (self._workspace / "gradlew").exists():
                return _START_COMMANDS["spring-gradle"]
            if (self._workspace / "pom.xml").exists():
                return _START_COMMANDS["spring"]

        # Check for Node.js frameworks
        if "express" in framework:
            # Check package.json for start script
            pkg_json = self._workspace / "package.json"
            if pkg_json.exists():
                return ["npm", "start"]
            return _START_COMMANDS["express"]

        # Check for Python frameworks
        if "fastapi" in framework:
            return _START_COMMANDS["fastapi"]
        if "flask" in framework:
            return _START_COMMANDS["flask"]
        if "django" in framework:
            return _START_COMMANDS["django"]

        # Check for Go
        if language == "go" or (self._workspace / "go.mod").exists():
            return _START_COMMANDS["go"]

        # Check for .NET
        if language in ("c#", "csharp", "dotnet") or list(self._workspace.glob("*.csproj")):
            return _START_COMMANDS["dotnet"]

        return None
