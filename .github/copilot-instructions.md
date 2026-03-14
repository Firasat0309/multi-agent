# GitHub Copilot Instructions — multi-agent-claude

## Purpose & Scope
This file governs all AI-assisted work on this repository: a **production-grade, multi-agent backend code-generation platform** powered by Claude/OpenAI/Gemini. All suggestions and edits must respect the architecture, conventions, and safety rules below.

---

## Architecture Principles

### 1. Incremental Refactoring Only
- Refactor **one concern at a time** — never rewrite multiple layers in a single change.
- Every change must leave all existing CLI (`core/cli.py`), API (`core/api.py`), and pipeline entry points (`core/pipeline.py`) intact and behaviorally identical unless explicitly asked to change them.
- Prefer **adding a new module** over editing a large existing one when introducing new responsibilities.

### 2. Backward Compatibility
- Never remove or rename public method signatures on `BaseAgent`, `PipelineExecutor`, `AgentManager`, `RepositoryManager`, or any `core/models.py` dataclass without an explicit migration plan.
- Deprecate before deleting: add a `DeprecationWarning` and keep the old symbol for at least one commit cycle.
- New fields added to `RepositoryBlueprint`, `FileBlueprint`, `AgentContext`, or `TaskResult` must have safe defaults so existing call sites continue to work.

### 3. Agent Separation of Responsibilities
Each agent owns exactly one concern. Never add logic that belongs to another agent's domain:

| Agent | Sole Responsibility |
|-------|-------------------|
| `ArchitectAgent` | Interpret user intent → produce `RepositoryBlueprint` |
| `PlannerAgent` | Blueprint → task graph + lifecycle state machine |
| `CoderAgent` | Generate / fix a single source file |
| `ReviewerAgent` | Critique code quality; emit REVIEW_FAILED or REVIEW_PASSED |
| `TestAgent` | Generate and execute test files only |
| `SecurityAgent` | Static security analysis (Bandit + LLM review) |
| `BuildVerifierAgent` | Run compiler/build tools; attribute errors to files |
| `PatchAgent` | Apply targeted diff patches; no full rewrites |
| `DeployAgent` | Generate Dockerfile, k8s manifests, compose files |
| `WriterAgent` | Documentation and README only |
| `IntegrationTestAgent` | End-to-end integration test generation only |

Do **not** add file-write logic to `ReviewerAgent`, orchestration logic to `CoderAgent`, or any LLM calls inside pipeline coordinators.

### 4. Pipeline & Orchestration
- The three-layer facade is **non-negotiable**: `pipeline.py` (facade) → `pipeline_run.py` / `pipeline_enhance.py` (workflow) → `pipeline_executor.py` (execution engine). Do not flatten these layers.
- All cross-file coordination must go through `EventBus`; agents must not call other agents directly.
- Tier-based execution order (via `TierScheduler`) must be preserved — never process a dependent file before its dependency tier has passed a build checkpoint (compiled languages).
- State transitions must only happen through `LifecycleEngine.process_event(path, EventType)` — never mutate file states directly.

### 5. Extensibility Patterns
- **New agent**: subclass `BaseAgent`, declare `role: AgentRole`, implement `async execute(context: AgentContext) -> TaskResult`. Register in `AgentManager._create_agent()` and `AgentRole` enum. No other files need to change.
- **New tool for agents**: add a `_tool_<name>` method to `BaseAgent` (or override in the subclass) and append a `ToolDefinition` entry to the `tools` property.
- **New pipeline phase**: add a phase constant, implement a `_run_<phase>_phase` method in `PipelineExecutor`, wire into the phase sequence list. Do not inline phase logic in the `run()` method.
- **New language support**: extend `core/language.py` and `core/ast_extractor.py`; no pipeline code should need to change.

---

## Coding Conventions

### Python Style
- Python ≥ 3.11; use `match/case`, `|` union types, `Self` where appropriate.
- All agent methods, file I/O, and LLM calls must be `async`. Use `aiofiles` for all file operations — never `open()` in async context.
- Use `pydantic>=2.5` models for all structured config and data transfer objects. No raw `dict` types crossing module boundaries.
- Type-annotate all public functions and class members. `Any` is forbidden except in tool dispatch internals.

### Error Handling
- Use structured logging (`logger.info/warning/error`) at every phase boundary. Do not use `print()`.
- LLM continuations: always check `stop_reason` (`"end_turn"` vs `"max_tokens"`); auto-continue up to the existing limit in `BaseAgent`.
- Build failures must be attributed per-file via `CompilerErrorAttributor` and trigger targeted fix loops — never a full pipeline retry.
- All exceptions in agent `execute()` must be caught, logged, and returned as `TaskResult(success=False, error=str(e))`. Do not let exceptions propagate out of agents.

### Security
- All file paths used by agents must be resolved and validated against the workspace root (see `BaseAgent._tool_read_file` / `_tool_write_file`). Reject path traversal (`..`) unconditionally.
- Do not expose raw subprocess output or stack traces in `TaskResult.output` that could leak internal paths.
- Never store LLM API keys in code; always read from environment variables via `config/settings.py`.
- All Docker sandbox commands must run in the restricted sandbox (via `SandboxOrchestrator`), never directly on the host.

### Metrics & Observability
- Every agent must populate `self._metrics` with at minimum `llm_calls` and `tokens_used`.
- New pipeline phases must emit OpenTelemetry spans and Prometheus metrics matching the existing pattern in `core/observability.py`.
- Heartbeat logs (every 15 s during long LLM awaits) must be preserved in `BaseAgent`.

---

## Testing Requirements
- Every new public function or class requires a corresponding test in `tests/`.
- Test file naming: `tests/test_<module_name>.py`.
- Use `pytest` with `pytest-asyncio` for async test cases.
- Mock all LLM calls in unit tests — never make real API calls in tests.
- Build verifier and sandbox tests must mock Docker to avoid requiring a running daemon.
- Do not modify existing passing tests unless fixing a genuine regression introduced by the current change.

---

## What NOT to Do
- Do not flatten the pipeline layers or merge pipeline phases into a single function.
- Do not add orchestration logic (agent sequencing, retries, tier looping) inside any agent class.
- Do not bypass `LifecycleEngine` state transitions.
- Do not call `asyncio.run()` inside any agent or core module — all async work must propagate up to the top-level entry points.
- Do not introduce new global state outside of `AgentManager`, `RepositoryManager`, or `LifecycleEngine`.
- Do not remove the multi-provider LLM abstraction in `core/llm_client.py`; any LLM interaction must go through it.
- Do not add synchronous blocking calls (requests, subprocess without `asyncio.create_subprocess_*`) inside async code paths.
