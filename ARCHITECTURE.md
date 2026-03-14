# Architecture — Multi-Agent Code Generation System

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Operating Modes](#2-operating-modes)
3. [Complete Flow: User Prompt to Generated Code](#3-complete-flow-user-prompt-to-generated-code)
4. [Agent Catalogue](#4-agent-catalogue)
   - 4.1 [ArchitectAgent](#41-architectagent)
   - 4.2 [PlannerAgent](#42-planneragent)
   - 4.3 [CoderAgent](#43-coderagent)
   - 4.4 [ReviewerAgent](#44-revieweragent)
   - 4.5 [PatchAgent](#45-patchagent)
   - 4.6 [BuildVerifierAgent](#46-buildverifieragent)
   - 4.7 [TestAgent](#47-testagent)
   - 4.8 [SecurityAgent](#48-securityagent)
   - 4.9 [DeployAgent](#49-deployagent)
   - 4.10 [WriterAgent](#410-writeragent)
   - 4.11 [IntegrationTestAgent](#411-integrationtestagent)
   - 4.12 [RepositoryAnalyzerAgent](#412-repositoryanalyzeragent)
   - 4.13 [ChangePlannerAgent](#413-changeplanneragent)
   - **Frontend agents (fullstack mode)**
   - 4.14 [ProductPlannerAgent](#414-productplanneragent)
   - 4.15 [APIContractAgent](#415-apicontractagent)
   - 4.16 [DesignParserAgent](#416-designparseragent)
   - 4.17 [ComponentPlannerAgent](#417-componentplanneragent)
   - 4.18 [ComponentDAGAgent](#418-componentdagagent)
   - 4.19 [ComponentGeneratorAgent](#419-componentgeneratoragent)
   - 4.20 [APIIntegrationAgent](#420-apiintegrationagent)
   - 4.21 [StateManagementAgent](#421-statemanagementagent)
5. [Orchestration Layer](#5-orchestration-layer)
   - 5.1 [Pipeline Facade](#51-pipeline-facade)
   - 5.2 [RunPipeline (generate)](#52-runpipeline-generate-mode)
   - 5.3 [EnhancePipeline (enhance)](#53-enhancepipeline-enhance-mode)
   - 5.4 [FullstackPipeline (fullstack)](#54-fullstackpipeline-fullstack-mode)
   - 5.5 [FrontendPipeline](#55-frontendpipeline)
   - 5.6 [PipelineExecutor](#56-pipelineexecutor)
   - 5.7 [AgentManager](#57-agentmanager)
   - 5.8 [LifecycleOrchestrator](#58-lifecycleorchestrator)
6. [Per-File State Machine](#6-per-file-state-machine)
7. [Dependency-Tier Scheduling](#7-dependency-tier-scheduling)
8. [Build Checkpoints](#8-build-checkpoints)
9. [Context Assembly](#9-context-assembly)
10. [Memory Stores](#10-memory-stores)
11. [EventBus](#11-eventbus)
12. [Sandbox Execution](#12-sandbox-execution)
13. [Observability](#13-observability)
14. [Directory Map](#14-directory-map)

---

## 1. System Overview

The system is a **hybrid orchestrated multi-agent pipeline** for autonomous
code generation and modification. It takes a natural-language prompt and
produces a complete, compilable, tested, and deployable software project.

The orchestrator does not perform software-design reasoning. It controls
sequencing, concurrency, dependency ordering, and validation gates. All
reasoning is done inside the agents through LLM calls.

The orchestrator is:

- **State-machine driven** — every file has its own `FileLifecycle` FSM
- **Graph-based** — dependency ordering uses a `networkx` DAG
- **Tier-aware** — files are grouped by dependency depth and processed in
  layers so foundational code compiles before dependent code is generated
- **Rule-based** for retries, gating, skip logic, and fallback behaviour

### Key design principles

1. Each agent owns exactly one concern (see [§ 4](#4-agent-catalogue)).
2. Agents never call each other directly — all cross-agent coordination goes
   through `EventBus`.
3. All file paths are validated against the workspace root before I/O; path
   traversal (`..`) is rejected unconditionally.
4. LLM API keys are never stored in code; they are read from environment
   variables at startup via `config/settings.py`.
5. All file operations use `aiofiles` for non-blocking async I/O.
6. Generated code that runs (builds, tests) is executed inside Docker
   sandboxes, never on the host.

---

## 2. Operating Modes

| Mode | Entry point | Description |
|------|------------|-------------|
| `generate` | `Pipeline.run()` | Create a new project from scratch |
| `enhance` | `Pipeline.enhance()` | Modify an existing repository |
| `fullstack` | `Pipeline.run_fullstack()` | Backend + React/Next.js frontend |

All three modes share the same `PipelineExecutor` as the core execution
engine, ensuring consistent checkpoint, tier, and fix-loop behaviour.

---

## 3. Complete Flow: User Prompt to Generated Code

This section covers two flows:
- **[3.1](#31-generate--enhance-flow)** — `generate` / `enhance` (backend-only)
- **[3.2](#32-fullstack-flow)** — `fullstack` (backend + React/Next.js frontend, run concurrently)

---

### 3.1 Generate / Enhance Flow

```
══════════════════════════════════════════════════════════════════
  ENTRY POINT
══════════════════════════════════════════════════════════════════
  User calls: codegen generate "Build a user management REST API …"
  CLI builds Settings from environment → constructs Pipeline facade
  Pipeline.run(prompt) begins

══════════════════════════════════════════════════════════════════
  PHASE 1 — ARCHITECTURE DESIGN
══════════════════════════════════════════════════════════════════
  RunPipeline instantiates ArchitectAgent
  ArchitectAgent sends prompt to LLM with a structured system prompt
  LLM returns a JSON RepositoryBlueprint:
    - project name, description, architecture style
    - tech_stack: { language, framework, db, build_tool }
    - file_blueprints[]: each file with path, purpose, depends_on[], exports[], layer
    - architecture_doc: markdown with overview, layer diagram, API endpoints, DB schema

══════════════════════════════════════════════════════════════════
  PHASE 2 — TASK PLANNING
══════════════════════════════════════════════════════════════════
  PlannerAgent.create_lifecycle_plan(blueprint) is called
  LifecyclePlanBuilder iterates blueprint.file_blueprints:
    - Creates a FileLifecycle state machine per file (PENDING → … → PASSED)
    - Wires file dependency edges into the LifecycleEngine DAG
    - Sets max_review_fixes=2 per file, max_test_fixes=3 per file
  LifecyclePlanBuilder also creates the global TaskGraph:
    - Sentinel task (unblocks global DAG after all files complete)
    - SecurityScan, GenerateDeploy, GenerateDocs, GenerateIntegrationTests
  TierScheduler.compute_tiers(file_paths, file_deps) groups files:
    - Tier 0: files with zero internal dependencies (models, DTOs, config)
    - Tier 1: files that depend only on Tier 0 (repositories, interfaces)
    - Tier 2: files that depend on Tier 0+1 (services, handlers)
    - Tier N: controllers, application entrypoints
  SandboxOrchestrator.setup() provisions two isolated Docker containers:
    - build_sandbox: has network access (for mvn/go mod/npm install)
    - test_sandbox:  network-isolated (tests must not make external calls)

══════════════════════════════════════════════════════════════════
  PHASE 3 — TIERED EXECUTION (PipelineExecutor)
══════════════════════════════════════════════════════════════════

  For each Tier 0 … Tier N:
  ┌──────────────────────────────────────────────────────────────┐
  │  PipelineExecutor._run_tier_lifecycles(engine, tier)         │
  │                                                              │
  │  All files in the tier run concurrently (up to              │
  │  max_concurrent_agents = 4 by default) with asyncio tasks   │
  │                                                              │
  │  Each file's lifecycle:                                      │
  │                                                              │
  │  PENDING ──DEPS_MET──► GENERATING                            │
  │                           │                                  │
  │                    CoderAgent.execute()                      │
  │                    Reads blueprint + dependency stubs        │
  │                    LLM generates the source file             │
  │                    Writes file to workspace                  │
  │                           │                                  │
  │                    CODE_GENERATED                            │
  │                           ▼                                  │
  │                       REVIEWING                              │
  │                           │                                  │
  │                    ReviewerAgent.execute()                   │
  │                    LLM checks: correctness, security,        │
  │                    blueprint conformance, layer violations   │
  │                    Returns {passed, findings[]}              │
  │                           │                                  │
  │              ┌────────────┼────────────┐                     │
  │        REVIEW_PASSED           REVIEW_FAILED                │
  │              │                      │                        │
  │           BUILDING              FIXING (up to 2×)           │
  │           (compiled             CoderAgent re-generates     │
  │            langs only,          with findings injected      │
  │            skipped for          into the prompt             │
  │            Python etc.)         FIX_APPLIED → REVIEWING     │
  │              │                                               │
  │          BuildVerifier                                       │
  │          runs compiler                                       │
  │          per-file                                           │
  │              │                                               │
  │        BUILD_PASSED → TESTING  (deferred to test phase)     │
  │        BUILD_FAILED → FIXING                                │
  └──────────────────────────────────────────────────────────────┘

  After all files in a tier reach a terminal state (PASSED, FAILED,
  DEGRADED) or the BUILDING phase:

  ┌──────────────────────────────────────────────────────────────┐
  │  BUILD CHECKPOINT (compiled languages only)                  │
  │                                                              │
  │  StubGenerator creates forward-reference stubs for the      │
  │  next tier (so Tier N+1 files can import Tier N types        │
  │  before they exist as real implementations).                 │
  │                                                              │
  │  BuildCheckpoint.run_once() — runs the full build command    │
  │  (mvn package, go build, tsc, cargo build …) against the   │
  │  whole workspace.                                            │
  │                                                              │
  │  If build fails:                                             │
  │  CompilerErrorAttributor maps each error line to a file     │
  │  CoderAgent fix tasks are dispatched for affected files     │
  │  Build is re-run — up to build_checkpoint_retries (3×)      │
  │                                                              │
  │  If checkpoint passes: advance to next tier                 │
  │  If all retries exhausted: fail the broken files;           │
  │  downstream files that exclusively depend on them are        │
  │  also failed (blocked). Files with at least one other        │
  │  good dependency continue.                                   │
  └──────────────────────────────────────────────────────────────┘

  EventBus cross-file coordination:
  - When a file is written (FILE_WRITTEN event), its dependents
    already in BUILDING/TESTING/PASSED are queued for re-verification
  - This ensures changes propagate to already-completed downstream files

══════════════════════════════════════════════════════════════════
  PHASE 4 — TEST GENERATION
══════════════════════════════════════════════════════════════════
  After all tiers + checkpoints pass, TestAgent runs for every file:
    - Generates ≥ 3 test methods per public method/function
    - Runs the test suite inside the test_sandbox (network-isolated)
    - On failure: inserts error output into a fix prompt → CoderAgent
                  re-generates the test → re-runs (up to 3 fix cycles)
    - Files with unresolvable test failures are marked DEGRADED
      (code is valid; tests are incomplete — reported, not fatal)

══════════════════════════════════════════════════════════════════
  PHASE 5 — GLOBAL TASKS (TaskDispatcher, parallel)
══════════════════════════════════════════════════════════════════
  All global tasks run after the sentinel task completes:

  SecurityAgent       scans entire workspace with Bandit + LLM review
                      outputs security_report.json
  DeployAgent         generates Dockerfile (multi-stage, non-root),
                      docker-compose.yml, k8s/ manifests
  WriterAgent         generates README.md, API.md, CHANGELOG.md
  IntegrationTestAgent  generates end-to-end HTTP/gRPC tests

══════════════════════════════════════════════════════════════════
  PHASE 6 — FINALISE
══════════════════════════════════════════════════════════════════
  WorkspaceIndexer re-indexes all files into RepoIndex
  DependencyGraphStore saved to disk
  EmbeddingStore updated for all new/modified files
  RunReporter writes run_report.json:
    { success, code_success, tests_passed, stats, elapsed,
      token_cost, blueprint, checkpoint_results }
  Sandbox containers torn down
```

---

### 3.2 Fullstack Flow

```
══════════════════════════════════════════════════════════════════
  ENTRY POINT
══════════════════════════════════════════════════════════════════
  User calls: codegen fullstack "Build a SaaS analytics dashboard"
  Pipeline.run_fullstack(prompt) → FullstackPipeline.execute()

══════════════════════════════════════════════════════════════════
  PHASE 1 — PRODUCT PLANNING
══════════════════════════════════════════════════════════════════
  ProductPlannerAgent converts prompt → ProductRequirements:
    has_frontend=True, has_backend=True
    tech_preferences: { frontend: "react", backend: "fastapi", db: "postgresql",
                        state: "zustand", styling: "tailwind" }
    user_stories[], features[]

══════════════════════════════════════════════════════════════════
  PHASE 2 — BACKEND ARCHITECTURE
══════════════════════════════════════════════════════════════════
  ArchitectAgent designs RepositoryBlueprint for the backend
  (same as generate mode — see §3.1 Phase 1)
  Prompt is enriched with tech_preferences from ProductRequirements

══════════════════════════════════════════════════════════════════
  PHASE 3 — API CONTRACT
══════════════════════════════════════════════════════════════════
  APIContractAgent receives ProductRequirements + RepositoryBlueprint
  LLM produces an OpenAPI 3.x APIContract:
    { title, version, base_url, endpoints[] }
  Each endpoint: { path, method, description, request_schema, response_schema,
                   auth_required, tags[] }
  This contract is the typed handshake between the FE and BE pipelines.
  Non-fatal if it fails: FE proceeds without typed bindings.

══════════════════════════════════════════════════════════════════
  PHASE 4 — PARALLEL BACKEND + FRONTEND  (asyncio.gather)
══════════════════════════════════════════════════════════════════

  ┌─────────────────────────────┐   ┌──────────────────────────────────────┐
  │  BACKEND  (RunPipeline)     │   │  FRONTEND  (FrontendPipeline)        │
  │  workspace/backend/         │   │  workspace/frontend/                 │
  │                             │   │                                      │
  │  Same 6-phase flow as §3.1  │   │  FE Phase 1 — DESIGN PARSING         │
  │  (Architecture → Planning   │   │    DesignParserAgent                 │
  │   → Tiers → Checkpoints     │   │    If FIGMA_TOKEN + figma_url set:   │
  │   → Tests → Global tasks)   │   │      MCPClient queries Figma API     │
  │                             │   │      extracts pages, styles, layout  │
  │  Uses enriched prompt that  │   │    Else: LLM interprets prose desc.  │
  │  includes the API contract  │   │    Output: UIDesignSpec              │
  │  base_url and endpoint list │   │      { framework, pages[], styles,   │
  │  so the BE implements the   │   │        components_hint[] }           │
  │  correct routes.            │   │                                      │
  │                             │   │  FE Phase 2 — COMPONENT PLANNING     │
  │                             │   │    ComponentPlannerAgent             │
  │                             │   │    UIDesignSpec + APIContract        │
  │                             │   │    → ComponentPlan                   │
  │                             │   │      { framework, state_solution,    │
  │                             │   │        routing_solution,             │
  │                             │   │        package_json,                 │
  │                             │   │        components[] }                │
  │                             │   │    Each component:                   │
  │                             │   │      { name, file_path, type,        │
  │                             │   │        props[], state_slices[],      │
  │                             │   │        api_endpoints[],              │
  │                             │   │        depends_on[] }                │
  │                             │   │                                      │
  │                             │   │  FE Phase 3 — COMPONENT DAG          │
  │                             │   │    ComponentDAGAgent (no LLM)        │
  │                             │   │    Topological sort of depends_on    │
  │                             │   │    Assigns tier number per component │
  │                             │   │    Detects cycles → same-tier        │
  │                             │   │    Ghost-dep detection: unplanned    │
  │                             │   │    deps get auto-synthesised stubs   │
  │                             │   │                                      │
  │                             │   │  FE Phase 4 — COMPONENT GENERATION   │
  │                             │   │    ComponentGeneratorAgent           │
  │                             │   │    Runs tier-by-tier (same model     │
  │                             │   │    as backend PipelineExecutor)      │
  │                             │   │    All components in a tier run      │
  │                             │   │    concurrently via asyncio.gather   │
  │                             │   │    Writes .tsx / .vue files to       │
  │                             │   │    workspace/frontend/src/           │
  │                             │   │                                      │
  │                             │   │  FE Phase 4.5 — TS COMPILATION       │
  │                             │   │    TSXCompiler.check(workspace)      │
  │                             │   │    Runs tsc --noEmit (if available)  │
  │                             │   │    Reports compile errors per file   │
  │                             │   │                                      │
  │                             │   │  FE Phase 4.6 — IMPORT VALIDATION    │
  │                             │   │    ImportValidator scans all imports │
  │                             │   │    Detects unresolved cross-         │
  │                             │   │    component import paths            │
  │                             │   │                                      │
  │                             │   │  FE Phase 5 — API INTEGRATION        │
  │                             │   │    APIIntegrationAgent               │
  │                             │   │    Generates src/lib/api.ts:         │
  │                             │   │      • Axios/fetch base client       │
  │                             │   │      • Auth interceptors (JWT)       │
  │                             │   │      • One typed service per tag     │
  │                             │   │      • SWR / React-Query hooks       │
  │                             │   │      • Full TypeScript types for     │
  │                             │   │        every request/response        │
  │                             │   │                                      │
  │                             │   │  FE Phase 6 — STATE MANAGEMENT       │
  │                             │   │    StateManagementAgent              │
  │                             │   │    Generates src/store/:             │
  │                             │   │      Zustand → one file per slice    │
  │                             │   │      Redux  → actions/reducers/      │
  │                             │   │               selectors per domain   │
  │                             │   │      Pinia  → one store per domain   │
  │                             │   │      Context → provider + hook pair  │
  │                             │   │                                      │
  │                             │   │  FINALISE                            │
  │                             │   │    workspace/frontend/ indexed into  │
  │                             │   │    its own EmbeddingStore + RepoIndex│
  └─────────────────────────────┘   └──────────────────────────────────────┘
                 │                                     │
                 └──────────── asyncio.gather ─────────┘
                                       │
                                       ▼
                           FullstackPipelineResult
                  success = be_ok AND fe_ok
                  workspace/
                    backend/    ← compiled, tested, deployed BE
                    frontend/   ← typed, validated FE
                  run_report.json merges BE + FE metrics
```

---

## 4. Agent Catalogue

All agents share the same base class `BaseAgent` and implement:

```python
async def execute(self, context: AgentContext) -> TaskResult
```

`AgentContext` carries: `task`, `blueprint`, `file_blueprint`,
`related_files` (dependency stubs or full source), `architecture_summary`,
`dependency_info`.

`TaskResult` carries: `success`, `output`, `errors[]`, `metrics{}`,
`files_modified[]`.

---

### 4.1 ArchitectAgent

| | |
|---|---|
| **Role** | `AgentRole.ARCHITECT` |
| **Task type** | `DESIGN_ARCHITECTURE` |
| **Invoked by** | `RunPipeline.execute()` directly (not via AgentManager) |

**Input:**
- Raw user prompt string

**What it does:**
1. Detects language from the prompt (e.g. "Spring Boot" → Java)
2. Detects or defaults a database
3. Designs a clean layered architecture: `model → repository → service → controller`
4. Calls the LLM with a structured system prompt requiring a pure JSON response

**Output:** `RepositoryBlueprint`
```python
@dataclass
class RepositoryBlueprint:
    name: str                          # "user-management-api"
    description: str                   # "REST API for user management …"
    architecture_style: str            # "REST" | "GraphQL" | "gRPC"
    tech_stack: dict[str, str]         # {"language":"java","framework":"spring-boot","db":"postgresql"}
    folder_structure: list[str]        # directory paths to create
    file_blueprints: list[FileBlueprint]
    architecture_doc: str              # markdown with layer diagram + API table
```

Each `FileBlueprint`:
```python
@dataclass
class FileBlueprint:
    path: str           # "src/main/java/com/example/model/User.java"
    purpose: str        # "JPA entity for user accounts"
    depends_on: list[str]   # paths of files this file imports from
    exports: list[str]      # ["User", "UserStatus"]
    language: str           # "java"
    layer: str              # "model" | "repository" | "service" | "controller" | …
```

**Guarantees enforced:**
- `depends_on` references are acyclic
- Models depend on nothing; repos depend on models; services depend on repos; controllers on services
- 8–30 files per project
- Build config file (pom.xml, package.json, go.mod, etc.) is always the first blueprint
- Application entry point and DB config files are always included

**Connected to:** `RunPipeline` passes the `RepositoryBlueprint` to `PlannerAgent`

---

### 4.2 PlannerAgent

| | |
|---|---|
| **Role** | `AgentRole.PLANNER` |
| **Task type** | `CREATE_PLAN` |
| **Invoked by** | `RunPipeline.execute()` directly |

**Input:** `RepositoryBlueprint`

**What it does:**
1. `LifecyclePlanBuilder.build(blueprint)` iterates every `FileBlueprint`
2. Creates a `FileLifecycle` state machine for each file (see §6)
3. Registers inter-file dependency edges in `LifecycleEngine`
4. Sets generation task type: `GENERATE_FILE` for new projects
5. Builds the global `TaskGraph` with sentinel + global phase tasks

**Output:** `(LifecycleEngine, TaskGraph)`
- `LifecycleEngine`: orchestrates all per-file FSMs, knows which files are ready to execute based on dependency resolution
- `TaskGraph`: a DAG of global tasks (security, deploy, docs, integration tests)

**Connected to:** `RunPipeline` passes both to `PipelineExecutor`

---

### 4.3 CoderAgent

| | |
|---|---|
| **Role** | `AgentRole.CODER` |
| **Task types** | `GENERATE_FILE`, `FIX_CODE` |
| **Invoked by** | `AgentManager._execute_lifecycle_phase()` |

**Input:** `AgentContext` containing:
- `task.file` — path to generate
- `file_blueprint` — purpose, exports, layer, depends_on
- `related_files` — AST stubs of dependency files, or full source if small
- `architecture_summary` — markdown from ArchitectAgent
- `task.metadata["review_errors"]` — reviewer findings (fix mode)
- `task.metadata["build_errors"]` — compiler output (fix mode)
- `task.metadata["test_errors"]` — test failures (fix mode)

**What it does:**
1. Builds a language-specific prompt with the blueprint context
2. Calls the LLM to generate (or repair) the file
3. Strips markdown fences, normalises line endings
4. For fix mode on large files (> 200 lines):  generates a unified diff instead of full rewrite (5–20× cheaper in tokens)
5. Validates the output:
   - Detects duplicate class/function definitions (LLM hallucination symptom)
   - Rejects content that grows > 135% of the original on rewrites (duplication guard)
6. Writes the file to workspace via `RepositoryManager` (atomic write with file lock)

**Output:** `TaskResult`
- `success=True`, `files_modified=["path/to/file"]`
- On failure: `success=False`, `errors=["reason"]`

**Connected to:**
- Triggered by `GENERATING` and `FIXING` phases
- On `CODE_GENERATED` event → FSM advances to `REVIEWING`
- On `FIX_APPLIED` event → FSM advances back to `REVIEWING`

---

### 4.4 ReviewerAgent

| | |
|---|---|
| **Role** | `AgentRole.REVIEWER` |
| **Task types** | `REVIEW_FILE`, `REVIEW_MODULE`, `REVIEW_ARCHITECTURE` |
| **Invoked by** | `AgentManager._execute_lifecycle_phase()` |

**Input:** `AgentContext` containing:
- The full source of the file under review
- AST stubs of its dependencies (for type/import correctness check)
- Architecture summary and blueprint

**What it does:**
1. Sends the file + context to the LLM
2. LLM returns structured JSON: `{passed, summary, findings[]}`
3. Each finding has: `severity (critical|warning|info)`, `file`, `line`, `message`, `suggestion`
4. Pass/fail verdict: `FAIL` if any finding is `critical`; `PASS` otherwise

**Output:** `TaskResult`
- `success=True` (task completed), `metrics["passed"]=True|False`
- `errors=[]` of finding messages if failed

**Severity rules (strictly enforced in prompt):**
- `critical`: will not compile, will crash, has a security vulnerability (injection, XSS, SSRF), or violates architecture blueprint
- `warning`: works but has significant quality issue (missing error handling, resource leak, race condition)
- `info`: style suggestion — does not block

**Connected to:**
- `REVIEW_PASSED` event → FSM advances to `BUILDING` (compiled) or `TESTING`
- `REVIEW_FAILED` event → FSM enters `FIXING`; `review_findings` stored in lifecycle
- On exception during review: auto-passes (degraded-gracefully) to avoid infinite block

---

### 4.5 PatchAgent

| | |
|---|---|
| **Role** | `AgentRole.PATCH_AGENT` |
| **Task type** | `MODIFY_FILE` |
| **Invoked by** | `AgentManager._execute_lifecycle_phase()` in **enhance mode** |

**Input:** `AgentContext` containing:
- Current file content
- The change request description
- `task.metadata["change_type"]`, `target_function`, `change_description`

**What it does:**
1. Reads the current file content
2. Sends file + change request to LLM — requests **only a unified diff** (not a rewrite)
3. Validates the diff: checks that it applies cleanly to the current file
4. Applies via `FileTools.apply_patch()`
5. On any failure → falls back to `CoderAgent` full-file rewrite (ensures pipeline never silently produces an empty result)

**Output:** `TaskResult`
- `success=True`, `files_modified=["path"]`

**Why it exists:** Targeted diffs are token-efficient and preserve surrounding code precisely. Full rewrites risk losing code outside the modified function.

**Connected to:**
- Used in `EnhancePipeline` instead of `CoderAgent` for the `GENERATING` phase
- On success: `CODE_GENERATED` → `REVIEWING`

---

### 4.6 BuildVerifierAgent

| | |
|---|---|
| **Role** | `AgentRole.BUILD_VERIFIER` |
| **Task type** | `VERIFY_BUILD` |
| **Invoked by** | `PipelineExecutor._run_checkpoint()` via fix dispatch |
| **Languages** | Java, Go, Rust, TypeScript, C# only — skipped for Python |

**Input:** `AgentContext` + `TerminalTools` (build sandbox)

**What it does:**
1. Calls `lang_profile.build_command` (e.g. `mvn package -q`, `go build ./...`, `tsc --noEmit`)
2. Captures stdout + stderr (capped to 3,000 chars — compiler output can be huge)
3. Passes output to `CompilerErrorAttributor.attribute(output, workspace_files)`
4. Attributor maps each error line/column to a specific source file path
5. Returns which files are broken so only they enter fix cycles

**Output:** `TaskResult`
- `success=True|False`
- `errors=["compiler output"]`
- `metrics["attributed_files"]=["path1","path2"]`

**Connected to:**
- `BUILD_PASSED` → FSM advances to `TESTING`
- `BUILD_FAILED` → FSM enters `FIXING`; `build_errors` stored in lifecycle
- At checkpoint level: broken files get targeted `CoderAgent` fix tasks

---

### 4.7 TestAgent

| | |
|---|---|
| **Role** | `AgentRole.TESTER` |
| **Task type** | `GENERATE_TEST` |
| **Invoked by** | `PipelineExecutor._run_test_phase()` |
| **Sandbox** | `test_sandbox` (network-isolated Docker container) |

**Input:** `AgentContext` containing:
- Full source of the file under test
- Any existing test file for this module (if enhancing)
- Language profile (test framework, test runner command)

**What it does:**
1. Sends source file to LLM with test generation prompt
2. Requirements enforced in prompt:
   - ≥ 3 test methods per public method
   - Happy path, error case, and edge case for each method
   - Must use exact import paths from source — no invented paths
   - Must compile and run independently
   - Assertions must check concrete values, not `assertNotNull`
3. Writes the test file to workspace
4. Runs the test suite via `TerminalTools` in the test sandbox
5. On failure: injects error output as fix context → re-calls LLM
6. Up to `max_test_fixes=3` repair cycles
7. `CoverageRunner` optionally reports per-file line coverage

**Output:** `TaskResult`
- `success=True` if tests pass
- `success=False` + `errors=["test output"]` if all fix cycles exhausted
- Files that exhaust test fix cycles are marked `DEGRADED` (not `FAILED`) — the source code is valid

**Connected to:**
- `TEST_PASSED` → FSM reaches `PASSED` (terminal)
- `TEST_FAILED` → FSM enters `FIXING` with `fix_trigger="test"`
- Post-all-tiers, before global tasks

---

### 4.8 SecurityAgent

| | |
|---|---|
| **Role** | `AgentRole.SECURITY` |
| **Task type** | `SECURITY_SCAN` |
| **Invoked by** | `TaskDispatcher` (global DAG phase) |
| **Sandbox** | `build_sandbox` (needs full Python environment for Bandit) |

**Input:** `AgentContext` with full workspace context

**What it does:**
1. Runs `bandit -r . -f json` via `TerminalTools` (for Python projects)
2. Also sends all source files to LLM for semantic vulnerability review
3. OWASP-aligned checks: SQL injection, command injection, XSS, SSRF, path traversal, insecure deserialization, hardcoded secrets, missing auth, weak crypto, etc.
4. Returns structured JSON: `{passed, vulnerabilities[], summary}`
5. Each vulnerability: `severity (critical|high|medium|low)`, `file`, `line`, `type`, `description`, `remediation`
6. Writes `security_report.json` to workspace

**Output:** `TaskResult`
- `success=True|False` based on whether critical vulnerabilities exist
- `files_modified=["security_report.json"]`

**Connected to:**
- Runs after sentinel task in global DAG
- Output written to workspace; does not block other global tasks

---

### 4.9 DeployAgent

| | |
|---|---|
| **Role** | `AgentRole.DEPLOYER` |
| **Task type** | `GENERATE_DEPLOY` |
| **Invoked by** | `TaskDispatcher` (global DAG phase) |

**Input:** `AgentContext` with `blueprint` (tech stack, port, health endpoint info)

**What it does:**
1. Detects app port and health endpoint from tech stack
2. Generates `Dockerfile` — multi-stage build, non-root user, `HEALTHCHECK`
3. Generates `docker-compose.yml` — app + DB + network configuration
4. Generates `k8s/deployment.yaml` — Deployment + resource limits + probes
5. Generates `k8s/service.yaml` — ClusterIP/NodePort service
6. Generates `k8s/configmap.yaml` — environment variables

**Consistency rules enforced in prompt:**
- Same port number across Dockerfile EXPOSE, docker-compose ports, K8s containerPort
- Same app name for all labels and container names
- Health check paths match the application's actual routes

**Output:** `TaskResult`
- `success=True`
- `files_modified=["Dockerfile","docker-compose.yml","k8s/deployment.yaml",…]`

---

### 4.10 WriterAgent

| | |
|---|---|
| **Role** | `AgentRole.WRITER` |
| **Task type** | `GENERATE_DOCS` |
| **Invoked by** | `TaskDispatcher` (global DAG phase) |

**Input:** `AgentContext` with blueprint + generated file context

**What it does:**
1. Generates `README.md` — installation, usage, environment variables, API endpoints (accurate to the generated code)
2. Generates `API.md` — full API reference with example requests/responses
3. Generates `CHANGELOG.md` — initial release entry

**Accuracy rules enforced in prompt:**
- Every endpoint documented must exist in a controller file provided in context
- Every command must be the correct build/run command for the language
- Port numbers must match application configuration
- No hallucinated features

**Output:** `TaskResult`
- `success=True`
- `files_modified=["docs/README.md","docs/API.md","docs/CHANGELOG.md"]`

---

### 4.11 IntegrationTestAgent

| | |
|---|---|
| **Role** | `AgentRole.INTEGRATION_TESTER` |
| **Task type** | `GENERATE_INTEGRATION_TEST` |
| **Invoked by** | `TaskDispatcher` (global DAG phase) |
| **Sandbox** | `test_sandbox` (network-isolated) |

**Input:** `AgentContext` with controller + service + repository source files

**What it does:**
- Generates end-to-end integration tests that exercise the full request → service → data-layer path
- For REST APIs: HTTP client tests (httpx, RestAssured, etc.) against the running app
- Mocks only external boundaries (real DB, external payment APIs, etc.)
- Tests at least one happy path, one auth-required path, and one error path per endpoint

**Output:** `TaskResult`
- `success=True`
- `files_modified=["tests/integration/…"]`

---

### 4.12 RepositoryAnalyzerAgent

| | |
|---|---|
| **Role** | `AgentRole.ANALYZER` |
| **Task type** | `ANALYZE_REPO` |
| **Invoked by** | `EnhancePipeline` — Phase 1 |

**Input:** Path to existing workspace directory

**What it does:**
1. Scans all source files in the workspace
2. Extracts `ModuleInfo` per file: detected layer, functions, classes, imports, exports, entry points
3. Infers language from file extensions
4. Produces a `RepoAnalysis` object describing the existing repository structure

**Output:** `RepoAnalysis`
```python
@dataclass
class RepoAnalysis:
    modules: list[ModuleInfo]
    tech_stack: dict[str, str]
    architecture_style: str
    summary: str
```

**Connected to:** `EnhancePipeline` uses `RepoAnalysis` to construct a
minimal `RepositoryBlueprint` before calling `ChangePlannerAgent`

---

### 4.13 ChangePlannerAgent

| | |
|---|---|
| **Role** | `AgentRole.CHANGE_PLANNER` |
| **Task type** | `PLAN_CHANGES` |
| **Invoked by** | `EnhancePipeline` — Phase 2 |

**Input:** `RepoAnalysis` + user prompt + `EmbeddingStore` for semantic search

**What it does:**
1. Uses semantic search to find files relevant to the requested change
2. Sends relevant files + user request to LLM
3. LLM returns a structured `ChangePlan`:
   - `changes[]`: existing files to modify with `ChangeAction` type, rationale, target function
   - `new_files[]`: new `FileBlueprint` entries to create
   - `summary`: one-sentence description of all changes

**Output:** `ChangePlan` — passed to `EnhanceLifecyclePlanBuilder` which
converts it into a `LifecycleEngine` with `MODIFY_FILE` task types (routed
to `PatchAgent`) and normal `GENERATE_FILE` task types for new files.

**If `require_plan_approval=True`:** `PlanApprover` displays the change plan
and waits for human `y/n` input before execution proceeds.

---

### 4.14 ProductPlannerAgent

| | |
|---|---|
| **Role** | `AgentRole.PRODUCT_PLANNER` |
| **Task type** | `PLAN_PRODUCT` |
| **Invoked by** | `FullstackPipeline` — Phase 1 |

**Input:** Raw user prompt string

**What it does:**
1. Determines whether the project needs a frontend, backend, or both
2. Extracts technology preferences from the prompt (`react`, `vue`, `nextjs`, `fastapi`, `spring-boot`, etc.)
3. Identifies state management preference (`zustand`, `redux`, `pinia`, `context`)
4. Generates user stories and feature list from the prompt
5. Returns structured JSON: `{title, description, user_stories[], features[], tech_preferences{}, has_frontend, has_backend}`

**Output:** `ProductRequirements` — passed to `ArchitectAgent` as enriched context, and to `FrontendPipeline` to drive all downstream FE decisions

**Connected to:** `FullstackPipeline.execute()` → passed to `APIContractAgent`, `ArchitectAgent`, and `FrontendPipeline`

---

### 4.15 APIContractAgent

| | |
|---|---|
| **Role** | `AgentRole.API_CONTRACT_GENERATOR` |
| **Task type** | `GENERATE_API_CONTRACT` |
| **Invoked by** | `FullstackPipeline` — Phase 3 |

**Input:** `ProductRequirements` + backend `RepositoryBlueprint`

**What it does:**
1. Inspects the backend blueprint's controller files and their declared routes
2. Generates a formal OpenAPI 3.x contract as structured JSON
3. Each endpoint: `path`, `method`, `description`, `request_schema`, `response_schema`, `auth_required`, `tags[]`
4. Provides a `base_url` (e.g. `/api/v1`) used consistently by both FE and BE

**Output:** `APIContract` — the shared handshake:
- Backend uses it to confirm its routes match the contract
- Frontend uses it to generate typed API client services and hooks

**Failure mode:** Non-fatal — if this fails, the frontend pipeline continues without typed bindings; the backend is unaffected.

**Connected to:** `FullstackPipeline` passes it to both `BackendPipeline` (as prompt enrichment) and `FrontendPipeline` (as `api_contract` param)

---

### 4.16 DesignParserAgent

| | |
|---|---|
| **Role** | `AgentRole.DESIGN_PARSER` |
| **Task type** | `PARSE_DESIGN` |
| **Invoked by** | `FrontendPipeline` — FE Phase 1 |

**Input:** `ProductRequirements` + optional `figma_url` string

**What it does:**
- **With Figma URL** (`FIGMA_TOKEN` set): uses `MCPClient` (Model Context Protocol) to query the live Figma API — reads file name, pages, node tree, typography tokens, and colour palette before calling the LLM
- **Without Figma URL**: LLM interprets the product description and infers a plausible UI structure
- Returns a `UIDesignSpec` with: `framework`, `design_description`, `figma_url`, `pages[]`, `global_styles{}`, `components_hint[]`

**Output:** `UIDesignSpec` — drives `ComponentPlannerAgent`

**Failure mode:** Non-fatal — if parsing fails, pipeline continues with an empty `UIDesignSpec` so later phases can still proceed.

---

### 4.17 ComponentPlannerAgent

| | |
|---|---|
| **Role** | `AgentRole.COMPONENT_PLANNER` |
| **Task type** | `PLAN_COMPONENTS` |
| **Invoked by** | `FrontendPipeline` — FE Phase 2 |

**Input:** `UIDesignSpec` + `APIContract` + `ProductRequirements`

**What it does:**
1. Applies Atomic Design: Atoms → Molecules → Organisms → Templates → Pages
2. For every component specifies: `name`, `file_path`, `component_type` (atom/molecule/organism/page/layout), `description`, `props[]`, `state_slices[]`, `api_endpoints[]`, `depends_on[]`
3. Resolves routing: assigns each page-level component to a URL path
4. Selects `state_solution` (zustand/redux/pinia/context) and `routing_solution` (react-router/nextjs/vue-router)
5. Produces `package_json` with exact `dependencies` and `devDependencies`

**Output:** `ComponentPlan` — the complete FE architecture blueprint:
```python
@dataclass
class ComponentPlan:
    framework: str           # "react" | "nextjs" | "vue" | "angular"
    state_solution: str      # "zustand" | "redux" | "pinia" | "context"
    api_base_url: str        # "/api/v1"
    routing_solution: str    # "react-router" | "nextjs" | "vue-router"
    package_json: dict       # exact deps for npm install
    components: list[UIComponent]
```

**Fatal if it fails** — without a `ComponentPlan` the frontend pipeline cannot continue.

---

### 4.18 ComponentDAGAgent

| | |
|---|---|
| **Role** | `AgentRole.COMPONENT_DAG_BUILDER` |
| **Task type** | `BUILD_COMPONENT_DAG` |
| **Invoked by** | `FrontendPipeline` — FE Phase 3 |
| **LLM call** | **None** — purely deterministic graph analysis |

**Input:** `ComponentPlan`

**What it does:**
1. Builds a `networkx.DiGraph` from `UIComponent.depends_on` fields
2. Topologically sorts the graph (Kahn's algorithm)
3. Assigns a `generation` tier number to each component
4. Detects dependency cycles → places cycle members in the same tier as their lowest-depth node
5. Returns `(ordered_components, tier_map)` — `tier_map[component_name] = tier_index`

**Ghost dependency detection** (done by `FrontendPipeline`):
If a `depends_on` reference names a component not in the `ComponentPlan`, a minimal stub `UIComponent` is synthesised and added to Tier 0 — so `ComponentGeneratorAgent` generates it rather than leaving a dangling import.

**Output:** `(list[UIComponent], dict[str, int])` — tier assignments used by `ComponentGeneratorAgent`

This agent mirrors the role of `TierScheduler` on the backend side.

---

### 4.19 ComponentGeneratorAgent

| | |
|---|---|
| **Role** | `AgentRole.COMPONENT_GENERATOR` |
| **Task type** | `GENERATE_COMPONENT` |
| **Invoked by** | `FrontendPipeline` — FE Phase 4 (tier-by-tier, concurrent within tier) |

**Input:** `AgentContext` containing:
- `task.metadata["component"]` — the `UIComponent` to generate
- `task.metadata["component_plan"]` — full plan (for framework/state/routing context)
- `task.metadata["api_contract"]` — for generating correct API call types
- `blueprint` — FE `RepositoryBlueprint` (synthetic, built from `ComponentPlan`)

**What it does:**
1. Sends component spec + context to LLM
2. Generates a complete `.tsx` / `.vue` / `.ts` source file
3. Requirements enforced in prompt:
   - Must import only from the component's declared `depends_on` paths
   - Props must match the ComponentPlan spec
   - API calls must use the typed service from `src/lib/api.ts`
   - State must use the selected `state_solution`
   - No inline styles (must use the declared styling system)
4. Writes the file via `RepositoryManager` (atomic write)
5. `EmbeddingStore` indexes each component as it is written (for later-tier semantic context)

**Concurrency:** All components in the same generation tier run via `asyncio.gather` — same pattern as the backend `PipelineExecutor`.

**Output:** `TaskResult` with `files_modified=[component.file_path]`

---

### 4.20 APIIntegrationAgent

| | |
|---|---|
| **Role** | `AgentRole.API_INTEGRATOR` |
| **Task type** | `INTEGRATE_API` |
| **Invoked by** | `FrontendPipeline` — FE Phase 5 |

**Input:** `AgentContext` with `api_contract` + `component_plan` in metadata

**What it does:**
1. Generates `src/lib/api.ts` — the base HTTP client:
   - Axios or `fetch` wrapper with base URL configuration
   - JWT auth interceptors (attach token, handle 401 refresh)
   - Error normalisation
2. Generates one typed service file per API tag (e.g. `src/services/userService.ts`)
3. Generates SWR / React-Query hooks per endpoint (e.g. `useUsers()`, `useCreateUser()`)
4. All TypeScript types are derived directly from the `APIContract` schemas — no hand-written types

**Output:** `TaskResult` with `files_modified=["src/lib/api.ts", "src/services/…", "src/hooks/…"]`

**Connected to:** Components generated in Phase 4 import from these service files — so Phase 5 must run after Phase 4.

---

### 4.21 StateManagementAgent

| | |
|---|---|
| **Role** | `AgentRole.STATE_MANAGER` |
| **Task type** | `MANAGE_STATE` |
| **Invoked by** | `FrontendPipeline` — FE Phase 6 |

**Input:** `AgentContext` with `component_plan` in metadata

**What it does:**
Generates the state layer based on `component_plan.state_solution`:

| `state_solution` | Files generated |
|---|---|
| `zustand` | `src/store/<feature>Store.ts` — one file per state slice |
| `redux` | `src/store/<feature>/actions.ts`, `reducers.ts`, `selectors.ts` + root `store.ts` |
| `pinia` | `src/stores/use<Feature>Store.ts` — one store per domain |
| `context` | `src/context/<Feature>Context.tsx` — Provider + typed `use<Feature>()` hook |

State slices are derived from `UIComponent.state_slices[]` fields — the state layer is never invented; it reflects what the components declared they need.

**Output:** `TaskResult` with `files_modified=["src/store/…"]`

**Connected to:** Components import from these store files; state layer therefore depends on the correct store filenames being known at component-generation time (Phase 4 uses the `state_solution` from `ComponentPlan` to determine import paths).

---

## 5. Orchestration Layer

### 5.1 Pipeline Facade

`core/pipeline.py` — `Pipeline` class. Thin wrapper that:
- Owns `LLMClient` and `LiveConsole` lifecycle
- Attaches a per-run file log handler
- Delegates to `RunPipeline`, `EnhancePipeline`, or `FullstackPipeline`
- Ensures console/logging cleanup even on failure

```python
async with Pipeline(settings) as p:
    result = await p.run("Build a user management API")
```

---

### 5.2 RunPipeline (generate mode)

`core/pipeline_run.py` — Four sequential phases:

| Phase | What happens |
|-------|-------------|
| 1 Architecture | `ArchitectAgent` → `RepositoryBlueprint` |
| 2 Planning | `PlannerAgent` → `LifecycleEngine` + `TaskGraph` + `TierScheduler` → `tiers[]` + `SandboxOrchestrator.setup()` |
| 3 Execution | `PipelineExecutor.execute(engine, global_graph, tiers, pipeline_def=GENERATE_PIPELINE)` |
| 4 Finalise | `index_workspace()` + `RunReporter.write_run_report()` + sandbox teardown |

The `GENERATE_PIPELINE` definition:
```
Phase "code_generation":
  file_tasks: [GENERATE_FILE, review=True, max_review_fixes=2]
  checkpoint: BuildCheckpoint(max_retries=3)   ← compiled langs only
Phase "testing":
  file_tasks: [GENERATE_TEST, max_test_fixes=3]
  skip_for_interpreted: False
Global tasks: [SECURITY_SCAN, GENERATE_DEPLOY, GENERATE_DOCS, GENERATE_INTEGRATION_TEST]
```

---

### 5.3 EnhancePipeline (enhance mode)

`core/pipeline_enhance.py` — Five sequential phases:

| Phase | What happens |
|-------|-------------|
| 1 Analysis | `RepositoryAnalyzerAgent` → `RepoAnalysis` |
| 2 Change Planning | `ChangePlannerAgent` → `ChangePlan` (+ optional human approval) |
| 3 Lifecycle Plan | `EnhanceLifecyclePlanBuilder.build(change_plan)` → `LifecycleEngine` with MODIFY_FILE tasks |
| 4 Execution | `PipelineExecutor.execute(engine, global_graph, tiers, pipeline_def=ENHANCE_PIPELINE)` |
| 5 Finalise | Re-index workspace + `RunReporter.write_modify_report()` |

Before Phase 4, a `WorkspaceSnapshot` is taken so the workspace can be rolled
back if execution fails critically.

---

### 5.4 FullstackPipeline (fullstack mode)

`core/pipeline_fullstack.py` — Four sequential phases:

| Phase | What happens |
|-------|-------------|
| 1 Product Planning | `ProductPlannerAgent.plan_product(prompt)` → `ProductRequirements` |
| 2 Backend Architecture | `ArchitectAgent.design_architecture(enriched_prompt)` → `RepositoryBlueprint` |
| 3 API Contract | `APIContractAgent.generate_contract(requirements, blueprint)` → `APIContract` (non-fatal if fails) |
| 4 Parallel BE + FE | `asyncio.gather(RunPipeline.execute(...), FrontendPipeline.execute(...))` |

**Workspace layout:**
```
workspace/
  backend/     ← RunPipeline output (compiled, tested, deployed)
  frontend/    ← FrontendPipeline output (typed, validated)
```

**Conditional execution:** If `ProductRequirements.has_backend=False`, the `RunPipeline` coroutine is omitted from `asyncio.gather`. Same for `has_frontend=False`. This allows generation of frontend-only or backend-only from the fullstack entrypoint.

**Root-level write lock:** A shared `asyncio.Lock` is passed to both pipelines to serialise writes to shared root files (`.gitignore`, root `docker-compose.yml`) that both pipelines may generate simultaneously.

---

### 5.5 FrontendPipeline

`core/pipeline_frontend.py` — Six sequential phases (see §3.2 for the detailed flow):

| Phase | Agent | Output |
|-------|-------|--------|
| 1 Design Parsing | `DesignParserAgent` | `UIDesignSpec` |
| 2 Component Planning | `ComponentPlannerAgent` | `ComponentPlan` + `package.json` + `.env.local` written |
| 3 Component DAG | `ComponentDAGAgent` (no LLM) | `(ordered_components, tier_map)` |
| 4 Component Generation | `ComponentGeneratorAgent` × N (tier-by-tier, concurrent within tier) | `.tsx`/`.vue` source files |
| 4.5 TS Compilation | `TSXCompiler.check()` | compile error list (non-blocking) |
| 4.6 Import Validation | `ImportValidator` | unresolved import list (non-blocking) |
| 5 API Integration | `APIIntegrationAgent` | `src/lib/api.ts` + service + hook files |
| 6 State Management | `StateManagementAgent` | `src/store/` files |
| Finalise | `index_workspace()` | FE `EmbeddingStore` + `RepoIndex` updated |

**Key differences from the backend `RunPipeline`:**
- No `LifecycleEngine` / `FilePhase` FSM — component generation uses a simpler `asyncio.gather` per tier
- No `BuildCheckpoint` loop — TypeScript compilation is a single non-blocking check
- No `ReviewerAgent` pass per component — review is omitted for speed (FE files are lighter)
- Uses its own `EmbeddingStore` and `RepositoryManager` scoped to `workspace/frontend/`
- MCP client (`MCPClient`) is wired in for Figma access and closed in the `finally` block

---

### 5.6 PipelineExecutor

`core/pipeline_executor.py` — The core unified execution engine. Used by all modes.

Responsibilities:
1. Wire `EventBus` subscribers for cross-file re-verification
2. For each tier (in order):
   - Call `_run_tier_lifecycles(engine, tier)` — parallel per-file FSM
   - If compiled language: create forward-reference stubs for next tier
   - Run `BuildCheckpoint` → attribute errors → dispatch fix tasks → retry up to N times
3. After all tiers: call `_run_test_phase(engine)`
4. Complete global DAG via `AgentManager.execute_graph(global_graph)`
5. Collect and return execution stats, checkpoint results, bus failure report

Key correctness guarantees:
- A tier does not start until the previous tier's checkpoint passes (or is declared unrecoverable)
- Files whose deps are ALL broken are themselves failed rather than hung PENDING
- Stale detection: if files are stuck PENDING with no in-flight tasks for `2× phase_timeout`, they are force-failed to prevent deadlock
- Reverify queue: files already in BUILDING/TESTING/PASSED that receive an upstream FILE_WRITTEN event are re-queued for verification at the next checkpoint

---

### 5.7 AgentManager

`core/agent_manager.py` — Thin coordination hub:

1. **`_create_agent(task_type)`** — factory returning the correct agent class from `TASK_AGENT_MAP`; wires the correct terminal (build vs test sandbox)
2. **`_execute_lifecycle_phase(engine, file_path, phase)`** — canonical per-file execution:
   - Handles `--skip-reviewer` flag
   - Builds `AgentContext` via `ContextBuilder`
   - Calls the agent, captures metrics (capped at 100 entries per agent)
   - Fires FSM events based on result
   - Publishes `EventBus` events for downstream coordination
3. **`_handle_execution_exception(engine, file_path, phase, exc)`** — graceful degradation:
   - `REVIEWING` exception → auto-pass (never block a file on a transient LLM error)
   - `FIXING` exception → `FIX_APPLIED` re-enters review cycle
   - All other phases → `RETRIES_EXHAUSTED`, increment `tasks_failed`
4. **`execute_graph(task_graph)`** — delegates to `TaskDispatcher` for global DAG
5. **`_get_agent_name_for_task_type(task_type)`** — derived from `TASK_AGENT_MAP` (single source of truth)

`TASK_AGENT_MAP` is a module-level dict mapping every `TaskType` to an agent class — the canonical registry for all agent routing.

---

### 5.8 LifecycleOrchestrator

`core/lifecycle_orchestrator.py` — Delegates to `AgentManager`:

- `execute_with_lifecycle(engine, graph)` — *(deprecated)* FSM event loop without checkpoints
- `execute_with_checkpoints(engine, graph, tiers, pipeline_def)` — *(deprecated)* delegates to `PipelineExecutor`
- `_execute_lifecycle_phase(engine, file_path, phase)` — thin shim calling `AgentManager._execute_lifecycle_phase`
- Static helpers `_build_lifecycle_metadata` and `_extract_event_data` — used by tests and `AgentManager` for backward compatibility

Both `execute_with_*` methods emit a `DeprecationWarning`. All production code
uses `PipelineExecutor.execute()` directly.

---

## 6. Per-File State Machine

`core/state_machine.py`

Every file has a `FileLifecycle` instance with a `phase: FilePhase`:

```
                   DEPS_MET
PENDING ──────────────────────────► GENERATING
                                        │
                                  CODE_GENERATED
                                        │
                                        ▼
                           ┌──── REVIEWING ◄───────────────────┐
                           │         │                          │
                     REVIEW_FAILED   REVIEW_PASSED         FIX_APPLIED
                           │         │                          │
                           │         ▼                          │
                           │      BUILDING ──BUILD_FAILED──► FIXING
                           │         │                          ▲
                           │    BUILD_PASSED             TEST_FAILED
                           │         │                          │
                           │         ▼                          │
                           └──────► TESTING ────────────────────┘
                                     │
                               TEST_PASSED
                                     │
                                     ▼
                                   PASSED  (terminal ✓)

At any phase: RETRIES_EXHAUSTED → FAILED   (terminal ✗)
If fixes exhausted but code exists: → DEGRADED  (terminal ⚠)
```

`FilePhase` values:

| Phase | Meaning |
|-------|---------|
| `PENDING` | Waiting for dependency files to complete |
| `GENERATING` | `CoderAgent` / `PatchAgent` writing the file |
| `REVIEWING` | `ReviewerAgent` checking correctness and compliance |
| `FIXING` | `CoderAgent` applying corrections from review/build/test findings |
| `BUILDING` | Per-file `BuildVerifierAgent` (non-checkpoint mode only) |
| `TESTING` | `TestAgent` generating and running tests |
| `PASSED` | Terminal — file fully generated, reviewed, tested |
| `FAILED` | Terminal — exhausted all retry limits |
| `DEGRADED` | Terminal — code valid, tests did not fully pass |

`LifecycleEngine` orchestrates all `FileLifecycle` instances:
- Tracks dependency edges between files: a file stays `PENDING` until all its declared depends-on files reach a terminal phase
- `get_actionable_files()` returns files that are not in-flight, not terminal, and whose dependencies are satisfied
- `all_terminal()` returns `True` when every file has reached `PASSED`, `FAILED`, or `DEGRADED`

---

## 7. Dependency-Tier Scheduling

`core/tier_scheduler.py`

Files are grouped into tiers by dependency depth using topological sort:

```
Input: file_paths, file_deps (from FileBlueprint.depends_on fields)

Tier 0: files with zero internal dependencies
         → models, DTOs, config primitives, build config

Tier 1: files that depend only on Tier 0
         → repositories, interfaces, lower-level services

Tier 2: files that depend on Tier 0 or Tier 1
         → higher-level services, handlers

Tier N: controllers, application entrypoints
```

**Why tiers matter:**

Backend projects do not become compilable one file at a time. They become
compilable one dependency layer at a time. By enforcing tier ordering:

- Foundational types exist before dependent code imports them
- Build errors in `User.java` are found and fixed before `UserRepository.java`
  is even generated — eliminating cascading failures
- Forward-reference stubs allow Tier N+1 to compile against Tier N signatures
  even while Tier N implementations are being fixed

Cycles in `depends_on` are detected and cycle members are placed in the same
tier as their lowest-depth dependency.

---

## 8. Build Checkpoints

`core/checkpoint.py`

After every tier completes (for compiled languages), a `BuildCheckpoint` runs:

```
1. StubGenerator.create_stubs(next_tier_files)
   — Creates minimal stub implementations for all Tier N+1 files
     so the compiler can resolve imports from Tier N+1 during the
     Tier N build without needing real implementations yet.

2. BuildCheckpoint.run_once(lang_profile.build_command)
   — Runs full workspace build: mvn package, go build ./..., tsc, etc.

3. If build fails:
   CompilerErrorAttributor.attribute(stderr, workspace_files)
   — Maps each compiler error to a specific file path

4. For each broken file:
   AgentManager._execute_lifecycle_phase(engine, file, FIXING)
   — CoderAgent receives the compiler error output in its prompt
   — Generates a repair

5. Rebuild up to max_retries (default 3)

6. If still failing after all retries:
   — Only files that transitively depend exclusively on broken files
     are blocked; other downstream files continue.
```

`CheckpointCycleResult` records: `passed`, `total_attempts`, `files_fixed[]`

---

## 9. Context Assembly

`core/context_builder.py`

For each agent invocation, `ContextBuilder.build(task)` assembles the minimal
relevant context to keep LLM prompts focused and token-efficient.

Priority ordering for included files:

| Priority | Source | Content |
|----------|--------|---------|
| 1 (highest) | Target file itself | Full source |
| 2 | Direct `depends_on` files | AST stubs (signatures only) |
| 3 | Semantic search hits | AST stubs (top 3 by cosine distance) |
| 4 | Module review scope | Full source (capped at 20 files) |

**AST stubs:** `core/ast_extractor.py` parses each dependency file using
`tree-sitter` and extracts only the interface-relevant parts:
- Imports
- Class names and inheritance
- Method signatures (name, parameters, return type)
- Field declarations
- Public API surface

This preserves what the LLM needs to know (what can be imported, what
methods exist, what types are exposed) while using far fewer tokens than
full source.

Total context is capped at `MAX_CONTEXT_CHARS = 120,000` characters across
all related files to prevent context overflow.

---

## 10. Memory Stores

Three complementary memory layers are maintained throughout execution:

### DependencyGraphStore (`memory/dependency_graph.py`)

A `networkx.DiGraph` where each node is a file path and each directed edge
`A → B` means file A imports from file B.

Used for:
- Building tier assignments in `TierScheduler`
- Computing impact of a changed file (which files will break?)
- Reactive re-verification via `EventBus`

### EmbeddingStore (`memory/embedding_store.py`)

A `ChromaDB` collection backed by `sentence-transformers/all-MiniLM-L6-v2`.

Used for:
- Semantic code search in `ContextBuilder` (find related files by meaning)
- Reducing context to the most relevant subset when > 3 semantically related files exist

Updated incrementally as each file is written by an agent.

### RepoIndex (`memory/repo_index.py`)

A structured catalog of all indexed files: path, language, imports, exports,
functions, classes, entry points.

Used for:
- `ContextBuilder` fast lookup of what is exported from each file
- `WorkspaceIndexer` rebuilds this after every run finalise step

---

## 11. EventBus

`core/event_bus.py`

Async publish/subscribe hub for cross-agent coordination.

Key events:

| Event | Published by | Subscribed by |
|-------|-------------|---------------|
| `FILE_WRITTEN` | `AgentManager` after any file write | `PipelineExecutor` re-verification handler |
| `REVIEW_PASSED` | `AgentManager` after review | Observability / logging |
| `REVIEW_FAILED` | `AgentManager` after review | Observability / logging |
| `TASK_COMPLETED` | `AgentManager` after any task success | Observability |
| `TASK_FAILED` | `AgentManager` / `PipelineExecutor` on build failure | Observability |
| `TEST_PASSED` | `AgentManager` after test success | Observability |
| `TEST_FAILED` | `AgentManager` after test failure | Observability |

The `FILE_WRITTEN` → re-verification handler queues dependent files for
re-check at the next checkpoint if they already passed their own generation.
This is the mechanism that propagates "an upstream file changed" to files
that may now have stale/broken imports.

Critical subscribers (like the re-verification handler) record failures
instead of swallowing them. `PipelineExecutor` checks `event_bus.has_failures()`
at the end of execution and includes them in the result.

---

## 12. Sandbox Execution

`core/sandbox_orchestrator.py`, `sandbox/sandbox_runner.py`

Two Docker containers are provisioned per run:

| Sandbox | Environment | Used for |
|---------|-------------|---------|
| `build_sandbox` | Full network access | `BuildVerifierAgent` (dependency downloads: `mvn`, `go mod download`, `npm install`) |
| `test_sandbox` | Network-isolated | `TestAgent`, `IntegrationTestAgent` (tests must not make external calls) |

Both containers:
- Are created from language-specific base images (`python:3.11-slim`, `maven:3.9-eclipse-temurin-21`, etc.)
- Mount the workspace directory read-write
- Have memory and CPU limits (`SANDBOX_MEMORY_LIMIT`, `SANDBOX_CPU_LIMIT`)
- Are torn down in the `finally` block of `RunPipeline.execute()`

**Local mode** (`--sandbox local`, `--allow-host-execution`): bypasses Docker
and runs commands directly on the host via `subprocess`. For development only — no isolation.

---

## 13. Observability

`core/observability.py`

- **OpenTelemetry**: spans emitted around each agent execution, phase boundary, and checkpoint run. OTLP exporter configured via `OTLP_ENDPOINT`.
- **Prometheus**: metrics exported on `PROMETHEUS_PORT` (default 9090):
  - `agent_tasks_total{agent, status}` — counter per agent per outcome
  - `agent_duration_seconds{agent}` — histogram of agent execution times
  - `checkpoint_attempts_total{tier}` — counter of build checkpoint attempts
  - `tokens_used_total{provider, direction}` — LLM token counters

Every agent execution calls `record_agent_start()` / `record_agent_end()` to
update the Prometheus histogram. Every agent populates `self._metrics` with
at minimum `llm_calls` and `tokens_used`.

---

## 14. Directory Map

```
multi-agent-claude/
├── agents/                   One file per specialised agent
│   ├── base_agent.py         Abstract base: execute(), system_prompt, tool dispatch
│   ├── architect_agent.py    Prompt → RepositoryBlueprint
│   ├── planner_agent.py      Blueprint → LifecycleEngine + TaskGraph
│   ├── coder_agent.py        GENERATE_FILE, FIX_CODE
│   ├── reviewer_agent.py     REVIEW_FILE, REVIEW_MODULE, REVIEW_ARCHITECTURE
│   ├── patch_agent.py        MODIFY_FILE (targeted diffs)
│   ├── build_verifier_agent.py  VERIFY_BUILD (compiler runner)
│   ├── test_agent.py         GENERATE_TEST (+ run + fix)
│   ├── security_agent.py     SECURITY_SCAN
│   ├── deploy_agent.py       GENERATE_DEPLOY
│   ├── writer_agent.py       GENERATE_DOCS
│   ├── integration_test_agent.py  GENERATE_INTEGRATION_TEST
│   ├── repository_analyzer_agent.py  ANALYZE_REPO (enhance mode)
│   ├── change_planner_agent.py  PLAN_CHANGES (enhance mode)
│   ├── product_planner_agent.py  PLAN_PRODUCT (fullstack mode)
│   ├── api_contract_agent.py    GENERATE_API_CONTRACT (fullstack)
│   ├── design_parser_agent.py   PARSE_DESIGN (Figma → layout)
│   ├── component_planner_agent.py  PLAN_COMPONENTS (frontend)
│   ├── component_dag_agent.py   BUILD_COMPONENT_DAG (frontend)
│   ├── component_generator_agent.py  GENERATE_COMPONENT (frontend)
│   ├── api_integration_agent.py  INTEGRATE_API (frontend)
│   └── state_management_agent.py  MANAGE_STATE (frontend)
│
├── core/                     Orchestration engine
│   ├── pipeline.py           Public facade
│   ├── pipeline_run.py       New-project workflow
│   ├── pipeline_enhance.py   Repository modification workflow
│   ├── pipeline_fullstack.py Full-stack workflow
│   ├── pipeline_frontend.py  Frontend half of fullstack
│   ├── pipeline_executor.py  Tier-based execution engine + checkpoints
│   ├── pipeline_definition.py  Declarative GENERATE_PIPELINE / ENHANCE_PIPELINE
│   ├── agent_manager.py      Agent factory + per-file lifecycle + metrics
│   ├── lifecycle_orchestrator.py  FSM event loop (delegates to AgentManager)
│   ├── state_machine.py      FileLifecycle FSM, FilePhase, EventType
│   ├── tier_scheduler.py     Dependency-tier computation
│   ├── task_engine.py        LifecyclePlanBuilder, TaskGraph, EnhanceLifecyclePlanBuilder
│   ├── task_dispatcher.py    Global DAG execution (security, deploy, docs)
│   ├── context_builder.py    Focused context assembly (AST stubs + semantic search)
│   ├── event_bus.py          Async publish/subscribe
│   ├── checkpoint.py         BuildCheckpoint, TestCheckpoint, CheckpointCycleResult
│   ├── error_attributor.py   CompilerErrorAttributor (error line → file mapping)
│   ├── stub_generator.py     Forward-reference stubs for next-tier compilation
│   ├── repository_manager.py Atomic writes, repo index, workspace snapshot
│   ├── sandbox_orchestrator.py  Docker container setup/teardown
│   ├── llm_client.py         Unified LLM client (Anthropic/OpenAI/Gemini)
│   ├── ast_extractor.py      tree-sitter AST → compact stubs
│   ├── language.py           LanguageProfile (build cmd, test cmd, extensions)
│   ├── observability.py      OpenTelemetry + Prometheus
│   ├── file_lock_manager.py  Per-file async locks (concurrent agent writes)
│   ├── coverage_runner.py    Test coverage reporting
│   ├── plan_approver.py      Interactive plan approval gate (enhance mode)
│   ├── run_reporter.py       Write run_report.json / modify_report.json
│   ├── workspace_indexer.py  Post-run workspace re-index
│   ├── workspace_snapshot.py Snapshot/rollback for enhance mode
│   ├── mcp_client.py         MCP tool server client (Figma integration)
│   ├── circuit_breaker.py    LLM call circuit breaker
│   ├── live_console.py       Rich live dashboard (FilePhase progress bars)
│   ├── import_validator.py   Import resolution validation
│   └── api.py                FastAPI HTTP server
│
├── memory/
│   ├── dependency_graph.py   networkx digraph (file → file deps)
│   ├── embedding_store.py    ChromaDB vector store + sentence-transformers
│   └── repo_index.py         Structural file catalog
│
├── config/
│   └── settings.py           Pydantic settings (reads from environment)
│
├── tools/
│   ├── terminal_tools.py     Run shell commands in sandbox or host
│   └── file_tools.py         apply_patch() (unified diff application)
│
├── sandbox/
│   └── sandbox_runner.py     Docker container lifecycle management
│
├── tests/                    pytest test suite (409 tests, 0 failures)
├── deploy/                   Docker Compose, Prometheus, Kubernetes manifests
└── workspace/                Default output directory for generated code
```

