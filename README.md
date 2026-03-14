# Multi-Agent Code Generator

Production-grade, autonomous code generation platform powered by a coordinated
pipeline of specialised AI agents.

Give it a plain-English prompt:

```
Build a user management REST API with PostgreSQL and JWT authentication
```

and it **designs the architecture → generates every source file → reviews and
fixes each one → runs build verification → generates tests → performs a security
scan → outputs Dockerfile + Kubernetes manifests** — all without human
intervention.

Supports **Python · Java / Spring Boot · Go · TypeScript / Node · Rust · C# /
.NET** and all three major LLM providers (**Anthropic Claude · OpenAI GPT ·
Google Gemini**).

---

## What It Generates

**Backend (`generate` / `enhance` modes)**

| Artifact | Description |
|----------|-------------|
| Source files | Every file in the designed architecture, layered correctly |
| Unit tests | ≥ 3 tests per public method, framework-idiomatic |
| Integration tests | End-to-end request → service → DB roundtrip tests |
| Security report | OWASP-aligned scan (SQL injection, XSS, SSRF, auth, crypto …) |
| Dockerfile | Multi-stage, non-root, with health checks |
| docker-compose.yml | App + DB services wired together |
| Kubernetes manifests | Deployment + Service + ConfigMap + liveness probes |
| README + API docs | Accurate to the generated code, not generic templates |

**Frontend (`fullstack` mode — React / Next.js)**

| Artifact | Description |
|----------|-------------|
| OpenAPI contract | Typed API handshake shared by backend and frontend |
| UI component tree | Atomic-design components derived from Figma or prose description |
| Typed API client | Auto-generated fetch/axios services + SWR/React-Query hooks |
| State stores | Zustand / Redux / Pinia / Context slice per feature |
| Routing | React Router or Next.js App Router pages wired to components |
| package.json | Exact dependency list matching the generated framework stack |
| Figma import | Optional — read live Figma layout via MCP to drive component names and styles |

---

## Supported Languages & Frameworks

**Backend**

| Language | Frameworks / Toolchain |
|----------|----------------------|
| Python | FastAPI, Flask, Django; pytest; pip / uv |
| Java | Spring Boot; JUnit 5; Maven / Gradle |
| Go | Gin, Chi, standard library; `go test`; `go mod` |
| TypeScript / Node | Express, Fastify, NestJS; Jest; npm / pnpm |
| Rust | Actix-web, Axum; cargo test |
| C# / .NET | ASP.NET Core; xUnit; dotnet |

**Frontend (fullstack mode)**

| Framework | State management | Styling |
|-----------|------------------|---------|
| React (Vite) | Zustand · Redux Toolkit · Context | Tailwind · CSS Modules · styled-components · MUI |
| Next.js (App Router) | Zustand · Redux Toolkit · Context | Tailwind · CSS Modules · MUI |
| Vue 3 | Pinia · Vuex | Tailwind · CSS Modules |
| Angular | NgRx · Services | Angular Material · Tailwind |

---

## Quick Start

### 1  Install

```bash
git clone <repo-url>
cd multi-agent-claude
pip install -e .           # runtime + CLI
pip install -e ".[dev]"    # + pytest, mypy, ruff
```

### 2  Set your API key

```bash
# Anthropic (recommended — highest code quality)
export ANTHROPIC_API_KEY="sk-ant-..."

# or OpenAI
export OPENAI_API_KEY="sk-proj-..."

# or Google Gemini
export GEMINI_API_KEY="AIza..."
```

### 3  Generate a project

```bash
codegen generate "Build a user management REST API with PostgreSQL and JWT auth"
```

Output appears in `workspace/` by default.

---

## Operating Modes

### `generate` — create a new project from scratch

```bash
codegen generate "Build a payment service with Stripe webhooks"
```

### `enhance` — modify an existing repository

```bash
codegen enhance "Add rate limiting and Redis caching to all endpoints" \
  --workspace ./my-existing-project
```

### `fullstack` — backend + React/Next.js frontend

```bash
codegen fullstack "Build a SaaS analytics dashboard with user management"

# Optionally import UI layouts from a Figma file
export FIGMA_TOKEN="figd_..."
export MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-figma"
codegen fullstack "Build a task tracker" \
  --figma-url "https://www.figma.com/file/YOUR_KEY/Design"
```

---

## CLI Reference

```
codegen generate  PROMPT [options]
codegen enhance   PROMPT [options]
codegen fullstack PROMPT [options]
codegen status    [WORKSPACE]

Options (all modes):
  --workspace   PATH    Output directory               default: workspace/
  --provider    TEXT    anthropic | openai | gemini    default: anthropic
  --model       TEXT    LLM model id                   default: claude-sonnet-4-20250514
  --sandbox     TEXT    docker | local                 default: docker
  --max-agents  INT     Concurrent agents (1–16)       default: 4
  --no-interactive      Disable live dashboard (CI mode)
  --skip-tester         Skip test generation
  --skip-reviewer       Skip code review
  --allow-host-execution  Run without Docker isolation

enhance only:
  --require-plan-approval   Pause for human approval before applying changes

fullstack only:
  --figma-url   URL     Figma file URL for design import
```

---

## HTTP API Server

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
uvicorn core.api:app --host 0.0.0.0 --port 8000
# Swagger UI → http://localhost:8000/docs
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate` | Start a new generation job |
| `POST` | `/enhance` | Start a repository enhancement job |
| `GET` | `/jobs/{id}` | Poll job status |
| `GET` | `/jobs/{id}/log` | Stream logs (Server-Sent Events) |
| `GET` | `/jobs/{id}/report` | Fetch `run_report.json` |
| `DELETE` | `/jobs/{id}` | Cancel a running job |
| `GET` | `/jobs` | List all jobs |
| `GET` | `/health` | Health check |

```bash
# Example: submit a job and poll
JOB=$(curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Build a REST API with PostgreSQL"}' | jq -r .job_id)

curl http://localhost:8000/jobs/$JOB
```

---

## Configuration

All settings are read from environment variables or a `.env` file in the repo
root. See [GETTING_STARTED.md — Configuration](GETTING_STARTED.md#3-configuration)
for the full reference including sandbox, memory, observability, and Figma settings.

---

## Project Layout

```
agents/                  One Python file per specialised agent

  ── Backend / core agents ──────────────────────────────────────────────────
  architect_agent.py           Prompt → RepositoryBlueprint
  planner_agent.py             Blueprint → LifecycleEngine + TaskGraph
  coder_agent.py               Generate a source file
  reviewer_agent.py            Review code; emit pass/fail + findings
  patch_agent.py               Targeted diff patches (enhance mode)
  build_verifier_agent.py      Compile the workspace; attribute errors
  test_agent.py                Generate + run tests; fix failures
  security_agent.py            OWASP vulnerability scan
  deploy_agent.py              Dockerfile + K8s manifests
  writer_agent.py              README, API docs, CHANGELOG
  integration_test_agent.py    End-to-end test generation
  repository_analyzer_agent.py Scan existing repo (enhance mode)
  change_planner_agent.py      Produce targeted change plan (enhance mode)

  ── Frontend agents (fullstack mode) ───────────────────────────────────────
  product_planner_agent.py     Prompt → ProductRequirements (has_frontend, has_backend, tech_preferences)
  api_contract_agent.py        BE blueprint → OpenAPI 3.x APIContract (FE/BE handshake)
  design_parser_agent.py       Figma URL or prose → UIDesignSpec (pages, styles, layout)
  component_planner_agent.py   UIDesignSpec → ComponentPlan (Atomic Design component list)
  component_dag_agent.py       ComponentPlan → topologically sorted build tiers (no LLM call)
  component_generator_agent.py Generate source code for one UI component per tier
  api_integration_agent.py     Generate typed API client (fetch/axios services + SWR hooks)
  state_management_agent.py    Generate Zustand / Redux / Pinia / Context store layer

core/                    Orchestration engine
  pipeline.py            Public facade (Pipeline.run / .enhance / .run_fullstack)
  pipeline_run.py        New-project workflow (4 phases)
  pipeline_enhance.py    Modification workflow (5 phases)
  pipeline_fullstack.py  Fullstack workflow — concurrent BE + FE execution
  pipeline_frontend.py   Frontend-only pipeline (6 phases)
  pipeline_executor.py   Tier-based execution engine with build checkpoints
  pipeline_definition.py  Declarative generate / enhance pipeline specs
  agent_manager.py       Agent factory + per-file lifecycle execution
  lifecycle_orchestrator.py  FSM event loop + global DAG hand-off
  state_machine.py       Per-file FileLifecycle state machine (FilePhase + EventType)
  tier_scheduler.py      Group files into dependency tiers
  task_engine.py         Lifecycle plan builder + global TaskGraph
  task_dispatcher.py     Global DAG execution (security, deploy, docs)
  context_builder.py     Assemble focused context for each agent (AST stubs + embeddings)
  event_bus.py           Async publish/subscribe for cross-agent events
  error_attributor.py    Map compiler errors to source files
  checkpoint.py          Build and test checkpoint logic
  repository_manager.py  Atomic file writes, repo index, snapshot
  sandbox_orchestrator.py  Docker sandbox setup and teardown
  llm_client.py          Unified LLM client (Anthropic / OpenAI / Gemini)
  observability.py       OpenTelemetry tracing + Prometheus metrics

memory/
  dependency_graph.py    File-to-file dependency store (used by ContextBuilder)
  embedding_store.py     ChromaDB vector store (semantic code search)
  repo_index.py          Structural catalog (imports, exports, classes, functions)

config/
  settings.py            Pydantic settings dataclass — reads from environment

tools/
  terminal_tools.py      Run shell commands in sandbox or host
  file_tools.py          Patch application (unified diff)

tests/                   pytest suite — 409 tests, 0 failures
deploy/                  Docker Compose, Prometheus, Kubernetes manifests
```

---

## How It Works

See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete description of every
agent, pipeline flows (generate / enhance / fullstack), the per-file state
machine, tier-based scheduling, context assembly, memory stores, and sandbox
execution.

---

## License

MIT

