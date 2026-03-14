# Getting Started

This guide walks you through installing, configuring, and running the
multi-agent code generation platform — for all three operating modes:
`generate` (new backend project), `enhance` (modify an existing repo), and
`fullstack` (backend + React/Next.js frontend).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Configuration](#3-configuration)
   - 3.1 [API Keys](#31-api-keys)
   - 3.2 [Environment File (.env)](#32-environment-file-env)
   - 3.3 [Full Environment Variable Reference](#33-full-environment-variable-reference)
4. [Generate — New Backend Project](#4-generate--new-backend-project)
5. [Enhance — Modify an Existing Repository](#5-enhance--modify-an-existing-repository)
6. [Fullstack — Backend + Frontend](#6-fullstack--backend--frontend)
   - 6.1 [Without Figma](#61-without-figma)
   - 6.2 [With Figma Import](#62-with-figma-import)
7. [HTTP API Server](#7-http-api-server)
8. [Running with Docker Compose](#8-running-with-docker-compose)
9. [Running the Test Suite](#9-running-the-test-suite)
10. [Verifying a Generated Workspace](#10-verifying-a-generated-workspace)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.11 | `python --version` to check |
| Docker | ≥ 24.0 | Required for sandbox isolation; skip with `--sandbox local` |
| Git | any | For cloning the repo |
| LLM API key | — | Anthropic (recommended), OpenAI, or Google Gemini |
| Node.js + npm | ≥ 18 | **Fullstack only** — for TypeScript compilation check (`tsc`) |
| Figma token | — | **Fullstack + Figma only** — optional |

> **Docker note:** The platform runs generated code (compilers, test suites)
> inside isolated Docker containers. If you skip Docker with `--sandbox local`
> the generated code runs directly on your machine — do not use this for
> untrusted prompts.

---

## 2. Installation

```bash
# 1. Clone
git clone <repo-url>
cd multi-agent-claude

# 2. Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install runtime dependencies
pip install -e .

# 4. (Optional) Install dev tools — required for running the test suite
pip install -e ".[dev]"
```

**Verify the CLI is installed:**

```bash
codegen --help
```

Expected output:

```
Usage: codegen [OPTIONS] COMMAND [ARGS]...

  Multi-Agent Backend Code Generator

Options:
  -v, --verbose  Enable debug logging
  --help         Show this message and exit.

Commands:
  enhance    Modify an existing project based on a natural language prompt.
  fullstack  Generate a complete fullstack project (backend + frontend)…
  generate   Generate a backend project from a natural language prompt.
  status     Show status of a workspace.
```

---

## 3. Configuration

### 3.1 API Keys

Set **one** of these before running any command:

```bash
# Anthropic Claude — recommended (highest code quality)
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# OpenAI GPT
export OPENAI_API_KEY="sk-proj-..."

# Google Gemini
export GEMINI_API_KEY="AIzaSy..."
```

On Windows (PowerShell):

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-api03-..."
```

---

### 3.2 Environment File (.env)

Create a `.env` file in the repo root — it is loaded automatically at startup:

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-api03-...
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514
WORKSPACE_DIR=workspace
SANDBOX_TYPE=docker
MAX_CONCURRENT_AGENTS=4
```

---

### 3.3 Full Environment Variable Reference

```bash
# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER=anthropic              # anthropic | openai | gemini
LLM_MODEL=claude-sonnet-4-20250514  # any model id supported by the provider
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
GEMINI_API_KEY=AIza...

# ── Workspace ─────────────────────────────────────────────────────────────────
WORKSPACE_DIR=workspace             # path to output directory (created if absent)

# ── Sandbox ───────────────────────────────────────────────────────────────────
SANDBOX_TYPE=docker                 # docker | local
                                    # local = no isolation, runs on host
                                    # docker = generated code runs in containers

# ── Execution ─────────────────────────────────────────────────────────────────
MAX_CONCURRENT_AGENTS=4             # 1–16 parallel agents per tier
BUILD_CHECKPOINT_RETRIES=3          # compile-and-fix cycles per tier
PHASE_TIMEOUT_SECONDS=600           # max wall-clock seconds per agent phase

# ── Memory / vector store ─────────────────────────────────────────────────────
CHROMA_PERSIST_DIR=.chroma          # ChromaDB on-disk persistence directory
EMBEDDING_MODEL=all-MiniLM-L6-v2    # sentence-transformers model for semantic search

# ── Observability ─────────────────────────────────────────────────────────────
PROMETHEUS_PORT=9090
OTLP_ENDPOINT=http://localhost:4317
ENABLE_TRACING=true                 # OpenTelemetry trace export

# ── Figma (fullstack mode only) ───────────────────────────────────────────────
FIGMA_TOKEN=figd_...
MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-figma"
```

---

## 4. Generate — New Backend Project

### Minimal example

```bash
codegen generate "Build a user management REST API with PostgreSQL and JWT auth"
```

Output appears in `workspace/` (created automatically).

### Specifying a language or framework

```bash
# Java / Spring Boot
codegen generate "Build an inventory management API with MySQL" \
  --provider anthropic

# Go / Gin
codegen generate "Build a URL shortener service with Redis" \
  --model claude-sonnet-4-20250514

# TypeScript / NestJS
codegen generate "Build a notifications microservice with WebSocket support"

# Python / FastAPI
codegen generate "Build a blog API with PostgreSQL and Alembic migrations"
```

The language is inferred from the prompt. If you want to force a specific
language, mention it explicitly: "using Go", "in Rust", "Spring Boot", etc.

### Custom output directory

```bash
codegen generate "Build a payment service" --workspace ./my-payment-service
```

### Using OpenAI or Gemini

```bash
export OPENAI_API_KEY="sk-proj-..."
codegen generate "Build a REST API" --provider openai --model gpt-4o

export GEMINI_API_KEY="AIza..."
codegen generate "Build a REST API" --provider gemini --model gemini-2.0-flash
```

### Skipping phases (for faster iteration)

```bash
# Skip reviewer only (faster, less safe)
codegen generate "Build a TODO API" --skip-reviewer

# Skip tests only
codegen generate "Build a TODO API" --skip-tester

# Skip both (minimal run — architecture + code only)
codegen generate "Build a TODO API" --skip-reviewer --skip-tester
```

### Running without Docker

```bash
# Use this only for development — no sandbox isolation
codegen generate "Build a TODO API" \
  --sandbox local \
  --allow-host-execution
```

> All available flags are listed in [README.md — CLI Reference](README.md#cli-reference).

### What gets generated

After a successful run, `workspace/` contains:

```
workspace/
  src/                  All source files (language-appropriate layout)
  tests/                Unit + integration tests
  Dockerfile            Multi-stage build
  docker-compose.yml    App + DB
  k8s/                  Kubernetes manifests
  security_report.json  OWASP vulnerability scan results
  docs/
    README.md           Project documentation (accurate to the code)
    API.md              Full API reference with request/response examples
    CHANGELOG.md
  run_report.json       Execution summary (timing, token cost, stats)
```

---

## 5. Enhance — Modify an Existing Repository

`enhance` analyses an existing codebase and applies targeted changes — it does
NOT rewrite the project from scratch.

### Basic usage

```bash
codegen enhance "Add rate limiting to all API endpoints" \
  --workspace ./my-existing-project
```

### With human approval of the change plan

Before any file is modified, the agent will display the proposed changes and
ask for confirmation:

```bash
codegen enhance "Migrate from SQLAlchemy 1.x to 2.x style" \
  --workspace ./my-project \
  --require-plan-approval
```

Output:

```
┌──────────────────────────────────────────────┐
│  Proposed Changes                            │
│                                              │
│  Modify: src/database.py                     │
│    Reason: Update Session usage to 2.x API  │
│  Modify: src/repositories/user_repo.py       │
│    Reason: Replace query() with select()     │
│  New:    tests/test_db_migration.py          │
│    Reason: Add migration smoke tests         │
│                                              │
│  Proceed? [y/N]                              │
└──────────────────────────────────────────────┘
```

### Example use cases

```bash
# Add a feature
codegen enhance "Add email verification on registration" \
  --workspace ./user-api

# Fix a bug family
codegen enhance "Fix all N+1 query issues in the repository layer" \
  --workspace ./my-api

# Security hardening
codegen enhance "Add input validation and sanitisation to all endpoints" \
  --workspace ./my-api

# Refactor
codegen enhance "Replace raw SQL with SQLAlchemy ORM" \
  --workspace ./legacy-api
```

> All available flags are listed in [README.md — CLI Reference](README.md#cli-reference).

---

## 6. Fullstack — Backend + Frontend

`fullstack` generates a complete project: a backend API (same as `generate`)
**and** a typed React/Next.js/Vue frontend — run in parallel.

### 6.1 Without Figma

```bash
codegen fullstack "Build a SaaS analytics dashboard with user management, \
  charts, and a subscription billing page"
```

The system will:
1. Plan product requirements (decides it needs frontend + backend)
2. Design the backend architecture
3. Generate an OpenAPI contract (FE/BE handshake)
4. Run the backend pipeline and frontend pipeline **simultaneously**

Output layout:

```
workspace/
  backend/
    src/              Backend source files
    tests/            Backend tests
    Dockerfile
    docker-compose.yml
    security_report.json
    docs/
  frontend/
    src/
      components/     React/Vue components (Atomic Design)
        atoms/
        molecules/
        organisms/
        pages/
      lib/
        api.ts        Typed API client (auto-generated from contract)
      services/       One service file per API tag
      hooks/          SWR / React-Query hooks
      store/          Zustand / Redux / Pinia stores
    package.json      Exact dependencies for npm install
    .env.local        NEXT_PUBLIC_API_URL etc.
  run_report.json
```

### Framework selection

The framework is inferred from the prompt. To be explicit:

```bash
codegen fullstack "Build a project manager with Next.js 14 App Router \
  and FastAPI backend"

codegen fullstack "Build a real-time chat app with Vue 3, Pinia, \
  and NestJS backend"

codegen fullstack "Build a dashboard with React, Zustand, and Spring Boot"
```

### 6.2 With Figma Import

When a Figma URL is provided the `DesignParserAgent` queries the live Figma
file via MCP to extract pages, component layout, typography tokens, and colour
palette — driving the generated component names and styles directly from your
design.

**Step 1 — Get a Figma personal access token**

In Figma: `Account Settings → Personal access tokens → Generate token`

**Step 2 — Configure MCP**

```bash
# Install the Figma MCP server (once)
npm install -g @modelcontextprotocol/server-figma

# Set environment variables
export FIGMA_TOKEN="figd_..."
export MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-figma"
```

Or add to `.env`:

```bash
FIGMA_TOKEN=figd_...
MCP_SERVER_COMMAND=npx -y @modelcontextprotocol/server-figma
```

**Step 3 — Run with the Figma URL**

```bash
codegen fullstack "Build a task tracker based on this design" \
  --figma-url "https://www.figma.com/file/YOUR_FILE_KEY/Your-Design-Name"
```

The `--figma-url` flag is also accepted without setting `MCP_SERVER_COMMAND` —
but in that case the Figma data is not fetched and you get LLM-inferred
components only.

> All available flags are listed in [README.md — CLI Reference](README.md#cli-reference).

---

## 7. HTTP API Server

Use the REST API to submit generation jobs from scripts, CI pipelines, or a
custom frontend.

### Start the server

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
uvicorn core.api:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: http://localhost:8000/docs

### Submit a `generate` job

```bash
JOB=$(curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Build a REST API with PostgreSQL",
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514"
  }' | python -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

echo "Job ID: $JOB"
```

### Submit a `fullstack` job

```bash
curl -s -X POST http://localhost:8000/fullstack \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Build a SaaS dashboard with React and FastAPI",
    "figma_url": ""
  }'
```

### Poll for status

```bash
curl http://localhost:8000/jobs/$JOB
```

```json
{
  "job_id": "abc123",
  "status": "running",
  "progress": 0.42,
  "current_phase": "FE: Component Generation"
}
```

### Stream logs (Server-Sent Events)

```bash
curl -N http://localhost:8000/jobs/$JOB/log
```

### Fetch the full report when complete

```bash
curl http://localhost:8000/jobs/$JOB/report
```

> Full endpoint table: [README.md — HTTP API Server](README.md#http-api-server).

---

## 8. Running with Docker Compose

The included `deploy/docker-compose.yml` starts the API server, Redis
(background job queue), Prometheus and Grafana.

```bash
# Copy the compose file to root (or run from deploy/)
cd deploy

# Set your API key (required by the compose environment)
export ANTHROPIC_API_KEY="sk-ant-..."

# Start all services
docker compose up -d

# Tail the API server logs
docker compose logs -f codegen
```

**Service URLs after startup:**

| Service | URL |
|---------|-----|
| API server (Swagger UI) | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

**Mount your own workspace:**

```yaml
# In docker-compose.yml, under the codegen service:
volumes:
  - ./my-output:/data/workspace
```

**Stopping:**

```bash
docker compose down          # stop containers
docker compose down -v       # stop + delete all volumes
```

---

## 9. Running the Test Suite

```bash
# Full suite (409 tests)
pytest tests/

# Stop on first failure with short traceback
pytest tests/ -x --tb=short

# Specific module
pytest tests/test_pipeline_executor.py -v

# With coverage report
pytest tests/ --cov=. --cov-report=term-missing

# Skip slow integration tests (if tagged)
pytest tests/ -m "not slow"
```

All LLM calls are mocked — no API key is needed to run the test suite.
No Docker daemon is needed — sandbox calls are mocked.

---

## 10. Verifying a Generated Workspace

After `generate` or `fullstack` completes, use the `status` command:

```bash
codegen status workspace/
```

Output:

```
         Workspace: workspace/
┌──────────────────────────┬────────┐
│ File                     │ Exists │
├──────────────────────────┼────────┤
│ architecture.md          │ Yes    │
│ file_blueprints.json     │ Yes    │
│ dependency_graph.json    │ Yes    │
│ repo_index.json          │ Yes    │
│ Source files             │ 18     │
│ Test files               │ 14     │
│ Deploy artifacts         │ 5      │
└──────────────────────────┴────────┘
```

Also check `workspace/run_report.json` for the full execution summary:

```json
{
  "success": true,
  "code_success": true,
  "tests_passed": 14,
  "tasks_completed": 18,
  "tasks_failed": 0,
  "elapsed_seconds": 187.4,
  "checkpoint_results": [...],
  "blueprint": { ... }
}
```

---

## 11. Troubleshooting

### `LLMConfigError: No API key found for provider 'anthropic'`

You have not set the API key environment variable. Set it and re-run:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or add it to `.env` in the repo root.

---

### `docker: command not found` / `Cannot connect to Docker daemon`

Docker is not installed or the daemon is not running.

- Install Docker Desktop from https://www.docker.com/products/docker-desktop
- Start Docker Desktop, then re-run
- Or use `--sandbox local --allow-host-execution` to bypass Docker (no isolation)

---

### `tsc: command not found` (fullstack mode only)

TypeScript compiler is not on `PATH`. The platform will skip the TS
compilation check and continue. To enable it:

```bash
npm install -g typescript
```

---

### Build checkpoint keeps failing for Java / Go

Increase the retry limit:

```bash
BUILD_CHECKPOINT_RETRIES=5 codegen generate "..."
```

Or increase the phase timeout for slow machines:

```bash
PHASE_TIMEOUT_SECONDS=900 codegen generate "..."
```

---

### Figma import returns empty `UIDesignSpec`

1. Verify `FIGMA_TOKEN` is set and has read access to the file
2. Verify `MCP_SERVER_COMMAND` is set (`npx -y @modelcontextprotocol/server-figma`)
3. Verify the Figma URL format: `https://www.figma.com/file/<FILE_KEY>/...`
4. Check logs for `Failed to initialize MCP client` — this usually means the MCP npm package is not installed

---

### `ChromaDB` / embedding model slow on first run

On the first run, `sentence-transformers` downloads `all-MiniLM-L6-v2`
(~22 MB). Subsequent runs use the local cache. Set `HF_HOME` to control
where the model is cached:

```bash
export HF_HOME=/path/to/fast/disk/.cache/huggingface
```

---

### Agent stuck / pipeline hangs

If a phase exceeds `PHASE_TIMEOUT_SECONDS` the file is marked `FAILED` and
the pipeline continues. If the whole run appears frozen:

1. Check the live dashboard (Rich progress bars) — it shows which phase each file is in
2. Use `--no-interactive` to get plain log output which is easier to inspect in CI
3. Reduce `--max-agents` to 1 to serialise execution and see clearer logs:

```bash
codegen generate "..." --max-agents 1 --no-interactive -v
```

---

### Running in CI / GitHub Actions

```yaml
- name: Generate project
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: |
    pip install -e .
    codegen generate "Build a REST API with tests" \
      --sandbox local \
      --allow-host-execution \
      --no-interactive \
      --skip-tester
```

Use `--sandbox local` in CI unless your runner has Docker available.
Use `--no-interactive` to disable the Rich live dashboard (not compatible with
non-TTY environments).

