# Getting Started

Multi-agent code generation platform powered by Claude (and optionally OpenAI / Gemini).
Generates or modifies full repositories from a plain-English prompt using a pipeline of
specialised AI agents running inside isolated Docker sandboxes.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Configuration](#3-configuration)
4. [Running with CLI](#4-running-with-cli)
5. [Running the API Server](#5-running-the-api-server)
6. [Running with Docker Compose (full stack)](#6-running-with-docker-compose-full-stack)
7. [Verifying the Setup](#7-verifying-the-setup)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

### 1.1 Python 3.11+

The project requires **Python 3.11 or newer**.

```bash
# Check your version
python --version          # or python3 --version

# Install on Ubuntu/Debian
sudo apt install python3.11 python3.11-venv python3.11-dev

# Install on macOS (Homebrew)
brew install python@3.11

# Install on Windows
# Download from https://www.python.org/downloads/
# Tick "Add to PATH" during setup
```

### 1.2 Docker

Docker is required for the sandboxed code execution environment (build and test tiers).
Without Docker, you can still run in `local` sandbox mode but code runs without isolation.

```bash
# Ubuntu/Debian
sudo apt install docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER   # run Docker without sudo (re-login after)

# macOS — install Docker Desktop from https://www.docker.com/products/docker-desktop/

# Windows — install Docker Desktop from https://www.docker.com/products/docker-desktop/
#   Enable WSL2 backend for best performance

# Verify
docker --version                # Docker version 25.x.x or newer
docker run --rm hello-world     # should print "Hello from Docker!"
```

### 1.3 Git

```bash
git --version    # any modern version is fine
```

### 1.4 LLM API Key (at least one)

| Provider | Environment Variable | Get Key |
|----------|---------------------|---------|
| Anthropic (Claude) — **recommended** | `ANTHROPIC_API_KEY` | https://console.anthropic.com |
| OpenAI | `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| Google Gemini | `GEMINI_API_KEY` | https://aistudio.google.com/app/apikey |

### 1.5 ChromaDB (Vector Database)

ChromaDB runs **in-process** — no separate server is needed. It is installed automatically
as a Python package and persists its data to a local directory (default: `.chroma/`).

> No extra installation steps are required for ChromaDB.

### 1.6 Redis (optional — API server job queue only)

Redis is only needed when running the **HTTP API server** for background job management.
It is **not** needed for CLI usage.

```bash
# Ubuntu/Debian
sudo apt install redis-server
sudo systemctl enable --now redis-server

# macOS
brew install redis
brew services start redis

# Windows — use the official MSI or run via Docker:
docker run -d -p 6379:6379 redis:7-alpine

# Verify
redis-cli ping    # should return PONG
```

---

## 2. Installation

### 2.1 Clone the repository

```bash
git clone <repository-url>
cd multi-agent-claude
```

### 2.2 Create and activate a virtual environment

```bash
# Create
python -m venv .venv

# Activate — Linux / macOS
source .venv/bin/activate

# Activate — Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate — Windows (CMD)
.venv\Scripts\activate.bat
```

### 2.3 Install the package and all dependencies

```bash
# Runtime dependencies + CLI entry-point
pip install -e .

# Include dev dependencies (linter, type checker, tests)
pip install -e ".[dev]"
```

This installs:

| Package | Purpose |
|---------|---------|
| `anthropic` | Claude LLM client |
| `openai` | OpenAI LLM client (optional) |
| `google-generativeai` | Gemini LLM client (optional) |
| `chromadb` | In-process vector database for semantic code search |
| `sentence-transformers` | Embedding model (`all-MiniLM-L6-v2`) used by ChromaDB |
| `docker` | Python SDK for Docker sandbox management |
| `networkx` | Dependency graph traversal |
| `fastapi` + `uvicorn` | HTTP API server |
| `redis` | Background job queue (API mode) |
| `rich` + `click` | CLI formatting and argument parsing |
| `tree-sitter` | AST parsing for code stubs |
| `bandit` | Security scanning agent |
| `prometheus-client` | Metrics export |
| `opentelemetry-*` | Distributed tracing |
| `pydantic` | Settings and data validation |

### 2.4 Pull the Docker sandbox image (recommended)

The sandbox agents execute generated code inside Docker containers.
Pulling the base image in advance avoids a delay on the first run:

```bash
docker pull python:3.11-slim
```

---

## 3. Configuration

All settings are read from **environment variables**. Copy the block below into a `.env`
file in the project root (or export them in your shell).

```bash
# ── LLM Provider ──────────────────────────────────────────────────────────────
# Choose one provider; set the matching API key.
LLM_PROVIDER=anthropic                        # anthropic | openai | gemini
LLM_MODEL=claude-sonnet-4-20250514            # model ID for the chosen provider

ANTHROPIC_API_KEY=sk-ant-...                  # required if LLM_PROVIDER=anthropic
OPENAI_API_KEY=sk-...                         # required if LLM_PROVIDER=openai
GEMINI_API_KEY=AI...                          # required if LLM_PROVIDER=gemini

# ── Workspace ─────────────────────────────────────────────────────────────────
WORKSPACE_DIR=workspace                       # output directory for generated code

# ── Sandbox ───────────────────────────────────────────────────────────────────
SANDBOX_TYPE=docker                           # docker | local
SANDBOX_MEMORY_LIMIT=512m                     # memory cap per container
SANDBOX_CPU_LIMIT=1.0                         # CPU cores per container
SANDBOX_TIMEOUT=300                           # seconds before a task is killed

# ── Agent concurrency ─────────────────────────────────────────────────────────
MAX_CONCURRENT_AGENTS=4                       # 1–16

# ── Vector DB (ChromaDB) ──────────────────────────────────────────────────────
CHROMA_PERSIST_DIR=.chroma                    # directory for vector store data
EMBEDDING_MODEL=all-MiniLM-L6-v2             # sentence-transformers model

# ── Context limits ────────────────────────────────────────────────────────────
MAX_CONTEXT_TOKENS=6000
MAX_RELATED_FILES=10

# ── Observability ─────────────────────────────────────────────────────────────
PROMETHEUS_PORT=9090
OTLP_ENDPOINT=http://localhost:4317           # OpenTelemetry collector endpoint
# Set to false to disable tracing (useful in dev)
# ENABLE_TRACING=false
```

Load a `.env` file automatically (Linux / macOS / Git Bash):

```bash
set -a && source .env && set +a
```

Or use [python-dotenv](https://pypi.org/project/python-dotenv/) if you prefer automatic loading.

### Model IDs reference

| Provider | Recommended Model ID |
|----------|---------------------|
| Anthropic | `claude-sonnet-4-20250514` |
| OpenAI | `gpt-4o` |
| Gemini | `gemini-1.5-pro` |

---

## 4. Running with CLI

The `codegen` command is installed as a script entry-point when you run `pip install -e .`.

### Generate a new project from scratch

```bash
codegen generate "Build a user management REST API with JWT auth, PostgreSQL, and pytest tests"
```

With all options explicit:

```bash
codegen generate "Build a user management REST API" \
  --workspace ./my-project \
  --provider anthropic \
  --model claude-sonnet-4-20250514 \
  --sandbox docker \
  --max-agents 4
```

Run without Docker (development / quick testing only — no isolation):

```bash
codegen generate "Hello world FastAPI app" \
  --sandbox local \
  --allow-host-execution
```

Non-interactive mode (no live dashboard — useful in CI):

```bash
codegen generate "Build a REST API" --no-interactive
```

### Enhance (modify) an existing project

```bash
codegen enhance "Add password reset via email and rate limiting" \
  --workspace ./my-project
```

Pause for human approval before executing the change plan:

```bash
codegen enhance "Refactor database layer to use SQLAlchemy 2.0" \
  --workspace ./my-project \
  --require-plan-approval
```

### Check workspace status

```bash
codegen status                  # checks default workspace
codegen status ./my-project     # checks a specific workspace
```

### Global options

| Flag | Description |
|------|-------------|
| `-v, --verbose` | Enable DEBUG-level logging |
| `--workspace PATH` | Directory to write generated code (default: `workspace`) |
| `--provider TEXT` | LLM provider: `anthropic` \| `openai` \| `gemini` |
| `--model TEXT` | Model ID (must match provider) |
| `--sandbox TEXT` | Sandbox type: `docker` \| `local` |
| `--max-agents INT` | Parallel agent limit (1–16, default 4) |
| `--no-interactive` | Disable live console dashboard |
| `--allow-host-execution` | Skip Docker requirement (local mode) |
| `--require-plan-approval` | Pause for approval before enhancement runs |

---

## 5. Running the API Server

The HTTP API allows you to submit jobs, poll their status, and stream logs.
Redis must be running (see [Prerequisites](#16-redis-optional--api-server-job-queue-only)).

### Start the server

```bash
uvicorn core.api:app --host 0.0.0.0 --port 8000
```

With auto-reload for development:

```bash
uvicorn core.api:app --host 0.0.0.0 --port 8000 --reload
```

The server is now available at `http://localhost:8000`.
Interactive docs (Swagger UI): `http://localhost:8000/docs`

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generate` | Start a new code generation job |
| `POST` | `/enhance` | Start a repository enhancement job |
| `GET` | `/jobs/{job_id}` | Poll job status |
| `GET` | `/jobs/{job_id}/log` | Stream job logs (Server-Sent Events) |
| `GET` | `/jobs/{job_id}/report` | Fetch the final run/modify report |
| `DELETE` | `/jobs/{job_id}` | Cancel a running job |
| `GET` | `/jobs` | List all jobs |
| `GET` | `/health` | Health check |

### Example: Generate a project via API

```bash
# Submit job
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Build a REST API for a todo app with SQLite",
    "workspace": "workspace",
    "model": "claude-sonnet-4-20250514",
    "max_agents": 4
  }'
# Response: { "job_id": "abc-123", "status": "queued", ... }

# Poll status
curl http://localhost:8000/jobs/abc-123

# Stream logs (Server-Sent Events — keep connection open)
curl -N http://localhost:8000/jobs/abc-123/log

# Fetch final report once status == "completed"
curl http://localhost:8000/jobs/abc-123/report
```

### Example: Enhance a project via API

```bash
curl -X POST http://localhost:8000/enhance \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Add Redis caching to all GET endpoints",
    "workspace": "workspace",
    "model": "claude-sonnet-4-20250514",
    "require_plan_approval": false
  }'
```

---

## 6. Running with Docker Compose (full stack)

Docker Compose spins up the app, Redis, Prometheus, and Grafana together.

```bash
cd deploy/

# Set your API key (required)
export ANTHROPIC_API_KEY=sk-ant-...

# Start all services
docker compose up -d

# Watch logs
docker compose logs -f codegen

# Stop everything
docker compose down
```

Services started:

| Service | URL | Description |
|---------|-----|-------------|
| **codegen** API | http://localhost:8000 | Main application (FastAPI) |
| **Redis** | localhost:6379 | Job queue |
| **Prometheus** | http://localhost:9090 | Metrics scraping |
| **Grafana** | http://localhost:3000 | Dashboards (admin / admin) |

> The `codegen` container mounts `/var/run/docker.sock` so it can spin up
> child Docker sandboxes from inside the container (Docker-in-Docker).

---

## 7. Verifying the Setup

```bash
# 1. Python version
python --version                            # Python 3.11.x or newer

# 2. CLI installed
codegen --help

# 3. Docker available
docker ps

# 4. API key works (Anthropic example)
python -c "
import anthropic, os
c = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
print(c.messages.create(model='claude-haiku-4-5-20251001', max_tokens=10,
    messages=[{'role':'user','content':'ping'}]).content[0].text)
"

# 5. ChromaDB in-process
python -c "import chromadb; print('chromadb OK', chromadb.__version__)"

# 6. API server health (if running)
curl http://localhost:8000/health

# 7. Run the test suite
pytest tests/ -v
```

---

## 8. Troubleshooting

### `codegen: command not found`

The virtual environment is not active or the package is not installed.

```bash
source .venv/bin/activate     # activate venv
pip install -e .              # reinstall
```

### `docker: permission denied`

Add your user to the `docker` group and re-login:

```bash
sudo usermod -aG docker $USER
# Log out and log back in, then:
docker ps
```

### `LLMConfigError: ANTHROPIC_API_KEY not set`

Export the key before running:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
codegen generate "..."
```

### ChromaDB / embedding model slow on first run

The `all-MiniLM-L6-v2` model (~90 MB) is downloaded from HuggingFace on the first
run and cached locally. Subsequent runs are instant. Ensure you have internet access
on first launch.

### `SandboxUnavailableError: Docker not available`

Either Docker is not running or the socket is not accessible. Start Docker and verify:

```bash
sudo systemctl start docker
docker ps
```

Or bypass Docker with local sandbox (no isolation — development only):

```bash
codegen generate "..." --sandbox local --allow-host-execution
```

### Redis connection refused (API server)

Start Redis before launching the API server:

```bash
sudo systemctl start redis-server    # Linux
brew services start redis             # macOS
# or via Docker:
docker run -d -p 6379:6379 redis:7-alpine
```

### Slow first run (Docker image pull)

Pull the sandbox image in advance:

```bash
docker pull python:3.11-slim
```

### Out of memory during generation

Reduce concurrency or increase the container memory limit:

```bash
MAX_CONCURRENT_AGENTS=2 SANDBOX_MEMORY_LIMIT=1g codegen generate "..."
```
