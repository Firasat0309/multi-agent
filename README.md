# Multi-Agent Backend Code Generator

Autonomous backend code generation platform powered by multiple LLMs.

---

## Prerequisites

### Python
- Python **3.11 or higher**
- Verify: `python --version`

### API Keys
At least one of the following:

| Provider | Environment Variable | Where to get it |
|----------|---------------------|-----------------|
| Anthropic (default) | `ANTHROPIC_API_KEY` | https://console.anthropic.com |
| OpenAI | `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| Google Gemini | `GEMINI_API_KEY` | https://aistudio.google.com/app/apikey |

### Optional Services
| Service | Purpose | Required |
|---------|---------|----------|
| Docker | Sandbox code execution | No (defaults to local) |
| Redis | Job queue for API server | No |
| Prometheus + Grafana | Metrics and monitoring | No |

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd multi-agent-claude

# 2. Install dependencies
pip install -e .

# 3. Install dev dependencies (for running tests)
pip install -e ".[dev]"
```

---

## Running with Anthropic (Claude)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

# Default model (claude-sonnet-4-20250514)
codegen generate "Build a user management REST API with PostgreSQL and JWT auth"

# Opus — most capable
codegen generate "Build a user management REST API" --provider anthropic --model claude-opus-4-6

# Haiku — fastest
codegen generate "Build a user management REST API" --provider anthropic --model claude-haiku-4-5-20251001
```

---

## Running with OpenAI (GPT)

```bash
export OPENAI_API_KEY="sk-proj-..."

# GPT-4o
codegen generate "Build a user management REST API" --provider openai --model gpt-4o

# GPT-4o mini — faster and cheaper
codegen generate "Build a user management REST API" --provider openai --model gpt-4o-mini
```

---

## Running with Google Gemini

```bash
export GEMINI_API_KEY="AIza..."

# Gemini 2.0 Flash — recommended
codegen generate "Build a user management REST API" --provider gemini --model gemini-2.0-flash

# Gemini 1.5 Pro — higher quality
codegen generate "Build a user management REST API" --provider gemini --model gemini-1.5-pro

# Gemini 1.5 Flash — faster
codegen generate "Build a user management REST API" --provider gemini --model gemini-1.5-flash
```

---

## All CLI Options

```bash
codegen generate "your prompt" \
  --provider anthropic   # anthropic | openai | gemini \
  --model    claude-sonnet-4-20250514 \
  --workspace ./output   # output directory \
  --sandbox  local       # local | docker \
  --max-agents 4         # concurrent agents (1-16)
```

---

## Running via API Server

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

# Start the server
uvicorn core.api:app --host 0.0.0.0 --port 8000

# Submit a job
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Build a REST API with PostgreSQL", "provider": "anthropic"}'

# Poll job status
curl http://localhost:8000/jobs/<job-id>
```

---

## Running Tests

```bash
pytest tests/
```

---

## Using Environment Variables Instead of Flags

```bash
export LLM_PROVIDER=gemini
export LLM_MODEL=gemini-2.0-flash
export GEMINI_API_KEY="AIza..."
export WORKSPACE_DIR=./my-output
export SANDBOX_TYPE=local
export MAX_CONCURRENT_AGENTS=4

codegen generate "Build a payment service"
```
