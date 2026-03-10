"""FastAPI server providing HTTP interface to the code generation pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.settings import Settings, get_settings
from core.pipeline import Pipeline, PipelineResult

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Agent Code Generator API",
    description="Autonomous backend code generation platform",
    version="1.0.0",
)


# ── Request / response models ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=10, description="Natural language project description")
    workspace: str = Field(default="workspace", description="Output directory name")
    model: str = Field(default="claude-sonnet-4-20250514", description="LLM model to use")
    max_agents: int = Field(default=4, ge=1, le=16, description="Max concurrent agents")
    allow_host_execution: bool = Field(default=False)


class EnhanceRequest(BaseModel):
    prompt: str = Field(..., min_length=10, description="Natural language change description")
    workspace: str = Field(..., description="Path to the existing workspace to modify")
    model: str = Field(default="claude-sonnet-4-20250514", description="LLM model to use")
    max_agents: int = Field(default=4, ge=1, le=16)
    require_plan_approval: bool = Field(
        default=False,
        description="When True, write pending_plan.json and wait for approval before executing",
    )


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str          # queued | running | completed | failed | cancelled
    phase: str = ""
    progress: float = 0.0
    elapsed_seconds: float = 0.0
    created_at: str = ""
    result: dict[str, Any] | None = None
    errors: list[str] = []


# ── In-memory job store ───────────────────────────────────────────────────────
# Use Redis / Postgres in multi-process deployments.

class _Job:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.status: str = "queued"
        self.phase: str = ""
        self.progress: float = 0.0
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.result: dict[str, Any] | None = None
        self.errors: list[str] = []
        self.elapsed_seconds: float = 0.0
        self._cancel_event: asyncio.Event = asyncio.Event()
        # Path to workspace for log streaming
        self.workspace: Path | None = None

    def to_status(self) -> JobStatus:
        return JobStatus(
            job_id=self.job_id,
            status=self.status,
            phase=self.phase,
            progress=self.progress,
            elapsed_seconds=round(self.elapsed_seconds, 1),
            created_at=self.created_at,
            result=self.result,
            errors=self.errors,
        )


_jobs: dict[str, _Job] = {}


# ── Helper ────────────────────────────────────────────────────────────────────

def _build_settings(request: GenerateRequest | EnhanceRequest, workspace_path: Path) -> Settings:
    base = get_settings()
    kw: dict[str, Any] = dict(
        workspace_dir=workspace_path,
        llm=base.llm,
        sandbox=base.sandbox,
        memory=base.memory,
        observability=base.observability,
        max_concurrent_agents=request.max_agents,
        allow_host_execution=request.allow_host_execution
        if hasattr(request, "allow_host_execution")
        else base.allow_host_execution,
    )
    if isinstance(request, EnhanceRequest):
        kw["require_plan_approval"] = request.require_plan_approval
    return Settings(**kw)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/generate", response_model=JobResponse)
async def generate(
    request: GenerateRequest, background_tasks: BackgroundTasks
) -> JobResponse:
    """Start a new code generation job.  Returns immediately with a ``job_id``."""
    job_id = str(uuid.uuid4())[:8]
    job = _Job(job_id)
    workspace = Path(request.workspace) / job_id
    job.workspace = workspace
    _jobs[job_id] = job

    background_tasks.add_task(_run_generate, job, request)
    return JobResponse(
        job_id=job_id,
        status="queued",
        message=f"Job {job_id} queued. Poll GET /jobs/{job_id} for status.",
    )


@app.post("/enhance", response_model=JobResponse)
async def enhance(
    request: EnhanceRequest, background_tasks: BackgroundTasks
) -> JobResponse:
    """Start a repository enhancement job."""
    job_id = str(uuid.uuid4())[:8]
    job = _Job(job_id)
    workspace = Path(request.workspace)
    job.workspace = workspace
    _jobs[job_id] = job

    background_tasks.add_task(_run_enhance, job, request)
    return JobResponse(
        job_id=job_id,
        status="queued",
        message=f"Job {job_id} queued. Poll GET /jobs/{job_id} for status.",
    )


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str) -> JobStatus:
    """Poll the status and result of a job."""
    job = _resolve_job(job_id)
    return job.to_status()


@app.get("/jobs/{job_id}/log")
async def stream_log(job_id: str) -> StreamingResponse:
    """Stream workspace/run.log in real-time using Server-Sent Events.

    Clients should connect with ``Accept: text/event-stream``.
    The stream closes automatically when the job reaches a terminal state.
    """
    job = _resolve_job(job_id)

    async def _event_stream() -> AsyncIterator[str]:
        log_path = (job.workspace or Path(".")) / "run.log"
        # Wait up to 10 s for the log file to appear (job may just be starting)
        for _ in range(100):
            if log_path.exists():
                break
            await asyncio.sleep(0.1)

        position = 0
        terminal_states = {"completed", "failed", "cancelled"}

        while True:
            if log_path.exists():
                try:
                    text = log_path.read_text(encoding="utf-8", errors="replace")
                    if len(text) > position:
                        new_text = text[position:]
                        position = len(text)
                        for line in new_text.splitlines():
                            yield f"data: {json.dumps({'line': line})}\n\n"
                except OSError:
                    pass

            if job.status in terminal_states:
                yield "data: {\"line\": \"[stream] Job finished.\"}\n\n"
                yield "event: done\ndata: {}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/jobs/{job_id}/report")
async def get_report(job_id: str) -> dict[str, Any]:
    """Return the run_report.json (generation) or modify_report.json (enhancement)."""
    job = _resolve_job(job_id)
    if job.status not in ("completed", "failed"):
        raise HTTPException(status_code=409, detail="Job not yet finished")

    workspace = job.workspace or Path(".")
    for name in ("run_report.json", "modify_report.json"):
        path = workspace / name
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Could not read report: {e}")

    raise HTTPException(status_code=404, detail="Report file not found")


@app.delete("/jobs/{job_id}", response_model=JobResponse)
async def cancel_job(job_id: str) -> JobResponse:
    """Request cancellation of a running job.

    The job may not stop immediately — it will transition to ``cancelled``
    when the current pipeline phase completes.
    """
    job = _resolve_job(job_id)
    if job.status not in ("queued", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel job in state '{job.status}'",
        )
    job._cancel_event.set()
    job.status = "cancelled"
    return JobResponse(
        job_id=job_id,
        status="cancelled",
        message=f"Cancellation requested for job {job_id}",
    )


@app.get("/jobs", response_model=list[JobStatus])
async def list_jobs() -> list[JobStatus]:
    """List all known jobs (most recent first)."""
    return [j.to_status() for j in reversed(list(_jobs.values()))]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "jobs": str(len(_jobs))}


# ── Background runners ────────────────────────────────────────────────────────

async def _run_generate(job: _Job, request: GenerateRequest) -> None:
    import time
    t0 = time.monotonic()
    job.status = "running"
    job.phase = "initializing"

    workspace = job.workspace or Path(request.workspace) / job.job_id
    settings = _build_settings(request, workspace)
    pipeline = Pipeline(settings, interactive=False)

    try:
        result = await pipeline.run(request.prompt)
        job.status = "completed" if result.success else "failed"
        job.result = _serialize_result(result)
        job.errors = result.errors
    except Exception as e:
        logger.exception("Pipeline failed for job %s", job.job_id)
        job.status = "failed"
        job.errors = [str(e)]
    finally:
        job.elapsed_seconds = time.monotonic() - t0


async def _run_enhance(job: _Job, request: EnhanceRequest) -> None:
    import time
    t0 = time.monotonic()
    job.status = "running"
    job.phase = "initializing"

    workspace = Path(request.workspace)
    settings = _build_settings(request, workspace)
    pipeline = Pipeline(settings, interactive=False)

    try:
        result = await pipeline.enhance(request.prompt)
        job.status = "completed" if result.success else "failed"
        job.result = _serialize_result(result)
        job.errors = result.errors
    except Exception as e:
        logger.exception("Enhancement failed for job %s", job.job_id)
        job.status = "failed"
        job.errors = [str(e)]
    finally:
        job.elapsed_seconds = time.monotonic() - t0


def _serialize_result(result: PipelineResult) -> dict[str, Any]:
    data: dict[str, Any] = {
        "workspace": str(result.workspace_path),
        "task_stats": result.task_stats,
        "elapsed_seconds": round(result.elapsed_seconds, 2),
    }
    if result.blueprint:
        data["project_name"] = result.blueprint.name
        data["language"] = result.blueprint.tech_stack.get("language", "")
    if result.token_cost:
        data["token_cost"] = {
            "input_tokens": result.token_cost.input_tokens,
            "output_tokens": result.token_cost.output_tokens,
            "cost_usd": round(result.token_cost.cost_usd, 6),
            "model": result.token_cost.model,
        }
    return data


def _resolve_job(job_id: str) -> _Job:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job
