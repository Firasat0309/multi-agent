"""FastAPI server providing HTTP interface to the code generation pipeline."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from config.settings import Settings, get_settings
from core.pipeline import Pipeline, PipelineResult

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Agent Code Generator API",
    description="Autonomous backend code generation platform",
    version="0.1.0",
)

# In-memory job store (use Redis/DB in production)
_jobs: dict[str, dict[str, Any]] = {}


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=10, description="Natural language project description")
    workspace: str = Field(default="workspace", description="Output directory name")
    model: str = Field(default="claude-sonnet-4-20250514", description="LLM model to use")
    max_agents: int = Field(default=4, ge=1, le=16, description="Max concurrent agents")


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    result: dict[str, Any] | None = None
    errors: list[str] = []


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks) -> GenerateResponse:
    """Start a code generation job."""
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "pending", "result": None, "errors": []}

    background_tasks.add_task(_run_pipeline, job_id, request)

    return GenerateResponse(
        job_id=job_id,
        status="pending",
        message=f"Job {job_id} created. Poll /jobs/{job_id} for status.",
    )


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str) -> JobStatus:
    """Get the status of a generation job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        result=job.get("result"),
        errors=job.get("errors", []),
    )


@app.get("/jobs")
async def list_jobs() -> list[dict[str, Any]]:
    """List all jobs."""
    return [
        {"job_id": jid, "status": data["status"]}
        for jid, data in _jobs.items()
    ]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


async def _run_pipeline(job_id: str, request: GenerateRequest) -> None:
    """Background task to run the pipeline."""
    _jobs[job_id]["status"] = "running"

    base = get_settings()
    # Build a fresh Settings per job — never mutate the singleton.
    settings = Settings(
        workspace_dir=Path(request.workspace) / job_id,
        llm=base.llm,
        sandbox=base.sandbox,
        memory=base.memory,
        observability=base.observability,
        max_concurrent_agents=request.max_agents,
    )

    pipeline = Pipeline(settings, interactive=False)

    try:
        result = await pipeline.run(request.prompt)
        _jobs[job_id]["status"] = "completed" if result.success else "failed"
        _jobs[job_id]["result"] = {
            "workspace": str(result.workspace_path),
            "task_stats": result.task_stats,
            "elapsed_seconds": result.elapsed_seconds,
            "project_name": result.blueprint.name if result.blueprint else None,
        }
        _jobs[job_id]["errors"] = result.errors
    except Exception as e:
        logger.exception(f"Pipeline failed for job {job_id}")
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["errors"] = [str(e)]
