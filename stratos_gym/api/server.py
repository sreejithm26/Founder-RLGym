"""
FastAPI application — StratOS-RL OpenEnv HTTP interface.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from stratos_gym.env import StratosEnv
from stratos_gym.models import StratosAction, ResetRequest, StateResult, StepResult
from stratos_gym.tasks import TASK_REGISTRY

logger = logging.getLogger(__name__)

app = FastAPI(title="StratOS-RL", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: StratosEnv = StratosEnv.in_process()


@app.get("/")
async def root() -> Dict[str, str]:
    return {"status": "ok", "env": "stratos-rl"}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
async def list_tasks() -> List[str]:
    return list(TASK_REGISTRY.keys())


@app.post("/reset")
async def reset_endpoint(
    request: Optional[ResetRequest] = None,
    task: Optional[str] = Query(default=None),
) -> StepResult:
    task_id = task or (request.task_id if request else None) or "task-1-growth"
    seed = int((request.seed if request else None) or 0)

    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task_id}")

    return await _env.reset(task_id=task_id, seed=seed)


@app.post("/step")
async def step_endpoint(action: StratosAction) -> StepResult:
    try:
        return await _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state")
async def state_endpoint() -> StateResult:
    try:
        return await _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
