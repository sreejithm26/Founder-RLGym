"""
FastAPI application — Founder Gym OpenEnv HTTP interface.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from founder_gym.env import FounderGymEnv
from founder_gym.models import FounderAction, ResetRequest, StateResult, StepResult
from founder_gym.tasks import TASK_REGISTRY

logger = logging.getLogger(__name__)

app = FastAPI(title="Founder Gym", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: FounderGymEnv = FounderGymEnv.in_process()


@app.get("/")
async def root() -> Dict[str, str]:
    return {"status": "ok", "env": "founder-gym"}


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
async def step_endpoint(action: FounderAction) -> StepResult:
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
