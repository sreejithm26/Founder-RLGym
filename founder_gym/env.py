"""
FounderGymEnv — the top-level OpenEnv environment class.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from typing import Any, Dict, Optional

import httpx

from founder_gym.graders import FounderGrader
from founder_gym.models import FounderAction, ResetRequest, StateResult, StepResult
from founder_gym.tasks import TASK_REGISTRY, BaseTask

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8000
_HEALTH_TIMEOUT = 30

_GRADER = FounderGrader()


class _InProcessBackend:
    def __init__(self) -> None:
        self._task: Optional[BaseTask] = None
        self._task_id: str = "task-1-growth"

    async def reset(self, task_id: str = "task-1-growth", seed: int = 0) -> StepResult:
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id {task_id}")
        self._task_id = task_id
        self._task = TASK_REGISTRY[task_id]()
        observation = self._task.reset(seed=seed)
        return StepResult(observation=observation, reward=0.0, done=False, info={})

    async def step(self, action: FounderAction) -> StepResult:
        if self._task is None:
            raise RuntimeError("Reset first.")
        
        observation, done = self._task.step(action)
        grade = self._grade(action)
        return StepResult(
            observation=observation,
            reward=float(grade["reward"]),
            done=done,
            info=grade.get("info", {}),
        )

    def _grade(self, action: FounderAction) -> Dict[str, Any]:
        assert self._task is not None and self._task.state is not None
        return _GRADER.grade(action, self._task.state, self._task.ground_truth, self._task.state.step_number)

    async def state(self) -> StateResult:
        if self._task is None or self._task.state is None:
            raise RuntimeError("Reset first.")
        snap = self._task.state.clone()
        return StateResult(
            task_id=self._task_id,
            step_number=snap.step_number,
            episode_seed=snap.episode_seed,
            founder_state=snap.to_dict(),
            is_done=snap.is_done,
        )


class FounderGymEnv:
    def __init__(self, base_url: Optional[str] = None, _in_process: bool = False) -> None:
        self._base_url = base_url
        self._in_process = _in_process
        self._client: Optional[httpx.AsyncClient] = None
        self._container_id: Optional[str] = None
        self._local: Optional[_InProcessBackend] = None
        self._task_id: str = "task-1-growth"

    @classmethod
    def in_process(cls) -> FounderGymEnv:
        env = cls(_in_process=True)
        env._local = _InProcessBackend()
        return env

    @classmethod
    async def from_docker_image(cls, image_name: str) -> FounderGymEnv:
        port = _DEFAULT_PORT
        result = subprocess.run(
            ["docker", "run", "-d", "--rm", "-p", f"{port}:{port}", image_name],
            capture_output=True, text=True, check=True
        )
        container_id = result.stdout.strip()
        base_url = f"http://localhost:{port}"
        env = cls(base_url=base_url)
        env._container_id = container_id
        env._client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
        
        deadline = time.monotonic() + _HEALTH_TIMEOUT
        while time.monotonic() < deadline:
            try:
                r = await env._client.get("/health")
                if r.status_code == 200: return env
            except Exception: pass
            await asyncio.sleep(1)
        await env.close()
        raise RuntimeError("Container health check failed.")

    async def reset(self, task_id: str = "task-1-growth", seed: int = 0) -> StepResult:
        self._task_id = task_id
        if self._in_process:
            assert self._local is not None
            return await self._local.reset(task_id, seed)
        assert self._client is not None
        r = await self._client.post("/reset", json=ResetRequest(task_id=task_id, seed=seed).model_dump())
        r.raise_for_status()
        return StepResult.model_validate(r.json())

    async def step(self, action: FounderAction) -> StepResult:
        if self._in_process:
            assert self._local is not None
            result = await self._local.step(action)
        else:
            assert self._client is not None
            r = await self._client.post("/step", json=action.model_dump())
            r.raise_for_status()
            result = StepResult.model_validate(r.json())
        return result

    async def state(self) -> StateResult:
        if self._in_process:
            assert self._local is not None
            return await self._local.state()
        assert self._client is not None
        r = await self._client.get("/state")
        r.raise_for_status()
        return StateResult.model_validate(r.json())

    async def close(self) -> None:
        if self._client: await self._client.aclose()
        if self._container_id:
            subprocess.run(["docker", "stop", self._container_id])
            subprocess.run(["docker", "rm", self._container_id])
