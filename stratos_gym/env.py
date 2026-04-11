from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from stratos_gym.graders.stratos_grader import StratosGrader
from stratos_gym.models import StratosAction, StratosObservation, StratosReward, StateResult
from stratos_gym.tasks.registry import TaskRegistry, BaseTask

_GRADER = StratosGrader()


class StratosEnv:
    """
    OpenEnv compliant environment for SaaS CEO simulation.
    Now follows standard Gym step signature: (obs, reward, done, info)
    """

    @staticmethod
    def in_process() -> StratosEnv:
        """Factory for in-process usage."""
        return StratosEnv()

    def __init__(self):
        self._registry = TaskRegistry()
        self._task: Optional[BaseTask] = None
        self._last_obs: Optional[StratosObservation] = None

    async def reset(self, task_id: str = "task-1-viral", seed: Optional[int] = None) -> StratosObservation:
        """Reset environment to a specific task scenario."""
        self._task = self._registry.get(task_id)
        
        # Reset task state
        obs = self._task.reset(seed=seed or 0)
        self._last_obs = obs
        return obs

    async def step(self, action: StratosAction) -> Tuple[StratosObservation, float, bool, Dict[str, Any]]:
        """
        Standard Gym Step.
        Returns: (observation, reward, done, info)
        """
        if self._task is None:
            raise RuntimeError("Environment must be reset before step.")

        # Advance physics
        obs, done = self._task.step(action)
        
        # Calculate Reward
        reward_obj = _GRADER.grade(
            action=action,
            state=self._task.state,
            ground_truth=self._task.ground_truth,
            step_number=self._task.state.step_number
        )
        
        # Prepare Info dict for debugging/logs
        info = {
            "reward_components": reward_obj.components,
            "logs": obs.logs,
            "task_name": self._task.name
        }
        
        self._last_obs = obs
        return obs, reward_obj.value, done, info

    async def state(self) -> StateResult:
        """Deep snapshot for debugging or grading."""
        if self._task is None or self._task.state is None:
            raise RuntimeError("No state available.")
        
        snap = self._task.state.clone()
        return StateResult(
            stratos_state=snap.to_dict(),
            task_id=self._task.name,
            step_number=snap.step_number,
            episode_seed=snap.episode_seed,
            is_done=snap.is_terminal()
        )

    async def list_tasks(self) -> List[str]:
        return self._registry.list_tasks()

    async def close(self):
        pass
