from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from stratos_gym.models import StratosAction, StratosObservation
from stratos_gym.state import SaaSState


class BaseTask(ABC):
    """Abstract base for StratOS-RL episode tasks."""
    max_steps: int = 25
    name: str = "base"

    def __init__(self) -> None:
        self.state: SaaSState | None = None

    @abstractmethod
    def reset(self, seed: int = 0) -> StratosObservation: ...

    def step(self, action: StratosAction) -> Tuple[StratosObservation, bool]:
        self._assert_ready()
        assert self.state is not None
        logs = self.state.apply_action(action)
        obs = self.state.to_observation()
        obs.logs = logs
        done = self.state.is_terminal()
        return obs, done

    @property
    @abstractmethod
    def ground_truth(self) -> Dict[str, Any]: ...

    def _assert_ready(self) -> None:
        if self.state is None:
            raise RuntimeError(f"Task '{self.name}' not reset.")


class ViralTask(BaseTask):
    """Scenario: 1M hits, 0 Servers. Scale AWS or die."""
    max_steps = 20
    name = "task-1-viral"
    def reset(self, seed: int = 0) -> StratosObservation:
        self.state = SaaSState(
            cash=5000.0, 
            active_customers=320, 
            infrastructure_capacity=100, 
            brand_momentum=12.0, 
            team_size=4,
            active_task=self.name, 
            max_steps=self.max_steps, 
            episode_seed=seed
        )
        return self.state.to_observation()
    @property
    def ground_truth(self) -> dict: return {"target_mrr": 10000.0}

class PriceTask(BaseTask):
    """Scenario: Price War. You are selling for $5 but cost is $10/user."""
    max_steps = 30
    name = "task-3-price"
    def reset(self, seed: int = 0) -> StratosObservation:
        self.state = SaaSState(
            cash=6000.0, 
            price_per_user=8.0, 
            active_customers=150,
            debt_limit=-15000.0, 
            active_task=self.name, 
            max_steps=self.max_steps, 
            episode_seed=seed
        )
        return self.state.to_observation()
    @property
    def ground_truth(self) -> dict: return {"target_mrr": 10000.0}


class EnterpriseTask(BaseTask):
    """Scenario: The Team Pivot. High quality demand, low team size."""
    max_steps = 25
    name = "task-2-enterprise"
    def reset(self, seed: int = 0) -> StratosObservation:
        self.state = SaaSState(
            cash=15000.0, 
            active_customers=10, 
            price_per_user=500.0, 
            product_quality=0.4, 
            team_size=2, 
            active_task=self.name, 
            max_steps=self.max_steps, 
            episode_seed=seed
        )
        return self.state.to_observation()
    @property
    def ground_truth(self) -> dict: return {"target_mrr": 10000.0}



class TaskRegistry:
    """Central registry for tasks."""
    def __init__(self) -> None:
        self._tasks = {
            "task-1-viral": ViralTask,
            "task-2-enterprise": EnterpriseTask,
            "task-3-price": PriceTask,
        }

    def get(self, task_id: str) -> BaseTask:
        if task_id not in self._tasks:
            raise ValueError(f"Unknown task: {task_id}")
        return self._tasks[task_id]()

    def list_tasks(self) -> typing.List[str]:
        return list(self._tasks.keys())
