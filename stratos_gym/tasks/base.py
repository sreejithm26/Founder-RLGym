from __future__ import annotations

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
    def reset(self, seed: int = 0) -> StratosObservation:
        """Initialise a new episode."""
        ...

    def step(self, action: StratosAction) -> Tuple[StratosObservation, bool]:
        """Apply action, advance simulation."""
        self._assert_ready()
        assert self.state is not None
        
        logs = self.state.apply_action(action)
        obs = self.state.to_observation()
        obs.logs = logs
        
        done = self.state.is_terminal()
        return obs, done

    @property
    @abstractmethod
    def ground_truth(self) -> Dict[str, Any]:
        """Ground-truth data for grading."""
        ...

    def _assert_ready(self) -> None:
        if self.state is None:
            raise RuntimeError(f"Task '{self.name}' not reset.")
