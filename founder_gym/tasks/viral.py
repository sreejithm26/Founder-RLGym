from founder_gym.tasks.base import BaseTask
from founder_gym.state import SaaSState
from founder_gym.models import FounderObservation

class ViralTask(BaseTask):
    """Medium mode: The Viral Crash."""
    max_steps = 20
    name = "task-2-viral"

    def reset(self, seed: int = 0) -> FounderObservation:
        self.state = SaaSState(
            cash=3000.0,
            active_basic=100,
            active_customers=100,
            mrr=2000.0,
            infrastructure_capacity=100, # Tiny
            current_traffic=800,         # Overloaded
            active_task=self.name,
            max_steps=self.max_steps,
            episode_seed=seed
        )
        return self.state.to_observation()

    @property
    def ground_truth(self) -> dict:
        return {"target_mrr": 10000.0}
