from stratos_gym.tasks.base import BaseTask
from stratos_gym.state import SaaSState
from stratos_gym.models import StratosObservation

class GrowthTask(BaseTask):
    """Easy mode: Bootstrapping 101."""
    max_steps = 25
    name = "task-1-growth"

    def reset(self, seed: int = 0) -> StratosObservation:
        self.state = SaaSState(
            cash=5000.0,
            infrastructure_capacity=1000,
            conversion_rate=0.08, # Boosted
            churn_rate=0.04,      # Better retention
            active_task=self.name,
            max_steps=self.max_steps,
            episode_seed=seed
        )
        return self.state.to_observation()

    @property
    def ground_truth(self) -> dict:
        return {"target_mrr": 10000.0}
