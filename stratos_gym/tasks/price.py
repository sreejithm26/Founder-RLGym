from stratos_gym.tasks.base import BaseTask
from stratos_gym.state import SaaSState
from stratos_gym.models import StratosObservation

class PriceTask(BaseTask):
    """Hard mode: The Efficiency War."""
    max_steps = 20
    name = "task-3-price"

    def reset(self, seed: int = 0) -> StratosObservation:
        self.state = SaaSState(
            cash=300.0, # Very low
            active_basic=25,
            active_customers=25,
            mrr=500.0,
            infrastructure_capacity=500,
            conversion_rate=0.02, # Hard
            churn_rate=0.1,       # Nightmare
            active_task=self.name,
            max_steps=self.max_steps,
            episode_seed=seed
        )
        return self.state.to_observation()

    @property
    def ground_truth(self) -> dict:
        return {"target_mrr": 10000.0}
