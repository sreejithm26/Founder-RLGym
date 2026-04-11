import numpy as np

class NoiseGenerator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def sample_demand_noise(self, scale: float = 1.0) -> float:
        return self.rng.normal(0, 0.05 * scale)

    def sample_cac_noise(self, scale: float = 1.0) -> float:
        return self.rng.uniform(-0.1 * scale, 1.5 * scale)

    def sample_churn_factor(self, base_churn: float) -> float:
        return max(0.001, base_churn + self.rng.normal(0, 0.005))

    def sample_competitor_evolution(self, drift: float = 1.0) -> float:
        return max(0.1, 1.0 + self.rng.normal(0, 0.1 * drift))
