from stratos_gym.tasks.base import BaseTask
from stratos_gym.tasks.growth import GrowthTask
from stratos_gym.tasks.viral import ViralTask
from stratos_gym.tasks.price import PriceTask

TASK_REGISTRY: dict[str, type[BaseTask]] = {
    "task-1-growth": GrowthTask,
    "task-2-viral": ViralTask,
    "task-3-price": PriceTask,
}

__all__ = [
    "BaseTask",
    "GrowthTask",
    "ViralTask",
    "PriceTask",
    "TASK_REGISTRY",
]
