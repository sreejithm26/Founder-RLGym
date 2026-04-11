from stratos_gym.tasks.registry import (
    BaseTask,
    ViralTask,
    PriceTask,
    EnterpriseTask,
)

TASK_REGISTRY: dict[str, type[BaseTask]] = {
    "task-1-viral": ViralTask,
    "task-2-enterprise": EnterpriseTask,
    "task-3-price": PriceTask,
}

__all__ = [
    "BaseTask",
    "ViralTask",
    "PriceTask",
    "EnterpriseTask",
    "TASK_REGISTRY",
]
