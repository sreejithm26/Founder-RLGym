"""
Pydantic v2 models for StratOS-RL (SaaS CEO Simulator).

All API boundary types live here:
  - StratosObservation — what the agent sees each step
  - StratosAction      — what the agent submits each step
  - StratosReward      — reward breakdown returned by graders
  - StepResult         — envelope returned by /step and /reset
  - StateResult        — envelope returned by /state (read-only)
  - ResetRequest       — optional body for POST /reset
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionType(str, Enum):
    ALLOCATE = "ALLOCATE"
    SET_PARAMETERS = "set_parameters"
    SET_PRICE = "set_price"
    HIRE = "hire"
    ALLOCATE_ALT = "allocate"


class StratosObservation(BaseModel):
    """Full observation delivered to the agent at each step."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dashboard": {
                    "bank": {"cash": 9500.0, "MRR": 500.0},
                    "aws": {"capacity": 1000, "utilization": 0.45},
                    "analytics": {"active_visitors": 450, "CAC": 12.5, "LTV": 60.0},
                    "market": {"market_share": 0.05, "brand": 1.0},
                },
                "logs": ["Successfully allocated $500.00"],
                "hint": "Try allocating to product to boost quality.",
                "step_number": 1,
            }
        }
    )

    dashboard: Dict[str, Any]
    logs: List[str]
    hint: str
    step_number: int


class StratosAction(BaseModel):
    """Action submitted by the agent each step."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action_type": "ALLOCATE",
                "price": 25.0,
                "a_marketing": 500.0,
                "a_product": 200.0,
                "a_infra": 100.0,
                "a_hiring": 0.0,
                "a_debt_repayment": 0.0,
                "reasoning": "Investing in marketing to drive traffic.",
                "confidence": 0.95,
            }
        }
    )

    action_type: ActionType = ActionType.ALLOCATE
    price: float = Field(default=0.0, description="Monthly subscription price per user. 0 keeps previous price.")
    a_marketing: float = Field(default=0.0)
    a_product: float = Field(default=0.0)
    a_infra: float = Field(default=0.0)
    a_hiring: float = Field(default=0.0)
    reasoning: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class StratosReward(BaseModel):
    """Reward breakdown returned by a grader."""

    value: float = Field(description="Scalar reward in [0.0, 1.0]")
    components: Dict[str, float] = Field(
        description="Per-component breakdown, e.g. growth, efficiency, infra"
    )
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def clamp_value(self) -> "StratosReward":
        """RL-Friendly clamping to avoid extreme outliers only."""
        self.value = float(min(200.0, max(-100.0, self.value)))
        return self


class StepResult(BaseModel):
    """Envelope returned by POST /reset and POST /step."""

    observation: StratosObservation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResult(BaseModel):
    """Read-only snapshot returned by GET /state."""

    task_id: str
    step_number: int
    episode_seed: int
    stratos_state: Dict[str, Any]  # serialized SaaSState
    is_done: bool


class ResetRequest(BaseModel):
    """Optional body for POST /reset."""

    task_id: Optional[str] = "task-1-growth"
    seed: Optional[int] = None
