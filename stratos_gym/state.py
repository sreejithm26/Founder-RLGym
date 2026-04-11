"""
SaaSState — Strategy-forcing physics for StratOS-RL.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stratos_gym.models import ActionType, StratosAction, StratosObservation
from stratos_gym.math.noise import NoiseGenerator

SUPPORT_CAPACITY_PER_FTE = 120


@dataclass
class SaaSState:
    """A task-aware state engine where the rules of the game change per scenario."""

    # Core Financials
    cash: float = 5000.0
    mrr: float = 0.0
    debt_limit: float = -5000.0
    
    # Temporal Signals
    prev_mrr: float = 0.0
    prev_cash: float = 5000.0
    last_marketing_spend: float = 0.0
    
    # Customers & Market
    active_basic: int = 0
    active_customers: int = 0
    active_enterprise: int = 0
    
    conversion_rate: float = 0.05
    churn_rate: float = 0.04
    effective_churn: float = 0.04
    
    infrastructure_capacity: int = 1000
    current_traffic: float = 0.0
    
    # Product & Team
    product_quality: float = 1.0
    brand_momentum: float = 0.0
    competitor_strength: float = 1.0
    
    team_size: int = 1
    employee_salary: float = 1000.0
    trainee_salary: float = 400.0
    
    # Economics
    cac: float = 20.0
    ltv: float = 400.0
    price_per_user: float = 25.0
    
    # Internal Queues
    hiring_queue: List[Dict[str, Any]] = field(default_factory=list)
    
    # Episode context
    step_number: int = 0
    episode_seed: int = 0
    max_steps: int = 30 # Standardized
    active_task: str = "task-1-viral"
    is_done: bool = False
    
    # Helpers
    rng: Any = field(default_factory=lambda: np.random.default_rng())
    noise: Optional[NoiseGenerator] = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.episode_seed)
        self.noise = NoiseGenerator(self.episode_seed)

    def calculate_burn(self) -> float:
        # Salaries
        trainee_exp = sum(h["count"] for h in self.hiring_queue) * self.trainee_salary
        active_exp = self.team_size * self.employee_salary
        
        # Scenario-specific infra costs (e.g. Price War makes infra more expensive per user)
        base_infra_mult = 0.2
        if self.active_task == "task-3-price":
            base_infra_mult = 0.8 # Lowered from 1.2
            
        infra_burn = (base_infra_mult * (self.active_customers ** 0.95)) + (self.infrastructure_capacity * 0.05)
        
        return trainee_exp + active_exp + infra_burn

    def to_observation(self) -> StratosObservation:
        burn = self.calculate_burn()
        net_flow = self.mrr - burn
        runway = 99 if net_flow >= 0 else round(abs(self.cash / max(1, burn - self.mrr)), 1)
        
        dashboard = {
            "bank": {
                "cash": round(self.cash, 2), 
                "MRR": round(self.mrr, 2),
                "burn_rate": round(burn, 2),
                "runway_months": runway
            },
            "unit_economics": {
                "CAC": round(self.cac, 2),
                "LTV": round(self.ltv, 2),
                "churn_rate": round(self.effective_churn, 3),
                "price_point": self.price_per_user
            },
            "internal_metrics": {
                "product_quality": round(self.product_quality, 2),
                "conversion_rate": round(self.conversion_rate, 4),
                "competitor_strength": round(self.competitor_strength, 2),
            },
            "ops": {
                "active_customers": self.active_customers,
                "support_utilization": round(self.active_customers / max(SUPPORT_CAPACITY_PER_FTE, self.team_size * SUPPORT_CAPACITY_PER_FTE), 2),
                "infra_utilization": round(self.active_customers / max(1, self.infrastructure_capacity), 2)
            }
        }
        return StratosObservation(
            dashboard=dashboard,
            logs=[],
            hint="",
            step_number=self.step_number,
        )

    def apply_action(self, action: StratosAction) -> List[str]:
        logs = []
        if self.is_done: return ["Terminated."]

        self.prev_mrr = self.mrr
        self.prev_cash = self.cash

        # 1. Budget Throttling
        burn_est = self.calculate_burn()
        available_liquidity = max(0.0, self.cash - self.debt_limit - (burn_est * 1.3))
        
        spending_req = (action.a_marketing + action.a_product + action.a_infra + action.a_hiring)
        
        if spending_req > available_liquidity and spending_req > 0:
            scale = available_liquidity / spending_req
            a_marketing, a_product, a_infra, a_hiring = action.a_marketing*scale, action.a_product*scale, action.a_infra*scale, action.a_hiring*scale
            logs.append(f"[SYSTEM] Overspent! Actions scaled to {int(scale*100)}% for liquidity safety.")
        else:
            a_marketing, a_product, a_infra, a_hiring = action.a_marketing, action.a_product, action.a_infra, action.a_hiring

        # 2. Strategy-Forcing Mutations
        self.cash -= (a_marketing + a_product + a_infra + a_hiring)
        self.last_marketing_spend = a_marketing
        
        if a_marketing > 0:
            boost = 0.15
            if self.active_task == "task-6-red-ocean": boost = 0.05 # Marketing is less effective
            self.brand_momentum += boost * math.log(1 + a_marketing)
            
        if a_product > 0:
            self.product_quality += 0.12 * math.tanh(a_product / 1500.0)
            self.product_quality = min(5.0, self.product_quality)
            
        if a_infra > 0:
            self.infrastructure_capacity += int(a_infra * 4.0)

        if a_hiring >= 400:
            n = int(a_hiring / 400.0)
            self.hiring_queue.append({"count": n, "steps_left": 2})
            logs.append(f"HR: Training started for {n} staff.")

        if action.price > 0:
            self.price_per_user = action.price

        logs.extend(self._tick())
        return logs

    def _tick(self) -> List[str]:
        logs = []
        if not self.noise: return logs
        
        # 1. Pipeline
        for h in self.hiring_queue:
            h["steps_left"] -= 1
        graduates = sum(h["count"] for h in self.hiring_queue if h["steps_left"] <= 0)
        self.team_size += graduates
        self.hiring_queue = [h for h in self.hiring_queue if h["steps_left"] > 0]

        # 2. Operating Cycle
        burn = self.calculate_burn()
        self.cash -= burn
        
        # Task Specific: Debt Spiral compounding interest
        if self.active_task == "task-5-debt" and self.cash < 0:
            interest = abs(self.cash) * 0.05
            self.cash -= interest
            logs.append(f"[DEBT] Paid ${interest:.2f} in emergency interest.")

        self.mrr = self.active_customers * self.price_per_user
        self.cash += self.mrr
        
        # 3. Rules of the Game (Non-linear physics)
        reference_price = 500.0 if self.active_task == "task-2-enterprise" else 25.0
        price_ratio = self.price_per_user / reference_price
        # Exponential price sensitivity
        if price_ratio > 1.0:
            if self.active_task == "task-3-price":
                e_mult = 3.0
                churn_power = 2.0
            elif self.active_task == "task-2-enterprise":
                e_mult = 0.6
                churn_power = 1.2
            else:
                e_mult = 2.0
                churn_power = 2.0
            elasticity = math.exp(-e_mult * (price_ratio - 1.0))
            churn_impact = price_ratio ** churn_power
        else:
            discount_boost = 0.1 if self.active_task == "task-2-enterprise" else 0.4
            elasticity = 1.0 + discount_boost * (1.0 - price_ratio)
            churn_impact = 1.0

        # Market Saturation (TAM = 10k)
        saturation = 1.0 - (self.active_customers / 10000.0)
        task_comp_growth = 0.05 if self.active_task == "task-6-red-ocean" else 0.012
        self.competitor_strength += task_comp_growth + (0.0001 * self.active_customers)
        
        # 4. Growth Cycle
        traffic = (60 + (self.brand_momentum * 150.0)) * saturation
        traffic *= self.noise.sample_demand_noise(0.08)
        
        prod_bonus = 1.0 + (self.team_size * 0.02)
        conv = (self.conversion_rate * elasticity * prod_bonus) / self.competitor_strength
        
        new_users = int(traffic * self.product_quality * conv)
        self.active_customers += new_users
        
        # 5. Churn & Crisis Checks
        infra_util = self.active_customers / max(1, self.infrastructure_capacity)
        # Task 1 Crisis Check: Viral overload kills the company
        if self.active_task == "task-1-viral" and infra_util > 2.0:
            logs.append("[FAIL] Infrastructure melted under viral load!")
            self.is_done = True
            
        support_capacity = max(SUPPORT_CAPACITY_PER_FTE, self.team_size * SUPPORT_CAPACITY_PER_FTE)
        support_ratio = self.active_customers / support_capacity
        support_penalty = max(0, (support_ratio - 1.0) * 0.4)
        
        quality_gate = 1.0
        if self.active_task == "task-2-enterprise" and self.product_quality < 1.0:
            quality_gate = 3.0 # Brutal churn if quality not met
            
        self.effective_churn = max(0.01, (self.churn_rate * churn_impact * quality_gate + support_penalty) / self.product_quality)
        self.effective_churn *= self.noise.sample_churn_factor(1.0)
        
        churned = int(self.active_customers * self.effective_churn)
        self.active_customers = max(0, self.active_customers - churned)

        # 6. Unit Econ
        if new_users > 0:
            self.cac = self.last_marketing_spend / new_users
        self.ltv = self.price_per_user / max(0.01, self.effective_churn)
        
        self.step_number += 1
        self.brand_momentum *= 0.75
        
        # 7. TERMINATION
        if self.cash < self.debt_limit:
            logs.append("[BANKRUPT] Out of cash.")
            self.is_done = True
        elif self.mrr >= 10000.0:
            logs.append("[IPO] Target reached!")
            self.is_done = True
        elif self.step_number >= self.max_steps:
            self.is_done = True

        return logs

    def is_terminal(self) -> bool:
        return self.is_done

    def clone(self) -> SaaSState:
        cloned = copy.deepcopy(self)
        if hasattr(self, "rng"): cloned.rng = copy.deepcopy(self.rng)
        return cloned

    def to_dict(self) -> Dict[str, Any]:
        d = copy.deepcopy(self.__dict__)
        if "rng" in d: d["rng"] = None # Avoid pickle issues
        if "noise" in d: d["noise"] = None
        return d
