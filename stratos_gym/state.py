"""
SaaSState — the mutable episode state for StratOS-RL.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stratos_gym.models import ActionType, StratosAction, StratosObservation
from stratos_gym.math.noise import NoiseGenerator


@dataclass
class SaaSState:
    """Full mutable state for one StratOS-RL episode."""

    # Core Financials
    cash: float = 5000.0
    mrr: float = 0.0
    debt_limit: float = -15000.0
    
    # Customers & Market
    active_basic: int = 0
    active_enterprise: int = 0
    active_customers: int = 0
    
    conversion_rate: float = 0.05
    churn_rate: float = 0.05
    churn_rate_enterprise: float = 0.01
    effective_churn: float = 0.05
    
    infrastructure_capacity: int = 1000
    infrastructure_utilization: float = 0.0
    current_traffic: int = 0
    last_infra_change_step: int = -10
    
    # Product & Team
    product_quality: float = 1.0
    brand: float = 1.0
    competitor_strength: float = 1.0
    market_share: float = 0.0
    customer_satisfaction: float = 0.7
    
    team_size: int = 1
    employee_salary: float = 200.0
    
    # Acquisition
    cac: float = 12.0
    ltv: float = 400.0
    price_per_user: float = 20.0
    price_enterprise: float = 500.0
    
    # Internal Queues
    leads_queue: List[float] = field(default_factory=lambda: [0.0] * 5)
    leads_queue_enterprise: List[float] = field(default_factory=lambda: [0.0] * 10)
    product_queue: List[float] = field(default_factory=lambda: [0.0] * 3)
    hiring_queue: List[int] = field(default_factory=lambda: [0] * 4) # 3-month ramp up
    
    brand_momentum: float = 0.0 # Decaying marketing effect
    grace_hires_remaining: int = 2 # First 2 hires are instant to avoid early-game death
    
    # Episode context
    step_number: int = 0
    episode_seed: int = 0
    max_steps: int = 25
    active_task: str = "task-1-growth"
    is_done: bool = False
    
    insolvency_steps: int = 0
    last_debt_payment: float = 0.0
    
    # Helpers
    rng: Any = field(default_factory=lambda: np.random.default_rng())
    noise: Optional[NoiseGenerator] = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.episode_seed)
        self.noise = NoiseGenerator(self.episode_seed)

    def seed(self, seed: int) -> None:
        self.episode_seed = seed
        self.rng = np.random.default_rng(seed)
        self.noise = NoiseGenerator(seed)

    def to_observation(self) -> StratosObservation:
        """Build StratosObservation from current state."""
        dashboard = {
            "bank": {"cash": round(self.cash, 2), "MRR": round(self.mrr, 2)},
            "aws": {
                "capacity": self.infrastructure_capacity,
                "utilization": round(self.infrastructure_utilization, 2),
            },
            "analytics": {
                "active_visitors": self.current_traffic,
                "CAC": round(self.cac, 2),
                "LTV": round(self.ltv, 2),
            },
            "market": {
                "market_share": round(self.market_share, 3),
                "brand": round(self.brand, 2),
            },
            "pricing": {
                "price_per_user": round(self.price_per_user, 2),
                "active_customers": self.active_customers,
            },
            "cohorts": {
                "basic": self.active_basic,
                "enterprise": self.active_enterprise,
            },
            "hr": {
                "team_size": self.team_size,
                "monthly_burn": round(self.team_size * self.employee_salary, 2),
            },
        }
        
        return StratosObservation(
            dashboard=dashboard,
            logs=[],  # Populated by apply_action
            hint="",  # Populated by task logic
            step_number=self.step_number,
        )

    def apply_action(self, action: StratosAction) -> List[str]:
        """Apply agent action and advance state using Saas CEO logic."""
        logs = []
        
        # 1. Action Validation & Spend Protection
        fixed_burn = 250.0 + (self.team_size * self.employee_salary) + (self.infrastructure_capacity * 0.1)
        total_debt = abs(min(0.0, self.cash))
        debt_installment = total_debt * 0.05
        monthly_reserve = fixed_burn + debt_installment
        
        max_credit_line = self.mrr * 6.0
        lender_trust = 1.0 - (total_debt / 20000.0) ** 2
        available_credit = max(0.0, (max_credit_line - total_debt) * max(0.0, lender_trust))
        physical_ceiling = max(0.0, self.cash - self.debt_limit - monthly_reserve)
        trust_ceiling = max(0.0, self.cash + available_credit - monthly_reserve)
        
        max_optional_liquidity = max(trust_ceiling, min(1000.0, physical_ceiling))
        
        a_marketing = action.a_marketing
        a_product = action.a_product
        a_infra = action.a_infra
        a_hiring = action.a_hiring
        
        requested = a_marketing + a_product + a_infra + a_hiring
        if requested > max_optional_liquidity and requested > 0:
            ratio = max_optional_liquidity / requested
            a_marketing *= ratio
            a_product *= ratio
            a_infra *= ratio
            a_hiring *= ratio
            logs.append(f"[SYSTEM] Budget Throttled: Action scaled to {int(ratio*100)}% to fit debt limit.")
            
        total_spend = a_marketing + a_product + a_infra + a_hiring
        self.cash -= total_spend
        self.last_debt_payment = action.a_debt_repayment
        
        # 2. Credit Squeeze
        credit_factor = 1.0
        if self.cash < 0:
            debt_ratio = abs(self.cash) / 15000.0
            credit_factor = max(0.5, 1.0 - (debt_ratio ** 1.5))
            
        # 3. Infra Expansion
        if a_infra > 0:
            if self.step_number - self.last_infra_change_step >= 2:
                self.last_infra_change_step = self.step_number
                self.infrastructure_capacity += int(a_infra * 2.0)
                logs.append(f"Expanded infra. New Cap: {self.infrastructure_capacity}")
            else:
                logs.append("Infra expansion skipped (cooldown).")
                self.cash += a_infra
                
        # 4. Pricing
        effective_price = action.price if action.price > 0 else self.price_per_user
        if effective_price > 200.0:
            logs.append(f"[ERROR] Pricing too high: ${effective_price:.2f} blocked.")
        elif effective_price > 0:
            self.price_per_user = effective_price
            self.ltv = effective_price / max(0.01, self.churn_rate)
            logs.append(f"Set subscription price to ${effective_price:.2f}/mo.")
            
        # 5. Marketing Queue
        if a_marketing > 0 and self.noise:
            marketing_basic = a_marketing * 0.9
            marketing_ent = a_marketing * 0.1
            
            saturation = self.active_customers / 450.0
            base_cost = 10.0 + self.noise.sample_cac_noise(0.5)
            quality_eff = 1.0 / (1.0 + (self.product_quality * 0.6))
            cost = base_cost * (1.0 + saturation ** 1.8) * quality_eff
            
            k_basic = self.rng.integers(0, 2)
            leads_basic = (marketing_basic * credit_factor) / cost
            while len(self.leads_queue) <= k_basic: self.leads_queue.append(0.0)
            self.leads_queue[k_basic] += leads_basic
            
            k_ent = self.rng.integers(2, 6)
            leads_ent = (marketing_ent * credit_factor) / (cost * 12.0)
            while len(self.leads_queue_enterprise) <= k_ent: self.leads_queue_enterprise.append(0.0)
            self.leads_queue_enterprise[k_ent] += leads_ent
            
            self.brand += 0.01 * math.log(1 + a_marketing)
            self.brand_momentum += 0.05 * math.log(1 + a_marketing)
            self.current_traffic += int(1.5 * math.log(1 + a_marketing) * self.brand * credit_factor + self.noise.sample_demand_noise())
            logs.append(f"Spent ${a_marketing:.2f} on marketing.")
            
        # 6. Product & Hiring
        if a_product > 0 or a_hiring > 0:
            val = 0.5 * math.log(1 + a_product) + 0.2 * math.log(1 + a_hiring)
            self.product_queue.append(val)
            if a_hiring >= 1000:
                hires = int(a_hiring / 1000)
                if self.grace_hires_remaining > 0:
                    self.team_size += hires
                    self.grace_hires_remaining -= 1
                    logs.append(f"Hired {hires} employees (Grace hire - instant).")
                else:
                    self.hiring_queue.append(hires)
                    logs.append(f"Hired {hires} employees (Starting ramp-up).")
                
        else:
            self.product_queue.append(0.0)
            self.hiring_queue.append(0)
            if self.product_quality > 1.0:
                self.product_quality *= 0.98
                
        # 7. Step Tick Logic (Simplified Integration)
        tick_logs, latency = self._tick()
        logs.extend(tick_logs)
        
        return logs

    def _tick(self) -> Tuple[List[str], float]:
        """Internal monthly cycle logic."""
        logs = []
        if not self.noise: return logs, 0.0
        
        # 0. Process Long-Horizon Queues
        if self.hiring_queue:
            new_hires = self.hiring_queue.pop(0)
            if new_hires > 0:
                self.team_size += new_hires
                logs.append(f"[HR] Training complete: {new_hires} new employees are now productive.")
        
        self.brand += self.brand_momentum
        self.brand_momentum *= 0.6 # Decays over time
        
        # Basic Acquisition
        new_basic = 0
        if self.leads_queue:
            leads = self.leads_queue.pop(0)
            market_eff = 1.0 / (1.0 + (self.active_customers / 550.0) ** 2)
            price_ratio = max(0.5, self.price_per_user / 30.0)
            elasticity = 1.0 / (price_ratio ** 1.5)
            volatility = 1.0 + self.noise.sample_demand_noise(0.05)
            conv = min(1.0, max(0.0001, self.conversion_rate * market_eff * elasticity * volatility * (0.5 + self.customer_satisfaction)))
            
            expected = leads * conv
            new_basic = int(expected) + (1 if self.rng.random() < (expected % 1) else 0)
            self.active_basic += new_basic
            if new_basic > 0:
                self.cac = (leads * 12.0) / new_basic + self.noise.sample_cac_noise()
        
        # Enterprise Acquisition
        if self.leads_queue_enterprise:
            ent_leads = self.leads_queue_enterprise.pop(0)
            expected_ent = ent_leads * self.conversion_rate * 0.2
            new_ent = int(expected_ent) + (1 if self.rng.random() < (expected_ent % 1) else 0)
            self.active_enterprise += new_ent
            if new_ent > 0: logs.append(f"Closed {new_ent} Enterprise deals!")
            
        self.active_customers = self.active_basic + self.active_enterprise
        self.market_share = min(1.0, self.active_customers / 5000.0) # Assume TAM is 5000 customers
        
        if self.product_queue:
            self.product_quality += self.product_queue.pop(0)
            
        # Satisfaction & Latency
        self.infrastructure_utilization = self.current_traffic / max(1, self.infrastructure_capacity)
        latency_penalty = max(0.0, self.infrastructure_utilization - 1.0)
        
        price_factor = 1.2 / (0.2 + (self.price_per_user / 25.0) ** 2) if self.price_per_user > 25 else 1.0
        quality_factor = min(1.2, 0.5 + 0.5 * math.log10(1 + self.product_quality))
        latency_factor = max(0.2, 1.0 - latency_penalty)
        
        target_csat = (price_factor * 0.4) + (quality_factor * 0.4) + (latency_factor * 0.2)
        self.customer_satisfaction = self.customer_satisfaction * 0.8 + target_csat * 0.2
        
        # Churn
        price_churn = (self.price_per_user / 30.0) ** 1.8 if self.price_per_user > 30 else 1.0
        support_workload = (self.active_basic / 100.0) + (self.active_enterprise / 10.0)
        support_mult = max(1.0, (support_workload / max(0.5, self.team_size)) ** 0.6)
        quality_mult = 1.2 / (0.2 + self.product_quality * 0.8)
        sat_mult = max(0.8, 2.5 - (self.customer_satisfaction * 1.8))
        
        self.effective_churn = max(0.01, self.churn_rate * price_churn * support_mult * quality_mult * sat_mult)
        
        churned_basic = int(self.active_basic * self.noise.sample_churn_factor(self.effective_churn))
        self.active_basic = max(0, self.active_basic - churned_basic)
        
        ent_churn = self.churn_rate_enterprise * sat_mult
        churned_ent = int(self.active_enterprise * self.noise.sample_churn_factor(ent_churn))
        self.active_enterprise = max(0, self.active_enterprise - churned_ent)
        
        # Financials
        self.active_customers = self.active_basic + self.active_enterprise
        revenue = (self.active_basic * self.price_per_user) + (self.active_enterprise * self.price_enterprise)
        self.mrr = revenue
        
        burn = (self.infrastructure_capacity * 0.1) + (self.team_size * self.employee_salary) + 250.0
        total_debt = abs(min(0.0, self.cash))
        interest = total_debt * 0.025
        self.cash = self.cash + revenue - burn - interest
        
        # Insolvency Check
        max_credit = revenue * 6.0
        trust = 1.0 - (total_debt / 20000.0) ** 2
        avail_credit = max(0.0, (max_credit - total_debt) * max(0.0, trust))
        
        if (max(0.0, self.cash) + revenue + avail_credit) < (burn + total_debt * 0.05):
            self.insolvency_steps += 1
            logs.append(f"[WARNING] Insolvency steps: {self.insolvency_steps}/10")
        else:
            self.insolvency_steps = 0
            
        self.current_traffic = max(0, int(self.current_traffic * 0.7 + self.noise.sample_demand_noise()))
        self.step_number += 1
        return logs, latency_penalty

    def is_terminal(self) -> bool:
        if self.step_number >= self.max_steps: return True
        if self.insolvency_steps >= 10: return True
        if self.cash < self.debt_limit: return True
        return False

    def clone(self) -> SaaSState:
        cloned = copy.deepcopy(self)
        cloned.rng = copy.deepcopy(self.rng)
        return cloned

    def to_dict(self) -> Dict[str, Any]:
        d = copy.deepcopy(self.__dict__)
        if "rng" in d: del d["rng"]
        if "noise" in d: del d["noise"]
        return d
