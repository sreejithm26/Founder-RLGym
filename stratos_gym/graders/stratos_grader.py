from __future__ import annotations

import math
from typing import Any, Dict, Optional

from stratos_gym.models import StratosAction, StratosObservation, StratosReward
from stratos_gym.state import SUPPORT_CAPACITY_PER_FTE, SaaSState


class StratosGrader:
    """A dense reward function normalized to the OpenEnv 0.0-1.0 range."""

    def grade(
        self,
        action: StratosAction,
        state: SaaSState,
        ground_truth: Dict[str, Any],
        step_number: int,
    ) -> StratosReward:
        """Calculate a dense, bounded reward signal for agent learning."""
        
        # 1. TERMINATION (CRITICAL FEEDBACK)
        target_mrr = float(ground_truth.get("target_mrr", 10000.0))

        if state.is_terminal():
            # Insolvency is the ultimate failure
            if state.cash < state.debt_limit:
                return StratosReward(value=0.0, components={"failure": 0.0}, done=True)
            
            # SUCCESS (Scaling to IPO)
            if state.mrr >= target_mrr:
                return StratosReward(value=1.0, components={"victory": 1.0}, done=True)

        # 2. TEMPORAL SIGNALS (WHAT CHANGED?)
        
        # A. Growth Velocity (Primary Reward)
        # We reward the DELTA of MRR.
        mrr_growth = state.mrr - state.prev_mrr
        # Use tanh to normalize growth signals (-2 to 2)
        mrr_reward = math.tanh(mrr_growth / 1000.0) * 5.0
        
        # B. Cash Gravity (Scaled Debt Penalty)
        cash_penalty = 0.0
        if state.cash < 0:
            # Quadratic scaling: debt is exponentially more dangerous
            debt_severity = abs(state.cash) / 1000.0
            cash_penalty = -(debt_severity ** 1.5)
            
        # C. Runway Punishment (Non-linear tanh penalty)
        runway_penalty = 0.0
        burn = state.calculate_burn()
        mrr = max(0.1, state.mrr)
        net_burn = max(0.1, burn - state.mrr)
        
        if state.cash > 0:
            runway = state.cash / net_burn if net_burn > 1.0 else 99
            # Penalty activates when runway < 4 months, approaches -8.0 as runway hits 0
            runway_penalty = math.tanh((runway - 4.0) / 2.0) * 4.0 - 4.0
        else:
            runway_penalty = -10.0 # Absolute floor

        # D. Unit Economics (Balanced Gradient)
        # Target Ratio is 3.0+
        ratio = state.ltv / max(1.0, state.cac)
        # Rewards efficiency improvements and punishes CAC-blindness
        unit_econ_reward = math.tanh((ratio - 3.0) / 2.0) * 2.0
        
        # E. Friction (Churn & Overload)
        # Churn is the 'anti-growth' signal
        churn_penalty = -8.0 * state.effective_churn
        
        # Support Overload
        capacity = max(float(SUPPORT_CAPACITY_PER_FTE), state.team_size * float(SUPPORT_CAPACITY_PER_FTE))
        utilization = state.active_customers / max(1, capacity)
        support_penalty = 0.0
        if utilization > 1.0:
            support_penalty = -4.0 * (utilization - 1.0)
        elif 0.7 <= utilization <= 0.95:
            support_penalty = 0.5 # Efficiency bonus

        # 3. COMPOSITE REWARD
        raw_total = (
            mrr_reward + 
            cash_penalty + 
            runway_penalty + 
            unit_econ_reward + 
            churn_penalty + 
            support_penalty
        )
        
        # Normalize shaping into the required [0.0, 1.0] interval.
        total_value = (math.tanh(raw_total / 8.0) + 1.0) / 2.0
        total_value = max(0.0, min(1.0, total_value))

        return StratosReward(
            value=round(float(total_value), 4),
            components={
                "growth": round(mrr_reward, 4),
                "fiscal": round(cash_penalty + runway_penalty, 4),
                "unity": round(unit_econ_reward, 4),
                "friction": round(churn_penalty + support_penalty, 4)
            },
            done=state.is_terminal()
        )
