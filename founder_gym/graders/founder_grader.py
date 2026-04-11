from __future__ import annotations

from typing import Any, Dict, List
from founder_gym.models import FounderAction, FounderReward
from founder_gym.state import SaaSState


class FounderGrader:
    """Grader for all Founder Gym tasks."""

    def grade(
        self,
        action: FounderAction,
        state: SaaSState,
        ground_truth: Dict[str, Any],
        step_number: int,
    ) -> Dict[str, Any]:
        """Grade the action and return a FounderReward-compatible dict."""
        
        # 1. Step Reward (Dense / Shaping)
        # MRR growth signal
        mrr_growth = (state.mrr / 10000.0) * 0.2
        
        # Profitability
        burn = (state.team_size * state.employee_salary) + (state.infrastructure_capacity * 0.1) + 500.0
        profit_margin = ((state.mrr - burn) / max(1.0, state.mrr)) * 0.1
        profit_margin = max(-0.1, min(0.1, profit_margin))
        
        # CSAT
        csat_bonus = (state.customer_satisfaction - 0.5) * 0.1
        
        step_reward = 0.1 + mrr_growth + profit_margin + csat_bonus
        
        # 2. Terminal Score (The actual [0, 1] grader output)
        done = state.is_terminal()
        
        if done:
            is_bankrupt = state.insolvency_steps >= 10 or state.cash < state.debt_limit
            if is_bankrupt:
                reward = (state.step_number / state.max_steps) * 0.1
            else:
                survival = min(1.0, state.step_number / state.max_steps)
                mrr_score = min(1.0, state.mrr / 10000.0)
                cash_score = max(0.0, min(1.0, state.cash / 10000.0))
                
                reward = (survival * 0.30) + (mrr_score * 0.60) + (cash_score * 0.10)
        else:
            reward = step_reward
            
        reward = float(min(1.0, max(0.0, reward)))
        
        return {
            "reward": reward,
            "done": done,
            "components": {
                "growth": mrr_growth,
                "efficiency": profit_margin,
                "csat": csat_bonus,
            },
            "info": {
                "step": step_number,
                "mrr": state.mrr,
                "cash": state.cash,
                "is_bankrupt": state.insolvency_steps >= 10 or state.cash < state.debt_limit
            }
        }
