from __future__ import annotations

import math
from typing import Any, Dict, List
from stratos_gym.models import StratosAction, StratosReward
from stratos_gym.state import SaaSState


class StratosGrader:
    """Grader for all StratOS-RL tasks with task-specific reward weighting."""

    # Reward weight profiles for different scientific scenarios
    TASK_PROFILES = {
        "task-1-growth": {
            "survival": 0.20,
            "mrr": 0.70,
            "cash": 0.10,
            "satisfaction": 0.00
        },
        "task-2-viral": {
            "survival": 0.20,
            "mrr": 0.20,
            "cash": 0.10,
            "satisfaction": 0.50  # High weight on handling the surge (satisfaction)
        },
        "task-3-price": {
            "survival": 0.20,
            "mrr": 0.30,
            "cash": 0.40,  # Focus on capital efficiency (profit/cash)
            "satisfaction": 0.10
        }
    }

    def grade(
        self,
        action: StratosAction,
        state: SaaSState,
        ground_truth: Dict[str, Any],
        step_number: int,
    ) -> Dict[str, Any]:
        """Grade the action using task-specific reward profiles."""
        
        # 1. Component Scoring
        # MRR Score (normalized to $10k target)
        mrr_score = min(1.0, state.mrr / 10000.0)
        
        # Efficiency Score (Cash reserves vs target)
        cash_score = max(0.0, min(1.0, state.cash / 10000.0))
        
        # Market Share Score
        mkt_score = min(1.0, state.market_share * 10.0) # Assume 10% is a good target
        
        # Stability Score (Customer Satisfaction)
        csat_score = state.customer_satisfaction
        
        # Survival Score (Progress through episode)
        survival_score = min(1.0, state.step_number / state.max_steps)
        
        # 2. Get Task Weights
        profile = self.TASK_PROFILES.get(state.active_task, self.TASK_PROFILES["task-1-growth"])
        
        # 3. Reward Calculation
        done = state.is_terminal()
        is_bankrupt = state.insolvency_steps >= 10 or state.cash < state.debt_limit
        
        if done:
            if is_bankrupt:
                # Heavy penalty for bankruptcy - only survival reward granted
                reward = (state.step_number / state.max_steps) * 0.05
            else:
                # Weighted terminal evaluation
                reward = (
                    (survival_score * profile["survival"]) +
                    (mrr_score * profile["mrr"]) +
                    (cash_score * profile["cash"]) +
                    (csat_score * profile["satisfaction"]) +
                    (mkt_score * 0.1 if state.active_task == "task-1-growth" else 0.0)
                )
        else:
            # Step-wise shaping (dense reward)
            # We use a mix of linear and log scaling to ensure strong early-game signal
            log_mrr = (math.log10(1 + state.mrr) / 4.0) # 0.0 to 1.0 (at 10k)
            
            step_reward = (
                (log_mrr * 0.4) +     # Increased weight on growth trajectory
                (csat_score * 0.2) + 
                (0.1 if not is_bankrupt else 0.0)
            )
            reward = step_reward
            
        reward = float(min(1.0, max(0.0, reward)))
        
        return {
            "reward": reward,
            "done": done,
            "components": {
                "mrr_norm": mrr_score,
                "cash_norm": cash_score,
                "csat": csat_score,
                "survival": survival_score
            },
            "info": {
                "task": state.active_task,
                "step": step_number,
                "mrr": state.mrr,
                "cash": state.cash,
                "is_bankrupt": is_bankrupt
            }
        }
