"""
StratOS-RL baseline inference script.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

from stratos_gym import StratosAction, StratosEnv

logging.basicConfig(level=logging.WARNING)

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASKS = [
    "task-1-viral",
    "task-2-enterprise",
    "task-3-price",
]

TASK_BRIEFS = {
    "task-1-viral": "You are already overloaded. Infra utilization starts far above safe limits. Spend on infrastructure first; marketing and hiring are traps until utilization is safe.",
    "task-2-enterprise": "Enterprise customers punish low product quality. Get product_quality to at least 1.0 before aggressive scaling; support capacity also matters.",
    "task-3-price": "The company is underpriced and unit economics are broken. Raising price is the first lever; acquiring more customers at a loss is usually fatal.",
}


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Any) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


_SYSTEM_PROMPT = """You are the CEO presiding over StratOS-RL, a high-fidelity SaaS simulation.
Your goal is to reach the task's target MRR of $10,000 while maintaining solvency.

SYSTEM DYNAMICS (Mental Model):
1. THE RAMP-UP: New hires take 2 months (2 steps) to become productive. You pay salary immediately, but capacity increases later.
2. MARKETING MOMENTUM: Spending on marketing builds 'Brand Momentum'. It provides a decay-based boost to leads over several months, not just one.
3. QUALITY-CHURN LOOP: Product R&D is a long-term play. It lowers churn and improves conversion over time.
4. DEBT & LENDER TRUST: You can go into negative cash (debt), but if you exceed your debt limit or burn too fast, 'Lender Trust' drops, interest rates spike, and you may face insolvency.
5. INFRASTRUCTURE: Scaling infra has a cooldown. You cannot expand every single month.
6. BOTTLENECK FIRST: Every task has a dominant constraint. Solve the constraint that causes immediate failure before chasing generic growth.

DECISION RULES:
- Price is strategic. If price is clearly broken, fix price before buying growth.
- Zero spend is allowed. If runway is critical, conserve cash.
- Early hiring is expensive because hires are unproductive for 2 steps.
- If infra or quality is the binding constraint, marketing can make things worse.
- Do not spread budget "evenly" by default. Concentrate on the highest-leverage bottleneck.
- Use spending discipline. Unless preventing immediate failure, avoid committing more than about 25% of current cash in a single step.
- Product investment compounds slowly; repeated moderate investments are usually safer than one huge bet.

OUTPUT FORMAT:
Respond with ONLY a valid JSON StratosAction:
{
  "action_type": "ALLOCATE",
  "price": float,
  "a_marketing": float,
  "a_product": float,
  "a_infra": float,
  "a_hiring": float,
  "reasoning": "Quick strategic summary"
}
"""

def build_decision_payload(
    task_id: str,
    obs: Dict[str, Any],
    reward_components: Optional[Dict[str, Any]] = None,
    logs: Optional[List[str]] = None,
) -> str:
    payload: Dict[str, Any] = {
        "task_id": task_id,
        "task_brief": TASK_BRIEFS.get(task_id, ""),
        "dashboard": obs,
    }
    if reward_components:
        payload["last_reward_components"] = reward_components
    if logs:
        payload["recent_logs"] = logs[-3:]
    return json.dumps(payload, indent=2)


async def get_llm_action(client: AsyncOpenAI, obs_json: str, conversation: List[Dict[str, str]]) -> StratosAction:
    conversation.append({"role": "user", "content": obs_json})
    # Compatibility for newer models requiring max_completion_tokens
    kwargs = {"max_completion_tokens": 512} if "gpt-5" in MODEL_NAME or "o1" in MODEL_NAME else {"max_tokens": 512}
    
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": _SYSTEM_PROMPT}] + conversation,
        temperature=0.0,
        **kwargs
    )
    content = response.choices[0].message.content or "{}"
    conversation.append({"role": "assistant", "content": content})
    
    content = content.strip().replace("```json", "").replace("```", "").strip()
    return StratosAction.model_validate(json.loads(content))


async def run_episode(env: StratosEnv, client: AsyncOpenAI, task_id: str) -> None:
    log_start(task=task_id, env="stratos-rl", model=MODEL_NAME)
    rewards: List[float] = []
    steps = 0
    conversation: List[Dict[str, str]] = []
    final_state: Optional[Dict[str, Any]] = None
    target_mrr = 10000.0
    last_reward_components: Optional[Dict[str, Any]] = None
    recent_logs: List[str] = []
    
    try:
        obs = await env.reset(task_id=task_id)
        done = False
        
        # Get the full state snapshot to find max_steps
        state_snapshot = await env.state()
        final_state = state_snapshot.stratos_state
        max_steps = final_state.get("max_steps", 25)
        
        while not done and steps < max_steps:
            steps += 1
            obs_json = build_decision_payload(
                task_id=task_id,
                obs=obs.dashboard,
                reward_components=last_reward_components,
                logs=recent_logs,
            )
            try:
                action = await get_llm_action(client, obs_json, conversation)
            except Exception as e:
                action = StratosAction(reasoning=f"Error: {e}")
            
            obs, reward, done, info = await env.step(action)
            rewards.append(reward)
            log_step(steps, action.model_dump_json(), reward, done, None)
            if "reward_components" in info:
                last_reward_components = info["reward_components"]
                print(f"       components={info['reward_components']}")
            recent_logs = obs.logs
            final_state = (await env.state()).stratos_state
            
    finally:
        score = sum(rewards) / max(steps, 1)
        if final_state:
            target_mrr = float(final_state.get("target_mrr", target_mrr))
            success = final_state.get("mrr", 0.0) >= target_mrr and final_state.get("cash", 0.0) >= final_state.get("debt_limit", float("-inf"))
        else:
            success = False
        log_end(success, steps, score, rewards)


async def main() -> None:
    client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    env = StratosEnv.in_process()
    for task_id in TASKS:
        await run_episode(env, client, task_id)

if __name__ == "__main__":
    asyncio.run(main())
