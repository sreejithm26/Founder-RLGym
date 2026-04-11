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
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASKS = ["task-1-growth", "task-2-viral", "task-3-price"]
MAX_STEPS = {"task-1-growth": 25, "task-2-viral": 20, "task-3-price": 20}
SUCCESS_SCORE_THRESHOLD = 0.5


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Any) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


_SYSTEM_PROMPT = """You are the CEO presiding over StratOS-RL, a high-fidelity SaaS simulation.
Your goal is to scale MRR to $10,000 while maintaining solvency.

SYSTEM DYNAMICS (Mental Model):
1. THE RAMP-UP: New hires take 3 months (3 steps) to become productive. You pay salary immediately, but capacity increases later.
2. MARKETING MOMENTUM: Spending on marketing builds 'Brand Momentum'. It provides a decay-based boost to leads over several months, not just one.
3. QUALITY-CHURN LOOP: Product R&D is a long-term play. It lowers churn and improves conversion over time.
4. DEBT & LENDER TRUST: You can go into negative cash (debt), but if you exceed your debt limit or burn too fast, 'Lender Trust' drops, interest rates spike, and you may face insolvency.
5. INFRASTRUCTURE: Scaling infra has a cooldown. You cannot expand every single month.

OUTPUT FORMAT:
Respond with ONLY a valid JSON StratosAction:
{
  "action_type": "ALLOCATE",
  "price": float,
  "a_marketing": float,
  "a_product": float,
  "a_infra": float,
  "a_hiring": float,
  "a_debt_repayment": float,
  "reasoning": "Quick strategic summary"
}
"""

async def get_llm_action(client: AsyncOpenAI, obs_json: str, conversation: List[Dict[str, str]]) -> StratosAction:
    conversation.append({"role": "user", "content": obs_json})
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": _SYSTEM_PROMPT}] + conversation,
        max_tokens=512,
        temperature=0.2,
    )
    content = response.choices[0].message.content or "{}"
    conversation.append({"role": "assistant", "content": content})
    
    content = content.strip().replace("```json", "").replace("```", "").strip()
    return StratosAction.model_validate(json.loads(content))


async def run_episode(env: StratosEnv, client: AsyncOpenAI, task_id: str) -> None:
    log_start(task=task_id, env="stratos-rl", model=MODEL_NAME)
    rewards: List[float] = []
    steps = 0
    max_steps = MAX_STEPS[task_id]
    conversation: List[Dict[str, str]] = []
    
    try:
        result = await env.reset(task_id=task_id)
        while not result.done and steps < max_steps:
            steps += 1
            obs_json = json.dumps(result.observation.model_dump(), indent=2)
            try:
                action = await get_llm_action(client, obs_json, conversation)
            except Exception as e:
                action = StratosAction(reasoning=f"Error: {e}")
            
            result = await env.step(action)
            rewards.append(result.reward)
            log_step(steps, action.model_dump_json(), result.reward, result.done, None)
            
    finally:
        score = sum(rewards) / max_steps
        log_end(score >= SUCCESS_SCORE_THRESHOLD, steps, score, rewards)


async def main() -> None:
    client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    env = StratosEnv.in_process()
    for task_id in TASKS:
        await run_episode(env, client, task_id)

if __name__ == "__main__":
    asyncio.run(main())
