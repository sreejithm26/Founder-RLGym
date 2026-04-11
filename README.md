---
title: StratOS-RL
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
license: mit
---

# 🪐 StratOS-RL: The SaaS Strategic RL Benchmark

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-green.svg)](https://github.com/openenv/spec)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**StratOS-RL** is a high-fidelity reinforcement learning environment designed to evaluate an agent's ability to manage complex, non-linear economic systems. Unlike traditional control tasks, StratOS-RL focuses on **Long-Horizon Planning**, **Delayed Feedback Loops**, and **Multi-Objective Resource Allocation**.

---

## 📊 Baseline Performance

The following results were collected during core experimentation, demonstrating the non-linear difficulty jump between tasks and the performance delta of various reasoning models.

| Task (Difficulty)         | Model       | Score      | Steps | Success | Key Behavior                       |
| ------------------------- | ----------- | ---------- | ----- | ------- | ---------------------------------- |
| **Task 1 – Viral** 🟢     | GPT-4o-mini | 0.578      | 20    | ❌       | Product loop, slow decline         |
|                           | GPT-4o      | 0.553      | 20    | ❌       | Marketing spam → plateau           |
|                           | GPT-5.4     | **0.739**  | **6** | ✅       | Fast win via pricing + infra       |
| **Task 2 – Enterprise** 🟡 | GPT-4o-mini | 0.462      | 25    | ❌       | Late-stage failure                 |
|                           | GPT-4o      | ~0.58      | 25    | ❌       | Builds, no monetization            |
|                           | GPT-5.4     | **0.644**  | **16**| ✅       | quality → demand → pricing         |
| **Task 3 – Pricing** 🔴    | GPT-4o-mini | 0.102      | 28    | ❌       | Cash death spiral                  |
|                           | GPT-4o      | 0.61       | 30    | ❌       | Oscillation (price ↔ growth)       |
|                           | GPT-5.4     | 0.596      | 30    | ❌       | Price elasticity + recovery loops  |

> [!IMPORTANT]
> The environment is tuned for **GPT-4o-level reasoning**. Smaller models frequently hallucinate fiscal bounds or fail to prioritize bottleneck resolution, leading to rapid insolvency.

---

## 🔬 The Scientific Challenge: "Temporal Credit Assignment in Stochastic Economies"

StratOS-RL is specifically engineered to be **insolvable by simple reactive agents**. It presents a "Hard AI" problem through three core mechanisms:

1.  **Stochastically Lagged Queues**: Marketing spend and R&D don't yield results immediately. We implement **Multi-Phase Leads Queues** and **Brand Momentum** decay models, forcing agents to value delayed rewards.
2.  **Human Capital Latency**: Hiring isn't instant. New employees enter a **3-step Training Queue**, creating a "Burn-before-Value" period that tests long-horizon liquidity management.
3.  **The Credit Squeeze Paradox**: High-growth strategies often lead to liquidity crises. The environment implements a "Lender Trust" metric that non-linearly scales interest rates and bankruptcy risk based on historical burn rates.

---

## 🛠️ Installation & Setup

### 1. Clone & Install
```bash
git clone https://github.com/your-username/stratos-rl.git
cd stratos-rl
pip install -e .
```

### 2. Launch the Environment Server
StratOS-RL uses an OpenEnv-compliant FastAPI server architecture.
```bash
# Start the local environment
python -m server.app
```

### 3. Run a Baseline Agent
```bash
export OPENAI_API_KEY=...
export API_BASE_URL=...
export MODEL_NAME=...
python inference.py
```

---

## 🎯 Evaluation Tasks

StratOS-RL includes three deterministic evaluation scenarios:

| Task ID | Name | Difficulty | Objective |
| :--- | :--- | :--- | :--- |
| `task-1-viral` | **The Viral Crash** | Easy | Managing sudden traffic influx with high momentum. |
| `task-2-enterprise` | **The Enterprise Pivot** | Medium | Managing high-LTV, high-churn enterprise accounts. |
| `task-3-price` | **The Efficiency War** | Hard | Survival in low-capital, high-churn environments. |

---

## 📊 Environment Specification

### Observation Space (`StratosObservation`)
The agent receives a full "CEO Dashboard" every step (1 month per step):
- `bank`: Current Cash and Monthly Recurring Revenue (MRR).
- `unit_economics`: Customer Acquisition Cost (CAC), Lifetime Value (LTV), churn, and price point.
- `internal_metrics`: Product quality, conversion rate, and competitor strength.
- `ops`: Active customers, support utilization, and infrastructure utilization.

### Action Space (`StratosAction`)
The agent must allocate resources across 5 dimensions:
- `price`: Subscription pricing strategy.
- `a_marketing`: Direct customer acquisition spend.
- `a_product`: R&D and product quality investment.
- `a_infra`: Server and infrastructure expansion.
- `a_hiring`: Talent acquisition and team scaling.

---

> [!NOTE]
> This environment is built for **OpenEnv compliance**. It provides a deterministic grader and standardized server/client interfaces for fair benchmarking.
