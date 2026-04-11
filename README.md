---
title: Founder Gym
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
license: mit
---

# Founder Gym (v2)

**SaaS CEO Simulator for Reinforcement Learning**

Restructured to match the `episteward` architecture for clean OpenEnv compatibility.

## Structure
- `founder_gym/`: Core package
  - `env.py`: OpenEnv environment entry point
  - `state.py`: SaaS simulation state & transition logic
  - `models.py`: Pydantic models (Action, Observation, etc.)
  - `tasks/`: Evaluation scenarios
  - `graders/`: Performance scoring logic
  - `math/`: Physics/Financial math modules
- `server/`: FastAPI server hosting the environment
- `inference.py`: Baseline LLM agent script

## Quick Start
```bash
pip install -e .
python -m server.app
```

## API
| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Environment identity check |
| `/health` | GET | Liveness probe |
| `/tasks` | GET | List available task IDs |
| `/reset` | POST | Reset environment (accepts `task_id` and `seed`) |
| `/step` | POST | Advance simulation with `FounderAction` |
| `/state` | GET | Read-only state snapshot |
