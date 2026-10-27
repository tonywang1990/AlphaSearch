"""Who decides what to train. Every policy sees the same round (an Observation, `round.py`) and
returns a Plan (`records.py`); the harness around it (selection, training, scoring, records) is
the same for all, so the baselines are the agent's controls at equal budget (452 §9).

    agent      an LLM session in the workspace, with the tools (policies/agent.py)
    surrogate  the random-forest surrogate's Thompson draws (policies/baselines.py)
    random     uniform untrained recipes (policies/baselines.py)

A new policy is a class with `name`, `fills_leftover`, `plan(obs, models, rng, log)` and
`reflect(obs, plan, lib, log)`, registered in POLICIES."""
from alphasearch.policies.agent import AgentPolicy
from alphasearch.policies.baselines import RandomPolicy, SurrogatePolicy

POLICIES = {p.name: p for p in (AgentPolicy, SurrogatePolicy, RandomPolicy)}
