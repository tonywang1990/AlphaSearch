"""The plan of a round: the agent's (an LLM session in the workspace) or a baseline's at the
same budget, and the experiments it names turned into recipes to train."""
import os
import time

import numpy as np

from alpharesearch.agent.llm import LLM, AgentError, json_block
from alpharesearch.infra.contract import name_of
from alpharesearch.search.surrogate import matches

TOOLS_AGENT = "Read,Grep,Glob,Write,Edit,Bash(./tool:*)"
ACTIONS = ("train", "suggest", "ensemble", "feature_set")


def plan_agent(R, cfg, log):
    """One fresh agent session over the workspace; one repair if the reply is not one JSON object."""
    ws = R.ws
    for root, _, _ in os.walk(ws):                       # read-only but work/
        if not os.path.abspath(root).startswith(os.path.abspath(os.path.join(ws, "work"))):
            os.chmod(root, 0o555)
    try:
        for attempt in range(3):
            try:
                reply = LLM(model=cfg["model"], effort=cfg["effort"], tools=TOOLS_AGENT, timeout=3600).call(
                    os.path.join(ws, "agent.md"), "Your workspace is the current directory. "
                    "Follow agent.md and reply with the JSON object only.", ws, log)
                break
            except AgentError:
                if attempt == 2:
                    raise
                time.sleep(120 * (attempt + 1))
    finally:
        for root, _, _ in os.walk(ws):
            os.chmod(root, 0o755)
    plan = json_block(reply or "") or {}
    if reply and "experiments" not in plan:
        fix = os.path.join(ws, "work", "fix_reply.md")
        open(fix, "w").write("The reply below is not ONE valid JSON object. Reply with the same plan "
                             "as one valid JSON object, nothing else.\n\n" + reply)
        plan = json_block(LLM(model=cfg["model"], effort="low", tools="Read").call(
            fix, "Fix the JSON.", ws, log) or "") or {}
    return plan


def plan_baseline(R, policy, models, rng):
    """The surrogate's Thompson draws, or random recipes and a random feature set."""
    names = list(R.untrained)
    if policy == "surrogate":
        pick = [names[j] for j in R.sur.thompson([R.untrained[n] for n in names], models, rng=rng)]
        return {"experiments": [{"id": "s", "action": "train", "recipes": pick,
                                 "hypothesis": "surrogate Thompson draws"}]}
    exps = [{"id": "r1", "action": "train", "hypothesis": "random recipes",
             "recipes": list(rng.choice(names, min(models // 2, len(names)), replace=False))}]
    if R.feature_sets:
        base = str(rng.choice(sorted(R.feature_sets)))
        cols = [c for c in R.columns if c not in set(R.feature_sets[base])]
        exps.append({"id": "r2", "action": "feature_set", "hypothesis": "random columns",
                     "name": f"rnd_{rng.integers(1e9)}", "base": base,
                     "features": list(rng.choice(cols, min(24, len(cols)), replace=False)),
                     "twins": list(rng.choice(R.F, min(4, len(R.F)), replace=False)), "k": models // 2})
    return {"experiments": exps}


def _twin(recipe, value):
    return {**recipe, "feature_set": value}


def realize(R, exp, left, rng):
    """-> (recipe names to train, the ledger row's facts): an experiment checked, screened and
    turned into at most `left` recipes."""
    a, row = exp.get("action"), {}
    if a not in ACTIONS:
        return [], {"refused": f"unknown action {a!r}"}
    if a == "ensemble":                                   # judged after selection, no training
        return [], {}
    if a == "train":
        want = list(dict.fromkeys(exp.get("recipes") or []))
        row["refused_recipes"] = {r: ("already trained" if r in R.coords else "not in available.txt")
                                  for r in want if r not in R.untrained}
        return [r for r in want if r in R.untrained][:left], row
    if a == "suggest":
        pool = [u for u in R.untrained if matches(R.untrained[u], exp.get("where"))]
        if exp.get("new_only"):
            pool = [u for u in pool if u not in R.seen]
        k = min(int(exp.get("n") or 8), left, len(pool))
        row["pool"] = len(pool)
        return ([pool[j] for j in R.sur.thompson([R.untrained[u] for u in pool], k, rng=rng)] if k else []), row
    # feature_set: screened against same-size random columns before anything trains
    base = R.feature_sets.get(exp.get("base")) or []
    feats = [c for c in dict.fromkeys(exp.get("features") or []) if c not in set(base)]
    if not feats:
        return [], {"refused": "no new columns"}
    sel = R.tool_sel
    row["screen"] = R.infra.screen(feats, sel.legs(sel.ix(R.F)))
    beats = [v["beats_null"] for c, v in row["screen"].items() if "/" not in c]
    if not beats or np.mean(beats) <= 0:
        return [], {**row, "refused": f"screen beats_null {np.mean(beats or [0]):+.2e}"}
    got = R.infra.propose_feature_set(exp.get("name", ""), feats, base=exp.get("base"))
    if not got.get("admitted"):
        return [], {**row, "refused": got.get("why")}
    value = got["value"]
    R.feature_sets[value] = got["columns"]
    R.refresh_recipes()
    cands = [n for n, r in R.untrained.items() if r.get("feature_set") == value]
    k = min(int(exp.get("k") or 8), left)
    mine = [t for t in dict.fromkeys(name_of(_twin(R.coords[m], value), R.desc.axes)
                                     for m in exp.get("twins") or [] if m in R.coords) if t in cands][:k]
    rest = [n for n in cands if n not in mine]
    if k > len(mine) and rest:
        mine += [rest[j] for j in R.sur.thompson([R.untrained[n] for n in rest], k - len(mine), rng=rng)]
    return mine, {**row, "feature_set": value}

