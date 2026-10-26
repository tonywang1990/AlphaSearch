"""Tools over the trained bank: moment arithmetic on the val days, no infra job."""
import re

import numpy as np

from alphasearch.search.surrogate import axis_summary, effects
from alphasearch.tools import opt, tool


@tool("whatif --add M... --drop M...", "F with members added or dropped: val score, paired gain over F, SE, t")
def whatif(obs, args):
    sel, F = obs.sel, obs.F
    s = [n for n in F if n not in opt(args, "--drop")] + opt(args, "--add")
    bad = [n for n in s if n not in sel.val.idx]
    if bad or not 0 < len(s) <= sel.cfg.k_max:
        return {"error": f"{len(s)} members (1-{sel.cfg.k_max}); not trained: {bad[:3]}"}
    f = sel.scorer()
    g, se = sel.gain(sel.ix(s), sel.ix(F))
    return {"members": s, "val": f(sel.ix(s)), "F_val": f(sel.ix(F)), "val_gain": g,
            "val_gain_se": se, "t": g / se if se else None}


@tool("members [regex]", "trained models by their marginal val gain to F (top 60)")
def members(obs, args):
    rx = re.compile(args[0]) if args else None
    rows = sorted(((g, n) for n, g in obs.gains.items() if not rx or rx.search(n)), reverse=True)
    return [{"member": n, "marginal_val_gain": round(float(g), 5), "in_F": n in obs.F} for g, n in rows[:60]]


@tool("effects", "mean gain per axis value and value pair, with SE and n")
def effects_(obs, args):
    return effects(obs.gains, obs.coords)


@tool("axes", "per axis value: mean gain of the trained models carrying it, and n")
def axes(obs, args):
    done = [n for n in obs.gains if n in obs.coords]
    return axis_summary([obs.coords[n] for n in done], [obs.gains[n] for n in done])


@tool("suggest [--n N] [--set FS]", "the surrogate's top untrained recipes, expected gain and sd")
def suggest(obs, args):
    n = int(opt(args, "--n")[0]) if opt(args, "--n") else 20
    want = opt(args, "--set")
    pool = [u for u in obs.untrained if not want or obs.untrained[u].get("feature_set") in want]
    if not pool:
        return []
    m = obs.sur.samples([obs.untrained[u] for u in pool])
    return [{"recipe": pool[i], "expected_gain": float(m.mean(0)[i]), "sd": float(m.std(0)[i])}
            for i in np.argsort(-m.mean(0))[:n]]


@tool("predcheck", "the surrogate's held-out Spearman on the trained models: the bar for your own predictor")
def predcheck(obs, args):
    return {"target": obs.cfg.sur_target, **obs.sur.predcheck(obs.coords)}


@tool("crossval", "select on half the val days from prod, score on the other: what the last batch and the bank add")
def crossval(obs, args):
    return obs.sel.crossval(obs.ref, recent=obs.recent)


@tool("search [--starts N]", "multi-start local search over ensembles: best sets, t over F, held-out gain (IC only)")
def search(obs, args):
    n = int(opt(args, "--starts")[0]) if opt(args, "--starts") else 8
    return obs.sel.search(obs.F, starts=n, seed=obs.it)
