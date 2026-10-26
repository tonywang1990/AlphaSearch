"""AlphaSearch's campaign loop: one research agent with tools (DOTA ADR 0452), over futs-infra's
research interface v1. Each round:

    select    F = the Frontier: greedy fill + gated swaps over the whole bank, from prod, on val
    brief     a fresh workspace: F, its diagnosis, the bank, moments, the grammar, the ledger,
              the budget, the experience store and the tool library
    plan      the policy writes 1-4 experiments (`train`, `suggest`, `ensemble`, `feature_set`);
              `agent`: an LLM session that calls the tools; `surrogate` / `random`: baselines
    train     every experiment's recipes in one infra job
    score     F' = the Frontier over the grown bank; infra's verdict on the TEST days for F' vs F
              and for each experiment alone (counted looks; the agent never sees them)
    reflect   one ledger row per experiment plus the agent's lessons; tools it wrote join the
              library once their own test passes in the sandbox"""
import json
import os
import shutil
import time
from dataclasses import dataclass, field

import numpy as np

from alphasearch.agent import experience, memory
from alphasearch.agent.log import Log
from alphasearch.agent.plan import plan_agent, plan_baseline, realize
from alphasearch.agent.tools import accept_tools, serve
from alphasearch.agent.workspace import PROMPT, brief
from alphasearch.ensemble.diagnose import diagnose
from alphasearch.ensemble.moments import Stats
from alphasearch.ensemble.select import SWAP_T, Cfg, Selector
from alphasearch.infra.contract import ApiError, check_version, name_of
from alphasearch.search.surrogate import Surrogate

ENSEMBLE_T = 2.0   # the agent's own ensemble must beat the Frontier's pick at paired val t >= 2

DEFAULTS = {"model": "claude-opus-5-5", "effort": "high", "select_objective": "ic", "swap_t": SWAP_T,
            "corrector_w": 0.5, "sur_target": "marginal", "experience": None, "memory": None,
            "prompt": None, "seed": 0, "frontier": False, "docs": (), "memory_stores": {}}


@dataclass
class Round:
    it: int
    ws: str
    infra: object
    desc: object
    cfg: dict
    F: list
    ref: list
    sel: Selector                 # the fixed selection; the agent's tools see the same val days
    gains: dict                   # every trained member's marginal gain to F
    swaps: dict                   # every non-member's best swap into F: (gain, the member it replaces)
    F_score: float
    coords: dict
    sur: Surrogate
    feature_sets: dict
    failed: set
    recent: list = field(default_factory=list)
    seen: dict = field(default_factory=dict)
    untrained: dict = field(default_factory=dict)

    @property
    def tool_sel(self):
        return self.sel

    @property
    def axes(self):
        return self.desc.axes

    @property
    def columns(self):
        return self.desc.columns

    def refresh_recipes(self):
        """The legal untrained recipes, after a feature set grew the grammar."""
        self.desc = self.infra.describe()
        self.untrained = {n: r for r in self.infra.recipes() if (n := name_of(r, self.desc.axes)) not in self.failed}


def selectors(infra, desc, cfg):
    """The fixed selection over the bank. `ic_drift` selects on the mean of the IC on every label
    infra carries (10s/30s/60s on LU: against front-loading), weights fitted to the same target;
    `ic` on the fit label alone; any other objective is scored by infra in batches."""
    labels = desc.labels if cfg["select_objective"] == "ic_drift" else [desc.fit_label]
    scfg = Cfg(k_max=desc.k_max, host=desc.host, corrector_w=cfg["corrector_w"], swap_t=cfg["swap_t"],
               objective="ic" if cfg["select_objective"] in ("ic", "ic_drift") else cfg["select_objective"])
    return Selector(Stats(infra.moments("validation"), labels), Stats(infra.moments("fit"), labels), scfg, infra)


def _verdict(infra, sel, new, old, log):
    if set(new) == set(old):
        return "flat"
    try:
        return infra.verdict(sel.legs(sel.ix(new)), sel.legs(sel.ix(old)))
    except ApiError as e:
        log("verdict", f"not scored: {e}")
        return "not_scored"


def _jsonl(path):
    return [json.loads(x) for x in open(path)] if os.path.exists(path) else []


def run(infra, cfg, out, policy="agent", iters=6, models=16):
    cfg = {**DEFAULTS, **cfg}
    desc = infra.describe()
    check_version(desc)
    os.makedirs(out, exist_ok=True)
    log, lib = Log(out), os.path.join(cfg["experience"] or out, "tools")   # tools outlive a campaign
    if not os.path.exists(os.path.join(out, "agent.md")):                 # the campaign's prompt, frozen
        shutil.copy(cfg["prompt"] or PROMPT, os.path.join(out, "agent.md"))
    if not cfg["memory"] and desc.served:       # the cross-symbol memory, from this week's prod (frozen)
        path = os.path.join(out, "memory.md")
        if not os.path.exists(path):
            open(path, "w").write(memory.render(desc.served, desc.symbol, cfg["memory_stores"]))
        cfg = {**cfg, "memory": path}
    iters_path, ledger_path = os.path.join(out, "iters.jsonl"), os.path.join(out, "ledger.jsonl")
    done, ledger = _jsonl(iters_path), _jsonl(ledger_path)
    failed = {r for d in done for r in d.get("failed", [])}
    feature_sets, ref = dict(desc.feature_sets), list(desc.reference)
    seen = experience.seen(cfg["experience"])
    if cfg["frontier"] and cfg["experience"] and not done:   # the bank is everything this symbol ever
        carry = [r for r in infra.recipes() if name_of(r, desc.axes) in seen]   # trained: free, uncounted
        if carry:
            infra.train(carry)
            log("carry", f"{len(carry)} models every earlier episode trained")
    rng = np.random.default_rng(cfg["seed"] * 1000 + len(done))
    R = None
    for it in range(len(done), iters):
        t0 = time.time()
        sel = selectors(infra, desc, cfg)
        # frontier: F re-picked from prod every round; else grown from the last round's F
        prev = ref if cfg["frontier"] or not done else done[-1]["F_after"]
        F = sel.select([prev])
        gains, swaps, base, y = sel.learn(F, cfg["sur_target"])
        coords = {m.name: m.recipe for m in infra.models()}
        target = {n: y[n] for n in gains if np.isfinite(gains[n])}
        R = Round(it=it, ws=os.path.join(out, f"it_{it}"), infra=infra, desc=desc, cfg=cfg, F=F, ref=ref,
                  sel=sel, gains=gains, swaps=swaps, F_score=base, coords=coords,
                  sur=Surrogate(list(desc.axes), seed=it).fit(coords, target), feature_sets=feature_sets,
                  failed=failed, recent=R.recent if R else [], seen=seen)
        R.refresh_recipes()
        os.makedirs(R.ws, exist_ok=True)
        budget = {"iterations_left": iters - it, "models_this_iteration": models}
        brief(R, ledger, budget, diagnose(sel, F) if sel.cfg.objective == "ic" else [], lib, seen)
        if policy == "agent":
            listener = serve(R, R.ws)
            try:
                plan = plan_agent(R, cfg, log)
            finally:
                listener.close()
        else:
            plan = plan_baseline(R, policy, models, rng)
        exps = (plan.get("experiments") or [])[:4]
        picks, rows, left = {}, [], models
        for e in exps:
            p, facts = realize(R, e, left, rng)
            taken = {r for q in picks.values() for r in q}
            p = [r for r in p if r not in taken]
            picks[e.get("id", str(len(rows)))] = p
            left -= len(p)
            rows.append({"iteration": it, "policy": policy, **{k: e.get(k) for k in (
                "id", "action", "hypothesis", "name", "base", "twins", "target_cell", "members")},
                "n_features": len(e.get("features") or []), "trained": p, **facts})
        if policy == "agent" and left > 0 and R.untrained:  # unspent budget: the surrogate spends it
            fill = [n for n in R.untrained if n not in {r for q in picks.values() for r in q}]
            draw = R.sur.thompson([R.untrained[n] for n in fill], min(left, len(fill)), rng=rng)
            picks["_fill"] = [fill[j] for j in draw]
            rows.append({"iteration": it, "policy": policy, "id": "_fill", "action": "train", "n_features": 0,
                         "hypothesis": "unspent budget, surrogate Thompson draws", "trained": picks["_fill"]})
        allp = sorted({r for q in picks.values() for r in q})
        _, bad = infra.train([R.untrained[n] for n in allp]) if allp else ([], {})
        failed |= set(bad)
        have = {m.name for m in infra.models()}
        R.recent = sorted(set(allp) & have)
        desc = infra.describe()
        sel2 = selectors(infra, desc, cfg)
        start = ref if cfg["frontier"] else F
        new = sel2.select([start])
        for e in exps:                                        # the agent's own ensemble, at its bar
            if e.get("action") == "ensemble":
                prop = [m for m in dict.fromkeys(e.get("members") or []) if m in sel2.val.idx]
                ok = 0 < len(prop) <= desc.k_max and (not desc.host or desc.host in prop)
                g, se = sel2.gain(sel2.ix(prop), sel2.ix(new)) if ok else (0.0, 0.0)
                e["_t"] = round(g / se, 2) if se else None
                e["_adopted"] = bool(ok and se and g / se >= ENSEMBLE_T)
                if e["_adopted"]:
                    new = prop
        f2 = sel2.scorer()
        verdict = _verdict(infra, sel2, new, F, log)
        for r in rows:                                        # each experiment alone
            e = next((x for x in exps if x.get("id") == r["id"]), {})
            if r.get("action") == "ensemble":
                r.update({"adopted": e.get("_adopted"), "t_vs_selected": e.get("_t")})
            other = {m for k, q in picks.items() if k != r["id"] for m in q}
            alone = sel2.select([start], exclude=other)
            r.update({"failed": [m for m in r["trained"] if m not in have],
                      "entered_F": sorted(set(new) & set(r["trained"])),
                      "val_gain": f2(sel2.ix(alone)) - f2(sel2.ix(F)),
                      "test_verdict": _verdict(infra, sel2, alone, F, log) if r["trained"] else "flat"})
        tools = accept_tools(R.ws, lib, plan.get("tools"), log) if policy == "agent" else []
        with open(ledger_path, "a") as fh:
            for r in rows + [{"iteration": it, "lessons": plan.get("lessons") or [], "tools": tools}]:
                fh.write(json.dumps(r, default=float) + "\n")
                ledger.append(r)
        row = {"iteration": it, "policy": policy, "F": F, "F_after": new, "F_val": sel.scorer()(sel.ix(F)),
               "F_after_val": f2(sel2.ix(new)), "test_verdict": verdict, "trained": allp,
               "failed": sorted(bad), "bank": len(have), "llm_usd": round(log.cost, 2),
               "secs": round(time.time() - t0), "stop": bool(plan.get("stop"))}
        with open(iters_path, "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        done.append(row)
        log("iteration", f"{it}: F val {row['F_val']:.4f} -> {row['F_after_val']:.4f}, test {verdict}, "
                         f"trained {len(allp)} ({len(bad)} failed)")
        if row["stop"] and not desc.universe:     # benchmark: the budget is fixed, unused is lost
            break
    if done and not os.path.exists(os.path.join(out, "episode.json")):
        sel = selectors(infra, desc, cfg)
        F = done[-1]["F_after"]
        ep = experience.record(out, os.path.basename(out.rstrip("/")), policy, F, sel.gains(F),
                               {m.name: m.recipe for m in infra.models()}, ledger, done, objective=desc.objective,
                               prompt=os.path.basename(cfg["prompt"] or PROMPT))
        if cfg["experience"]:
            experience.add(cfg["experience"], ep)
    return out
