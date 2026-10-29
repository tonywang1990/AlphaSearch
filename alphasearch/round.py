"""One round of a campaign, as five phases. Each takes what the previous one returned; the
campaign (`campaign.py`) calls them in order and the journal records what they found.

    observe   select F over the bank (the fixed frontier search), learn every model's gain and
              best swap to F, fit the surrogate, list the trainable recipes: the Observation
    plan      the policy reads the Observation (the agent through its workspace and tools) and
              returns a Plan (policies/)
    act       the Plan's experiments turned into recipes (actions.py), leftover budget to the
              surrogate if the policy asks, all trained in one infra job
    evaluate  F' = the search over the grown bank; the agent's ensemble adopted at ENSEMBLE_T;
              infra's sealed verdict on the test days for F' vs F and for each experiment alone
    record    the ledger rows (val facts, the verdicts sealed from the agent) and the round row"""
import os
from dataclasses import dataclass, field

import numpy as np

from alphasearch import actions
from alphasearch.config import ENSEMBLE_T
from alphasearch.ensemble.moments import Stats
from alphasearch.ensemble.select import Cfg, Selector
from alphasearch.ensemble.walkforward import merge
from alphasearch.infra.contract import ApiError, name_of
from alphasearch.records import LEDGER_KEYS
from alphasearch.search.surrogate import Surrogate


@dataclass
class Observation:
    """What a policy knows about the round (the agent reads it as files, `context/`)."""
    it: int
    ws: str
    infra: object
    desc: object
    cfg: object
    F: list                       # the frontier search's pick, the round's starting ensemble
    ref: list                     # prod's members: the 1.0 of every score
    sel: Selector                 # the fixed selection over the bank, on the val days
    gains: dict                   # every trained member's marginal gain to F
    swaps: dict                   # every non-member's best swap into F: (gain, the member it replaces)
    F_score: float
    coords: dict                  # every trained member's recipe
    sur: Surrogate
    feature_sets: dict
    failed: set
    recent: list = field(default_factory=list)     # the last round's newly trained models
    seen: dict = field(default_factory=dict)       # every recipe the experience store has seen
    untrained: dict = field(default_factory=dict)  # every legal recipe trainable now

    @property
    def columns(self):
        return self.desc.columns

    def refresh_recipes(self):
        """The legal untrained recipes (again, after a feature set grew the grammar)."""
        self.desc = self.infra.describe()
        self.untrained = {n: r for r in self.infra.recipes() if (n := name_of(r, self.desc.axes)) not in self.failed}


def selector(infra, desc, cfg):
    """The fixed selection over the bank. `ic_drift` selects on the mean IC over every label infra
    carries (10s/30s/60s on LU), weights fitted to the same target; `ic` on the fit label; any other
    objective is scored by infra in batches."""
    labels = desc.labels if cfg.select_objective == "ic_drift" else [desc.fit_label]
    scfg = Cfg(k_max=desc.k_max, host=desc.host, corrector_w=cfg.corrector_w, swap_t=cfg.swap_t,
               objective="ic" if cfg.select_objective in ("ic", "ic_drift") else cfg.select_objective,
               wf_chunks=cfg.wf_chunks, worst_weight=cfg.worst_weight, weight_shrink=cfg.weight_shrink,
               regime_cells=cfg.regime_cells, keep_reference=cfg.keep_reference)
    if scfg.objective == "wf":               # the SoM aggregator: fit + validation, walked forward
        agg = Stats(merge(infra.moments("fit"), infra.moments("validation")), labels)
        return Selector(agg, agg, scfg, infra)
    val = Stats(infra.moments("validation"), labels)
    fit = val if cfg.weights_block == "validation" else Stats(infra.moments("fit"), labels)
    return Selector(val, fit, scfg, infra)


def observe(infra, desc, cfg, it, out, start, carry):
    """F from `start` (prod in frontier mode, else the last round's F). `carry`: what outlives a
    round (failed, feature_sets, seen, recent), shared with the campaign."""
    sel = selector(infra, desc, cfg)
    F = sel.select([start])
    gains, swaps, base, y = sel.learn(F, cfg.sur_target)
    coords = {m.name: m.recipe for m in infra.models()}
    target = {n: y[n] for n in gains if np.isfinite(gains[n])}
    obs = Observation(it=it, ws=os.path.join(out, f"it_{it}"), infra=infra, desc=desc, cfg=cfg, F=F,
                      ref=list(desc.reference), sel=sel, gains=gains, swaps=swaps, F_score=base, coords=coords,
                      sur=Surrogate(list(desc.axes), seed=it).fit(coords, target), **carry)
    obs.refresh_recipes()
    os.makedirs(obs.ws, exist_ok=True)
    return obs


@dataclass
class Act:
    picks: dict                   # experiment id -> the recipes it trained
    rows: list                    # one ledger row per experiment
    trained: list                 # every recipe trained this round
    failed: dict                  # recipe -> why it failed


def act(obs, plan, policy, models, rng):
    picks, rows, left = {}, [], models
    for e in plan.experiments:
        p, facts = actions.realize(obs, e, left, rng)
        taken = {r for q in picks.values() for r in q}
        p = [r for r in p if r not in taken]          # a recipe picked by two experiments is spent once
        picks[e.get("id", str(len(rows)))] = p
        left -= len(p)
        rows.append({"iteration": obs.it, "policy": policy.name, **{k: e.get(k) for k in LEDGER_KEYS},
                     "n_features": len(e.get("features") or []), "trained": p, **facts})
    if policy.fills_leftover and left > 0 and obs.untrained:
        picks["_fill"] = actions.fill(obs, picks, left, rng)
        rows.append({"iteration": obs.it, "policy": policy.name, "id": "_fill", "action": "train", "n_features": 0,
                     "hypothesis": "unspent budget, surrogate Thompson draws", "trained": picks["_fill"]})
    trained = sorted({r for q in picks.values() for r in q})
    _, bad = obs.infra.train([obs.untrained[n] for n in trained]) if trained else ([], {})
    obs.failed |= set(bad)
    return Act(picks=picks, rows=rows, trained=trained, failed=bad)


def _verdict(infra, sel, new, old, log):
    if set(new) == set(old):
        return "flat"
    try:
        return infra.verdict(sel.legs(sel.ix(new)), sel.legs(sel.ix(old)))
    except ApiError as e:
        log("verdict", f"not scored: {e}")
        return "not_scored"


@dataclass
class Outcome:
    F_after: list
    sel: Selector                 # the selection over the grown bank
    verdict: str                  # infra's sealed word on the test days, F' vs F
    have: set                     # the bank's members after training


def evaluate(obs, plan, a, log):
    infra, cfg, F = obs.infra, obs.cfg, obs.F
    have = {m.name for m in infra.models()}
    obs.recent = sorted(set(a.trained) & have)
    desc = infra.describe()
    sel = selector(infra, desc, cfg)
    start = obs.ref if cfg.frontier else F
    new = sel.select([start])
    for e in plan.experiments:                          # the agent's own ensemble, at its bar
        if e.get("action") == "ensemble":
            prop = [m for m in dict.fromkeys(e.get("members") or []) if m in sel.val.idx]
            ok = 0 < len(prop) <= desc.k_max and (not desc.host or desc.host in prop)
            g, se = sel.gain(sel.ix(prop), sel.ix(new)) if ok else (0.0, 0.0)
            e["_t"] = round(g / se, 2) if se else None
            e["_adopted"] = bool(ok and se and g / se >= ENSEMBLE_T)
            if e["_adopted"]:
                new = prop
    f = sel.scorer()
    verdict = _verdict(infra, sel, new, F, log)
    for r in a.rows:                                    # each experiment alone
        e = next((x for x in plan.experiments if x.get("id") == r["id"]), {})
        if r.get("action") == "ensemble":
            r.update({"adopted": e.get("_adopted"), "t_vs_selected": e.get("_t")})
        other = {m for k, q in a.picks.items() if k != r["id"] for m in q}
        alone = sel.select([start], exclude=other)
        r.update({"failed": [m for m in r["trained"] if m not in have],
                  "entered_F": sorted(set(new) & set(r["trained"])),
                  "val_gain": f(sel.ix(alone)) - f(sel.ix(F)),
                  "test_verdict": _verdict(infra, sel, alone, F, log) if r["trained"] else "flat"})
    return Outcome(F_after=new, sel=sel, verdict=verdict, have=have), desc


def record(journal, obs, plan, a, o, tools, policy, log, secs):
    row = {"iteration": obs.it, "policy": policy.name, "F": obs.F, "F_after": o.F_after,
           "F_val": obs.sel.scorer()(obs.sel.ix(obs.F)), "F_after_val": o.sel.scorer()(o.sel.ix(o.F_after)),
           "test_verdict": o.verdict, "trained": a.trained, "failed": sorted(a.failed), "bank": len(o.have),
           "llm_usd": round(log.cost, 2), "secs": round(secs), "stop": plan.stop,
           "F_after_legs": o.sel.legs(o.sel.ix(o.F_after)),             # exactly what this round would serve
           **({"F_after_cells": o.sel.wf.table(o.sel.ix(o.F_after))} if o.sel.wf else {})}
    journal.add_round(a.rows + [{"iteration": obs.it, "lessons": plan.lessons, "tools": tools}], row)
    log("iteration", f"{obs.it}: F val {row['F_val']:.4f} -> {row['F_after_val']:.4f}, test {o.verdict}, "
                     f"trained {len(a.trained)} ({len(a.failed)} failed)")
    return row
