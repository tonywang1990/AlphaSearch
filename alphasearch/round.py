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
import functools
import json
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
    credibility: dict | None      # is F more than val fitting: cross-fit t, time halves, periods (Selector.credibility)
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


@functools.lru_cache(maxsize=4)
def _cost(path):
    """The operator's cost table: member -> its served features (client names, `.s1` dropped)."""
    if not path:
        return None
    return {k.removesuffix(".s1"): frozenset(v) for k, v in json.load(open(path)).items()}


def _costs(infra, desc, cfg):
    """member -> served features: the operator's table, then each trained member's own report (a live campaign's
    members exist only once trained). A sequence member (it serves NA inputs) also pays `seq_cost` pseudo-features for
    its recurrence; the NA inputs it shares with other sequence legs are counted once by the union."""
    if not cfg.cost_lambda:                # no price: no cost model (a set's features are not needed)
        return None
    table = dict(_cost(cfg.cost_table) or {})
    for m in infra.models():
        if m.name not in table:
            seq = any(f.startswith("na::") for f in m.features)
            table[m.name] = frozenset(m.features) | {f"SEQ::{m.name}::{i}" for i in range(cfg.seq_cost if seq else 0)}
    unpriced = [n for n in [*desc.reference, *(m.name for m in infra.models())] if not table.get(n)]
    if unpriced:
        print(f"[cost] no served features known for {unpriced}: priced at 0", flush=True)
    for n in unpriced:
        table.setdefault(n, frozenset())
    return table


def selector(infra, desc, cfg):
    """The fixed selection over the bank. `ic_drift` selects on the mean IC over every label infra
    carries (10s/30s/60s on LU), weights fitted to the same target; `ic` on the fit label; any other
    objective is scored by infra in batches."""
    labels = desc.labels if cfg.select_objective == "ic_drift" else [cfg.select_label or desc.fit_label]
    scfg = Cfg(k_max=desc.k_max, host=desc.host, corrector_w=cfg.corrector_w, swap_t=cfg.swap_t,
               exclude=() if cfg.reference_in_bank else tuple(desc.reference), veto_min=cfg.veto_min,
               objective="ic" if cfg.select_objective in ("ic", "ic_drift") else cfg.select_objective,
               wf_chunks=cfg.wf_chunks, worst_weight=cfg.worst_weight, weight_shrink=cfg.weight_shrink,
               weight_ridge=cfg.weight_ridge, period_floor=cfg.period_floor, gate_t=cfg.gate_t,
               regime_cells=cfg.regime_cells, keep_reference=cfg.keep_reference,
               cost=_costs(infra, desc, cfg), cost_lambda=cfg.cost_lambda)
    if scfg.objective == "wf":               # the SoM aggregator: fit + validation, walked forward
        agg = Stats(merge(infra.moments("fit"), infra.moments("validation")), labels)
        return Selector(agg, agg, scfg, infra)
    val = Stats(infra.moments("validation"), labels)
    fit = val if cfg.weights_block == "validation" else Stats(infra.moments("fit"), labels)
    veto = Stats(infra.moments("validation"), [cfg.veto_label]) if cfg.veto_label else None
    return Selector(val, fit, scfg, infra, veto=veto)


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
                      credibility=sel.credibility(F, start),
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
    seconds: dict                 # recipe -> what training it cost (live)


def act(obs, plan, policy, models, rng, until=None):
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
    ready, bad = obs.infra.train([obs.untrained[n] for n in trained], until=until) if trained else ([], {})
    obs.failed |= set(bad)
    return Act(picks=picks, rows=rows, trained=trained, failed=bad, seconds={m.name: m.seconds for m in ready})


def _verdict(infra, sel, new, old, log, ref=()):
    """F' vs F on the sealed test days; an empty F (prod kept out of the bank) is judged against prod."""
    old = old or list(ref)
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
    start = (obs.ref if cfg.reference_in_bank else []) if cfg.frontier else F
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
    verdict = _verdict(infra, sel, new, F, log, obs.ref)
    for r in a.rows:                                    # each experiment alone
        e = next((x for x in plan.experiments if x.get("id") == r["id"]), {})
        if r.get("action") == "ensemble":
            r.update({"adopted": e.get("_adopted"), "t_vs_selected": e.get("_t")})
        other = {m for k, q in a.picks.items() if k != r["id"] for m in q}
        alone = sel.select([start], exclude=other)
        r.update({"failed": [m for m in r["trained"] if m not in have],
                  "entered_F": sorted(set(new) & set(r["trained"])),
                  "val_gain": f(sel.ix(alone)) - f(sel.ix(F)),
                  "test_verdict": _verdict(infra, sel, alone, F, log, obs.ref) if r["trained"] else "flat"})
    return Outcome(F_after=new, sel=sel, verdict=verdict, have=have), desc


def record(journal, obs, plan, a, o, tools, policy, log, secs):
    row = {"iteration": obs.it, "policy": policy.name, "F": obs.F, "F_after": o.F_after,
           "F_val": obs.sel.scorer()(obs.sel.ix(obs.F)), "F_credibility": obs.credibility,
           "F_after_val": o.sel.scorer()(o.sel.ix(o.F_after)),
           "test_verdict": o.verdict, "trained": a.trained, "failed": sorted(a.failed), "bank": len(o.have),
           "llm_usd": round(log.cost, 2), "secs": round(secs), "stop": plan.stop, "train_seconds": a.seconds,
           "F_after_legs": o.sel.legs(o.sel.ix(o.F_after)),             # exactly what this round would serve
           **({"F_after_cells": o.sel.wf.table(o.sel.ix(o.F_after))} if o.sel.wf else {})}
    journal.add_round(a.rows + [{"iteration": obs.it, "lessons": plan.lessons, "tools": tools}], row)
    log("iteration", f"{obs.it}: F val {row['F_val']:.4f} -> {row['F_after_val']:.4f}, test {o.verdict}, "
                     f"trained {len(a.trained)} ({len(a.failed)} failed)")
    return row
