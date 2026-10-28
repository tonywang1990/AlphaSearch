"""A campaign: R rounds of B models on one symbol, over futs-infra's research interface v1. This file
is the whole algorithm (dota/docs/agent-search.md §2); each phase lives in `round.py`.

    setup     freeze the brief and this week's cross-symbol memory; resume from the journal; in
              frontier mode, carry every model the experience store has seen into the bank (free)
    rounds    observe -> workspace -> plan -> act -> evaluate -> reflect -> record
    episode   the campaign's val facts and lessons join the experience store (online memory)

The fixed search picks F; the policy only decides what to train; infra trains and scores, and its
test verdicts never reach the agent. State lives in files, so a rerun with the same `out` resumes."""
import json
import os
import shutil
import time
from dataclasses import asdict

import numpy as np

from alphasearch import round as rnd
from alphasearch.context import experience, memory, workspace
from alphasearch.ensemble.diagnose import diagnose
from alphasearch.infra.contract import ApiError, check_version, name_of
from alphasearch.policies import POLICIES
from alphasearch.records import Journal
from alphasearch.runtime.log import Log

PHASES = ("observe", "workspace", "plan", "act", "evaluate", "reflect", "record")


def _setup(infra, cfg, out, desc, log, journal):
    """(the memory.md path or None, what outlives a round)."""
    if not os.path.exists(os.path.join(out, "agent.md")):              # the campaign's brief, frozen
        shutil.copy(cfg.prompt or workspace.PROMPT, os.path.join(out, "agent.md"))
        json.dump(asdict(cfg), open(os.path.join(out, "config.json"), "w"), indent=1)
    mem = cfg.memory
    if not mem and desc.served:           # the cross-symbol memory, from this week's prod (frozen)
        mem = os.path.join(out, "memory.md")
        if not os.path.exists(mem):
            open(mem, "w").write(memory.render(desc.served, desc.symbol, cfg.memory_stores))
    seen = experience.seen(cfg.experience)
    if cfg.frontier and cfg.experience and not journal.iters:   # the bank is everything this symbol
        carry = [r for r in infra.recipes() if name_of(r, desc.axes) in seen]   # trained: free, uncounted
        if carry:
            infra.train(carry)
            log("carry", f"{len(carry)} models every earlier episode trained")
    return mem, dict(failed=journal.failed, feature_sets=dict(desc.feature_sets), seen=seen, recent=[])


def _eval_report(infra, desc, cfg, out, legs, log):
    """The SoM eval block, read ONCE per campaign: the gate's Table 1 of the final F vs the reference
    on the days after every day the search read; the next campaign's agent sees it (experience.md)."""
    path = os.path.join(out, "eval_report.json")
    if os.path.exists(path):
        return json.load(open(path))
    if not (cfg.report and desc.report):
        return None
    try:
        rep = infra.test_report(legs)
    except ApiError as e:
        log("eval", f"not reported: {e}")
        return None
    json.dump(rep, open(path, "w"), default=float)
    return rep


def run(infra, cfg, out):
    desc = infra.describe()
    check_version(desc)
    os.makedirs(out, exist_ok=True)
    log, journal, policy = Log(out), Journal(out), POLICIES[cfg.policy](cfg)
    lib = os.path.join(cfg.experience or out, "tools")                 # the agent's tools outlive a campaign
    mem, carry = _setup(infra, cfg, out, desc, log, journal)
    rng = np.random.default_rng(cfg.seed * 1000 + len(journal.iters))
    for it in range(len(journal.iters), cfg.rounds):
        t0 = time.time()
        laps = [t0]                                     # each phase's end, for the session viewer
        start = desc.reference if cfg.frontier or not journal.iters else journal.iters[-1]["F_after"]
        obs = rnd.observe(infra, desc, cfg, it, out, list(start), carry)
        laps.append(time.time())
        workspace.build(obs, journal=journal, budget={"iterations_left": cfg.rounds - it, "models_this_iteration":
                                                      cfg.models}, lib=lib, memory=mem,
                        diagnosis=diagnose(obs.sel, obs.F) if obs.sel.cfg.objective == "ic" else [])
        laps.append(time.time())
        plan = policy.plan(obs, cfg.models, rng, log)
        laps.append(time.time())
        a = rnd.act(obs, plan, policy, cfg.models, rng)
        laps.append(time.time())
        o, desc = rnd.evaluate(obs, plan, a, log)
        laps.append(time.time())
        tools = policy.reflect(obs, plan, lib, log)
        laps.append(time.time())
        carry["recent"] = obs.recent
        row = rnd.record(journal, obs, plan, a, o, tools, policy, log, time.time() - t0)
        laps.append(time.time())
        log("phases", str(it), **{p: round(b - a, 1) for p, a, b in zip(PHASES, laps[:-1], laps[1:], strict=True)})
        if row["stop"] and not desc.universe:          # benchmark: the budget is fixed, unused is lost
            break
    if journal.iters and not os.path.exists(os.path.join(out, "episode.json")):
        sel = rnd.selector(infra, desc, cfg)
        F = journal.iters[-1]["F_after"]
        ev = _eval_report(infra, desc, cfg, out, sel.legs(sel.ix(F)), log)
        ep = experience.record(out, os.path.basename(out.rstrip("/")), cfg.policy, F, sel.gains(F),
                               {m.name: m.recipe for m in infra.models()}, journal.ledger, journal.iters,
                               objective=desc.objective, prompt=os.path.basename(cfg.prompt or workspace.PROMPT),
                               eval_report=ev)
        if cfg.experience:
            experience.add(cfg.experience, ep)
    return out
