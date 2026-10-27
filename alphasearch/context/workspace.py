"""The round's workspace: everything the agent reads, written fresh each round (a long campaign never
depends on one long context). Read-only but `work/`, which the agent may write and run. Val-tier
only: no test verdict, no row, no date appears in any file.

The workspace is the list PROVIDERS, in order; each provider writes its files for the round. To
add context (or remove it for an ablation) add or drop a provider here and say what it is in the
brief."""
import json
import os
import shutil

from alphasearch.context import bankview
from alphasearch.tools.server import launcher

PROMPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "agent.md")


def _json(ws, name, obj):
    json.dump(obj, open(os.path.join(ws, name), "w"), indent=1, default=float)


def state(obs, ctx):
    """state.json: the round, the objective, F and its val score next to prod's, the budget."""
    sel, f = obs.sel, obs.sel.scorer()
    _json(obs.ws, "state.json", {
        "iteration": obs.it, "objective": obs.desc.objective, "val_numbers_in": obs.cfg.select_objective,
        "swap_t": sel.cfg.swap_t, "k_max": sel.cfg.k_max, "budget": ctx["budget"],
        "F": [{"member": n, "weight": round(float(x), 4)} for n, x in sel.legs(sel.ix(obs.F))],
        "F_val": f(sel.ix(obs.F)), "prod_val": f(sel.ix(obs.ref)) if obs.ref else None, "host": sel.cfg.host,
        "note": "scores are ratios to the prod reference (1.0 = prod) averaged over periods"
        if obs.ref else f"scores are the raw {obs.desc.objective}"})


def diagnosis(obs, ctx):
    """diagnosis.json: F by period x session next to prod, weakest first, with untaken signal."""
    _json(obs.ws, "diagnosis.json", ctx["diagnosis"])


def grammar(obs, ctx):
    """grammar.json, grammar_notes.json, feature_sets.json: the axes, what each value means, the sets."""
    _json(obs.ws, "grammar.json", obs.desc.axes)
    _json(obs.ws, "grammar_notes.json", obs.desc.notes)
    _json(obs.ws, "feature_sets.json", obs.desc.feature_set_defs or obs.feature_sets)


def ledger(obs, ctx):
    """ledger.json (val fields only) and bank_top.json (the top 40 by marginal gain)."""
    _json(obs.ws, "ledger.json", ctx["journal"].visible())
    _json(obs.ws, "bank_top.json", [{"member": n, "marginal_val_gain": round(float(g), 5)}
                                    for n, g in sorted(obs.gains.items(), key=lambda kv: -kv[1])[:40]])


def lists(obs, ctx):
    """store_columns.txt and available.txt: the raw columns, every recipe trainable now."""
    open(os.path.join(obs.ws, "store_columns.txt"), "w").write("\n".join(obs.columns) + "\n")
    open(os.path.join(obs.ws, "available.txt"), "w").write("\n".join(sorted(obs.untrained)) + "\n")


def memory(obs, ctx):
    """memory.md (other symbols, offline), experience.md + recipes.json (this symbol's own campaigns)."""
    cfg = obs.cfg
    if ctx["memory"]:
        shutil.copy(ctx["memory"], os.path.join(obs.ws, "memory.md"))
    for name in ("experience.md", "recipes.json"):
        if cfg.experience and os.path.exists(os.path.join(cfg.experience, name)):
            shutil.copy(os.path.join(cfg.experience, name), os.path.join(obs.ws, name))


def docs(obs, ctx):
    """The reference docs the brief names (the feature taxonomy, the dead ends)."""
    for doc in obs.cfg.docs:
        shutil.copy(doc, os.path.join(obs.ws, os.path.basename(doc)))


def bank(obs, ctx):
    """bank.json, moments.npz, set_score.py: the whole bank."""
    bankview.write(obs, obs.ws, obs.seen)


def library(obs, ctx):
    """lib/ (the tool library), agent.md (the brief), ./tool (the launcher)."""
    if os.path.isdir(ctx["lib"]):
        shutil.copytree(ctx["lib"], os.path.join(obs.ws, "lib"), dirs_exist_ok=True)
    shutil.copy(os.path.join(obs.ws, "..", "agent.md"), os.path.join(obs.ws, "agent.md"))
    launcher(obs.ws, ctx["lib"])


def _pct(base, cand):
    return None if base is None or cand is None or abs(base) < 1e-9 else round(100 * (cand - base) / abs(base), 1)


def _compare(rep):
    """{period: {session: {metric: [F', reference, delta %]}}}: the gate's Table 1 layout."""
    out = {}
    for period, sessions in rep["subject"].items():
        ref = rep["compare_to"].get(period, {})
        out[period] = {s: {m: [v, ref.get(s, {}).get(m), _pct(ref.get(s, {}).get(m), v)] for m, v in cells.items()}
                       for s, cells in sessions.items() if isinstance(cells, dict)}
    return out


def report(obs, ctx):
    """report.json: the eval gate's Table 1 of each round's F' vs the reference on the held-out test
    days (strictly after every day the search selects on), the latest in full, every round's EVf@10s."""
    rows = [r for r in ctx["journal"].iters if r.get("report")]
    if not rows:
        return
    last = rows[-1]
    _json(obs.ws, "report.json", {
        "what": "the eval gate's Table 1 on the TEST days (after the fit and validation days, never selected on): "
                "[F', reference, delta %] per period, session and metric; the gate judges EVf@10s per period "
                "and session, and a storm EVf loss kills a candidate whatever calm does",
        "latest": {"iteration": last["iteration"], "F": last["F_after"], "table": _compare(last["report"])},
        "history": [{"iteration": r["iteration"], "EVf@10s_delta_pct": {
            f"{p}/{s}": c.get("EVf@10s", [None, None, None])[2] for p, ss in _compare(r["report"]).items()
            for s, c in ss.items()}} for r in rows]})


PROVIDERS = (state, diagnosis, grammar, ledger, lists, memory, docs, bank, report, library)


def build(obs, **ctx):
    """Write the round's workspace: `ctx` holds journal, budget, diagnosis, memory (path), lib."""
    os.makedirs(os.path.join(obs.ws, "work"), exist_ok=True)
    for provider in PROVIDERS:
        provider(obs, ctx)
