"""The round's workspace: everything the agent reads, written fresh each round (a long campaign
never depends on one long context). Read-only but `work/`, which the agent may write and run.
Val-tier only: no test verdict, no row, no date appears in any file.

The whole bank is in it (DOTA's `bankview`): `bank.json`, one row per trained model (its axes,
IC per label and period, correlation with F, partial IC given F, marginal gain, best swap into F
and whom it replaces, how often earlier final Fs held it), and `moments.npz`, the val moments
behind every score, with `set_score.py` reproducing any set's val score from them."""
import inspect
import json
import os
import shutil

import numpy as np

from alphasearch.agent.tools import launcher

PROMPT = os.path.join(os.path.dirname(__file__), "prompts", "agent.md")


def set_score(m, idx, w, labels=None):
    """A set's val score from moments.npz `m`: per period the blend's IC, w.c/sqrt(w'Sw), averaged
    over `labels` (default: the ones selection uses), as a ratio to the prod reference's, then
    averaged over periods. `w` None: the fit block's weights, S_fit^-1 c_fit (what F uses)."""
    L = [list(m["labels"]).index(x) for x in labels] if labels else list(m["select_labels"])
    idx = np.asarray(idx)
    if w is None:
        cf = m["c_fit"][L].mean(0)[idx]
        w = np.linalg.solve(m["S_fit"][np.ix_(idx, idx)], cf)

    def ic(p, jj, ww):            # NaN where S is not positive definite (the selection's guard)
        sub = m["S_val"][p][np.ix_(jj, jj)]
        if len(jj) > 1 and np.linalg.eigvalsh(sub)[0] < 1e-6:
            return float("nan")
        return float(ww @ m["c_val"][L, p].mean(0)[jj] / np.sqrt(ww @ sub @ ww))
    ref = np.asarray(m["ref_idx"])
    vals = [ic(p, idx, w) / (ic(p, ref, m["ref_w"]) if len(ref) else 1.0) for p in range(len(m["periods"]))]
    return float(np.mean([v for v in vals if np.isfinite(v)]))


def _moments(sel, labels):
    """(c_val[L,P,K], S_val[P,K,K], c_fit[L,K], S_fit[K,K]) on the val Stats, every label."""
    st, m = sel.val, sel.val.m
    P, K = len(m.periods), len(st.names)
    c_val, S_val = np.full((len(labels), P, K), np.nan), np.zeros((P, K, K))
    for p in range(P):
        units = np.asarray(m.period) == p
        if units.any():
            c_val[:, p] = st.label_ics(st.ones(), units)
            S_val[p] = st.moments(st.ones(), units)[1]
    return c_val, S_val, np.array(sel.fit.label_ics(sel.fit.ones()))[:, sel.fit_order], sel.S_fit


def bankview(R, ws, seen):
    sel, names = R.tool_sel, R.tool_sel.names
    idx = {n: i for i, n in enumerate(names)}
    labels = list(sel.val.m.label_names)
    c_val, S_val, c_fit, S_fit = _moments(sel, labels)
    sl = sel.val.L
    Fj, Fw = sel.blend(sel.ix(R.F))
    ref = sel.val.ref
    np.savez(os.path.join(ws, "moments.npz"), names=np.array(names), labels=np.array(labels),
             periods=np.array(sel.val.m.periods), select_labels=np.array(sl), c_val=c_val.astype(np.float32),
             S_val=S_val.astype(np.float32), c_fit=c_fit.astype(np.float32), S_fit=S_fit.astype(np.float32),
             F_idx=np.array(Fj, int), F_w=Fw, ref_idx=np.array([i for i, _ in ref], int),
             ref_w=np.array([w for _, w in ref], float))
    # pooled over val periods in the selection labels: F's prediction and each model against it
    cP, SP = np.nanmean(c_val[sl].mean(0), axis=0), S_val.mean(0)
    vF = float(Fw @ SP[np.ix_(Fj, Fj)] @ Fw)
    rF, icF = SP[:, Fj] @ Fw / np.sqrt(vF), float(Fw @ cP[Fj] / np.sqrt(vF))
    with np.errstate(invalid="ignore", divide="ignore"):
        pc = (cP - rF * icF) / np.sqrt((1 - rF ** 2) * (1 - icF ** 2))
    periods = list(sel.val.m.periods)
    rows = []
    for n in names:
        j, swap = idx[n], R.swaps.get(n)
        gain = R.gains.get(n)
        rows.append({"model": n, **(R.coords.get(n) or {}),
                     "in_F": j in Fj, "weight_in_F": round(float(Fw[Fj.index(j)]), 4) if j in Fj else None,
                     "ic_val": {lab: {per: round(float(c_val[li, p, j]), 4) for p, per in enumerate(periods)}
                                for li, lab in enumerate(labels)},
                     "corr_with_F": round(float(rF[j]), 4), "pcorr_given_F": round(float(pc[j]), 4),
                     "marginal_gain": round(float(gain), 5) if gain is not None and np.isfinite(gain) else None,
                     "best_swap_gain": round(float(swap[0]), 5) if swap and np.isfinite(swap[0]) else None,
                     "best_swap_out": swap[1] if swap else None,
                     "in_earlier_final_F": (seen or {}).get(n, {}).get("in_final_F")})
    rows.sort(key=lambda r: -max(r["best_swap_gain"] or -9, r["marginal_gain"] or -9))
    open(os.path.join(ws, "set_score.py"), "w").write("import numpy as np\n\n\n" + inspect.getsource(set_score))
    json.dump({"F_val": round(float(R.F_score), 5),
               "F_val_from_moments": round(set_score(np.load(os.path.join(ws, "moments.npz")), Fj, Fw), 5),
               "rows": rows}, open(os.path.join(ws, "bank.json"), "w"), indent=0)


def brief(R, ledger, budget, diag, lib, seen=None):
    ws, cfg, sel = R.ws, R.cfg, R.tool_sel
    os.makedirs(os.path.join(ws, "work"), exist_ok=True)
    f = sel.scorer()
    files = {
        "state.json": {"iteration": R.it, "objective": R.desc.objective, "val_numbers_in": cfg["select_objective"],
                       "swap_t": sel.cfg.swap_t, "k_max": sel.cfg.k_max, "budget": budget,
                       "F": [{"member": n, "weight": round(float(x), 4)} for n, x in sel.legs(sel.ix(R.F))],
                       "F_val": f(sel.ix(R.F)), "prod_val": f(sel.ix(R.ref)) if R.ref else None, "host": sel.cfg.host,
                       "note": "scores are ratios to the prod reference (1.0 = prod) averaged over periods"
                       if R.ref else f"scores are the raw {R.desc.objective}"},
        "diagnosis.json": diag,
        "grammar.json": R.desc.axes,
        "grammar_notes.json": R.desc.notes,
        "feature_sets.json": R.desc.feature_set_defs or R.feature_sets,
        # the agent never sees a test verdict: steering by what held on test selects on the days
        # the arms are compared on (the baselines never read them)
        "ledger.json": [{k: v for k, v in r.items() if not k.startswith("test")} for r in ledger],
        "bank_top.json": [{"member": n, "marginal_val_gain": round(float(g), 5)}
                          for n, g in sorted(R.gains.items(), key=lambda kv: -kv[1])[:40]],
    }
    for name, obj in files.items():
        json.dump(obj, open(os.path.join(ws, name), "w"), indent=1, default=float)
    open(os.path.join(ws, "store_columns.txt"), "w").write("\n".join(R.columns) + "\n")
    open(os.path.join(ws, "available.txt"), "w").write("\n".join(sorted(R.untrained)) + "\n")
    if cfg.get("memory"):                       # what campaigns on OTHER symbols learned
        shutil.copy(cfg["memory"], os.path.join(ws, "memory.md"))
    for name in ("experience.md", "recipes.json"):  # this symbol's own earlier campaigns
        if cfg.get("experience") and os.path.exists(os.path.join(cfg["experience"], name)):
            shutil.copy(os.path.join(cfg["experience"], name), os.path.join(ws, name))
    for doc in cfg.get("docs") or ():           # reference docs the brief names (taxonomy, dead ends)
        shutil.copy(doc, os.path.join(ws, os.path.basename(doc)))
    bankview(R, ws, seen)
    if os.path.isdir(lib):
        shutil.copytree(lib, os.path.join(ws, "lib"), dirs_exist_ok=True)
    shutil.copy(os.path.join(ws, "..", "agent.md"), os.path.join(ws, "agent.md"))
    launcher(ws, lib)
