"""The round's workspace: everything the agent reads, written fresh each round (a long campaign
never depends on one long context). Read-only but `work/`, which the agent may write and run.
Val-tier only: no test verdict, no row, no date appears in any file."""
import json
import os
import shutil

import numpy as np

from alpharesearch.agent.tools import launcher

PROMPT = os.path.join(os.path.dirname(__file__), "prompts", "agent.md")

SET_SCORE = '''"""A set's val score from moments.npz: per period the IC w.c/sqrt(w'Sw) over the selection
labels, divided by prod's, averaged over periods; w = S_fit^-1 c_fit over the set (sum |w| = 1)."""
import numpy as np


def set_score(m, idx):
    idx = list(idx)
    w = np.linalg.lstsq(m["S_fit"][np.ix_(idx, idx)], m["c_fit"][idx], rcond=None)[0]
    w = w / np.abs(w).sum()
    out = []
    for p in range(len(m["periods"])):
        S, c = m["S_val"][p], m["c_val"][:, p].mean(0)
        f = lambda jj, ww: float(ww @ c[jj]) / np.sqrt(float(ww @ S[np.ix_(jj, jj)] @ ww))
        out.append(f(idx, w) / f(list(m["ref_idx"]), m["ref_w"]))
    return float(np.mean(out))
'''


def _moments_npz(R, path):
    """moments.npz: each model's val IC per label and period, the val correlation matrix per
    period, the fit block (c_fit, S_fit), F and prod's legs: what `set_score.py` needs."""
    st, sel = R.tool_sel.val, R.tool_sel
    from alpharesearch.ensemble.moments import solo_ic, undilute
    m, P = st.m, len(st.m.periods)
    c_val = np.full((len(m.labels), P, len(st.names)), np.nan)
    S_val = np.zeros((P, len(st.names), len(st.names)))
    for p in range(P):
        u = (np.asarray(m.period) == p).astype(float)
        if not u.any():
            continue
        szz = u @ m.szz
        nrm = np.sqrt(np.where(szz > 0, szz, 1.0))
        S_val[p] = undilute(np.tensordot(u, m.zz, 1) / np.outer(nrm, nrm), np.tensordot(u, m.zzm, 1))
        for i, L in enumerate(m.labels):
            c_val[i, p] = solo_ic(u @ m.n, u @ m.sz, szz, u @ L["sy"], u @ L["syy"], u @ L["szy"])
    jj, w = sel.blend(sel.ix(R.F))
    ref = [(st.idx[n], x) for n, x in m.reference if n in st.idx]
    np.savez(path, names=np.array(st.names), periods=np.array(m.periods), c_val=c_val, S_val=S_val,
             c_fit=sel.c_fit, S_fit=sel.S_fit, F_idx=np.array(jj), F_w=w,
             ref_idx=np.array([j for j, _ in ref]), ref_w=np.array([x for _, x in ref]))


def _bank(R):
    """bank.json: one row per trained model: its axes, solo val IC, marginal gain, weight in F."""
    sel = R.tool_sel
    c = sel.val.moments(sel.val.ones())[0]
    jj, w = sel.blend(sel.ix(R.F))
    wf = {sel.names[j]: float(x) for j, x in zip(jj, w, strict=True)}
    return [{"member": n, **R.coords.get(n, {}), "val_ic": round(float(c[i]), 5),
             "marginal_gain": round(float(R.gains.get(n, np.nan)), 6), "weight_in_F": round(wf.get(n, 0.0), 4)}
            for i, n in enumerate(sel.names)]


def brief(R, ledger, budget, diag, lib, experience=None, memory=None):
    ws = R.ws
    os.makedirs(os.path.join(ws, "work"), exist_ok=True)
    f = R.tool_sel.scorer()
    sel = R.tool_sel
    files = {
        "state.json": {"iteration": R.it, "objective": R.desc.objective,
                       "val_numbers_in": sel.cfg.objective, "swap_t": sel.cfg.swap_t,
                       "k_max": sel.cfg.k_max, "host": sel.cfg.host, "budget": budget,
                       "F": [{"member": n, "weight": round(float(x), 4)} for n, x in sel.legs(sel.ix(R.F))],
                       "F_val": f(sel.ix(R.F)), "prod_val": f(sel.ix(R.ref)) if R.ref else None,
                       "note": "scores are ratios to the prod reference (1.0 = prod) averaged over periods"},
        "diagnosis.json": diag,
        "grammar.json": R.desc.axes,
        "feature_sets.json": R.feature_sets,
        # the agent never sees a test verdict: steering by what held on test selects on the days
        # the arms are compared on (the baselines never read them)
        "ledger.json": [{k: v for k, v in r.items() if not k.startswith("test")} for r in ledger],
        "bank_top.json": [{"member": n, "marginal_val_gain": round(float(g), 5)}
                          for n, g in sorted(R.gains.items(), key=lambda kv: -kv[1])[:40]],
        "bank.json": _bank(R),
    }
    for name, obj in files.items():
        json.dump(obj, open(os.path.join(ws, name), "w"), indent=1, default=float)
    _moments_npz(R, os.path.join(ws, "moments.npz"))
    open(os.path.join(ws, "set_score.py"), "w").write(SET_SCORE)
    open(os.path.join(ws, "store_columns.txt"), "w").write("\n".join(R.columns) + "\n")
    open(os.path.join(ws, "available.txt"), "w").write("\n".join(sorted(R.untrained)) + "\n")
    if memory:
        shutil.copy(memory, os.path.join(ws, "memory.md"))
    for name in ("experience.md", "recipes.json"):
        if experience and os.path.exists(os.path.join(experience, name)):
            shutil.copy(os.path.join(experience, name), os.path.join(ws, name))
    if os.path.isdir(lib):
        shutil.copytree(lib, os.path.join(ws, "lib"), dirs_exist_ok=True)
    shutil.copy(os.path.join(ws, "..", "agent.md"), os.path.join(ws, "agent.md"))
    launcher(ws, lib)
