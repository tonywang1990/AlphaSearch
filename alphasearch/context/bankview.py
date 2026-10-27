"""The whole bank, for the agent (DOTA's `bankview`): `bank.json`, one row per trained model (its
axes, IC per label and period, correlation with F, partial IC given F, marginal gain, best swap
into F and whom it replaces, how often earlier final Fs held it), and `moments.npz`, the val
moments behind every score, with `set_score.py` reproducing any set's val score from them."""
import inspect
import json
import os

import numpy as np


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


def write(R, ws, seen):
    sel, names = R.sel, R.sel.names
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
