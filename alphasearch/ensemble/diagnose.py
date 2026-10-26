"""Where F fails and what the bank leaves on the table, per period x session cell, from the
val moments: F's IC next to the reference's, and the trained members with the largest partial
correlation to the label given F (what the fill did not take)."""
import numpy as np

from alphasearch.ensemble.moments import ic


def pcorrs(c, S, jj, w, pool):
    """Each pool member's residual partial correlation to the label given the blend."""
    pool = np.asarray(pool, int)
    v = float(np.sqrt(max(w @ S[np.ix_(jj, jj)] @ w, 0.0))) if len(jj) else 0.0
    if v <= 0:
        return c[pool]
    ic_e = float(w @ c[jj]) / v
    r = (S[np.ix_(pool, jj)] @ w) / v
    return (c[pool] - r * ic_e) / np.sqrt(np.clip((1 - r ** 2) * (1 - ic_e ** 2), 1e-12, None))


def diagnose(sel, F, top=5, min_units=3):
    """[cell report], weakest ratio to the reference first."""
    st, m = sel.val, sel.val.m
    jj, w = sel.blend(sel.ix(F))
    pool = [j for j in range(len(st.names)) if j not in set(jj)]
    out = []
    for p, pname in enumerate(m.periods):
        for s, sname in enumerate(m.sessions):
            units = (np.asarray(m.period) == p) & (np.asarray(m.session) == s)
            if units.sum() < min_units:
                continue
            c, S = st.moments(st.ones(), units)
            got, ref = ic(c, S, jj, w), st.ref_ic(c, S)
            pc = np.nan_to_num(pcorrs(c, S, jj, w, pool))
            out.append({"cell": f"{pname}/{sname}", "days": int(units.sum()),
                        "ic": round(float(got), 4), "ref_ic": round(float(ref), 4),
                        "ratio": round(float(got / ref), 3) if ref > 0 else None,
                        "untaken": [{"member": st.names[pool[o]], "pcorr": round(float(pc[o]), 4)}
                                    for o in np.argsort(-np.abs(pc))[:top]]})
    return sorted(out, key=lambda r: r["ratio"] if r["ratio"] is not None else 9.0)
