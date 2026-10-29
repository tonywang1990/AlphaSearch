"""Fill-aware checks: one infra job each (`score_many`), so the agent spends them on finalists.

IC selects (ADR 0008) but the gate judges EVf, the 1% tail weighted by the chance a taking order
fills: a set can match prod on IC and on the tail's gross edge yet lose EVf because its strongest
calls land where the book does not fill (the LU gate, 2026-10-29: recent night IC +0.7%, gross tail
+0.8%, EVf -11%). This tool is how a gain is checked against fills before it is kept."""
import numpy as np

from alphasearch.tools import opt, tool


@tool("evf --add M... --drop M...", "F with members added or dropped vs F and prod on fill-aware EV (val days; one infra job)")
def evf(obs, args):
    sel, F = obs.sel, obs.F
    s = [n for n in F if n not in opt(args, "--drop")] + opt(args, "--add")
    bad = [n for n in s if n not in sel.val.idx]
    if bad or not 0 < len(s) <= sel.cfg.k_max:
        return {"error": f"{len(s)} members (1-{sel.cfg.k_max}); not trained: {bad[:3]}"}
    ref = [(n, float(w)) for n, w in sel.val.m.reference]
    new, old, prod = obs.infra.score_many([sel.legs(sel.ix(s)), sel.legs(sel.ix(F)), ref], "evf")

    def paired(a, b):
        d = np.array([a.daily[k] - b.daily[k] for k in a.daily if k in b.daily])
        d = d[np.isfinite(d)]
        se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 2 else None
        return float(d.mean()) if len(d) else None, se, len(d)

    g, se, n = paired(new, old)
    gp, sep, _ = paired(new, prod)
    return {"members": s, "evf": new.value, "F_evf": old.value, "prod_evf": prod.value,
            "gain_vs_F": g, "gain_vs_F_se": se, "t_vs_F": g / se if se else None,
            "gain_vs_prod_pct": 100 * (new.value / prod.value - 1) if prod.value else None,
            "t_vs_prod": gp / sep if sep else None, "days": n}
