"""Walk-forward selection on the SoM aggregator (the search's only days): each period's days in time
order are cut into `chunks`; on every chunk after the first, a set is scored with weights fitted on
the earlier chunks of all three periods, per period x session cell, as a ratio to the reference.
A fold's score is mean(cells) + worst_weight * min(cells); the objective is the mean over folds.

So no number the search selects on comes from the days its weights were fitted on, a set that only
works in one stretch of the window scores poorly, and a session or period that loses cannot hide
behind one that wins (the LU gate: pooled in-window IC rose every round while the held-out night
and recent cells fell). Weights are pooled across periods and shrunk toward equal: 10-20 days per
period cannot carry 12 free weights."""
import numpy as np

from alphasearch.ensemble.moments import Stats, ic
from alphasearch.infra.contract import Moments


def merge(a: Moments, b: Moments) -> Moments:
    """Two blocks' unit moments over the same members as one block (the SoM aggregator = fit + validation)."""
    assert a.names == b.names and a.label_names == b.label_names and a.periods == b.periods
    cat = lambda x, y: np.concatenate([np.asarray(x), np.asarray(y)])
    return Moments(names=a.names, day=cat(a.day, b.day), session=cat(a.session, b.session),
                   period=cat(a.period, b.period), periods=a.periods, sessions=a.sessions,
                   n=cat(a.n, b.n), sz=cat(a.sz, b.sz), szz=cat(a.szz, b.szz), zz=cat(a.zz, b.zz),
                   zzm=cat(a.zzm, b.zzm), reference=a.reference, label_names=a.label_names,
                   labels=[{k: cat(x[k], y[k]) for k in x} for x, y in zip(a.labels, b.labels, strict=True)])


def shrunk_weights(c, S, shrink):
    """w = (S + shrink * diag(S))^-1 c, sum |w| = 1: the IC-optimal weights, ridge-shrunk."""
    S = np.asarray(S, float)
    try:
        w = np.linalg.solve(S + shrink * np.diag(np.diag(S)), np.asarray(c, float))
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(S + shrink * np.diag(np.diag(S)), c, rcond=None)[0]
    return w / np.abs(w).sum()


class WalkForward:
    def __init__(self, st: Stats, chunks=4, worst_weight=0.5, shrink=0.25):
        self.st, self.worst_weight, self.shrink = st, worst_weight, shrink
        sess = np.asarray(st.m.session)
        order = {}                                     # chunk index of every day, per period
        for k in np.unique(st.period):
            days = np.flatnonzero(st.period == k)       # day indices in time order (st.days is sorted)
            for i, part in enumerate(np.array_split(days, chunks)):
                order.update({int(d): i for d in part})
        chunk = np.array([order[d] for d in range(len(st.days))])
        self.folds = []
        for i in range(1, chunks):
            c, S = st.moments((chunk < i).astype(float))
            cells = []
            for k in np.unique(st.period):
                for s in np.unique(sess):
                    cc, SS = st.moments(((chunk == i) & (st.period == k)).astype(float), units=sess == s)
                    r = st.ref_ic(cc, SS)
                    if np.isfinite(r) and abs(r) > 1e-9:
                        cells.append((cc, SS, r, f"{st.m.periods[k]}/{st.m.sessions[s]}"))
            self.folds.append((c, S, cells))
        self.c_all, self.S_all = st.moments(st.ones())

    def weights(self, jj):
        """The weights a selected set is served with: every aggregator day, shrunk."""
        return shrunk_weights(self.c_all[jj], self.S_all[np.ix_(jj, jj)], self.shrink)

    def cells(self, jj):
        """(folds x cells) ratios to the reference, each fold's blend fitted on the earlier chunks."""
        jj = list(jj)
        out = []
        for c, S, cells in self.folds:
            w = shrunk_weights(c[jj], S[np.ix_(jj, jj)], self.shrink)
            out.append([ic(cc, SS, jj, w) / r for cc, SS, r, _ in cells])
        return np.array(out)

    def score(self, jj):
        if not len(jj):
            return float("nan")
        r = self.cells(jj)
        if not np.all(np.isfinite(r)):
            return float("nan")
        return float(np.mean(r.mean(1) + self.worst_weight * r.min(1)))

    def gain(self, new, old):
        """(gain, SE) of `new` over `old`: the score difference, its SE from the fold x cell differences."""
        d = (self.cells(new) - self.cells(old)).ravel()
        d = d[np.isfinite(d)]
        se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 2 else 0.0
        return self.score(new) - self.score(old), se

    def table(self, jj):
        """{fold i: {cell: ratio}}: what a set's selection score is made of (the agent reads it)."""
        r = self.cells(jj)
        return {f"fold{i + 1}": {name: round(float(x), 4) for (_, _, _, name), x in zip(cells, row, strict=True)}
                for i, ((_, _, cells), row) in enumerate(zip(self.folds, r, strict=True))}
