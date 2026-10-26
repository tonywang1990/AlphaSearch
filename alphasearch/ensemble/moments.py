"""Ensemble arithmetic on infra's per-unit sums (from DOTA's `daystats`, ADR 0452 P3).

The moments of ANY day subset (a period, a split half, a bootstrap resample with repeats) are
a weighted sum of per-unit sums, so a blend's IC and its paired day-by-day test cost K x K
algebra, never a pass over ticks. Day-level calls take `counts`, one weight per day."""
import numpy as np

from alphasearch.infra.contract import Moments


def undilute(S, Q):
    """The zero-filled gram -> the correlation ON THE PAIRWISE OVERLAP. Zero-filling outside
    each member's validity dilutes a pair whose masks differ by sqrt(Q_ij Q_ji)/sqrt(Q_ii Q_jj);
    left in, the objective drifts with depth."""
    d = np.sqrt(np.clip(np.diag(Q), 1e-300, None))
    den = np.sqrt(np.clip(Q * Q.T, 1e-300, None))
    return np.clip(S * np.outer(d, d) / den, -1.0, 1.0)


def solo_ic(n, sz, szz, sy, syy, szy):
    """Each member's IC from its sums against one label."""
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = szy / n - (sz / n) * (sy / n)
        return cov / np.sqrt((szz / n - (sz / n) ** 2) * (syy / n - (sy / n) ** 2))


def ic(c, S, jj, w):
    """The IC of blend `w` over members `jj`; NaN for a flat blend, or for a set whose
    overlap-corrected S is not positive definite (there w'Sw nears 0 and the IC is unbounded:
    LU multistart random starts scored val 1e8)."""
    sub = S[np.ix_(jj, jj)]
    if len(jj) > 1 and np.linalg.eigvalsh(sub)[0] < 1e-6:
        return float("nan")
    v = float(w @ sub @ w)
    return float(w @ c[jj]) / np.sqrt(v) if v > 0 else float("nan")


def fit_weights(c, S):
    """w = S^-1 c scaled to sum |w| = 1: the weights that maximise the blend's IC. A singular S
    (an exact copy of a member in the set) takes the minimum-norm solution."""
    S, c = np.asarray(S, float), np.asarray(c, float)
    try:
        w = np.linalg.solve(S, c)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(S, c, rcond=None)[0]
    return w / np.abs(w).sum()


class Stats:
    """One block's moments, with the day structure the selection and its tests need."""

    def __init__(self, m: Moments, labels=None):
        """`labels`: the label names IC is the mean over (default: every label in `m`); one label
        is IC, the 10s/30s/60s three are DOTA's `ic_drift`."""
        self.m, self.names = m, list(m.names)
        self.L = [m.label_names.index(x) for x in labels] if labels else list(range(len(m.labels)))
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.days, self.uday = np.unique(m.day, return_inverse=True)
        first = np.array([np.flatnonzero(self.uday == d)[0] for d in range(len(self.days))])
        self.period = np.asarray(m.period)[first]
        self.ref = [(self.idx[n], w) for n, w in m.reference if n in self.idx]
        eye = np.eye(len(self.days))
        cs = [self.moments(eye[i]) for i in range(len(self.days))]
        self.c_day, self.S_day = np.array([c for c, _ in cs]), np.array([S for _, S in cs])

    def moments(self, counts, units=None):
        """(c, S) over the days weighted by `counts` (a day drawn twice counts twice), optionally
        restricted to the units where `units` is True (a period x session cell)."""
        u = np.asarray(counts, float)[self.uday]
        if units is not None:
            u = np.where(units, u, 0.0)
        m = self.m
        szz = u @ m.szz
        norms = np.sqrt(szz)
        norms[norms == 0] = 1.0
        S = undilute(np.tensordot(u, m.zz, 1) / np.outer(norms, norms), np.tensordot(u, m.zzm, 1))
        c = np.mean([self.label_ics(counts, units)[i] for i in self.L], axis=0)
        return np.where(np.isfinite(c), c, np.nan), S

    def label_ics(self, counts, units=None):
        """Each label's solo IC vector over the weighted days."""
        u = np.asarray(counts, float)[self.uday]
        if units is not None:
            u = np.where(units, u, 0.0)
        m = self.m
        n, sz, szz = u @ m.n, u @ m.sz, u @ m.szz
        return [solo_ic(n, sz, szz, u @ L["sy"], u @ L["syy"], u @ L["szy"]) for L in m.labels]

    def ones(self):
        return np.ones(len(self.days))

    def ref_ic(self, c, S):
        if not self.ref:
            return 1.0
        return ic(c, S, [j for j, _ in self.ref], np.array([w for _, w in self.ref]))

    def per_period(self, counts):
        """[(c, S, the reference's IC)] per period over the weighted days."""
        out = []
        for k in np.unique(self.period):
            w = np.where(self.period == k, counts, 0.0)
            if w.sum() > 0:
                c, S = self.moments(w)
                out.append((k, c, S, self.ref_ic(c, S)))
        return out

    def halves(self):
        """Alternate days within each period: two day-count vectors."""
        h = [np.zeros(len(self.days)), np.zeros(len(self.days))]
        for k in np.unique(self.period):
            d = np.flatnonzero(self.period == k)
            h[0][d[0::2]], h[1][d[1::2]] = 1.0, 1.0
        return h


def score(per, jj, w):
    """The period-mean IC ratio of the blend to the reference."""
    vals = [ic(c, S, jj, w) / r for _, c, S, r in per]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def day_ics(st, jj, w):
    """The blend's IC on each day alone."""
    sub = st.S_day[:, jj][:, :, jj]
    v = np.einsum("i,dij,j->d", w, sub, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, st.c_day[:, jj] @ w / np.sqrt(v), np.nan)


def paired(st, counts, a, b):
    """(gain, SE) of blend `a` over blend `b` (each (jj, w)) on the weighted days: per period,
    day-by-day IC differences over the period's reference IC, sd/sqrt(days), periods combined
    as the mean; the gain from the pooled moments."""
    per = st.per_period(counts)
    gain = score(per, *a) - score(per, *b)
    d = day_ics(st, *a) - day_ics(st, *b)
    ses = []
    for k, _, _, r in per:
        sel = st.period == k
        x = np.repeat(d[sel], counts[sel].astype(int)) / r
        x = x[np.isfinite(x)]
        if len(x) >= 3:
            ses.append(float(np.std(x, ddof=1) / np.sqrt(len(x))))
    return gain, float(np.sqrt(np.sum(np.square(ses))) / len(ses)) if ses else float("nan")


def paired_daily(a, b, period_of):
    """(gain, SE) from two scored sets' per-day values (objectives with no moment form)."""
    days = sorted(set(a.daily) & set(b.daily))
    d = np.array([a.daily[x] - b.daily[x] for x in days])
    per = np.array([period_of.get(x, 0) for x in days])
    ses = [float(np.std(d[per == k], ddof=1) / np.sqrt((per == k).sum()))
           for k in np.unique(per) if (per == k).sum() >= 3]
    se = float(np.sqrt(np.sum(np.square(ses))) / len(ses)) if ses else float("nan")
    return a.value - b.value, se
