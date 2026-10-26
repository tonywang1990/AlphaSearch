"""A synthetic futs-infra for tests and dry runs: the v1 contract over generated ticks.

The label is a few latent factors with day-varying loadings plus noise; a recipe's prediction
loads on those factors through its axis values (so recipes sharing values are correlated and
some cells pay), plus its own noise. Like the real infra, the rows never leave this object:
the client gets moments, scores, verdict words and screens."""
import hashlib

import numpy as np

from alphasearch.infra.contract import ApiError, Budget, Description, Model, Moments, Scored, name_of

AXES = {"arch": ["tree", "linear", "gru"], "horizon": ["10s", "30s", "60s"],
        "target": ["ret", "skip"], "feature_set": ["agg_a", "agg_b", "book"]}
PERIODS, SESSIONS = ["calm", "storm", "recent"], ["night", "morning", "afternoon"]
BLOCKS = {"fit": range(0, 24), "validation": range(24, 36), "test": range(36, 48)}


def _seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class FakeInfra:
    def __init__(self, seed=0, factors=8, ticks=120, verdicts=40, k_max=6, verdict_t=2.0):
        self.rng, self.F, self.k_max, self.verdict_t = np.random.default_rng(seed), factors, k_max, verdict_t
        self.seed, self.verdicts_left = seed, verdicts
        days = max(r.stop for r in BLOCKS.values())
        self.unit_day = np.repeat(np.arange(days), len(SESSIONS))
        self.unit_session = np.tile(np.arange(len(SESSIONS)), days)
        self.unit = np.repeat(np.arange(len(self.unit_day)), ticks)
        self.day = self.unit_day[self.unit]
        self.X = self.rng.standard_normal((len(self.unit), factors))
        beta = self.rng.normal(0.0, 1.0, (days, factors)) * 0.3 + np.linspace(1, 0.2, factors)
        self.y = (self.X * beta[self.day]).sum(1) + 3.0 * self.rng.standard_normal(len(self.unit))
        self.labels = {"10s": self.y, "30s": self.y + 2.0 * self.rng.standard_normal(len(self.unit)),
                       "60s": 0.5 * self.y + 3.0 * self.rng.standard_normal(len(self.unit))}
        self.columns = [f"x{i}" for i in range(factors)] + [f"noise{i}" for i in range(12)]
        self.store = np.hstack([self.X, self.rng.standard_normal((len(self.unit), 12))])
        self.feature_sets = {"agg_a": self.columns[:4], "agg_b": self.columns[4:8], "book": self.columns[2:6]}
        self.axes = {a: list(v) for a, v in AXES.items()}
        self.preds, self.trained, self.failed = {}, {}, {}
        ref = [{"arch": "tree", "horizon": "30s", "target": "ret", "feature_set": "agg_a"},
               {"arch": "linear", "horizon": "10s", "target": "ret", "feature_set": "agg_b"}]
        self.train(ref)
        self.reference = [name_of(r, self.axes) for r in ref]

    # -- the grammar ----------------------------------------------------------------------
    def describe(self):
        return Description(api_version="1.0", campaign=f"fake/{self.seed}", symbol="FAKE",
                           axes=self.axes, objective="ic", k_max=self.k_max, reference=self.reference,
                           columns=list(self.columns), feature_sets={k: list(v) for k, v in self.feature_sets.items()},
                           feature_set_rule={"min_columns": 2, "max_columns": 200}, labels=list(self.labels),
                           fit_label="10s", notes={"arch": {"gru": "a recurrent net on the book"}},
                           served=[{"slot": "fake", "symbol": "FAKE", "release": "prod_2", "model": "som_fake",
                                    "since": "prod_2", "prev_model": None, "legs": []},
                                   {"slot": "other", "symbol": "OTHER", "release": "prod_2", "model": "som_other_2f0",
                                    "since": "prod_2", "prev_model": "hybrid", "legs": [
                                        {"member": "linear.1s.y.agg_all.all.40d.hot.pooled.x1.s1", "kind": "linear",
                                         "weight": 1.0, "recipe": {"arch": "linear", "horizon": "1s", "target": "y",
                                                                   "feature_set": "agg_all", "feature_selection": "all",
                                                                   "session": "pooled"}}]}])

    def budget(self):
        return Budget(seconds_left=1e9, verdicts_left=self.verdicts_left, deadline=None, max_parallel=4)

    def _legal(self, r):
        if set(r) != set(self.axes) or any(r[a] not in self.axes[a] for a in self.axes):
            return "not a grammar coordinate"
        if r["arch"] == "linear" and r["target"] == "skip":
            return "law: linear x skip cannot build"
        return None

    def recipes(self, where=None):
        from itertools import product

        from alphasearch.search.surrogate import matches
        out = [dict(zip(self.axes, v, strict=True)) for v in product(*self.axes.values())]
        return [r for r in out if self._legal(r) is None and name_of(r, self.axes) not in self.trained
                and matches(r, where)]

    def validate(self, recipes):
        return {name_of(r, self.axes): self._legal(r) for r in recipes}

    # -- training -------------------------------------------------------------------------
    def _loadings(self, r):
        a = np.zeros(self.F)
        for axis, v in r.items():
            g = np.random.default_rng(_seed(axis, v, self.seed))
            a += g.normal(0, 0.5, self.F) * (1.0 if axis != "feature_set" else 0.0)
        cols = self.feature_sets[r["feature_set"]]
        for c in cols:
            if c.startswith("x"):
                a[int(c[1:])] += 0.8
        return a

    def train(self, recipes):
        ready, failed = [], {}
        for r in recipes:
            n = name_of(r, self.axes)
            why = self._legal(r) or self.failed.get(n)
            if why:
                failed[n] = why
                continue
            if n not in self.trained:
                g = np.random.default_rng(_seed("noise", n, self.seed))
                p = self.X @ self._loadings(r) + 4.0 * g.standard_normal(len(self.unit))
                valid = np.ones(len(p), bool)
                if r["arch"] == "gru":                          # a model missing some ticks
                    valid = g.uniform(size=len(p)) > 0.05
                self.preds[n] = (np.where(valid, p, 0.0), valid)
                self.trained[n] = Model(name=n, recipe=dict(r), id=hashlib.sha256(n.encode()).hexdigest()[:16])
            ready.append(self.trained[n])
        return ready, failed

    def models(self):
        return list(self.trained.values())

    # -- search tier ----------------------------------------------------------------------
    def _rows(self, block):
        return np.isin(self.day, list(BLOCKS[block]))

    def _z(self, names, rows):
        """Per-day standardized predictions (zero where invalid) and validity, on `rows`."""
        Z = np.zeros((rows.sum(), len(names)))
        M = np.zeros_like(Z)
        d = self.day[rows]
        for k, n in enumerate(names):
            p, v = self.preds[n][0][rows], self.preds[n][1][rows]
            for day in np.unique(d):
                q = (d == day) & v
                if q.sum() > 1:
                    Z[q, k] = (p[q] - p[q].mean()) / (p[q].std() or 1.0)
                    M[q, k] = 1.0
        return Z, M

    def moments(self, block):
        if block not in ("fit", "validation"):
            raise ApiError("not_permitted", f"moments on {block!r}")
        names, rows = sorted(self.trained), self._rows(block)
        Z, M = self._z(names, rows)
        u = self.unit[rows]
        units = np.unique(u)
        K = len(names)
        agg = {k: np.zeros((len(units), K)) for k in ("n", "sz", "szz")}
        lab = [{k: np.zeros((len(units), K)) for k in ("sy", "syy", "szy")} for _ in self.labels]
        zz, zzm = np.zeros((len(units), K, K)), np.zeros((len(units), K, K))
        for i, x in enumerate(units):
            q = u == x
            z, m = Z[q], M[q]
            agg["n"][i], agg["sz"][i], agg["szz"][i] = m.sum(0), z.sum(0), (z * z).sum(0)
            for y, L in zip(self.labels.values(), lab, strict=True):
                yy = y[rows][q]
                L["sy"][i], L["syy"][i], L["szy"][i] = m.T @ yy, m.T @ (yy * yy), z.T @ yy
            zz[i], zzm[i] = z.T @ z, (z * z).T @ m
        days = self.unit_day[units]
        return Moments(names=names, day=days, session=self.unit_session[units],
                       period=np.minimum((days - min(BLOCKS[block])) * 3 // len(BLOCKS[block]), 2),
                       periods=PERIODS, sessions=SESSIONS, n=agg["n"], sz=agg["sz"], szz=agg["szz"],
                       zz=zz, zzm=zzm, labels=lab, label_names=list(self.labels),
                       reference=[(n, 1.0 / len(self.reference)) for n in self.reference])

    def _pred(self, legs, rows):
        Z, _ = self._z([n for n, _ in legs], rows)
        return Z @ np.array([w for _, w in legs])

    def _daily(self, legs, block, objective):
        rows = self._rows(block)
        p, y, d = self._pred(legs, rows), self.y[rows], self.day[rows]
        out = {}
        for day in np.unique(d):
            q = d == day
            if objective == "ic":
                out[str(day)] = float(np.corrcoef(p[q], y[q])[0, 1])
            else:                                               # tail EV: the top 1% |pred|
                top = np.argsort(-np.abs(p[q]))[: max(1, q.sum() // 100)]
                out[str(day)] = float(np.mean(np.sign(p[q][top]) * y[q][top]))
        return out

    def score_many(self, sets, objective):
        out = []
        for legs in sets:
            daily = self._daily(legs, "validation", objective)
            out.append(Scored(value=float(np.mean(list(daily.values()))), daily=daily))
        return out

    def verdict(self, subject, compare_to):
        if self.verdicts_left <= 0:
            raise ApiError("budget_exhausted", "no verdict looks left")
        self.verdicts_left -= 1
        a, b = self._daily(subject, "test", "ic"), self._daily(compare_to, "test", "ic")
        d = np.array([a[k] - b[k] for k in a])
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std() > 0 else 0.0
        return "better" if t >= self.verdict_t else "worse" if t <= -self.verdict_t else "flat"

    def _resid(self, parent, rows):
        p = self._pred(parent, rows)
        y = self.y[rows]
        return y - p * (p @ y) / max(p @ p, 1e-12)

    def screen(self, columns, parent):
        rows = self._rows("validation")
        r, ix = self._resid(parent, rows), [self.columns.index(c) for c in columns]

        def r2(cols):
            X = self.store[rows][:, cols]
            half = self.day[rows] % 2 == 0
            b = np.linalg.lstsq(X[half], r[half], rcond=None)[0]
            e = r[~half] - X[~half] @ b
            return 1 - (e @ e) / (r[~half] @ r[~half])
        null = [r2(list(self.rng.choice(len(self.columns), len(ix), replace=False))) for _ in range(4)]
        got = r2(ix)
        return {"all": {"r2": got, "null_r2": float(np.mean(null)), "beats_null": got - float(np.mean(null))}}

    def pcorr(self, columns, parent, cell=None):
        rows = self._rows("validation")
        r = self._resid(parent, rows)
        return {c: {"all": float(np.corrcoef(self.store[rows][:, self.columns.index(c)], r)[0, 1])}
                for c in columns}

    def propose_feature_set(self, name, columns, base=None):
        cols = list(dict.fromkeys((self.feature_sets.get(base) or []) + list(columns)))
        if not 2 <= len(cols) <= 200 or any(c not in self.columns for c in cols):
            return {"admitted": False, "why": "columns must be 2-200 store columns"}
        value = f"rs_{name}"
        if value in self.axes["feature_set"]:
            return {"admitted": False, "why": f"{value} exists"}
        self.feature_sets[value] = cols
        self.axes["feature_set"].append(value)
        return {"admitted": True, "value": value, "columns": cols}

    def close(self, outcome, reason):
        self.closed = (outcome, reason)
