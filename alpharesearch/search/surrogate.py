"""A model of which recipes pay off, refitted every round on every trained member (SMAC-style).

The target is a member's gain to the current F (`Selector.gains`); the features are its recipe
axes, one-hot, so any legal recipe, trained or not, can be scored. A random forest's trees are
the posterior samples: Thompson sampling draws one tree per pick."""
import numpy as np

PAIRS = (("arch", "horizon"), ("arch", "feature_set"), ("horizon", "target"), ("arch", "target"))


def matches(coords, where):
    """True if every axis in `where` holds one of its values (a trailing `*` is a prefix)."""
    for axis, values in (where or {}).items():
        v = str(coords.get(axis, ""))
        vals = [values] if isinstance(values, str) else list(values)
        if not any(v.startswith(x[:-1]) if x.endswith("*") else v == x for x in vals):
            return False
    return True


class Surrogate:
    def __init__(self, axes, trees=300, seed=0):
        self.axes, self.trees, self.seed = list(axes), trees, seed
        self.cols, self.rf, self.target = None, None, {}

    def _x(self, cs):
        return np.array([[float(c.get(a) == v) for a, v in self.cols] for c in cs])

    def fit(self, coords, gains):
        """`coords`: {name: {axis: value}} of trained members, `gains`: {name: gain}."""
        from sklearn.ensemble import RandomForestRegressor
        names = [n for n in gains if n in coords and np.isfinite(gains[n])]
        self.target = {n: gains[n] for n in names}
        cs = [coords[n] for n in names]
        self.cols = sorted({(a, c[a]) for c in cs for a in self.axes if a in c})
        if not cs:                       # nothing trained yet: the prior, every recipe alike
            return self
        self.rf = RandomForestRegressor(n_estimators=self.trees, min_samples_leaf=2,
                                        max_features=0.5, random_state=self.seed)
        self.rf.fit(self._x(cs), np.array([gains[n] for n in names], float))
        return self

    def samples(self, cs):
        """(trees x recipes) predictions: the posterior draws."""
        if self.rf is None:
            return np.random.default_rng(self.seed).normal(0.0, 1e-3, (self.trees, len(cs)))
        x = self._x(cs)
        return np.array([t.predict(x) for t in self.rf.estimators_])

    def thompson(self, cs, k, rng=None):
        """Indices of `k` distinct recipes, each the argmax of one drawn tree."""
        rng = rng or np.random.default_rng(self.seed)
        draws = self.samples(cs)
        draws = draws + rng.uniform(0, 1e-12, draws.shape)   # exact ties broken by `rng`
        picked = []
        for t in rng.permutation(len(draws)):
            order = [i for i in np.argsort(-draws[t]) if i not in picked]
            if order:
                picked.append(int(order[0]))
            if len(picked) == k:
                break
        return picked

    def predcheck(self, coords, halves=5):
        """The surrogate's held-out Spearman on the trained members: the bar an agent's own
        predictor must beat."""
        from scipy.stats import spearmanr
        names, rng, out = sorted(self.target), np.random.default_rng(0), []
        if len(names) < 6:
            return {"n_trained": len(names), "surrogate_heldout_spearman": None}
        for _ in range(halves):
            p = rng.permutation(len(names))
            a, b = [names[i] for i in p[: len(p) // 2]], [names[i] for i in p[len(p) // 2:]]
            s = Surrogate(self.axes, seed=0).fit(coords, {n: self.target[n] for n in a})
            out.append(spearmanr(s.samples([coords[n] for n in b]).mean(0), [self.target[n] for n in b])[0])
        return {"n_trained": len(names), "surrogate_heldout_spearman": round(float(np.nanmean(out)), 3),
                "how": f"{halves} random halves of the trained models: fit on one, Spearman on the other"}


def effects(gains, coords, axes, in_f=()):
    """What each grammar value and value pair has paid: the mean gain of the trained members
    carrying it, its SE and n, best first."""
    rows = [(coords[n], g) for n, g in gains.items() if n in coords and n not in in_f and np.isfinite(g)]

    def table(keys):
        cell = {}
        for c, g in rows:
            cell.setdefault(tuple(c.get(k) for k in keys), []).append(g)
        out = [{"cell": dict(zip(keys, k, strict=True)), "n": len(v), "mean": round(float(np.mean(v)), 5),
                "se": round(float(np.std(v, ddof=1) / np.sqrt(len(v))), 5) if len(v) > 1 else None,
                "best": round(float(max(v)), 5)} for k, v in cell.items()]
        return sorted(out, key=lambda r: -r["mean"])
    return {"n_models": len(rows), "single": {a: table((a,)) for a in axes},
            "pairs": {f"{a}x{b}": table((a, b))[:12] for a, b in PAIRS if a in axes and b in axes}}
