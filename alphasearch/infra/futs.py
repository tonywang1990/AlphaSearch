"""The real adapter: `Infra` over futs-infra's public module `futs_infra.api` (interface v1).
The only file in AlphaSearch that imports futs-infra. Infra names models by its own asset
names and ids; the client names them by recipe (`name_of`), so this file maps between them."""
import numpy as np

from alphasearch.infra.contract import ApiError, Budget, Description, Model, Moments, Scored, check_version, name_of


def _api():
    try:
        import futs_infra.api as api
    except ImportError as e:              # pragma: no cover - needs the installed wheel
        raise ApiError("unavailable", "futs-infra with interface v1 is not installed") from e
    return api


class FutsInfra:
    def __init__(self, campaign):
        self.api = _api()
        self.r = self._call(self.api.Research, campaign)
        self._moments, self._bank_key = {}, None
        self.desc = self.describe()
        check_version(self.desc)

    def _call(self, f, *a, **k):
        try:
            out = f(*a, **k)
            return out.wait() if isinstance(out, self.api.Job) else out
        except self.api.ApiError as e:
            raise ApiError(e.code, e.message, e.retryable) from e

    # -- names ------------------------------------------------------------------------------
    def _bank(self):
        entries = self._call(self.r.models)
        self._by_id = {e.id: e for e in entries}
        self._name = {e.name: self._client(e) for e in entries}       # infra name -> client name
        self._id = {self._client(e): e.id for e in entries}           # client name -> id
        return entries

    def _client(self, entry):
        return name_of(entry.recipe, self._axes) if entry.recipe else entry.name

    def _legs(self, legs):
        self._bank()
        return [(self._id[n], float(w)) for n, w in legs]

    # -- the contract -----------------------------------------------------------------------
    def describe(self):
        d = self._call(self.r.describe)
        self._axes = {a: list(v) for a, v in d.axes.items()}
        rs = d.research
        by_id = {e.id: self._client(e) for e in self._bank()}
        return Description(api_version=d.api_version, campaign=d.campaign, symbol=d.symbol, axes=self._axes,
                           objective=rs.metric, k_max=rs.k_max, reference=[by_id[i] for i, _ in rs.reference],
                           host=by_id.get(rs.host), columns=list(rs.columns),
                           feature_sets={k: list(v) for k, v in rs.feature_sets.items()},
                           feature_set_rule=rs.feature_set_rule.model_dump() if rs.feature_set_rule else None,
                           universe=rs.universe, labels=list(rs.labels), fit_label=d.scoring_horizon,
                           notes={a: dict(v) for a, v in rs.notes.items()}, feature_set_defs=dict(rs.feature_set_defs))

    def budget(self):
        b = self._call(self.r.budget)
        return Budget(seconds_left=b.seconds_left, verdicts_left=b.queries_left["verdict"],
                      deadline=b.deadline, max_parallel=b.max_parallel)

    def recipes(self, where=None):
        return [dict(r) for r in self._call(self.r.recipes, where)]

    def validate(self, recipes):
        if self._call(self.r.describe).research.universe:
            legal = {name_of(r, self._axes) for r in self.recipes()}
            return {name_of(r, self._axes): None if name_of(r, self._axes) in legal else "not in the universe"
                    for r in recipes}
        rep = self._call(self.r.validate, {name_of(r, self._axes): r for r in recipes})
        return {k: None if v.status == "valid" else "; ".join(i.message for i in v.issues)
                for k, v in rep.results.items()}

    def train(self, recipes):
        res = self._call(self.r.train, {name_of(r, self._axes): r for r in recipes})
        self._bank()
        ready, failed = [], {}
        for label, outcomes in res.results.items():
            ok = [o for o in outcomes if o.status == "ready"]
            if ok and ok[0].member.id in self._by_id:
                e = self._by_id[ok[0].member.id]
                ready.append(Model(name=self._client(e), recipe=dict(e.recipe or {}), id=e.id))
            elif not ok:
                failed[label] = "; ".join(o.error or "failed" for o in outcomes)
        return ready, failed

    def models(self):
        return [Model(name=self._client(e), recipe=dict(e.recipe or {}), id=e.id)
                for e in self._bank() if e.role == "member"]

    def moments(self, block):
        key = tuple(sorted(e.id for e in self._bank()))
        if key != self._bank_key:            # one job for both blocks, per bank state
            result, arrays = self._call(self.r.moments, ("fit", "validation"))
            self._moments, self._bank_key = {"_names": result.names, "_ref": result.reference, **arrays}, key
        a, names = self._moments[block], [self._name[n] for n in self._moments["_names"]]
        return Moments(names=names, day=a["day"], session=a["session"], period=a["period"],
                       periods=[str(p) for p in a["periods"]], sessions=[str(s) for s in a["sessions"]],
                       n=a["n"], sz=a["sz"], szz=a["szz"], zz=a["zz"], zzm=a["zzm"],
                       labels=[{k: a[k][i] for k in ("sy", "syy", "szy")} for i in range(len(a["labels"]))],
                       label_names=[str(x) for x in a["labels"]],
                       reference=[(self._name[n], w) for n, w in self._moments["_ref"]])

    def score_many(self, sets, objective):
        res = self._call(self.r.score_many, [self._legs(s) for s in sets], objective=objective)
        return [Scored(value=np.nan if s.value is None else s.value, daily=dict(s.daily)) for s in res.scores]

    def verdict(self, subject, compare_to):
        return self._call(self.r.verdict, self._legs(subject), compare_to=self._legs(compare_to)).verdict

    def screen(self, columns, parent):
        return self._call(self.r.screen, columns, parent=self._legs(parent)).cells

    def pcorr(self, columns, parent, cell=None):
        return self._call(self.r.pcorr, columns, parent=self._legs(parent), cell=cell).columns

    def propose_feature_set(self, name, columns, base=None):
        return self._call(self.r.propose_feature_set, name, columns, base=base).model_dump()

    def close(self, outcome, reason):
        self._call(self.r.close, outcome, reason)
