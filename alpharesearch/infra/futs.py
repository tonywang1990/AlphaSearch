"""The real adapter: `Infra` over futs-infra's public module `futs_infra.api` (interface v1).
The only file in AlphaResearch that imports futs-infra; result models are mapped field for
field (the v1 schemas use the contract's names), and every infra failure surfaces as the
contract's `ApiError` with infra's own code."""
import numpy as np

from alpharesearch.infra.contract import (ApiError, Budget, Description, Model, Moments, Scored,
                                          check_version, name_of)


def _api():
    try:
        import futs_infra.api as api
    except ImportError as e:              # pragma: no cover - needs the installed wheel
        raise ApiError("unavailable", "futs-infra with interface v1 is not installed") from e
    return api


class FutsInfra:
    def __init__(self, campaign):
        self.api = _api()
        self.r = self.api.Research(campaign)
        self.d = self._call(self.r.describe)
        self.desc = Description(
            api_version=self.d.api_version, campaign=self.d.campaign, symbol=self.d.symbol,
            axes={a: list(v) for a, v in self.d.axes.items()}, objective=self.d.objective.metric,
            k_max=self.d.objective.k_max, reference=list(self.d.reference), host=self.d.host,
            columns=list(self.d.columns), feature_sets={k: list(v) for k, v in self.d.feature_sets.items()})
        check_version(self.desc)

    def _call(self, f, *a, **k):
        try:
            out = f(*a, **k)
            return out.wait() if hasattr(out, "wait") else out
        except self.api.ApiError as e:
            raise ApiError(e.code, e.message, e.retryable) from e

    def describe(self):
        return self.desc

    def budget(self):
        b = self._call(self.r.budget)
        return Budget(seconds_left=b.seconds_left, verdicts_left=b.queries_left["verdict"],
                      deadline=b.deadline, max_parallel=b.max_parallel)

    def recipes(self, where=None):
        return [dict(r) for r in self._call(self.r.recipes, where=where)]

    def validate(self, recipes):
        rep = self._call(self.r.validate, {name_of(r, self.desc.axes): r for r in recipes})
        return {k: None if v.status == "valid" else "; ".join(i.message for i in v.issues)
                for k, v in rep.results.items()}

    def _model(self, h):
        return Model(name=name_of(h.recipe, self.desc.axes), recipe=dict(h.recipe), id=h.id)

    def train(self, recipes):
        res = self._call(self.r.train, {name_of(r, self.desc.axes): r for r in recipes})
        ready, failed = [], {}
        for label, outcomes in res.results.items():
            ok = [o for o in outcomes if o.status == "ready"]
            if ok:
                ready.append(self._model(ok[0].member))
            else:
                failed[label] = "; ".join(o.error or "failed" for o in outcomes)
        return ready, failed

    def models(self):
        return [self._model(h) for h in self._call(self.r.models)]

    def _handles(self, legs):
        by = {m.name: m for m in self._call(self.r.models)}
        return [(by[n].handle, w) for n, w in legs]

    def moments(self, block):
        m = self._call(self.r.moments, blocks=(block,))[block]
        arr = lambda x: np.asarray(x, float)
        return Moments(names=[name_of(h.recipe, self.desc.axes) for h in m.members],
                       day=np.asarray(m.day), session=np.asarray(m.session), period=np.asarray(m.period),
                       periods=list(m.periods), sessions=list(m.sessions), n=arr(m.n), sz=arr(m.sz),
                       szz=arr(m.szz), zz=arr(m.zz), zzm=arr(m.zzm),
                       labels=[{k: arr(L[k]) for k in ("sy", "syy", "szy")} for L in m.labels],
                       reference=[(name_of(h.recipe, self.desc.axes), w) for h, w in m.reference])

    def score_many(self, sets, objective):
        res = self._call(self.r.score_many, [self._cand(s) for s in sets], block="validation",
                         objective=objective)
        return [Scored(value=s.value, daily=dict(s.daily)) for s in res.scores]

    def _cand(self, legs):
        return self._call(self.r.create_candidate, self._handles(legs)).candidate

    def verdict(self, subject, compare_to):
        return self._call(self.r.verdict, self._cand(subject), compare_to=self._cand(compare_to)).verdict

    def screen(self, columns, parent):
        return self._call(self.r.screen, columns, parent=self._cand(parent)).model_dump()

    def pcorr(self, columns, parent, cell=None):
        return self._call(self.r.pcorr, columns, parent=self._cand(parent), cell=cell).model_dump()

    def propose_feature_set(self, name, columns, base=None):
        return self._call(self.r.propose_feature_set, name, columns, base=base).model_dump()

    def close(self, outcome, reason):
        self._call(self.r.close, outcome, reason)
