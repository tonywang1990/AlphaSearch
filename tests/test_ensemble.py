import numpy as np
import pytest

from alpharesearch.ensemble.moments import Stats, fit_weights, ic, paired, subset
from alpharesearch.ensemble.select import Cfg, Selector
from alpharesearch.infra.fake import FakeInfra


@pytest.fixture(scope="module")
def infra():
    f = FakeInfra(seed=1)
    f.train(f.recipes()[:20])
    return f


def test_moments_reproduce_pooled_pearson(infra):
    """The IC from moments is the Pearson IC on the pooled rows (members valid on every tick)."""
    st = Stats(infra.moments("validation"))
    names = [n for n in st.names if not n.startswith("gru")][:4]
    rows = infra._rows("validation")
    Z, _ = infra._z(names, rows)
    y = infra.y[rows]
    c, S = st.moments(st.ones())
    jj = [st.idx[n] for n in names]
    for k, j in enumerate(jj):
        assert c[j] == pytest.approx(np.corrcoef(Z[:, k], y)[0, 1], abs=1e-9)
    w = np.array([0.4, -0.1, 0.3, 0.2])
    assert ic(c, S, jj, w) == pytest.approx(np.corrcoef(Z @ w, y)[0, 1], abs=1e-9)


def test_day_counts_are_day_subsets(infra):
    m = infra.moments("validation")
    st = Stats(m)
    half = st.halves()[0]
    sub = Stats(subset(m, st.days[half > 0]))
    a, b = st.moments(half), sub.moments(sub.ones())
    assert np.allclose(np.nan_to_num(a[0]), np.nan_to_num(b[0])) and np.allclose(a[1], b[1])


def test_selection_beats_prod_and_respects_k_max(infra):
    sel = Selector(Stats(infra.moments("validation")), Stats(infra.moments("fit")), Cfg(k_max=infra.k_max))
    ref = infra.describe().reference
    F = sel.select([ref])
    assert 0 < len(F) <= infra.k_max
    f = sel.scorer()
    assert f(sel.ix(F)) >= f(sel.ix(ref))
    g, se = paired(sel.val, sel.val.ones(), sel.blend(sel.ix(F)), sel.blend(sel.ix(ref)))
    assert np.isfinite(g) and se > 0


def test_weights_maximise_ic():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5))
    S = A @ A.T + 5 * np.eye(5)
    S = S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    c = rng.uniform(0.01, 0.1, 5)
    w = fit_weights(c, S)
    best = ic(c, S, list(range(5)), w)
    for _ in range(50):
        assert ic(c, S, list(range(5)), w + rng.normal(0, 0.05, 5)) <= best + 1e-12


def test_host_mode_keeps_the_host(infra):
    host = infra.describe().reference[0]
    sel = Selector(Stats(infra.moments("validation")), Stats(infra.moments("fit")), Cfg(k_max=4, host=host))
    F = sel.select()
    assert host in F and dict(sel.legs(sel.ix(F)))[host] == 1.0


def test_ev_selection_goes_through_infra_in_batches(infra):
    calls = []
    real = infra.score_many

    def spy(sets, objective):
        calls.append(len(sets))
        return real(sets, objective)
    infra.score_many = spy
    try:
        sel = Selector(Stats(infra.moments("validation")), Stats(infra.moments("fit")),
                       Cfg(k_max=3, objective="ev"), infra)
        F = sel.select([infra.describe().reference])
    finally:
        infra.score_many = real
    assert 0 < len(F) <= 3 and max(calls) > 1
