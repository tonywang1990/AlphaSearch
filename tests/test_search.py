import numpy as np

from alphasearch.search.surrogate import Surrogate


def test_thompson_preserves_recipe_order_when_jitter_still_ties(monkeypatch):
    surrogate = Surrogate([])
    # Adding jitter can round to identical doubles; CPU sort paths must then agree.
    draws = np.tile([1e16, 2e16], (3, 20))
    monkeypatch.setattr(surrogate, "samples", lambda cs: draws)
    assert surrogate.thompson([{}] * 40, 3, rng=np.random.default_rng(0)) == [1, 3, 5]
