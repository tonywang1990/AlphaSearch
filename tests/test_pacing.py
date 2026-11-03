"""A live campaign paces itself (ADR 0472-research-api-v1-live): the round's size from measured model cost, the
deadline and the budget; the loop ends on the agent's stop, a plateau, the deadline or the budget."""
import json
import os
from types import SimpleNamespace

from alphasearch import campaign, pacing
from alphasearch.config import Config
from alphasearch.infra.contract import Budget
from alphasearch.infra.fake import FakeInfra
from alphasearch.records import Plan


def _journal(*rows):
    return SimpleNamespace(iters=list(rows))


def _round(val, verdict="flat", secs=None):
    return {"F_after_val": val, "test_verdict": verdict, "train_seconds": secs or {}}


def test_the_round_fits_the_deadline_the_budget_and_what_a_model_costs():
    cfg = Config(models=16, final_reserve_min=45, model_minutes=40)
    j = _journal(_round(1.0, secs={"tree.10s.y.agg_store.all.40d.hot.split.x1": 3600,
                                   "ridge.1s.y.agg_book.s10.40d.hot.split.x1": 1200,
                                   "gru.1s.y.na_all.all.40d.hot.split.x1": 1800,
                                   "ridge.5s.y.agg_xs.all.40d.hot.split.x1": 0}))
    assert pacing.arch_minutes(j) == {"gru": 30.0, "ridge": 20.0, "tree": 60.0}      # a cached reuse is no cost
    assert pacing.arch_minutes(j, by_set=True) == {"gru.na_all": 30.0, "ridge.agg_book": 20.0, "tree.agg_store": 60.0}
    b = Budget(seconds_left=10 * 3600, verdicts_left=9, deadline=1000 + (45 + 95) * 60, max_parallel=4)
    n, block, why = pacing.pace(b, j, cfg, now=1000)
    assert (n, why) == (12, None) and block["minutes_left"] == 95 and block["parallel"] == 4   # 4 × floor(95 / 30)
    n, _, why = pacing.pace(Budget(40 * 60, 9, b.deadline, 4), j, cfg, now=1000)
    assert n == 1 and why is None                                                  # 40 compute min: one median model
    assert pacing.pace(Budget(10 * 3600, 9, 1000 + 50 * 60, 4), j, cfg, now=1000)[2].startswith("deadline")
    assert pacing.pace(Budget(10 * 60, 9, None, 4), j, cfg, now=1000)[2].startswith("budget")
    assert pacing.pace(Budget(1e9, 9, None, 4), _journal(), cfg)[0] == 16          # no deadline, nothing measured


def test_a_plateau_is_val_gains_too_small_with_no_better_verdict():
    cfg = Config(plateau_rounds=2, plateau_gain=0.002)
    assert pacing.plateau(_journal(_round(1.0), _round(1.01)), cfg) is None        # too few rounds to judge
    assert pacing.plateau(_journal(_round(1.0), _round(1.001), _round(1.0015)), cfg).startswith("plateau")
    assert pacing.plateau(_journal(_round(1.0), _round(1.001, "better"), _round(1.0015)), cfg) is None
    assert pacing.plateau(_journal(_round(1.0), _round(1.01), _round(1.02)), cfg) is None


def test_the_agent_sizes_its_round_and_a_live_campaign_stops_with_its_reason(tmp_path, monkeypatch):
    assert Plan.parse({"train": 3}).train == 3 and Plan.parse({}).train is None
    infra = FakeInfra(seed=3)
    seen = []
    real = campaign.rnd.act
    monkeypatch.setattr(campaign.rnd, "act",
                        lambda obs, plan, policy, n, rng, until=None: seen.append(n) or real(obs, plan, policy, n, rng))
    monkeypatch.setattr(campaign.POLICIES["surrogate"], "plan",
                        lambda self, obs, models, rng, log: Plan(experiments=[], train=2, stop=obs.it == 1))
    campaign.run(infra, Config(policy="surrogate", rounds=5, models=6, seed=1), str(tmp_path / "a"))
    assert seen == [2, 2]                                                          # the agent's size, under the cap
    assert json.load(open(tmp_path / "a" / "stop.json"))["reason"].startswith("agent")
    infra = FakeInfra(seed=3)
    infra.budget = lambda: Budget(seconds_left=60, verdicts_left=9, deadline=None, max_parallel=2)
    campaign.run(infra, Config(policy="surrogate", rounds=5, models=6, seed=1), str(tmp_path / "b"))
    assert not os.path.exists(tmp_path / "b" / "iters.jsonl")                      # 1 compute min: no round starts
    assert json.load(open(tmp_path / "b" / "stop.json"))["reason"].startswith("budget")


def test_a_budget_refusal_mid_round_ends_the_campaign_cleanly(tmp_path):
    """Past the deadline infra refuses every job, scoring included: the campaign stops with the last round's F."""
    from alphasearch.infra.contract import ApiError
    infra = FakeInfra(seed=3)
    train = infra.train
    calls = []

    def refuse_second(recipes, until=None):
        calls.append(until)
        if len(calls) > 1:
            raise ApiError("budget_exhausted", "Campaign compute allowance exhausted")
        return train(recipes)
    infra.train = refuse_second
    campaign.run(infra, Config(policy="surrogate", rounds=5, models=3, seed=1), str(tmp_path / "c"))
    assert len(open(tmp_path / "c" / "iters.jsonl").readlines()) == 1
    assert json.load(open(tmp_path / "c" / "stop.json"))["reason"].startswith("budget or deadline")
