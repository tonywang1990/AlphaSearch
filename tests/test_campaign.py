import json
import os
import subprocess

import pytest

from alphasearch import campaign
from alphasearch.config import Config
from alphasearch.infra.fake import FakeInfra
from alphasearch.policies import agent as agent_policy


def _rows(path):
    return [json.loads(x) for x in open(path)]


@pytest.mark.parametrize("policy", ["surrogate", "random"])
def test_baseline_campaign_runs_and_resumes(tmp_path, policy):
    infra = FakeInfra(seed=2)
    out = str(tmp_path / "c")
    campaign.run(infra, Config(policy=policy, rounds=2, models=6), out)
    it = _rows(os.path.join(out, "iters.jsonl"))
    assert [r["iteration"] for r in it] == [0, 1]
    assert all(r["test_verdict"] in ("better", "worse", "flat", "not_scored") for r in it)
    assert it[0]["bank"] > 2 and os.path.exists(os.path.join(out, "episode.json"))
    campaign.run(infra, Config(policy=policy, rounds=2, models=6), out)   # resume: nothing reruns
    assert len(_rows(os.path.join(out, "iters.jsonl"))) == 2


def test_agent_round_uses_tools_and_never_sees_test(tmp_path, monkeypatch):
    """The agent path end to end with a scripted agent: it calls ./tool through the workspace
    launcher, proposes experiments, and its next brief carries no test verdict."""
    seen = {}

    def scripted(R, cfg, log):
        ws = R.ws
        run = lambda *a: json.loads(subprocess.run(["./tool", *a], cwd=ws, capture_output=True, text=True,
                                                   check=True).stdout)
        seen["members"] = run("members")
        seen["whatif"] = run("whatif", "--drop", R.F[0])
        seen["suggest"] = run("suggest", "--n", "3")
        seen["screen"] = run("screen", "x0", "x1", "x5")
        seen["crossval"] = run("crossval")
        seen["ledger"] = json.load(open(os.path.join(ws, "ledger.json")))
        recipes = [s["recipe"] for s in seen["suggest"]]
        return {"experiments": [
            {"id": "e1", "action": "train", "hypothesis": "h", "recipes": recipes[:2]},
            {"id": "e2", "action": "feature_set", "hypothesis": "h", "name": "fx", "base": "agg_a",
             "features": ["x5", "x6", "x7"], "twins": R.F[:1], "k": 2},
            {"id": "e3", "action": "ensemble", "hypothesis": "h", "members": R.F}],
            "lessons": ["l"], "stop": False}
    monkeypatch.setattr(agent_policy, "session", scripted)
    infra = FakeInfra(seed=3)
    out = str(tmp_path / "a")
    campaign.run(infra, Config(policy="agent", rounds=2, models=6), out)
    assert seen["members"] and "val_gain" in seen["whatif"] and "latest_models_value" in seen["crossval"]
    assert "beats_null" in seen["screen"]["all"]
    assert seen["ledger"] and not any(k.startswith("test") for r in seen["ledger"] for k in r)
    led = _rows(os.path.join(out, "ledger.jsonl"))
    assert any(r.get("feature_set", "").startswith("rs_") or "refused" in r for r in led if r.get("id") == "e2")
    assert any("test_verdict" in r for r in led)


def test_frontier_chain_carries_the_bank_and_shows_it(tmp_path):
    """agent-search.md's configuration: ic_drift, no swap bar, frontier, best-swap surrogate, a
    store across campaigns. Campaign 2 starts with every model campaign 1 trained (free), its
    brief shows the whole bank, and the store names the best final F."""
    import numpy as np
    store = str(tmp_path / "store")
    cfg = Config(policy="surrogate", select_objective="ic_drift", swap_t=0.0, frontier=True,
                 sur_target="best_swap", experience=store, seed=1, models=6)
    first = FakeInfra(seed=5)
    campaign.run(first, Config(**{**cfg.__dict__, "rounds": 2}), str(tmp_path / "c1"))
    trained = {n for r in _rows(tmp_path / "c1" / "iters.jsonl") for n in r["trained"]}
    second = FakeInfra(seed=5)
    campaign.run(second, Config(**{**cfg.__dict__, "rounds": 1}), str(tmp_path / "c2"))
    bank = json.load(open(tmp_path / "c2" / "it_0" / "bank.json"))
    assert trained <= {r["model"] for r in bank["rows"]}                       # carried, free
    assert len(_rows(tmp_path / "c2" / "iters.jsonl")[0]["trained"]) == 6        # the budget counts new ones
    row = next(r for r in bank["rows"] if not r["in_F"])
    assert {"ic_val", "corr_with_F", "pcorr_given_F", "best_swap_gain", "best_swap_out",
            "in_earlier_final_F"} <= set(row)
    m = np.load(tmp_path / "c2" / "it_0" / "moments.npz")
    assert list(m["labels"]) == ["10s", "30s", "60s"] and list(m["select_labels"]) == [0, 1, 2]
    assert bank["F_val_from_moments"] == pytest.approx(bank["F_val"], abs=1e-4)
    assert "Best final F on val" in open(os.path.join(store, "experience.md")).read()


def test_memory_is_this_weeks_prod_on_the_other_symbols(tmp_path):
    """No --memory: the campaign renders memory.md from what infra says prod serves, excluding its
    own symbol, and every round's workspace carries that frozen copy."""
    out = str(tmp_path / "m")
    campaign.run(FakeInfra(seed=6), Config(policy="surrogate", rounds=1, models=4), out)
    text = open(os.path.join(out, "it_0", "memory.md")).read()
    assert "OTHER" in text and "som_other_2f0" in text and "before it hybrid" in text
    assert "**FAKE**" not in text and "linear 1s y on agg_all" in text
