import json
import os
import subprocess

import pytest

from alpharesearch.agent import loop
from alpharesearch.infra.fake import FakeInfra


def _rows(path):
    return [json.loads(x) for x in open(path)]


@pytest.mark.parametrize("policy", ["surrogate", "random"])
def test_baseline_campaign_runs_and_resumes(tmp_path, policy):
    infra = FakeInfra(seed=2)
    out = str(tmp_path / "c")
    loop.run(infra, {}, out, policy=policy, iters=2, models=6)
    it = _rows(os.path.join(out, "iters.jsonl"))
    assert [r["iteration"] for r in it] == [0, 1]
    assert all(r["test_verdict"] in ("better", "worse", "flat", "not_scored") for r in it)
    assert it[0]["bank"] > 2 and os.path.exists(os.path.join(out, "episode.json"))
    loop.run(infra, {}, out, policy=policy, iters=2, models=6)          # resume: nothing reruns
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
    monkeypatch.setattr(loop, "plan_agent", scripted)
    infra = FakeInfra(seed=3)
    out = str(tmp_path / "a")
    loop.run(infra, {}, out, policy="agent", iters=2, models=6)
    assert seen["members"] and "val_gain" in seen["whatif"] and "latest_models_value" in seen["crossval"]
    assert "beats_null" in seen["screen"]["all"]
    assert seen["ledger"] and not any(k.startswith("test") for r in seen["ledger"] for k in r)
    led = _rows(os.path.join(out, "ledger.jsonl"))
    assert any(r.get("feature_set", "").startswith("rs_") or "refused" in r for r in led if r.get("id") == "e2")
    assert any("test_verdict" in r for r in led)


def test_split_val_gives_the_agent_other_days(tmp_path):
    infra = FakeInfra(seed=4)
    sel, tsel = loop.selectors(infra, infra.describe(), {**loop.DEFAULTS, "split_val": True})
    assert not set(sel.val.days) & set(tsel.val.days)
