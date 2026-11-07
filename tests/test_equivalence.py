"""The harness reproduces its recorded decisions and numerical behaviour (tests/golden/campaigns.json,
re-recorded for permutation-invariant selection ties; docs/design.md): four campaigns (surrogate, random,
a two-campaign frontier chain, a scripted agent), compared field by field: every round row, every
ledger row, the last workspace's bank.json and state.json and its file list. A refactor that changes
a decision, a number beyond roundoff or a workspace file fails here."""

import json
import os
import subprocess

import pytest

from alphasearch import campaign
from alphasearch.config import Config
from alphasearch.infra.fake import FakeInfra
from alphasearch.policies import agent as agent_policy

GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "campaigns.json")


def _rows(p):
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def _snap(out):
    its = [
        {k: v for k, v in r.items() if k not in ("secs", "llm_usd", "F_after_legs", "F_after_cells")}  # records only
        for r in _rows(os.path.join(out, "iters.jsonl"))
    ]
    ws = os.path.join(out, sorted(d for d in os.listdir(out) if d.startswith("it_"))[-1])
    return {
        "iters": its,
        "ledger": _rows(os.path.join(out, "ledger.jsonl")),
        "bank": json.load(open(os.path.join(ws, "bank.json"))),
        "state": json.load(open(os.path.join(ws, "state.json"))),
        "files": sorted(f for f in os.listdir(ws) if not f.startswith(".")),
    }


def _scripted(obs, cfg, log):
    run = lambda *a: json.loads(
        subprocess.run(["./tool", *a], cwd=obs.ws, capture_output=True, text=True, check=True).stdout
    )
    sug = run("suggest", "--n", "3")
    return {
        "experiments": [
            {"id": "e1", "action": "train", "hypothesis": "h", "recipes": [s["recipe"] for s in sug][:2]},
            {"id": "e2", "action": "suggest", "hypothesis": "h", "where": {"arch": ["tree"]}, "n": 2},
            {
                "id": "e3",
                "action": "feature_set",
                "hypothesis": "h",
                "name": "fx",
                "base": "agg_a",
                "features": ["x5", "x6", "x7"],
                "twins": obs.F[:1],
                "k": 2,
            },
            {"id": "e4", "action": "ensemble", "hypothesis": "h", "members": obs.F},
        ],
        "lessons": ["l"],
        "tools": [],
        "stop": False,
    }


def _norm(x):
    return json.loads(json.dumps(x, default=float, sort_keys=True))


def _assert_same(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            _assert_same(a, e)
    elif isinstance(expected, float):
        # Different BLAS builds change the final bits, not the research decision.
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    else:
        assert actual == expected


def test_golden_comparison_tolerates_only_numerical_roundoff():
    expected = {"gain": 0.1277712040601715, "members": ["tree"], "round": 1}
    _assert_same({**expected, "gain": 0.12777120406017173}, expected)
    for changed in ({"gain": 0.1277713}, {"members": ["gru"]}, {"round": 2}):
        with pytest.raises(AssertionError):
            _assert_same({**expected, **changed}, expected)
    with pytest.raises(AssertionError):
        _assert_same({**expected, "extra": 0}, expected)


def test_the_harness_reproduces_its_golden_campaigns(tmp_path, monkeypatch):
    golden = json.load(open(GOLDEN))
    campaign.run(FakeInfra(seed=2), Config(policy="surrogate", seed=3, rounds=3, models=6), f"{tmp_path}/s")
    campaign.run(FakeInfra(seed=2), Config(policy="random", seed=4, rounds=3, models=6), f"{tmp_path}/r")
    chain = dict(
        policy="surrogate",
        select_objective="ic_drift",
        swap_t=0.0,
        frontier=True,
        sur_target="best_swap",
        experience=f"{tmp_path}/store",
        seed=1,
        rounds=2,
        models=6,
    )
    campaign.run(FakeInfra(seed=5), Config(**chain), f"{tmp_path}/f1")
    campaign.run(FakeInfra(seed=5), Config(**chain), f"{tmp_path}/f2")
    monkeypatch.setattr(agent_policy, "session", _scripted)
    campaign.run(FakeInfra(seed=3), Config(policy="agent", seed=2, rounds=2, models=6), f"{tmp_path}/a")
    got = {
        "surrogate": _snap(f"{tmp_path}/s"),
        "random": _snap(f"{tmp_path}/r"),
        "frontier": {**_snap(f"{tmp_path}/f2"), "experience_md": open(f"{tmp_path}/store/experience.md").read()},
        "agent": _snap(f"{tmp_path}/a"),
    }
    for name in golden:
        for key in golden[name]:
            _assert_same(_norm(got[name][key]), golden[name][key])
