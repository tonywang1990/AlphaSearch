"""The tools and their guard rails: the registry matches the brief; the sandbox holds only the
workspace; an agent's tool joins the library only through its own passing test under work/
(dota/docs/agent-search.md §14)."""
import os
import re
import shutil
import subprocess

import numpy as np
import pytest

from alphasearch.context.workspace import PROMPT
from alphasearch.ensemble.moments import ic
from alphasearch.runtime.sandbox import accept_tools, sandbox
from alphasearch.tools import TOOLS

needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap is not installed")


def test_every_tool_is_in_the_brief_and_every_brief_tool_exists():
    brief = set(re.findall(r"\./tool (\w+)", open(PROMPT).read()))
    assert brief == set(TOOLS), (brief - set(TOOLS), set(TOOLS) - brief)
    assert all(t.usage.split()[0] == name and t.doc for name, t in TOOLS.items())


@needs_bwrap
def test_the_sandbox_sees_the_workspace_only(tmp_path):
    ws = tmp_path / "ws"
    (ws / "work").mkdir(parents=True)
    (ws / "bank.json").write_text("{}")
    (ws / "work" / "probe.py").write_text(
        "import os, socket\n"
        "print('ws', sorted(os.listdir('/ws')))\n"
        "print('home', os.listdir('/home') if os.path.exists('/home') else [])\n"
        "for p in ('/ws/bank.json', '/ws/x', '/tmp/../ws/y'):\n"
        "    try:\n        open(p, 'a').write('x'); print('wrote', p)\n"
        "    except OSError: print('refused', p)\n"
        "open('/ws/work/ok', 'w').write('x'); print('wrote work')\n"
        "try:\n    socket.create_connection(('1.1.1.1', 443), timeout=3); print('network')\n"
        "except OSError: print('no network')\n")
    code, out = sandbox(str(ws), "work/probe.py", [])
    assert code == 0, out
    assert "refused /ws/bank.json" in out and "refused /ws/x" in out and "wrote work" in out
    assert "no network" in out and "home []" in out


@needs_bwrap
def test_a_tool_joins_the_library_only_through_its_passing_test(tmp_path):
    ws, lib = tmp_path / "ws", tmp_path / "lib"
    (ws / "work").mkdir(parents=True)
    (ws / "work" / "good.py").write_text("X = 1\n")
    (ws / "work" / "test_good.py").write_text("print('ok')\n")
    (ws / "work" / "bad.py").write_text("X = 1\n")
    (ws / "work" / "test_bad.py").write_text("raise SystemExit(1)\n")
    (ws / "outside.py").write_text("X = 1\n")
    got = accept_tools(str(ws), str(lib), [
        {"name": "good", "path": "work/good.py", "test": "work/test_good.py"},
        {"name": "bad", "path": "work/bad.py", "test": "work/test_bad.py"},
        {"name": "outside", "path": "outside.py", "test": "work/test_good.py"}], lambda *a, **k: None)
    assert [t["accepted"] for t in got] == [True, False, False]
    assert sorted(os.listdir(lib)) == ["good.py", "test_good.py"]


def test_a_non_psd_set_scores_nan():
    """The overlap-corrected S need not be positive definite; there the moment IC is unbounded (LU
    multistart scored val ~1e8), so such a set scores NaN."""
    S = np.array([[1.0, 0.99, -0.99], [0.99, 1.0, 0.99], [-0.99, 0.99, 1.0]])
    assert np.linalg.eigvalsh(S)[0] < 0
    assert np.isnan(ic(np.array([0.1, 0.1, 0.1]), S, [0, 1, 2], np.array([1.0, -1.0, 1.0])))


def test_no_test_field_reaches_the_experience_store(tmp_path):
    from alphasearch import campaign
    from alphasearch.config import Config
    from alphasearch.infra.fake import FakeInfra
    store = str(tmp_path / "store")
    cfg = Config(policy="surrogate", rounds=2, models=4, experience=store)
    campaign.run(FakeInfra(seed=8), cfg, str(tmp_path / "c"))
    text = subprocess.run(["cat", *[os.path.join(dp, f) for dp, _, fs in os.walk(store) for f in fs]],
                          capture_output=True, text=True).stdout
    assert '"test' not in text and "test_verdict" not in text


def test_evf_scores_a_change_against_f_and_prod_by_val_day(tmp_path):
    """`./tool evf`: the changed F, F and prod scored in one score_many job on fill-aware EV, paired
    by val day; a bad member list is refused before any job."""
    from alphasearch.config import Config
    from alphasearch.infra.fake import FakeInfra
    from alphasearch.round import observe
    from alphasearch.tools import answer
    infra = FakeInfra(seed=2)
    infra.train([infra.recipes()[i] for i in range(4)])
    desc, calls = infra.describe(), []
    real = infra.score_many
    infra.score_many = lambda sets, objective, cell=None: calls.append(objective) or real(sets, objective, cell)
    obs = observe(infra, desc, Config(policy="surrogate", select_objective="wf"), 0, str(tmp_path), desc.reference,
                  dict(failed=set(), feature_sets={}, seen=set(), recent=[]))
    extra = next(n for n in obs.sel.val.idx if n not in obs.F)
    out = answer(obs, "evf", ["--add", extra])
    assert calls == ["evf"] and out["members"] == obs.F + [extra] and out["days"] > 2
    assert np.isfinite(out["t_vs_F"]) and np.isfinite(out["gain_vs_prod_pct"])
    assert "error" in answer(obs, "evf", ["--add", "not.a.member"]) and calls == ["evf"]
    cell = answer(obs, "evf", ["--add", extra, "--cell", "calm/day"])
    assert cell["cell"] == "calm/day" and 0 < cell["days"] <= out["days"] and cell["evf"] != out["evf"]
