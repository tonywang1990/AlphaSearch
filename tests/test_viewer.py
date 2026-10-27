"""The session viewer builds every campaign's tree: rounds in order, the seven phases in each, one act
child per experiment, and a page whose embedded data parses back."""
import json
import re

from alphasearch import campaign
from alphasearch.config import Config
from alphasearch.infra.fake import FakeInfra
from alphasearch.viewer import write
from alphasearch.viewer.trajectory import PHASES, build


def test_the_viewer_shows_every_round_and_phase(tmp_path):
    out = str(tmp_path / "c")
    campaign.run(FakeInfra(seed=2), Config(policy="surrogate", seed=3, rounds=3, models=6), out)
    c = build([out])["campaigns"][0]
    rounds = [n for n in c["c"] if n["k"] == "round"]
    assert [r["t"] for r in rounds] == ["Round 0", "Round 1", "Round 2"]
    assert all(tuple(p["t"] for p in r["c"]) == PHASES for r in rounds)
    assert all(p["s"].endswith("s") for p in rounds[0]["c"])                 # each phase's wall time
    ledger = [json.loads(x) for x in open(f"{out}/ledger.jsonl")]
    assert len(rounds[0]["c"][3]["c"]) == sum(1 for r in ledger if r["iteration"] == 0 and "action" in r)
    page = open(write([out], str(tmp_path / "v.html"))).read()
    data = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1)
    assert json.loads(data.replace("<\\/", "</"))["campaigns"][0]["t"] == "c"
