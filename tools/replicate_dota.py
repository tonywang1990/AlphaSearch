"""Owner-run acceptance: replay a recorded DOTA (futs PR 452) benchmark run through futs-infra v1.

Registers one campaign per run over the benchmark snapshot, runs AlphaSearch with the run's policy
and seed, then compares round by round: F before and after, the recipes trained, and the
test-day score of each F (the verdict jobs' owner measurements; the client never sees them).

FUTS_CAMPAIGNS=... PYTHONPATH=<alphasearch> <venv>/bin/python tools/replicate_dota.py \\
    --ref /qr_scratch/tony/dota/agent/LU_EIs0 --policy surrogate --seed 0 --snapshot ... --out runs/
    [--register <register_universe_benchmark.py>] [--reference NAME:W ...]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from alphasearch import campaign
from alphasearch.config import Config
from alphasearch.infra.futs import FutsInfra


def _rows(path):
    return [json.loads(x) for x in open(path)] if os.path.exists(path) else []


def owner_scores(campaign):
    """{(frozenset child, frozenset parent): (child score, parent score)} from the verdict jobs."""
    from lib.api.registration import identity
    root = Path(os.environ["FUTS_CAMPAIGNS"]) / identity(campaign)
    out, one = {}, {}
    for job in (root / "jobs").glob("*/job.json"):
        rec = json.loads(job.read_text())
        meas = job.parent / "owner-measurement.json"
        if rec.get("operation") == "verdict" and meas.exists():
            m = json.loads(meas.read_text())
            pay = rec["payload"]
            child, parent = (frozenset(n for n, _ in pay[k]) for k in ("subject", "compare_to"))
            out[(child, parent)] = (m["child"], m["parent"], m["verdict"])
            one[child], one[parent] = m["child"], m["parent"]   # a set's own score, whatever it was compared to
    return out, one


def compare(ref_dir, out, campaign, seed_suffix=".s1"):
    """Round-by-round rows: same F, same F', same trained set, and both test scores."""
    ref, ours = _rows(Path(ref_dir) / "iters.jsonl"), _rows(Path(out) / "iters.jsonl")
    scores, one = owner_scores(campaign)
    infra = lambda names: frozenset(n + seed_suffix for n in names)
    rows = []
    for a, b in zip(ref, ours, strict=False):
        key = (infra(b["F_after"]), infra(b["F"]))
        child, parent, word = scores.get(key, (one.get(key[0]), one.get(key[1]), b["test_verdict"]))
        rows.append(dict(
            iteration=a["iteration"], same_F={n + seed_suffix for n in b["F"]} == set(a["F"]),
            same_F_after={n + seed_suffix for n in b["F_after"]} == set(a["F_after"]),
            same_trained={n + seed_suffix for n in b["trained"]} == set(a["trained"]),
            dota_test=a["F_test"], ours_test=parent, dota_after_test=a["F_after_test"], ours_after_test=child,
            verdict=word))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="the recorded DOTA run directory")
    p.add_argument("--policy", choices=["surrogate", "random", "agent"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--out", required=True, help="where the replay's campaign directories go")
    p.add_argument("--register", default=None, help="the owner's register_universe_benchmark.py")
    p.add_argument("--reference", action="append", default=[])
    p.add_argument("--ingredients", default=None, help="the grammar the universe was cut from (notes, set definitions)")
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--models", type=int, default=16)
    p.add_argument("--extra", default="{}", help="JSON loop config (agent arms: prompt, memory, experience)")
    p.add_argument("--name", default=None, help="the infra campaign (default replay/<ref>); a fresh run needs its own")
    a = p.parse_args()
    name = a.name or f"replay/{Path(a.ref).name}"
    from lib.api.registration import identity
    if not (Path(os.environ["FUTS_CAMPAIGNS"]) / identity(name)).exists():
        subprocess.run([sys.executable, "-I", a.register, "register", "--snapshot", a.snapshot, "--name", name,
                        *[x for r in a.reference for x in ("--reference", r)],
                        *(["--ingredients", a.ingredients] if a.ingredients else [])], check=True)
    out = os.path.join(a.out, Path(a.ref).name)
    cfg = Config(**{"policy": a.policy, "seed": a.seed, "rounds": a.iters, "models": a.models,
                    "select_objective": "ic", "swap_t": 2.0, **json.loads(a.extra)})
    campaign.run(FutsInfra(name), cfg, out)
    rows = compare(a.ref, out, name)
    json.dump(rows, open(os.path.join(out, "replication.json"), "w"), indent=1)
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
