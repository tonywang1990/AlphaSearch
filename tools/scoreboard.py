"""Owner-run: the replayed frontier arms next to PR 452's (dota/docs/agent-search.md's result table).
Test score (`ev_drift`, 1.0 = prod) by CUMULATIVE models trained across a chain's campaigns, mean over
the chains present in both, and the fit test = a + b * log2(cumulative models).

python tools/scoreboard.py --dota /qr_scratch/tony/dota/agent --ours <frontier dir> [--chains 0 1]
"""
import argparse
import json
import os

import numpy as np

ARMS = {"agent": "LU_FDAc", "surrogate": "LU_FDSc", "random": "LU_FRc"}
BINS = ((16, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 1024))


def points(root, prefix, chains, key):
    """[(chain, cumulative models, test)] over each chain's campaigns e1, e2, ... in order."""
    out = []
    for c in chains:
        cum = 0
        for k in range(1, 9):
            f = os.path.join(root, f"{prefix}{c}e{k}", "replication.json" if key == "ours" else "iters.jsonl")
            if not os.path.exists(f):
                break
            rows = json.load(open(f)) if key == "ours" else [json.loads(x) for x in open(f)]
            for r in rows:
                cum += 16
                v = r["ours_after_test"] if key == "ours" else r["F_after_test"]
                if v is not None:
                    out.append((c, cum, v))
    return out


def summary(pts):
    cells = []
    for lo, hi in BINS:
        v = [p[2] for p in pts if lo <= p[1] < hi]
        cells.append(f"{np.mean(v):.4f}({len(v)})" if v else "-")
    slope = np.polyfit(np.log2([p[1] for p in pts]), [p[2] for p in pts], 1)[0] if len(pts) > 3 else float("nan")
    return cells, slope


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dota", required=True)
    p.add_argument("--ours", required=True)
    p.add_argument("--chains", type=int, nargs="+", default=[0, 1])
    a = p.parse_args()
    print("test (ev_drift, 1.0 = prod) by cumulative models; mean (n); ours = AlphaSearch on futs-infra 0.2.0")
    print(f"{'arm':18s}" + "".join(f"{f'[{lo},{hi})':>14s}" for lo, hi in BINS) + f"{'per doubling':>14s}")
    for arm, prefix in ARMS.items():
        for tag, root in (("dota", a.dota), ("ours", a.ours)):
            pts = points(root, prefix, a.chains, tag)
            if pts:
                cells, slope = summary(pts)
                print(f"{arm + ' ' + tag:18s}" + "".join(f"{c:>14s}" for c in cells) + f"{slope:>+14.4f}")


if __name__ == "__main__":
    main()
