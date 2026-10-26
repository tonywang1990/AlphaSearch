"""The cross-symbol memory (offline): what won on OTHER symbols, refreshed every campaign.

DOTA's `memory.md` was written by hand once; models are refit and re-picked weekly, so here it is
rendered from what infra says production serves at registration (`describe().served`), plus,
when given, the best final F of AlphaSearch's own campaigns on the other symbols. Nothing about
the target symbol is in it."""
import collections
import json
import os
import re

import numpy as np


def _same_symbol(slot, symbol):
    return re.sub(r"\d+$", "", slot).upper() == symbol.upper()      # AG1 is AG's second slot


def _leg(leg):
    r = leg.get("recipe")
    shape = (f"{r['arch']} {r['horizon']} {r['target']} on {r['feature_set']} ({r['feature_selection']}, "
             f"{r['session']})") if r else leg["member"]
    return f"{shape}, weight {leg['weight']:+.2f}"


def render(served, symbol, stores=None):
    """`memory.md` from the served ensembles (excluding `symbol`) and `stores` ({symbol: experience dir})."""
    others = [s for s in served if not _same_symbol(s["slot"], symbol)]
    release = max((s["release"] for s in others), default="?")
    md = [f"# What production serves on other symbols (as of {release}; nothing here comes from {symbol})", "",
          "Refreshed for every campaign from the live deployment: models are refit weekly and a symbol's "
          "served ensemble can change between campaigns. Weights are the served recipe weights.", "",
          "## Served ensembles", ""]
    legs = []
    for s in sorted(others, key=lambda s: s["slot"]):
        change = (f"; since {s['since']}, before it {s['prev_model']}"
                  if s.get("prev_model") and s["prev_model"] != s["model"] else f"; since {s['since']}")
        md.append(f"- **{s['slot'].upper()}**: `{s['model']}`{change}")
        for leg in sorted(s["legs"], key=lambda x: -abs(x["weight"])):
            md.append(f"  - {_leg(leg)}")
            legs.append((s["slot"], leg))
        if not s["legs"]:
            md.append(f"  - a `{s['model']}` slot: the builder is the shape (no searched legs)")
    shaped = [(slot, leg) for slot, leg in legs if leg.get("recipe")]
    if shaped:
        cell = collections.Counter(f"{leg['recipe']['arch']} {leg['recipe']['horizon']}" for _, leg in shaped)
        anchors = collections.Counter(
            f"{a['recipe']['arch']} {a['recipe']['horizon']}" for a in
            (max((leg for s, leg in shaped if s == slot), key=lambda x: abs(x["weight"]))
             for slot in {s for s, _ in shaped}))
        sizes = [len(s["legs"]) for s in others if s["legs"]]
        neg = sum(leg["weight"] < 0 for _, leg in shaped)
        md += ["", "## Across symbols (counted from the legs above)", "",
               f"- Ensembles serve {min(sizes)} to {max(sizes)} legs (median {int(np.median(sizes))}).",
               "- The heaviest leg, per symbol: " + ", ".join(f"{k} ({n})" for k, n in anchors.most_common()),
               "- Legs by arch and horizon: " + ", ".join(f"{k} ({n})" for k, n in cell.most_common(10)),
               f"- {neg} of {len(shaped)} legs carry a negative weight (hedges)."]
    for sym, store in sorted((stores or {}).items()):
        if _same_symbol(sym, symbol) or not os.path.isdir(os.path.join(store, "episodes")):
            continue
        d = os.path.join(store, "episodes")
        eps = [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d))]
        best = max(eps, key=lambda e: e["trajectory"][-1]["F_val"] if e["trajectory"] else -9, default=None)
        if best:
            md += ["", f"## AlphaSearch on {sym.upper()} ({len(eps)} campaigns)", "",
                   f"Best final F on val ({best['name']}, {best['trajectory'][-1]['F_val']:.4f}): "
                   + ", ".join(f"`{n}`" for n in best["F"])]
    return "\n".join(md) + "\n"
