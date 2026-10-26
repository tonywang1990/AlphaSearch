"""Online learning across campaigns on one symbol (DOTA ADR 0452): every finished campaign leaves
an episode (val days only, never a test verdict) in a per-symbol store; the next campaign's agent
reads it as `experience.md` plus `recipes.json`."""
import json
import os

import numpy as np

from alphasearch.search.surrogate import effects


def record(out, name, policy, F, gains, coords, ledger, iters):
    """`out/episode.json`: the finished campaign, val-only."""
    keys = ("action", "hypothesis", "where", "n", "trained", "entered_F", "val_gain", "adopted")
    ep = {"name": name, "policy": policy, "F": F,
          "trajectory": [{"models": sum(len(d["trained"]) for d in iters[:i + 1]), "F_val": d["F_after_val"]}
                         for i, d in enumerate(iters)],
          "F_start_val": iters[0]["F_val"] if iters else None,
          "gains": {n: round(float(g), 6) for n, g in gains.items() if np.isfinite(g)},
          "coords": {n: coords[n] for n in gains if n in coords},
          "experiments": [{k: r[k] for k in keys if k in r} for r in ledger if "action" in r],
          "lessons": [x for r in ledger for x in r.get("lessons") or []]}
    path = os.path.join(out, "episode.json")
    json.dump(ep, open(path, "w"), default=float)
    return path


def add(store, episode):
    os.makedirs(os.path.join(store, "episodes"), exist_ok=True)
    ep = json.load(open(episode))
    json.dump(ep, open(os.path.join(store, "episodes", ep["name"] + ".json"), "w"))
    render(store)


def load(store):
    d = os.path.join(store or "", "episodes")
    if not os.path.isdir(d):
        return []
    files = sorted(os.listdir(d), key=lambda f: os.path.getmtime(os.path.join(d, f)))
    return [json.load(open(os.path.join(d, f))) for f in files]


def recipes(eps):
    """{recipe: {"gains": [...], "in_final_F": k}} over the episodes that trained it."""
    out = {}
    for e in eps:
        for n, g in e["gains"].items():
            r = out.setdefault(n, {"gains": [], "in_final_F": 0})
            r["gains"].append(g)
            r["in_final_F"] += n in e["F"]
    return out


def render(store):
    """`experience.md` (what the agent reads first) and `recipes.json` (every recipe)."""
    eps, md = load(store), []
    rec = recipes(eps)
    json.dump({n: {"mean_val_gain": round(float(np.mean(r["gains"])), 6), "episodes": len(r["gains"]),
                   "in_final_F": r["in_final_F"]} for n, r in rec.items()},
              open(os.path.join(store, "recipes.json"), "w"), indent=0)
    md += ["# Your own earlier campaigns on this symbol (val days only; nothing here is a test result)", "",
           f"{len(eps)} episode(s). A recipe's gain is its marginal val gain to that episode's final F. "
           "A recipe that paid in several episodes is evidence; one that paid once may be noise.", "",
           "## Episodes", "", "| episode | policy | F val by models trained | final F |", "|---|---|---|---|"]
    for e in eps:
        traj = " ".join(f"{t['models']}:{t['F_val']:.4f}" for t in e["trajectory"])
        md.append(f"| {e['name']} | {e['policy']} | start {e['F_start_val'] or 0:.4f}; {traj} | {', '.join(e['F'])} |")
    rows = sorted(((float(np.mean(r["gains"])), len(r["gains"]), r["in_final_F"], n) for n, r in rec.items()),
                  reverse=True)
    md += ["", "## Recipes that paid (top 30; recipes.json has every one)", "",
           "| recipe | mean val gain | episodes | in a final F |", "|---|---|---|---|"]
    md += [f"| {n} | {g:+.5f} | {k} | {f} |" for g, k, f, n in rows[:30]]
    dead = [n for g, k, f, n in rows if k >= 2 and f == 0 and g <= 0]
    md += ["", f"## Trained in 2+ episodes, never in a final F, gain <= 0 ({len(dead)})", "", ", ".join(dead[:40]) or "none"]
    cs = {n: c for e in eps for n, c in e.get("coords", {}).items()}
    axes = sorted({a for c in cs.values() for a in c})
    eff = effects({n: float(np.mean(r["gains"])) for n, r in rec.items()}, cs, axes)
    md += ["", "## Cells (mean gain, SE, n)", ""]
    for a, t in list(eff["single"].items()) + list(eff["pairs"].items()):
        md.append(f"- **{a}**: " + "; ".join(
            f"{'/'.join(map(str, r['cell'].values()))} {r['mean']:+.4f} n{r['n']}" for r in t[:8]))
    lessons = [x for e in reversed(eps) for x in e["lessons"]][:20]
    md += ["", "## Lessons you wrote (latest episode first)", ""] + [f"- {x}" for x in lessons]
    open(os.path.join(store, "experience.md"), "w").write("\n".join(md) + "\n")
