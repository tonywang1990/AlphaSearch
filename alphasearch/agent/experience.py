"""Online learning across campaigns on one symbol (DOTA ADR 0452): every finished campaign leaves
an episode (val days only, never a test verdict) in a per-symbol store; the next campaign's agent
reads it as `experience.md` plus `recipes.json`."""
import json
import os

import numpy as np

from alphasearch.search.surrogate import effects


def record(out, name, policy, F, gains, coords, ledger, iters, objective=None, prompt=None, known_best=None):
    """`out/episode.json`: the finished campaign, val-only. `known_best`: F selected over every
    recipe this symbol's episodes trained (the frontier bank), with its val score and pool size."""
    keys = ("action", "hypothesis", "where", "n", "trained", "entered_F", "val_gain", "adopted")
    ep = {"name": name, "policy": policy, "objective": objective, "prompt": prompt, "F": F,
          "trajectory": [{"models": sum(len(d["trained"]) for d in iters[:i + 1]), "F_val": d["F_after_val"]}
                         for i, d in enumerate(iters)],
          "F_start_val": iters[0]["F_val"] if iters else None,
          "gains": {n: round(float(g), 6) for n, g in gains.items() if np.isfinite(g)},
          "coords": {n: coords[n] for n in gains if n in coords},
          "experiments": [{k: r[k] for k in keys if k in r} for r in ledger if "action" in r],
          "lessons": [x for r in ledger for x in r.get("lessons") or []]}
    if known_best is not None:
        ep["known_best"] = known_best
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


def seen(store):
    """{recipe: {mean_val_gain, episodes, in_final_F}} of every recipe the store's episodes trained."""
    p = os.path.join(store or "", "recipes.json")
    return json.load(open(p)) if store and os.path.exists(p) else {}


def recipes(eps):
    """{recipe: {"gains": [...], "in_final_F": k}} over the episodes that trained it."""
    out = {}
    for e in eps:
        for n, g in e["gains"].items():
            r = out.setdefault(n, {"gains": [], "in_final_F": 0})
            r["gains"].append(g)
            r["in_final_F"] += n in e["F"]
    return out


def _cell(r):
    se = "" if r["se"] is None else f" ±{r['se']:.4f}"
    return f"{'/'.join(str(v) for v in r['cell'].values())} {r['mean']:+.4f}{se} n{r['n']}"


def render(store):
    """`experience.md` (the summary the agent reads first) and `recipes.json` (every recipe)."""
    eps = load(store)
    rec = recipes(eps)
    json.dump({n: {"mean_val_gain": round(float(np.mean(r["gains"])), 6), "episodes": len(r["gains"]),
                   "in_final_F": r["in_final_F"]} for n, r in rec.items()},
              open(os.path.join(store, "recipes.json"), "w"), indent=0)
    md = ["# Your own earlier campaigns on this symbol (val days only; nothing here is a test number)",
          "",
          f"{len(eps)} episode(s), each from the prod members on this grammar. A recipe's gain is its "
          "marginal val gain to that episode's final F (a member: what dropping it loses; any other: "
          "what adding it gains). Gains are measured on the same val days you are scored by, so a "
          "recipe that paid in several episodes is evidence; one that paid once may be noise.", "",
          "## Episodes", "", "| episode | policy | F val by models trained | final F |", "|---|---|---|---|"]
    for e in eps:
        traj = " ".join(f"{t['models']}:{t['F_val']:.4f}" for t in e["trajectory"])
        md.append(f"| {e['name']} | {e['policy']} | start {e['F_start_val']:.4f}; {traj} | "
                  f"{', '.join(e['F'])} |")
    kb = next((e["known_best"] for e in reversed(eps) if e.get("known_best")), None)
    if kb:
        md += ["", "## The best known set: rebuild this first", "",
               f"F selected on val (from the prod members) over EVERY recipe your episodes trained "
               f"({kb['pool']} models), after the latest episode: val {kb['val']:.4f}. It beats each "
               "episode's own final F only when it draws on more good models than any one episode had:", "",
               f"`{json.dumps(kb['F'])}`"]
    elif eps:
        best = max(eps, key=lambda e: e["trajectory"][-1]["F_val"] if e["trajectory"] else 0)
        md += ["", f"Best final F on val ({best['name']}): `{json.dumps(best['F'])}`"]
    rows = sorted(((float(np.mean(r["gains"])), len(r["gains"]), r["in_final_F"], n) for n, r in rec.items()),
                  reverse=True)
    md += ["", "## Recipes that paid (top 30 by mean gain; recipes.json has every one)", "",
           "| recipe | mean val gain | episodes | in a final F |", "|---|---|---|---|"]
    md += [f"| {n} | {g:+.5f} | {k} | {f} |" for g, k, f, n in rows[:30]]
    dead = [n for g, k, f, n in rows if k >= 2 and f == 0 and g <= 0]
    md += ["", f"## Trained in 2+ episodes, never in a final F, gain <= 0 ({len(dead)})", "",
           ", ".join(dead[:40]) or "none"]
    cs = {n: c for e in eps for n, c in e.get("coords", {}).items()}
    eff = effects({n: float(np.mean(r["gains"])) for n, r in rec.items()}, cs)
    md += ["", "## Cells (all episodes; mean of each recipe's mean gain, SE, n)", ""]
    for a, t in list(eff["single"].items()) + list(eff["pairs"].items()):
        md.append(f"- **{a}**: " + "; ".join(_cell(r) for r in t[:8]))
    md += ["", "## What your experiments did (val gain each alone; entered = models that entered F)", ""]
    by = {}
    for e in eps:
        for x in e["experiments"]:
            b = by.setdefault(x.get("action"), {"n": 0, "val": [], "trained": 0, "entered": 0})
            b["n"] += 1
            b["val"].append(x.get("val_gain") or 0)
            b["trained"] += len(x.get("trained") or [])
            b["entered"] += len(x.get("entered_F") or [])
    md += [f"- {a}: {b['n']} experiments, mean val gain {np.mean(b['val']):+.5f}, "
           f"{b['entered']}/{b['trained']} models entered F" for a, b in by.items()]
    lessons = [x for e in reversed(eps) for x in e["lessons"]][:20]
    md += ["", "## Lessons you wrote (latest episode first)", ""] + [f"- {x}" for x in lessons]
    open(os.path.join(store, "experience.md"), "w").write("\n".join(md) + "\n")
