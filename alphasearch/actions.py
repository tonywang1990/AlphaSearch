"""A plan's experiments turned into recipes to train. One handler per action, registered in
ACTIONS: it checks the experiment, returns at most `left` recipe names and the facts its ledger
row records (what was refused and why). A new action is a handler here plus its line in the brief.

    train        named recipes that are untrained and legal
    suggest      the surrogate's Thompson draws inside `where` (a trailing `*` is a prefix)
    ensemble     nothing to train: judged after selection (round.evaluate, at ENSEMBLE_T)
    feature_set  a composed set from store columns, screened against same-size random sets first,
                 admitted by infra's rule, then its twins of F members and surrogate fill-ins"""
import numpy as np

from alphasearch.infra.contract import name_of, split_name
from alphasearch.search.surrogate import matches


def train(obs, exp, left, rng):
    """Named recipes from available.txt; `<recipe>@key=value,...` sets approved trainer options (options.json)
    on any legal recipe, trained or not: the variant is another model."""
    picks, refused = [], {}
    for r in dict.fromkeys(exp.get("recipes") or []):
        base, opts = split_name(r)
        recipe = obs.untrained.get(base) or next(({a: v for a, v in rec.items() if a in obs.desc.axes}
                                                   for n, rec in obs.coords.items() if split_name(n)[0] == base), None)
        allowed = obs.desc.options.get(recipe["arch"], {}) if recipe else {}
        bad = sorted(k for k, v in opts.items() if v not in allowed.get(k, ()))
        why = ("already trained" if r in obs.coords else
               "not in available.txt" if recipe is None or (not opts and base not in obs.untrained) else
               f"options not approved for {recipe['arch']}: {bad}" if bad else None)
        if why:
            refused[r] = why
            continue
        picks.append((r, {**recipe, **opts}))
    obs.untrained.update(picks[:left])                  # a variant is trainable once picked
    return [r for r, _ in picks[:left]], {"refused_recipes": refused}


def suggest(obs, exp, left, rng):
    pool = [u for u in obs.untrained if matches(obs.untrained[u], exp.get("where"))]
    if exp.get("new_only"):                             # never trained by any earlier episode
        pool = [u for u in pool if u not in obs.seen]
    k = min(int(exp.get("n") or 8), left, len(pool))
    picks = [pool[j] for j in obs.sur.thompson([obs.untrained[u] for u in pool], k, rng=rng)] if k else []
    return picks, {"pool": len(pool)}


def ensemble(obs, exp, left, rng):
    return [], {}


def feature_set(obs, exp, left, rng):
    if obs.desc.feature_set_rule is None:
        return [], {"refused": "this campaign admits no new feature sets (a fixed universe)"}
    base = obs.feature_sets.get(exp.get("base")) or []
    feats = [c for c in dict.fromkeys(exp.get("features") or []) if c not in set(base)]
    if not feats:
        return [], {"refused": "no new columns"}
    sel, row = obs.sel, {}
    row["screen"] = obs.infra.screen(feats, sel.legs(sel.ix(obs.F)))
    beats = [v["beats_null"] for c, v in row["screen"].items() if "/" not in c]
    if not beats or np.mean(beats) <= 0:
        return [], {**row, "refused": f"screen beats_null {np.mean(beats or [0]):+.2e}"}
    got = obs.infra.propose_feature_set(exp.get("name", ""), feats, base=exp.get("base"))
    if not got.get("admitted"):
        return [], {**row, "refused": got.get("why")}
    value = got["value"]
    obs.feature_sets[value] = got["columns"]
    obs.refresh_recipes()
    cands = [n for n, r in obs.untrained.items() if r.get("feature_set") == value]
    k = min(int(exp.get("k") or 8), left)
    mine = [t for t in dict.fromkeys(name_of({**obs.coords[m], "feature_set": value}, obs.desc.axes)
                                     for m in exp.get("twins") or [] if m in obs.coords) if t in cands][:k]
    rest = [n for n in cands if n not in mine]
    if k > len(mine) and rest:
        mine += [rest[j] for j in obs.sur.thompson([obs.untrained[n] for n in rest], k - len(mine), rng=rng)]
    return mine, {**row, "feature_set": value}


ACTIONS = {"train": train, "suggest": suggest, "ensemble": ensemble, "feature_set": feature_set}


def realize(obs, exp, left, rng):
    """-> (recipe names to train, the ledger row's facts)."""
    handler = ACTIONS.get(exp.get("action"))
    if handler is None:
        return [], {"refused": f"unknown action {exp.get('action')!r}"}
    return handler(obs, exp, left, rng)


def fill(obs, picks, left, rng):
    """Unspent budget: the surrogate's Thompson draws over what no experiment picked."""
    fill = [n for n in obs.untrained if n not in {r for q in picks.values() for r in q}]
    return [fill[j] for j in obs.sur.thompson([obs.untrained[n] for n in fill], min(left, len(fill)), rng=rng)]
