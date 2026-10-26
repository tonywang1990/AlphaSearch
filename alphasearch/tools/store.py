"""Tools over the raw store columns: infra computes them on the val rows (the rows never leave it)."""
import json
import os
import re

from alphasearch.tools import opt, tool

MAX_COLS = 300


def _check(obs, want):
    bad = [c for c in want if c not in set(obs.columns)]
    if bad or not 1 <= len(want) <= MAX_COLS:
        return {"error": f"{len(want)} columns (1-{MAX_COLS}); not in the store: {bad[:3]}"}


@tool("pcorr <regex> [--cell period/session]", "store columns by partial correlation to the label after F")
def pcorr(obs, args):
    rx = re.compile(args[0]) if args else re.compile(".")
    want = [c for c in obs.columns if rx.search(c)]
    return _check(obs, want) or obs.infra.pcorr(want, obs.sel.legs(obs.sel.ix(obs.F)),
                                                cell=(opt(args, "--cell") or [None])[0])


@tool("screen <file.json | column ...>", "the R² a column set adds over F out of day, vs same-size random sets")
def screen(obs, args):
    from_file = len(args) == 1 and args[0].endswith(".json")
    want = json.load(open(os.path.join(obs.ws, args[0]))) if from_file else args
    want = list(dict.fromkeys(want["features"] if isinstance(want, dict) else want))
    return _check(obs, want) or obs.infra.screen(want, obs.sel.legs(obs.sel.ix(obs.F)))
