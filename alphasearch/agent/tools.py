"""The agent's tools: fixed algorithms it calls while it thinks, never one that decides for it.
All of them answer on the val days (the agent's half when split) and nothing else.

The loop serves them over a unix socket in the workspace (`serve`); `./tool` is a thin client
(`main`). `./tool run` executes the agent's own Python in a bwrap sandbox that holds only the
workspace (bank.json, moments.npz: summaries, no rows), its `work/` and the tool library."""
import json
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np

from alphasearch.search.surrogate import axis_summary, effects

SOCK, CALLS, MAX_COLS = ".tool.sock", "tool_calls.jsonl", 300


def _opt(args, k):
    """The values after flag `k`, up to the next flag."""
    if k not in args:
        return []
    rest = args[args.index(k) + 1:]
    return rest[:next((i for i, a in enumerate(rest) if a.startswith("--")), len(rest))]


def answer(R, cmd, args):
    """One tool call against the round `R` (`loop.Round`)."""
    sel, F = R.tool_sel, R.F
    if cmd == "whatif":
        s = [n for n in F if n not in _opt(args, "--drop")] + _opt(args, "--add")
        bad = [n for n in s if n not in sel.val.idx]
        if bad or not 0 < len(s) <= sel.cfg.k_max:
            return {"error": f"{len(s)} members (1-{sel.cfg.k_max}); not trained: {bad[:3]}"}
        f = sel.scorer()
        g, se = sel.gain(sel.ix(s), sel.ix(F))
        return {"members": s, "val": f(sel.ix(s)), "F_val": f(sel.ix(F)), "val_gain": g,
                "val_gain_se": se, "t": g / se if se else None}
    if cmd == "members":
        rx = re.compile(args[0]) if args else None
        rows = sorted(((g, n) for n, g in R.gains.items() if not rx or rx.search(n)), reverse=True)
        return [{"member": n, "marginal_val_gain": round(float(g), 5), "in_F": n in F} for g, n in rows[:60]]
    if cmd == "effects":
        return effects(R.gains, R.coords)
    if cmd == "axes":
        done = [n for n in R.gains if n in R.coords]
        return axis_summary([R.coords[n] for n in done], [R.gains[n] for n in done])
    if cmd == "suggest":
        n = int(_opt(args, "--n")[0]) if _opt(args, "--n") else 20
        want = _opt(args, "--set")
        pool = [u for u in R.untrained if not want or R.untrained[u].get("feature_set") in want]
        if not pool:
            return []
        m = R.sur.samples([R.untrained[u] for u in pool])
        return [{"recipe": pool[i], "expected_gain": float(m.mean(0)[i]), "sd": float(m.std(0)[i])}
                for i in np.argsort(-m.mean(0))[:n]]
    if cmd == "predcheck":
        return {"target": R.cfg.get("sur_target", "marginal"), **R.sur.predcheck(R.coords)}
    if cmd == "crossval":
        return sel.crossval(R.ref, recent=R.recent)
    if cmd == "search":
        n = int(_opt(args, "--starts")[0]) if _opt(args, "--starts") else 8
        return sel.search(F, starts=n, seed=R.it)
    if cmd in ("screen", "pcorr"):
        cols = R.columns
        if cmd == "screen":
            from_file = len(args) == 1 and args[0].endswith(".json")
            want = json.load(open(os.path.join(R.ws, args[0]))) if from_file else args
            want = list(dict.fromkeys(want["features"] if isinstance(want, dict) else want))
        else:
            rx = re.compile(args[0]) if args else re.compile(".")
            want = [c for c in cols if rx.search(c)]
        bad = [c for c in want if c not in set(cols)]
        if bad or not 1 <= len(want) <= MAX_COLS:
            return {"error": f"{len(want)} columns (1-{MAX_COLS}); not in the store: {bad[:3]}"}
        legs = sel.legs(sel.ix(F))
        if cmd == "screen":
            return R.infra.screen(want, legs)
        return R.infra.pcorr(want, legs, cell=(_opt(args, "--cell") or [None])[0])
    return {"error": f"unknown tool {cmd!r}"}


def serve(R, ws):
    """The tool server for one round: answers until the returned listener is closed."""
    from multiprocessing.connection import Listener
    sock = os.path.join(ws, SOCK)
    if os.path.exists(sock):
        os.remove(sock)
    listener = Listener(sock, family="AF_UNIX")

    def loop():
        while True:
            try:
                conn = listener.accept()
            except OSError:
                return
            with conn:
                try:
                    req = conn.recv()
                    try:
                        out = answer(R, req["cmd"], req["args"])
                    except Exception as e:                # noqa: BLE001 - an answer, not a crash
                        out = {"error": f"{type(e).__name__}: {e}"}
                    conn.send(out)
                except (EOFError, OSError):
                    continue
    threading.Thread(target=loop, daemon=True).start()
    return listener


def sandbox(ws, script, args, lib=None, timeout=900):
    """(exit code, output) of the agent's `script` under bwrap: /ws the workspace read-only but
    `work/`, /lib_tools the campaign's tool library; no network, no repo, no data beyond the
    workspace's summaries."""
    env = sys.prefix
    base = sys.base_prefix                              # a venv's bin/python links into its base env
    cmd = ["bwrap", "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
           "--ro-bind", "/lib64", "/lib64", "--ro-bind", base, base, "--ro-bind", env, env, "--proc", "/proc",
           "--dev", "/dev", "--tmpfs", "/tmp", "--unshare-all", "--die-with-parent",
           "--ro-bind", os.path.abspath(ws), "/ws",
           "--bind", os.path.abspath(os.path.join(ws, "work")), "/ws/work", "--chdir", "/ws"]
    if lib and os.path.isdir(lib):
        cmd += ["--ro-bind", lib, "/lib_tools"]
    cmd += ["--setenv", "OMP_NUM_THREADS", "4", os.path.join(env, "bin", "python"), script, *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"
    return r.returncode, (r.stdout + r.stderr)[-20000:]


def launcher(ws, lib):
    """Write the workspace's `./tool`."""
    json.dump({"lib": lib}, open(os.path.join(ws, ".tool_meta.json"), "w"))
    path, root = os.path.join(ws, "tool"), os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    open(path, "w").write(f'#!/bin/sh\ncd "$(dirname "$0")" && PYTHONNOUSERSITE=1 PYTHONPATH={root} '
                          f'exec {sys.executable} -m alphasearch.agent.tools . "$@"\n')
    os.chmod(path, 0o755)


def main(argv):
    ws, cmd, args = argv[0], argv[1], argv[2:]
    with open(os.path.join(ws, "..", CALLS), "a") as fh:
        fh.write(json.dumps({"t": time.time(), "ws": os.path.basename(os.path.abspath(ws)),
                             "cmd": cmd, "args": args}) + "\n")
    if cmd == "run":
        if not args:
            sys.exit("usage: ./tool run <script.py> [args]  (scripts live under work/ or /lib_tools)")
        code, out = sandbox(ws, args[0], args[1:], json.load(open(os.path.join(ws, ".tool_meta.json")))["lib"])
        print(out)
        sys.exit(code)
    from multiprocessing.connection import Client
    with Client(os.path.join(ws, SOCK), family="AF_UNIX") as c:
        c.send({"cmd": cmd, "args": args})
        print(json.dumps(c.recv(), indent=1, default=float))


def accept_tools(ws, lib, tools, log):
    """Tools the agent wrote: each joins `lib` only if its own test passes in the sandbox."""
    import shutil
    out = []
    for t in tools or []:
        path, test = t.get("path", ""), t.get("test", "")
        if not (path.startswith("work/") and test.startswith("work/")
                and os.path.exists(os.path.join(ws, path)) and os.path.exists(os.path.join(ws, test))):
            out.append({**t, "accepted": False, "why": "path and test must be files under work/"})
            continue
        code, txt = sandbox(ws, test, [], lib)
        if code == 0:
            os.makedirs(lib, exist_ok=True)
            for p in (path, test):
                shutil.copy(os.path.join(ws, p), os.path.join(lib, os.path.basename(p)))
        out.append({**t, "accepted": code == 0, "test_output": txt[-1500:]})
        log("tool", f"{t.get('name')}: {'accepted' if code == 0 else 'rejected'}")
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
