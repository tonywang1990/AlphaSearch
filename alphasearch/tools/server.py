"""How the agent reaches the tools: a unix-socket server in the round's workspace, and `./tool`, a
thin client the brief tells it to run (`Bash(./tool:*)` is its only command). `./tool run` executes
in the sandbox client-side; every other call goes to the server. Every call is logged beside the
workspace."""
import json
import os
import sys
import threading
import time

from alphasearch.runtime.sandbox import sandbox
from alphasearch.tools import TOOLS, answer

SOCK, CALLS = ".tool.sock", "tool_calls.jsonl"


def serve(obs, ws):
    """The tool server for one round: answers until the returned listener is closed."""
    from multiprocessing.connection import Listener
    sock = os.path.join(ws, SOCK)
    if os.path.exists(sock):                 # a killed run's: a resume must bind again
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
                        out = answer(obs, req["cmd"], req["args"])
                    except Exception as e:                # noqa: BLE001 - an answer, not a crash
                        out = {"error": f"{type(e).__name__}: {e}"}
                    conn.send(out)
                except (EOFError, OSError):               # the client gave up
                    continue
    threading.Thread(target=loop, daemon=True).start()
    return listener


def launcher(ws, lib):
    """Write the workspace's `./tool`."""
    json.dump({"lib": lib}, open(os.path.join(ws, ".tool_meta.json"), "w"))
    path, root = os.path.join(ws, "tool"), os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    open(path, "w").write(f'#!/bin/sh\ncd "$(dirname "$0")" && PYTHONNOUSERSITE=1 PYTHONPATH={root} '
                          f'exec {sys.executable} -m alphasearch.tools.server . "$@"\n')
    os.chmod(path, 0o755)


def main(argv):
    if len(argv) < 2:
        sys.exit("usage: ./tool " + " | ".join(t.usage for t in TOOLS.values()))
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


if __name__ == "__main__":
    main(sys.argv[1:])
