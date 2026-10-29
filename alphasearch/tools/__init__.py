"""The agent's tools: fixed algorithms it calls while it thinks, never one that decides for it.

Every tool is one function registered with `@tool(usage, doc)`; it takes the round's Observation
and its argv and returns JSON. The socket server (`tools/server.py`) dispatches by name, so adding a
tool is one function here or in a sibling module; `tests/test_tools.py` checks every registered
tool is documented in the brief and every `./tool X` in the brief is registered. All tools answer
on the val days only."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    name: str
    usage: str        # the argv shape, as the brief shows it
    doc: str          # one line: what it answers
    fn: object


TOOLS: dict[str, Tool] = {}


def tool(usage, doc):
    def register(fn):
        name = usage.split()[0]
        TOOLS[name] = Tool(name, usage, doc, fn)
        return fn
    return register


def opt(args, k):
    """The values after flag `k`, up to the next flag."""
    if k not in args:
        return []
    rest = args[args.index(k) + 1:]
    return rest[:next((i for i, a in enumerate(rest) if a.startswith("--")), len(rest))]


def answer(obs, cmd, args):
    """One tool call against the round's Observation."""
    if cmd not in TOOLS or TOOLS[cmd].fn is None:
        return {"error": f"unknown tool {cmd!r}"}
    return TOOLS[cmd].fn(obs, args)


TOOLS["run"] = Tool("run", "run <script.py> [args]", "your own Python in the sandbox (client-side: tools/server.py)",
                    None)

from alphasearch.tools import bank, fill, store  # noqa: E402,F401 - registers the tools
