"""The agent: one fresh LLM session per round in a read-only workspace (but `work/`), with the
tools behind `./tool` and the brief as its system prompt. State lives in files, so a long
campaign never depends on one long context, and a resume starts the next round cleanly."""
import glob
import os
import shutil
import time

from alphasearch.records import Plan
from alphasearch.runtime.llm import LLM, AgentError, json_block, transcript_path
from alphasearch.runtime.sandbox import accept_tools
from alphasearch.tools.server import serve

TOOLS_AGENT = "Read,Grep,Glob,Write,Edit,Bash(./tool:*)"   # the only command it may run is ./tool


def session(obs, cfg, log):
    """The session's reply as an object; one repair call if it is not one JSON object."""
    ws = obs.ws
    for root, _, _ in os.walk(ws):                       # read-only but work/
        if not os.path.abspath(root).startswith(os.path.abspath(os.path.join(ws, "work"))):
            os.chmod(root, 0o555)
    try:
        for attempt in range(3):                         # an API error (529) killed a campaign once
            try:
                reply = LLM(model=cfg.model, effort=cfg.effort, tools=TOOLS_AGENT, timeout=3600).call(
                    os.path.join(ws, "agent.md"), "Your workspace is the current directory. "
                    "Follow agent.md and reply with the JSON object only.", ws, log)
                break
            except AgentError:
                if attempt == 2:
                    raise
                time.sleep(120 * (attempt + 1))
    finally:                                             # writable again, or the socket cannot unlink
        for root, _, _ in os.walk(ws):
            os.chmod(root, 0o755)
    plan = json_block(reply or "") or {}
    if reply and "experiments" not in plan:              # code inside the JSON once parsed to an inner object
        fix = os.path.join(ws, "work", "fix_reply.md")
        open(fix, "w").write("The reply below is not ONE valid JSON object. Reply with the same plan "
                             "as one valid JSON object, nothing else.\n\n" + reply)
        plan = json_block(LLM(model=cfg.model, effort="low", tools="Read").call(
            fix, "Fix the JSON.", ws, log) or "") or {}
    _archive(ws)
    return plan


def _archive(ws):
    """The round's session transcripts, kept beside the campaign (<out>/sessions/it_N/) for the
    session viewer: the CLI's own copy under ~/.claude/projects is not ours to keep."""
    dest = os.path.join(os.path.dirname(os.path.abspath(ws)), "sessions", os.path.basename(os.path.abspath(ws)))
    os.makedirs(dest, exist_ok=True)
    for f in glob.glob(os.path.join(os.path.dirname(transcript_path(os.path.abspath(ws), "x")), "*.jsonl")):
        shutil.copy(f, dest)


class AgentPolicy:
    name = "agent"
    fills_leftover = True          # budget it leaves unspent goes to the surrogate's draws

    def __init__(self, cfg):
        self.cfg = cfg

    def plan(self, obs, models, rng, log):
        listener = serve(obs, obs.ws)
        try:
            return Plan.parse(session(obs, self.cfg, log))
        finally:
            listener.close()

    def reflect(self, obs, plan, lib, log):
        return accept_tools(obs.ws, lib, plan.tools, log)
