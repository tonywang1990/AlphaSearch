"""One LLM call: `claude -p` in an isolated workspace, with the retries a long unattended
run needs (ported from DOTA, futs PR 452).

    LLM(model, effort, tools).call(spec_path, prompt, cwd, log) -> reply text

A safeguard stop is retried once in a fresh session at `retry_effort`; a usage limit waits
and retries (never a silent fallback); every call's transcript is audited for reaches
outside its workspace.
"""
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass

class AgentError(Exception):
    pass


class AgentLimit(AgentError):
    """The account's usage limit, not the agent: every call fails until it resets."""


# Steps 16-19 of the AU walkthrough: every call failed with "You've hit your session
# limit · resets 6:50pm", so four steps ran on the fallback alone and fed the plateau
# stop. A limit says nothing about the node, so it must pause the search, never fall back.
LIMIT = re.compile(r"session limit|usage limit|rate limit|limit reached|resets \d", re.I)


def agent_contract(spec_path):
    """The agent body WITHOUT its YAML frontmatter. The frontmatter is Claude Code's
    own registration block; passed through it begins with `---`, which the CLI parses as
    a flag (`unknown option '---\nname: explore...'`)."""
    if not os.path.exists(spec_path):
        return ""
    text = open(spec_path).read()
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end >= 0:
            return text[end + 4:].lstrip()
    return text


def _agent_cmd(model, effort=None, tools="", cwd=None):
    """The `claude -p` invocation, testable without a subprocess.

    `--tools` decides which tools EXIST; `--allowed-tools` only pre-approves. The first
    opus run was launched with `--allowed-tools Read,Grep,Glob` alone, and its transcript
    shows Bash was still there: `ls -R` ran (a built-in safe command), two `python3` calls
    were denied. An agent that can `ls /qr_scratch/tony/som/wip/` can learn which caches
    exist; one that can `cat` a bank can learn what is trained. `--permission-prompts
    none` denies anything that would have prompted, closing the safe-list gap.
    `tools=""` means no tools at all. `--strict-mcp-config` with no `--mcp-config` loads
    no MCP servers: the account's connectors (a docs tool's instructions) were being
    injected into every agent session.

    Each allowed tool is SCOPED to `cwd` (`Read(//<cwd>/**)`): a bare `Read` pre-approves
    reads anywhere, and the first dota offline phase's policy agent used it to Grep
    /qr_scratch/tony/dream_rsi and read an earlier run's policy (2026-09-26). Scoped, an
    outside read needs approval, and `--permission-prompts none` denies it (checked
    with a live call against a file outside the workspace).

    A tool given with its own rule (`Bash(./tool:*)`) exists, and only that rule is approved:
    the research agent may run its workspace's `./tool` and no other command."""
    names = [t for t in (tools or "").split(",") if t]
    allow = ",".join(t if "(" in t else f"{t}(/{os.path.abspath(cwd)}/**)" for t in names) \
        if cwd else ",".join(names)
    cmd = ["claude", "-p", "--output-format", "json",
           "--tools", ",".join(dict.fromkeys(t.split("(")[0] for t in names)),
           "--allowed-tools", allow,
           "--permission-prompts", "none", "--strict-mcp-config", "--model", model]
    if effort:
        cmd += ["--effort", effort]
    return cmd


def transcript_path(cwd, session_id):
    """Where the CLI wrote this session: ~/.claude/projects/<cwd with / and _ -> ->/<id>."""
    enc = re.sub(r"[/_]", "-", cwd.rstrip("/"))
    return os.path.expanduser(f"~/.claude/projects/{enc}/{session_id}.jsonl")


# newer CLIs spill an oversized tool result under /tmp/claude-<uid>/<cwd-slug>/<session>/tool-results
SPILL = re.compile(r"^/tmp/claude-\d+/[^/]+/[^/]+/tool-results/")
OUTSIDE = re.compile(r"(^|[\s'\"=<>(])(/|~|\.\.)")
DENIED = ("No such tool", "disabled for this session", "requires approval",
          "not allowed", "permission")


def audit_transcript(path, cwd):
    """Every tool call the agent made, joined to its result, and every call that reached
    OUTSIDE its workspace — split into what EXECUTED and what was merely ATTEMPTED.

    Read/Grep/Glob take absolute paths and the CLI has no deny-path option, so the
    workspace is a soft boundary; this makes it a detected one. The split matters: the
    first hardened run showed a `Write` to the CLI's auto-memory dir, and the audit called
    it a breach. The tool_result said "No such tool available: Write" — the harness told
    the model to save a memory, the model asked, `--tools` had removed the tool, nothing
    landed. An attempt the sandbox stopped is evidence the sandbox works; only what ran
    outside is a breach, and the record must not conflate them."""
    if not os.path.exists(path):
        return {"found": False}
    uses, results, stopped = {}, {}, 0
    with open(path) as f:
        for line in f:
            try:
                msg = (json.loads(line).get("message") or {})
            except json.JSONDecodeError:
                continue
            stopped += msg.get("stop_reason") == "refusal"
            for b in (msg.get("content") or []) if isinstance(msg.get("content"), list) else []:
                if b.get("type") == "tool_use":
                    uses[b.get("id")] = b
                elif b.get("type") == "tool_result":
                    results[b.get("tool_use_id")] = b
    # the CLI spills an oversized tool result to <session>/tool-results/ and has the
    # agent Read it back — the agent's own output from inside the workspace, not a reach
    # outside it (first seen: a Grep over feature_catalog.json, walkthrough step 1)
    spill = os.path.abspath(path[:-len(".jsonl")])
    by, files, breach, attempt = {}, set(), [], []
    for uid, b in uses.items():
        by[b["name"]] = by.get(b["name"], 0) + 1
        inp = b.get("input") or {}
        r = results.get(uid) or {}
        c = r.get("content")
        txt = c if isinstance(c, str) else " ".join(
            x.get("text", "") for x in c if isinstance(x, dict)) if isinstance(c, list) else ""
        denied = bool(r.get("is_error")) and any(k in txt for k in DENIED)
        failed = bool(r.get("is_error")) and not denied
        p = inp.get("file_path") or inp.get("path") or ""
        p = os.path.join(cwd, p) if p else p      # relative = the agent's cwd, not ours (LU KBc1e4)
        cmd = inp.get("command", "")
        # each part of a compound command on its own: the workspace's ./tool, or a read-only
        # command the CLI runs unprompted inside cwd (a live call was denied `cat /etc/hostname`)
        # that names no path outside it (/dev/null is not a reach)
        parts = [x.strip() for x in re.split(r"[;&|]+", cmd) if x.strip()]
        inside = lambda x: x.startswith("./tool ") or not OUTSIDE.search(
            x.replace(os.path.abspath(cwd), "").replace("/dev/null", ""))
        if b["name"] == "Bash" and parts and all(inside(x) for x in parts):
            if cmd.startswith("./tool "):
                files.add("./tool " + cmd[7:40])
            continue
        if b["name"] == "Bash":
            tag = f"Bash: {inp.get('command', '')[:80]}"
        elif p and not os.path.abspath(p).startswith((os.path.abspath(cwd), spill)) \
                and not SPILL.match(os.path.abspath(p)):
            tag = f"{b['name']}: {p}"
        else:
            if p and not os.path.abspath(p).startswith(spill) and not SPILL.match(os.path.abspath(p)):
                files.add(p)
            continue
        if denied or failed:
            attempt.append(f"{tag}  [{'denied' if denied else 'failed'}]")
        else:
            breach.append(tag)
    return {"found": True, "by_tool": by, "files": sorted(files),
            "breach": breach, "attempt": attempt, "stopped": stopped}


def _call_agent(spec_path, prompt, model, log, timeout=900, cwd=None, effort=None,
                tools=""):
    """One isolated `claude -p` turn, from a neutral cwd: the agent answers the brief
    instead of inheriting a repo's context (the first non-isolated attempt replied about
    a docs hook rather than the question). With `tools`, `cwd` should be a `workspace()`.

    The prompt goes on STDIN, never argv: a brief is tens of kilobytes, and argv has a
    limit that a campaign would eventually find on its own."""
    full = f"{agent_contract(spec_path)}\n\n---\n\n{prompt}"
    t0 = time.time()
    cmd = _agent_cmd(model, effort, tools, cwd)
    try:
        r = subprocess.run(cmd, input=full, capture_output=True, text=True,
                           timeout=timeout, cwd=cwd or "/tmp")
        try:
            d = json.loads(r.stdout)
        except json.JSONDecodeError:
            d = {}
        if r.returncode != 0 or d.get("is_error"):
            # A safeguard stop can also END the session with an error instead of a normal
            # turn (walkthrough step 3). The transcript says which: a stop is retried by
            # `call_agent`, anything else is a failure.
            if tools and cwd and d.get("session_id"):
                a = audit_transcript(transcript_path(cwd, d["session_id"]), cwd)
                if a.get("stopped"):
                    log("agent.call", os.path.basename(spec_path), model=model,
                        effort=effort or "default", secs=round(time.time() - t0, 1),
                        session=d.get("session_id"),
                        cost_usd=round(d.get("total_cost_usd") or 0, 4), stopped=True)
                    return "", a
            msg = f"exit {r.returncode}: {r.stderr[:200] or str(d.get('result'))[:200]}"
            raise (AgentLimit if LIMIT.search(msg) else AgentError)(msg)
        reply = d.get("result") or ""
        log("agent.call", os.path.basename(spec_path),
            model=model, effort=effort or "default", tools=tools or "none",
            secs=round(time.time() - t0, 1), turns=d.get("num_turns"),
            session=d.get("session_id"),
            cost_usd=round(d.get("total_cost_usd") or 0, 4),
            prompt_chars=len(full), reply=reply[:4000])
        a = {}
        if tools and cwd and d.get("session_id"):
            a = audit_transcript(transcript_path(cwd, d["session_id"]), cwd)
            if a.get("found"):
                log("agent.tools", f"{sum(a['by_tool'].values())} call(s)", **a["by_tool"],
                    files=[p.replace(cwd.rstrip("/") + "/", "") for p in a["files"]])
                if a["attempt"]:
                    log("agent.denied", f"{len(a['attempt'])} call(s) outside the workspace "
                                        f"were stopped", calls=a["attempt"])
                if a["breach"]:
                    log("agent.breach", f"{len(a['breach'])} call(s) EXECUTED outside the "
                                        f"workspace", calls=a["breach"])
        return reply, a
    except AgentLimit:
        raise
    except Exception as e:                       # noqa: BLE001 — advisory: never fatal
        log("agent.fail", f"{os.path.basename(spec_path)}: {e}",
            secs=round(time.time() - t0, 1))
        raise AgentError(str(e)) from e


def json_block(text):
    """The first {...} in a reply. Agents narrate; the contract is the object. Decoded
    by the JSON parser itself, so a brace inside a string is just text."""
    dec, i = json.JSONDecoder(), text.find("{")
    while i >= 0:
        try:
            obj, _ = dec.raw_decode(text, i)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        i = text.find("{", i + 1)
    return None


DEFAULT_MODEL = "claude-opus-5-5"   # pinned, not the `opus` alias: a run records what it ran on


@dataclass
class LLM:
    model: str = DEFAULT_MODEL
    effort: str = "high"
    tools: str = "Read,Grep,Glob"
    retry_effort: str = "high"
    timeout: int = 1800
    limit_wait: int = 900
    limit_tries: int = 48

    def call(self, spec_path, prompt, cwd, log, effort=None):
        """The reply text. A stopped answer gets one fresh session at `retry_effort`; a
        usage limit pauses; an unreachable agent raises AgentError."""
        efforts = [effort or self.effort] + ([self.retry_effort] if self.retry_effort else [])
        reply = ""
        for attempt, eff in enumerate(efforts):
            for wait in range(self.limit_tries + 1):
                try:
                    reply, a = _call_agent(spec_path, prompt, self.model, log, self.timeout,
                                           cwd, eff, self.tools)
                    break
                except AgentLimit as e:
                    if wait == self.limit_tries:
                        raise
                    log("agent.limit", f"usage limit — waiting {self.limit_wait}s "
                                       f"({wait + 1}/{self.limit_tries}): {str(e)[:120]}")
                    time.sleep(self.limit_wait)
            if not a.get("stopped") or json_block(reply) is not None:
                return reply
            log("agent.stopped", f"safeguards stopped the answer at effort {eff} "
                                 f"(attempt {attempt + 1} of {len(efforts)})")
        return reply
