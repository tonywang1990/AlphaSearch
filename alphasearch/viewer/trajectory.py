"""A campaign directory -> its trajectory as a tree of nodes, for the session viewer (page.html).

    campaign   config, brief, memory, carry -> rounds -> episode
    round      the round row; children: the seven phases
    phase      observe | workspace | plan | act | evaluate | reflect | record; plan's children are
               the agent session's turns (each tool call with its result), act's the experiments

A node is {"t": title, "s": subtitle, "k": kind, "in": [...], "proc": [...], "out": [...], "c": [...]}
and each section {"h": heading, "ty": json | text | file | table | chat, "v": value}. File contents go
once into `files` (by hash) however many rounds carry them. The owner reads this: the sealed test
verdicts are shown, marked as such."""
import glob
import hashlib
import json
import os

from alphasearch.runtime.llm import transcript_path

PHASES = ("observe", "workspace", "plan", "act", "evaluate", "reflect", "record")   # campaign.PHASES

MAX_FILE, MAX_RESULT = 60_000, 8_000


def _rows(path):
    return [json.loads(x) for x in open(path)] if os.path.exists(path) else []


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


class Files:
    """Every file shown, once, by content hash."""
    def __init__(self):
        self.store = {}

    def ref(self, path, name=None):
        if not os.path.exists(path):
            return None
        size = os.path.getsize(path)
        if path.endswith((".npz", ".sock")) or ".sock" in path:
            text = f"(binary, {size:,} bytes)"
        else:
            raw = open(path, "rb").read(MAX_FILE + 1)
            text = raw[:MAX_FILE].decode("utf-8", "replace") + (f"\n… cut at {MAX_FILE:,} of {size:,} bytes"
                                                                if size > MAX_FILE else "")
        key = hashlib.sha1(text.encode()).hexdigest()[:12]
        self.store.setdefault(key, text)
        return {"name": name or os.path.basename(path), "size": size, "ref": key}


def sec(h, ty, v):
    return {"h": h, "ty": ty, "v": v}


def node(t, k, s="", inp=(), proc=(), out=(), c=()):
    return {"t": t, "s": s, "k": k, "in": [x for x in inp if x], "proc": [x for x in proc if x],
            "out": [x for x in out if x], "c": list(c)}


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
    return json.dumps(content)


def _cut(text, n=MAX_RESULT):
    return text if len(text) <= n else text[:n] + f"\n… cut at {n:,} of {len(text):,} chars"


def session_turns(path, files):
    """One transcript -> (prompt file, its turns as nodes, the final reply text, usage)."""
    uses, results, turns, prompt, reply = {}, {}, [], None, ""
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") not in ("user", "assistant"):
            continue
        content = (r.get("message") or {}).get("content")
        if r["type"] == "user" and isinstance(content, str) and prompt is None:
            prompt = content
            continue
        for b in content if isinstance(content, list) else []:
            if b.get("type") == "tool_result":
                results[b.get("tool_use_id")] = b
            elif r["type"] == "assistant":
                turns.append(b)
                if b.get("type") == "tool_use":
                    uses[b.get("id")] = b
                elif b.get("type") == "text" and b.get("text", "").strip():
                    reply = b["text"]
    nodes, n_tool = [], 0
    for b in turns:
        if b.get("type") == "tool_use":
            n_tool += 1
            inp = b.get("input") or {}
            what = inp.get("command") or inp.get("file_path") or inp.get("pattern") or json.dumps(inp)[:80]
            what = what.replace(os.path.dirname(path), "")
            res = results.get(b.get("id")) or {}
            err = bool(res.get("is_error"))
            nodes.append(node(f"{b['name']}: {os.path.basename(what) if b['name'] != 'Bash' else what}"[:90],
                              "tool", "error" if err else "",
                              inp=[sec("Call", "json", inp)],
                              out=[sec("Result (error)" if err else "Result", "text",
                                       _cut(_text(res.get("content"))))]))
        elif b.get("type") == "text" and b.get("text", "").strip():
            nodes.append(node("says", "text", out=[sec("Text", "text", _cut(b["text"], 20_000))]))
        elif b.get("type") == "thinking" and (b.get("thinking") or "").strip():
            nodes.append(node("thinks", "thinking", out=[sec("Thinking", "text", _cut(b["thinking"], 20_000))]))
    return prompt, nodes, reply, n_tool


def _sessions(out, it):
    """The round's transcripts: the archived copy, else the CLI's own."""
    ws = os.path.abspath(os.path.join(out, f"it_{it}"))
    got = sorted(glob.glob(os.path.join(out, "sessions", f"it_{it}", "*.jsonl")))
    return got or sorted(glob.glob(os.path.join(os.path.dirname(transcript_path(ws, "x")), "*.jsonl")),
                         key=os.path.getmtime)


def _events_by_round(events):
    """Events split at each round's last line (`phases`; `iteration` in runs before it was logged);
    the rest is the episode's."""
    end = "phases" if any(e.get("kind") == "phases" for e in events) else "iteration"
    out, cur = [], []
    for e in events:
        cur.append(e)
        if e.get("kind") == end:
            out.append(cur)
            cur = []
    return out, cur


def _plan_object(reply):
    from alphasearch.runtime.llm import json_block
    return json_block(reply or "") or {}


def _table(rows, keys):
    return [{k: r.get(k) for k in keys} for r in rows]


def observe(ws, row, cfg, secs, files):
    state = _load(os.path.join(ws, "state.json")) or {}
    top = _load(os.path.join(ws, "bank_top.json")) or []
    avail = os.path.join(ws, "available.txt")
    n_avail = sum(1 for _ in open(avail)) if os.path.exists(avail) else None
    start = "prod's members (frontier mode: F is re-picked from prod over the whole bank each round)" \
        if cfg.get("frontier") else "the previous round's F′"
    return node("observe", "phase", secs,
                inp=[sec("Start", "text", start),
                     sec("Bank", "text", f"{row.get('bank', 0) - len(row.get('trained', []))} trained models before")],
                proc=[sec("What the code does", "text",
                          "1. The fixed search selects F from the start over the bank on the val days "
                          f"({cfg.get('select_objective')}, swap_t {cfg.get('swap_t')}).\n"
                          "2. It learns every model's marginal gain to F and every non-member's best swap.\n"
                          f"3. The surrogate refits on those gains (target {cfg.get('sur_target')}).\n"
                          "4. It lists every legal recipe not yet trained.")],
                out=[sec("F (the round's starting ensemble)", "table",
                         state.get("F") or [{"member": m} for m in row["F"]]),
                     sec("Scores (1.0 = prod)", "json", {"F_val": state.get("F_val", row.get("F_val")),
                                                         "prod_val": state.get("prod_val"),
                                                         "untrained recipes": n_avail}),
                     sec("Top of the bank by marginal gain", "table", top[:20]),
                     sec("Diagnosis", "file", files.ref(os.path.join(ws, "diagnosis.json")))])


def workspace(ws, secs, files):
    refs = []
    for root, _, names in os.walk(ws):
        for n in sorted(names):
            rel = os.path.relpath(os.path.join(root, n), ws)
            if not n.startswith(".") and rel != "tool":
                refs.append(files.ref(os.path.join(root, n), rel))
    return node("workspace", "phase", secs, proc=[sec("What the code does", "text",
                "context/workspace.py runs each provider in PROVIDERS; each writes its files into this new "
                "directory. Everything is read-only for the agent except work/.")],
                out=[sec(f"Files the agent can read ({len(refs)})", "files", refs)])


def plan(out, it, ws, cfg, rows, events, secs, files):
    calls = [e for e in events if e.get("kind") == "agent.call"]
    exps = [r for r in rows if "action" in r]
    if cfg.get("policy") != "agent":
        how = "Thompson draws from the surrogate over every untrained recipe" if cfg["policy"] == "surrogate" \
            else "uniform random untrained recipes"
        return node("plan", "phase", secs, proc=[sec("Policy", "text", f"{cfg['policy']}: {how}")],
                    out=[sec("Plan", "table", _table(exps, ("id", "action", "hypothesis")))]), {}
    kids, reply = [], ""
    for i, path in enumerate(_sessions(out, it)):
        prompt, turns, r, n_tool = session_turns(path, files)
        call = calls[i] if i < len(calls) else {}
        label = "session" if i == 0 else "repair session (reply was not one JSON object)"
        kids.append(node(label, "session", f"{n_tool} tool calls · {call.get('turns', '?')} turns · "
                         f"${call.get('cost_usd', 0):.2f} · {call.get('secs', '?')}s",
                         inp=[sec("Prompt", "text", _cut(prompt or "", 4000))],
                         out=[sec("Final reply", "text", r)], c=turns))
        reply = r or reply
    reply = reply or (calls[-1].get("reply", "") if calls else "")
    obj = _plan_object(reply)
    cost = sum(c.get("cost_usd") or 0 for c in calls)
    return node("plan", "phase", f"{secs} · ${cost:.2f}" if secs else f"${cost:.2f}",
                inp=[sec("Brief", "file", files.ref(os.path.join(ws, "agent.md"))),
                     sec("Model", "json", {k: calls[0].get(k) for k in ("model", "effort", "tools")} if calls else {})],
                proc=[sec("What happens", "text", "One claude -p session in the workspace. Its only command is "
                          "./tool; it reads files, runs tools and its own scripts in the sandbox, then replies. "
                          "Open the session below for every turn.")],
                out=[sec("Plan (parsed)", "json", obj), sec("Raw reply", "text", reply)], c=kids), obj


def act(rows, plan_obj, row, secs):
    by_id = {e.get("id"): e for e in plan_obj.get("experiments") or []}
    kids = []
    for r in rows:
        if "action" not in r:
            continue
        e = by_id.get(r["id"]) or {k: r.get(k) for k in ("id", "action", "hypothesis")}
        facts = {k: v for k, v in r.items() if k in ("pool", "refused_recipes", "refused", "screen", "feature_set")}
        how = {"train": "kept the named recipes that are untrained and legal",
               "suggest": "the surrogate's Thompson draws inside `where`",
               "ensemble": "nothing to train; judged in evaluate",
               "feature_set": "screen vs random sets, infra's admission rule, then twins + surrogate fill"}
        how_txt = "unspent budget: the surrogate's Thompson draws" if r["id"] == "_fill" else how.get(r["action"], "")
        kids.append(node(f"{r['id']} · {r['action']}", "experiment", f"{len(r.get('trained') or [])} trained",
                         inp=[sec("Experiment", "json", e)],
                         proc=[sec("Handler", "text", how_txt), sec("Facts", "json", facts) if facts else None],
                         out=[sec("Trained", "table", [{"recipe": x} for x in r.get("trained") or []])]))
    return node("act", "phase", secs,
                inp=[sec("Budget", "text", f"{len(row.get('trained', []))} models this round")],
                proc=[sec("What the code does", "text", "Each experiment's handler (actions.py) turns it into recipes, "
                          "in order, each taking only what budget is left; a recipe two experiments pick is trained "
                          "once; then one infra.train call for the batch.")],
                out=[sec("Trained", "text",
                         f"{len(row.get('trained', []))} models, {len(row.get('failed', []))} failed"),
                     sec("Failed", "json", row.get("failed")) if row.get("failed") else None], c=kids)


def evaluate(rows, row, secs):
    F, G = row["F"], row["F_after"]
    diff = [{"member": m, "change": "kept" if m in F else "IN"} for m in G] + \
           [{"member": m, "change": "out"} for m in F if m not in G]
    per = _table([r for r in rows if "action" in r], ("id", "action", "entered_F", "val_gain", "test_verdict",
                                                     "adopted", "t_vs_selected"))
    return node("evaluate", "phase", secs,
                inp=[sec("Bank", "text", f"{row.get('bank')} models after training")],
                proc=[sec("What the code does", "text", "Re-select F′ over the grown bank; an `ensemble` experiment "
                          "replaces it only at paired val t ≥ 2; infra's sealed test verdict for F′ vs F and for "
                          "each experiment alone (its models only).")],
                out=[sec("Scores", "json", {"F_val": row.get("F_val"), "F_after_val": row.get("F_after_val"),
                                            "test_verdict (sealed from the agent)": row.get("test_verdict")}),
                     sec("F → F′", "table", diff), sec("Each experiment", "table", per)])


def reflect(rows, events, secs):
    tools = next((r.get("tools") for r in rows if "lessons" in r), []) or []
    return node("reflect", "phase", secs, inp=[sec("Tools the agent returned", "table",
                                                   _table(tools, ("name", "path", "test", "what")))],
                proc=[sec("What the code does", "text", "Each tool's test runs in the sandbox; a tool joins the "
                          "library (lib/, kept across campaigns) only if its test passes. No LLM runs here.")],
                out=[sec("Result", "table", _table(tools, ("name", "accepted", "test_output")))])


def record(rows, row, secs):
    lessons = next((r.get("lessons") for r in rows if "lessons" in r), []) or []
    return node("record", "phase", secs,
                out=[sec("Lessons (written by the agent in its reply)", "text", "\n\n".join(f"• {x}" for x in lessons)),
                     sec("Round row (iters.jsonl)", "json", row),
                     sec("Ledger rows (ledger.jsonl)", "json", rows)])


def round_node(out, it, row, rows, events, cfg, files):
    ws = os.path.join(out, f"it_{it}")
    lap = next((e for e in events if e.get("kind") == "phases"), {})
    s = lambda p: f"{lap[p]}s" if p in lap else ""
    p, obj = plan(out, it, ws, cfg, rows, events, s("plan"), files)
    kids = [observe(ws, row, cfg, s("observe"), files), workspace(ws, s("workspace"), files), p,
            act(rows, obj, row, s("act")), evaluate(rows, row, s("evaluate")), reflect(rows, events, s("reflect")),
            record(rows, row, s("record"))]
    sub = f"val {row['F_val']:.4f} → {row['F_after_val']:.4f} · test {row['test_verdict']} · " \
          f"{len(row.get('trained', []))} trained · {row.get('secs', 0) // 60}m"
    return node(f"Round {it}", "round", sub, inp=[sec("F", "table", [{"member": m} for m in row["F"]])],
                out=[sec("Round row", "json", row)], c=kids)


def campaign_node(out, files):
    out = os.path.abspath(out)
    cfg = _load(os.path.join(out, "config.json")) or {}
    iters, ledger = _rows(os.path.join(out, "iters.jsonl")), _rows(os.path.join(out, "ledger.jsonl"))
    cfg.setdefault("policy", (iters[0] if iters else {}).get("policy", "?"))
    by_round, tail = _events_by_round(_rows(os.path.join(out, "events.jsonl")))
    kids = [round_node(out, r["iteration"], r, [x for x in ledger if x.get("iteration") == r["iteration"]],
                       by_round[i] if i < len(by_round) else [], cfg, files) for i, r in enumerate(iters)]
    ep = _load(os.path.join(out, "episode.json"))
    if ep:
        kids.append(node("episode", "episode", "into the experience store",
                         proc=[sec("What the code does", "text", "context/experience.py records the campaign's val "
                                   "facts and lessons and adds them to the per-symbol store the next campaign reads.")],
                         out=[sec("episode.json", "json", ep)]))
    rep = _load(os.path.join(out, "replication.json"))
    usd = sum(e.get("cost_usd") or 0 for g in by_round for e in g)
    vs = [r["test_verdict"] for r in iters]
    return node(os.path.basename(out), "campaign", f"{cfg.get('policy')} · {len(iters)} rounds · ${usd:.2f} · test "
                + " ".join(v[0] for v in vs),
                inp=[sec("Config", "json", cfg), sec("Brief", "file", files.ref(os.path.join(out, "agent.md"))),
                     sec("Cross-symbol memory", "file", files.ref(cfg.get("memory") or os.path.join(out, "memory.md"))),
                     sec("Setup events", "table", _table([e for e in (by_round[0] if by_round else [])
                                                          if e.get("kind") == "carry"], ("kind", "msg")))],
                out=[sec("Test score vs PR 452, per round (owner only)", "table", rep) if rep else None,
                     sec("Final F", "table", [{"member": m} for m in iters[-1]["F_after"]] if iters else [])],
                c=kids)


def build(outs):
    files = Files()
    return {"campaigns": [campaign_node(o, files) for o in outs], "files": files.store}
