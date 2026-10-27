# AlphaSearch design

AlphaSearch searches futs-infra's model space for one symbol at a time: an LLM research agent
proposes which models to train, a fixed search picks the ensemble from everything trained, and a
sealed scorer judges it. It is DOTA's concluded agent search (futs PR 452,
`dota/docs/agent-search.md`) rebuilt on futs-infra's **research interface v1** (futs ADR 0472).

## The split with futs-infra

| futs-infra owns | AlphaSearch owns |
|---|---|
| data, stores, caches; the grammar's laws; training, fees, fill models | which experiments to run |
| blocks of days: fit, validation (search), test (verdict only), final (oos) | the loop, the brief, the memory, the records |
| per-unit moments, batched scores, screens, verdicts; what prod serves | the fixed search (fill, swaps), the surrogate |
| budgets, exposure, one frozen final submission, builds | the tools and the sandbox the agent's code runs in |

The agent never receives a row, a timestamp, a test number or an oos number: val-tier evidence
arrives as moments (sufficient statistics), test days as a counted `better / worse / flat`, the
final frame only through the owner's `evaluate`.

## The harness, in one page

```
campaign.run(infra, Config, out)                                         campaign.py
  setup     freeze the brief and this week's cross-symbol memory; resume from the journal;
            frontier mode: carry every model the experience store has seen (free)
  per round                                                              round.py
    observe    F = the fixed search over the bank; gains, best swaps, the surrogate -> Observation
    workspace  the files the agent reads, from named providers           context/workspace.py
    plan       the policy's Plan (agent | surrogate | random)            policies/
    act        experiments -> recipes (one handler per action), train   actions.py
    evaluate   F' over the grown bank; the agent's ensemble at t >= 2; sealed verdicts
    reflect    the agent's tools join the library if their tests pass   runtime/sandbox.py
    record     ledger rows (val facts) + the round row (sealed verdict) records.Journal
  episode   val facts and lessons join the experience store            context/experience.py
```

The principles it is built on:

1. **The loop reads like the algorithm.** `campaign.py` is the whole campaign; each phase of a round
   is one function in `round.py` taking what the previous one returned (`Observation`, `Plan`,
   `Act`, `Outcome`). To change what a round does, change one phase.
2. **One policy interface.** The agent and the baselines see the same Observation and return a
   `Plan`; everything around them is shared, so the baselines are the agent's controls at equal
   budget. A policy is `plan(obs, models, rng, log)` + `reflect(...)` (`policies/__init__.py`).
3. **Fixed algorithms are tools; the LLM decides.** The search, the surrogate, the screens are
   deterministic code the agent calls (`tools/`), never one that decides for it; the agent only
   proposes what to train (452 §1: separate "what to train" from "what to combine").
4. **Tools are a registry.** One `@tool(usage, doc)` function per tool; the socket server dispatches
   by name; a test checks the brief and the registry name the same tools.
5. **Context is assembled, not accreted.** The workspace is `context/workspace.PROVIDERS`, an ordered
   list of functions, one per file group; adding context (or ablating it) is one entry.
6. **State in files, sessions stateless.** Each round is a fresh agent session over a fresh
   workspace; the journal (`iters.jsonl`, `ledger.jsonl`) and the experience store are the state, so
   a rerun with the same `out` resumes.
7. **Guard rails at the boundary.** Test fields are stripped where the ledger is exposed
   (`Journal.visible`); the agent's code runs only in bwrap (`runtime/sandbox.py`); its only command
   is `./tool`; its tools join the library only through their own passing test; the transcript is
   audited for reads outside the workspace (`runtime/llm.py`).
8. **Behaviour is pinned.** `tests/test_equivalence.py` replays four recorded campaigns on the
   synthetic infra field by field; a change that moves a decision or a number fails it.
9. **Config is data.** `config.Config` holds every knob with its default and the evidence for it.

## Layers (one-way; `tests/test_layers.py` enforces them)

```
alphasearch/
  config.py       Config: every knob, its default, why
  campaign.py     setup, the round loop, the episode
  round.py        Observation and the five phases
  records.py      Plan (a parsed reply), Journal (iters/ledger; test fields sealed)
  actions.py      plan -> recipes: train, suggest, ensemble, feature_set; the leftover fill
  policies/       agent (the LLM session + tool server), surrogate, random
  tools/          the registry; bank tools (whatif, members, effects, axes, suggest, predcheck,
                  crossval, search); store tools (pcorr, screen); server.py (socket + ./tool)
  context/        workspace (providers), bankview (bank.json, moments.npz, set_score.py),
                  memory (cross-symbol, refreshed from prod), experience (the per-symbol store)
  runtime/        llm (claude -p, retries, audit), sandbox (bwrap, accept_tools), log
  ensemble/       moments, the fixed search, diagnosis: pure numpy
  search/         the surrogate, Thompson draws, effects: pure
  infra/          contract.py (the v1 seam), futs.py (the only futs-infra import), fake.py (tests)
  prompts/        agent.md: brief v15
```

`ensemble/`, `search/` and `runtime/` know nothing of the harness; tools and context read a round,
never drive it; only `campaign.py`, `round.py` and `actions.py` compose the parts.

## How to change it

| To ... | Edit | And |
|---|---|---|
| add a tool | a `@tool(usage, doc)` function in `tools/bank.py` or `tools/store.py` | name it in the brief (`tests/test_tools.py` checks) |
| add a policy (another agent design, a new baseline) | a class in `policies/`, registered in `POLICIES` | add it to `Config.policy`'s choices |
| give the agent more context, or ablate some | a provider in `context/workspace.PROVIDERS` | say what the file is in the brief |
| add an action | a handler in `actions.ACTIONS` | its line in the brief's reply schema |
| change the fixed search | `ensemble/select.py` | expect `test_equivalence` to fail: record why, re-record the golden |
| change the brief | `prompts/agent.md` (frozen per campaign at its start) | a campaign's `agent.md` records what it ran |
| add a memory source | `context/memory.py` (rendered once per campaign) | keep the target symbol out of it |
| reach a new infra capability | `infra/contract.py`, then `infra/futs.py` and `infra/fake.py` | a futs ADR for the infra side |

Re-recording the golden (only for an intended behaviour change): run the four scenarios in
`tests/test_equivalence.py` and write their snapshot to `tests/golden/campaigns.json`, saying in the
commit what changed and why.

## What came over from PR 452, and what did not

- **Kept**: the frontier search (fill + gated swaps from prod over the whole bank), `ic_drift`
  selection, the best-swap surrogate, the agent's ensemble at t >= 2, the whole-bank view, the
  experience store and carry, brief v15, the tool set, the sandbox, the transcript audit.
- **Changed**: the val rows and store frames are not in the sandbox (`pcorr`/`screen` are infra
  calls; 452's v15 agents never read them); test numbers are infra verdict words; the cross-symbol
  memory is rendered from what prod serves each week instead of a hand-written file.
- **Not ported** (452's own deletes): the tree search and offline policy learner, split-val,
  `--stock`, `known_best`, the residual tree.
