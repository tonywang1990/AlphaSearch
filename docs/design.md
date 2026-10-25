# AlphaResearch design

AlphaResearch is the research agent that searches futs-infra's model space for one symbol at a
time. It is the one-agent DOTA loop (futs ADR 0452) plus the research half of SoM, rebuilt as
its own repository over futs-infra's **research interface v1** (futs ADRs
`research-api-v1` and `research-api-v1-infra`, on PR 409's separation).

## The split

| futs-infra owns | AlphaResearch owns |
|---|---|
| market data, stores, caches; recipe resolution and the grammar's laws | which experiments to run |
| training, calibration, scoring, the fill model and fees | the loop, prompts, memory, ledger |
| blocks of days: fit, validation (search), test (verdict only), final (oos) | selection policy (fill, swaps, ensemble search) |
| per-unit moments, batched scores, screens, verdicts | the surrogate and the agent's own predictors |
| budgets, exposure, one frozen final submission, builds | tools and the sandbox the agent's code runs in |

The agent never receives a row, a timestamp, a test number or an oos number. Val-tier
evidence arrives as moments (sufficient statistics); test days as a counted `better / worse /
flat`; the final frame only through the owner's `evaluate`.

## Layers (one-way; `tests/test_layers.py` enforces them)

```
alpharesearch/
  infra/      contract.py   the v1 interface as the client sees it (dataclasses + Protocol)
              futs.py       the real adapter: the ONLY import of futs-infra (futs_infra.api)
              fake.py       synthetic infra over generated ticks: tests and dry runs
  ensemble/   moments.py    IC, paired tests, day subsets from per-unit sums   (som Space + dota daystats)
              select.py     fill, gated swaps, Frontier select, crossval, search, gains (dota frontier/select)
              diagnose.py   F by period x session vs prod, untaken signal       (dota diagnose)
  search/     surrogate.py  random-forest surrogate, Thompson draws, effects, predcheck (dota surrogate)
  agent/      loop.py       the campaign round loop                              (dota agent.py)
              plan.py       agent / baseline plans, experiments -> recipes       (dota agent_plan.py)
              workspace.py  the round's brief: state, bank, moments.npz, ...     (dota agent_plan.brief, bankview)
              tools.py      ./tool server + client, the bwrap sandbox            (dota env/som/tools.py)
              experience.py the per-symbol store of earlier campaigns            (dota experience.py)
              llm.py, log.py, prompts/agent.md                                   (dota core/, prompts/agent_v15.md)
```

`ensemble/` and `search/` are pure computation; `agent/` composes them; only `infra/futs.py`
touches futs-infra. A new agent design replaces `agent/` and keeps the rest.

## What came over, and what did not

- **Kept**: the Frontier (fill + swaps at paired val t >= 2, from prod, over the whole bank), the
  agent's `ensemble` bar (t >= 2), split-val, host mode, `ic` selection with EV-family selection
  through infra batches, the surrogate and `best_swap` target, crossval, ensemble search,
  feature-set experiments with a null screen first, agent-written tools behind a sandboxed
  test, the experience store, the v15 brief's measured guidance.
- **Changed**: the val rows and store frames are no longer in the sandbox; `pcorr` and `screen`
  are infra calls. Test numbers became infra verdict words, also for the per-experiment rows.
  Recipes are axis dicts; their names are the values joined in the grammar's axis order.
- **Not ported**: the tree search and offline policy learner (ADR 0452 deletes them), SoM's
  menu/plateau/bundle policies, benchmark ("universe") mode, carrying models across campaigns
  (needs infra's v1.x model import), the sandbox over raw val rows (v1.x `analyze` if earned).

## Status

Runs end to end on `FakeInfra` (`python -m alpharesearch run --fake 0 --out runs/x --policy
surrogate`). The real adapter is written against the v1 contract; it needs futs-infra 0.2.0
(the P1 build). P1's acceptance includes replaying a recorded PR 452 LU round through it.
