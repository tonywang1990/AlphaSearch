# CLAUDE.md

AlphaSearch: the research agent over futs-infra. Read `docs/design.md` before changing a layer.

- **The seam is futs-infra's research interface v1.** `alphasearch/infra/contract.py` is what
  the client may rely on; only `infra/futs.py` imports futs-infra (`futs_infra.api`), never its
  `lib`/`config` internals. A capability the contract lacks is a futs ADR + infra release, not a
  workaround here.
- **The harness reads top-down**: `campaign.py` (the campaign) -> `round.py` (five phases) ->
  `policies/`, `actions.py`, `tools/`, `context/`. `docs/design.md` has the table of where each kind
  of change goes; follow it rather than adding a new place.
- **Layers are one-way** (`tests/test_layers.py`): `ensemble/`, `search/`, `runtime/` know nothing of
  the harness; tools and context read a round, never drive it.
- **Behaviour is pinned** (`tests/test_equivalence.py`): a refactor must pass it unchanged; an
  intended behaviour change re-records the golden and says why in the commit.
- **Honesty rules**: the agent sees val-tier summaries only; no test verdict reaches a workspace
  (`Journal.visible` strips `test*` keys); oos is the owner's `evaluate`, never called here.
- **Lean**: delete by default, no defensive code for what cannot happen, Python files <= 400 lines,
  comments only where they carry a measured reason. Use `FakeInfra` for every test.
- **Run tests**: `pytest` (synthetic infra, ~10 s).
