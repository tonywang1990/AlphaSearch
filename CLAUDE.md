# CLAUDE.md

AlphaResearch: the research agent over futs-infra. Read `docs/design.md` before changing a layer.

- **The seam is futs-infra's research interface v1.** `alpharesearch/infra/contract.py` is what
  the client may rely on; only `infra/futs.py` imports futs-infra (`futs_infra.api`), never its
  `lib`/`config` internals. A capability the contract lacks is a futs ADR + infra release, not a
  workaround here.
- **Layers are one-way** (`tests/test_layers.py`): `infra.contract` <- `ensemble`, `search` <- `agent`.
  `ensemble/` and `search/` are pure computation.
- **Honesty rules**: the agent sees val-tier summaries only; no test verdict reaches a workspace
  (`workspace.brief` strips `test*` keys); oos is the owner's `evaluate`, never called here.
- **Lean**: delete by default, no defensive code for what cannot happen, Python files <= 400 lines,
  comments only where they carry a measured reason. Use `FakeInfra` for every test.
- **Run tests**: `pytest` (synthetic infra, ~10 s).
