# AlphaSearch

The research agent for futs (named after AlphaGo): one LLM agent with tools that proposes models
and ensembles for a futures symbol, while futs-infra trains, scores and gates them. It talks to
futs-infra only through the research interface v1 (API majors 1 and 2). The design, its layers and its origins
(DOTA and SoM in futs) are in [`docs/design.md`](docs/design.md).

## Run

```bash
make setup                                 # uv 0.11.29; installs locked tools and commit hooks
uv run --no-sync python -m alphasearch run --fake 0 --out runs/dry --policy surrogate --rounds 3
make check                                 # lint, formatting, types, all synthetic tests
make smoke                                 # build and test the installed wheel outside the checkout
```

`<campaign>` is the name futs-infra's owner launcher generated for an approved symbol/budget
request. Rerun the same command to resume; the campaign's state lives in `--out`
(`iters.jsonl`, `ledger.jsonl`, one workspace per round). Every option is a field of
`alphasearch/config.py`. How the harness is laid out, and where to make a change (a tool, a
policy, context, an action, memory), is in `docs/design.md`.

## Development and CI

Use Linux and Python 3.12.13 (pinned in `.python-version`). Install Bubblewrap (`apt install
bubblewrap`) for sandbox tests and Node 22 for Pyright. CI requires Bubblewrap to work;
local tests may skip it only when it is absent. `make format` applies Ruff fixes and formatting;
`make check` checks without rewriting. Pyright checks the package in basic mode, the infra contract in strict mode, and both implementations
of `Infra` in `tests/type_contract.py`. Python files have a 700-line hard cap; prefer 400
and keep one responsibility in larger modules.

`uv.lock` pins the development environment; NumPy/SciPy/sklearn constraints preserve the
golden baseline. Update deliberately with `uv lock --upgrade` and rerun `make check smoke`.
Golden decisions, strings and IDs compare exactly; numeric leaves tolerate only 1e-12
relative/absolute roundoff. Review intended behavioural changes before re-recording a golden.

The `CI / quality` job runs on PRs, main/master pushes and version tags. It uploads the
tested wheel/sdist and SHA256SUMS as `alphasearch-<commit>`. Make `quality` required in
the repository's branch rules after its first hosted run. Hooks provide local feedback;
the required CI check enforces the policy. Tag delivery does not deploy a model.

## Installed infra

Real campaigns require the owner's exact futs-infra wheel and provisioned runtime.
PR472 supplies the `futs_infra.api` facade; PR409 alone is insufficient. Install the approved
AlphaSearch and infra wheels into the owner-managed environment **before** registering a
campaign, and run `python -I -m alphasearch run <campaign> --out runs/<name>` there.
Follow futs' installation instructions for native artifacts and runtime pins.

The former `[infra]` extra is replaced by explicit owner installation: an internal package
must not be resolved from public PyPI. The optional import has one localized Pyright
missing-import exception, so fake CI remains independent of private infra credentials.
Package CI does not establish real adapter compatibility: futs' owner-host compatibility
test installs both wheels and exercises the adapter through a registered synthetic campaign.
Run it for API/adapter changes and before releasing a new pair; retain both wheel hashes.
Do not `uv sync` a registered runtime: it can remove the owner-installed packages and
invalidate the campaign's pinned inventory.
