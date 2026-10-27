# AlphaSearch

The research agent for futs (named after AlphaGo): one LLM agent with tools that proposes models
and ensembles for a futures symbol, while futs-infra trains, scores and gates them. It talks to
futs-infra only through the research interface v1. The design, its layers and its origins
(DOTA and SoM in futs) are in [`docs/design.md`](docs/design.md).

## Run

```bash
pip install -e ".[dev]"                      # plus futs-infra>=0.2 for real campaigns
python -m alphasearch run --fake 0 --out runs/dry --policy surrogate --rounds 3  # synthetic infra
python -m alphasearch run <campaign> --out runs/<name>                           # a real campaign
pytest
```

`<campaign>` is the name futs-infra's owner launcher generated for an approved symbol/budget
request. Rerun the same command to resume; the campaign's state lives in `--out`
(`iters.jsonl`, `ledger.jsonl`, one workspace per round). Every option is a field of
`alphasearch/config.py`. How the harness is laid out, and where to make a change (a tool, a
policy, context, an action, memory), is in `docs/design.md`.
