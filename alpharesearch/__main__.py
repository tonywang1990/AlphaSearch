"""python -m alpharesearch run <campaign> --out DIR [options]

<campaign> is the name futs-infra's owner launcher generated for this run (a symbol/budget
request against an approved template); `--fake SEED` runs against the synthetic infra instead."""
import argparse

from alpharesearch.agent.loop import DEFAULTS, run


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m alpharesearch")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run (or resume) a campaign")
    r.add_argument("campaign", nargs="?", help="the infra campaign name (omit with --fake)")
    r.add_argument("--out", required=True, help="the campaign's directory; rerun to resume")
    r.add_argument("--fake", type=int, default=None, metavar="SEED", help="the synthetic infra")
    r.add_argument("--policy", choices=["agent", "surrogate", "random"], default="agent")
    r.add_argument("--iters", type=int, default=6)
    r.add_argument("--models", type=int, default=16, help="models trained per round")
    r.add_argument("--select-objective", default="ic", help="F is selected in this (ic, or an infra "
                   "objective such as ev_drift scored in batches); verdicts use the campaign's objective")
    r.add_argument("--swap-t", type=float, default=DEFAULTS["swap_t"])
    r.add_argument("--split-val", action="store_true", help="the agent's tools see half the val days, "
                   "F is selected on the other half")
    r.add_argument("--sur-target", choices=["marginal", "best_swap"], default="marginal")
    r.add_argument("--experience", default=None, metavar="STORE", help="the per-symbol store of earlier campaigns")
    r.add_argument("--memory", default=None, help="a memory.md of what won on other symbols")
    r.add_argument("--prompt", default=None, help="the agent's brief (default prompts/agent.md)")
    r.add_argument("--model", default=DEFAULTS["model"])
    r.add_argument("--effort", default=DEFAULTS["effort"])
    r.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.fake is not None:
        from alpharesearch.infra.fake import FakeInfra
        infra = FakeInfra(seed=a.fake)
    else:
        assert a.campaign, "a campaign name, or --fake SEED"
        from alpharesearch.infra.futs import FutsInfra
        infra = FutsInfra(a.campaign)
    cfg = {k: getattr(a, k) for k in ("select_objective", "swap_t", "split_val", "sur_target", "experience",
                                      "memory", "prompt", "model", "effort", "seed")}
    print(run(infra, cfg, a.out, a.policy, a.iters, a.models))


if __name__ == "__main__":
    main()
