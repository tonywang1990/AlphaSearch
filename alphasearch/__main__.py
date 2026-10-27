"""python -m alphasearch run <campaign> --out DIR [options]
python -m alphasearch view <campaign dir> [...] -o view.html

<campaign> is the name futs-infra's owner launcher generated for this run (a symbol/budget
request against an approved template); `--fake SEED` runs against the synthetic infra instead.
Every option is a `Config` field (config.py), where its default and its reason live."""
import argparse
from dataclasses import fields

from alphasearch.campaign import run
from alphasearch.config import Config


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m alphasearch")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run (or resume) a campaign")
    r.add_argument("campaign", nargs="?", help="the infra campaign name (omit with --fake)")
    r.add_argument("--out", required=True, help="the campaign's directory; rerun to resume")
    r.add_argument("--fake", type=int, default=None, metavar="SEED", help="the synthetic infra")
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if f.name == "memory_stores":
            r.add_argument("--memory-store", action="append", default=[], metavar="SYM=STORE",
                           help="an experience store of AlphaSearch on another symbol, summarized into the memory")
        elif f.type is bool:
            r.add_argument(flag, action="store_true")
        elif f.name == "docs":
            r.add_argument(flag, nargs="*", default=())
        else:
            r.add_argument(flag, type={int: int, float: float}.get(f.type, str), default=f.default)
    v = sub.add_parser("view", help="write the session viewer for campaign directories")
    v.add_argument("dirs", nargs="+")
    v.add_argument("-o", "--out", default="view.html")
    a = p.parse_args(argv)
    if a.cmd == "view":
        from alphasearch.viewer import write
        print(write(a.dirs, a.out))
        return
    if a.fake is not None:
        from alphasearch.infra.fake import FakeInfra
        infra = FakeInfra(seed=a.fake)
    else:
        assert a.campaign, "a campaign name, or --fake SEED"
        from alphasearch.infra.futs import FutsInfra
        infra = FutsInfra(a.campaign)
    values = {f.name: getattr(a, f.name) for f in fields(Config) if f.name != "memory_stores"}
    cfg = Config(**values, memory_stores=dict(x.split("=", 1) for x in a.memory_store))
    print(run(infra, cfg, a.out))


if __name__ == "__main__":
    main()
