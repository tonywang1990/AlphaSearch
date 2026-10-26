"""The layering is one-way. The fixed algorithms (ensemble/, search/) and the runtime (runtime/) know
nothing of the harness; tools and context read the round, never drive it; policies return plans;
only campaign.py, round.py and actions.py compose them. Only infra/futs.py talks to futs-infra."""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent / "alphasearch"
ALLOWED = {        # a package -> the alphasearch modules it may import
    "ensemble": {"alphasearch.ensemble", "alphasearch.infra.contract"},
    "search": {"alphasearch.search"},
    "runtime": {"alphasearch.runtime"},
    "infra": {"alphasearch.infra", "alphasearch.search.surrogate"},          # fake: the region matcher
    "tools": {"alphasearch.tools", "alphasearch.search", "alphasearch.runtime"},
    "context": {"alphasearch.context", "alphasearch.search", "alphasearch.tools.server"},
    "viewer": {"alphasearch.viewer", "alphasearch.runtime"},                    # reads a campaign's files only
    "policies": {"alphasearch.policies", "alphasearch.records", "alphasearch.runtime", "alphasearch.tools.server"},
}


def _imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            yield from (a.name for a in node.names)


def test_layers():
    bad = []
    for path in ROOT.rglob("*.py"):
        layer = path.relative_to(ROOT).parts[0]
        for mod in _imports(path):
            top = mod.split(".")[0]
            if top in ("futs_infra", "lib", "config") and path.name != "futs.py":
                bad.append(f"{path.name}: {mod}")
            if top == "alphasearch" and layer in ALLOWED and mod != "alphasearch" \
                    and not any(mod == a or mod.startswith(a + ".") for a in ALLOWED[layer]):
                bad.append(f"{layer}/{path.name}: {mod}")
    assert not bad, bad


def test_file_size_cap():
    big = [p.name for p in ROOT.rglob("*.py") if len(p.read_text().splitlines()) > 400]
    assert not big, big
