"""The layering is one-way: infra.contract <- ensemble, search <- agent; only infra/futs.py
talks to futs-infra; ensemble/ and search/ are pure computation (no agent, no I/O to infra)."""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent / "alphasearch"
ALLOWED = {
    "infra": {"alphasearch.infra", "alphasearch.search.surrogate"},   # fake: the region matcher
    "ensemble": {"alphasearch.ensemble", "alphasearch.infra.contract"},
    "search": {"alphasearch.search"},
    "agent": {"alphasearch.agent", "alphasearch.ensemble", "alphasearch.search", "alphasearch.infra.contract"},
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
