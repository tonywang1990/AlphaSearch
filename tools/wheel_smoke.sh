#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
smoke=$(mktemp -d)
trap 'rm -rf "$smoke"' EXIT
cd "$repo"
uv export --locked --no-dev --no-emit-project --output-file "$smoke/requirements.txt" >/dev/null
uv venv --python "$(cat .python-version)" "$smoke/env"
uv pip install --python "$smoke/env/bin/python" --require-hashes -r "$smoke/requirements.txt"
uv pip install --python "$smoke/env/bin/python" --no-deps dist/alphasearch-*.whl
cd "$smoke"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$smoke/env/bin/python" -I -c 'from importlib.resources import files; import alphasearch, sys; from pathlib import Path; assert Path(alphasearch.__file__).is_relative_to(sys.prefix); assert files("alphasearch").joinpath("prompts/agent.md").read_text(); assert files("alphasearch").joinpath("viewer/page.html").read_text()'
"$smoke/env/bin/python" -I -m alphasearch run --fake 0 --out "$smoke/run" --policy surrogate --rounds 3
"$smoke/env/bin/python" -I -m alphasearch view "$smoke/run" --out "$smoke/view.html"
test -s "$smoke/run/iters.jsonl"
test -s "$smoke/view.html"
