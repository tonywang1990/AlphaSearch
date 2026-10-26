"""The session viewer: one self-contained HTML page over one or more campaign directories, the
trajectory on the left (campaign > round > phase > session turn or experiment), the selected step's
input, processing and output on the right. Owner-side: it shows the sealed test verdicts.

    python -m alphasearch view <campaign dir> [...] -o view.html"""
import json
import os

from alphasearch.viewer.trajectory import build

PAGE = os.path.join(os.path.dirname(__file__), "page.html")


def write(outs, path):
    data = json.dumps(build(outs), default=float, separators=(",", ":")).replace("</", "<\\/")
    open(path, "w").write(open(PAGE).read().replace("%%DATA%%", data))
    return path
