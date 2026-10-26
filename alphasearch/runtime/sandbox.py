"""The agent's own code runs here and nowhere else: bwrap, the workspace read-only but `work/`, the
tool library, no network, no repo, no data beyond the workspace's summaries."""
import os
import shutil
import subprocess
import sys


def sandbox(ws, script, args, lib=None, timeout=900):
    """(exit code, output) of the agent's `script` under bwrap."""
    env = sys.prefix
    cmd = ["bwrap", "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
           "--ro-bind", "/lib64", "/lib64", "--ro-bind", env, env, "--proc", "/proc",
           "--dev", "/dev", "--tmpfs", "/tmp", "--unshare-all", "--die-with-parent",
           "--ro-bind", os.path.abspath(ws), "/ws",
           "--bind", os.path.abspath(os.path.join(ws, "work")), "/ws/work", "--chdir", "/ws"]
    if lib and os.path.isdir(lib):
        cmd += ["--ro-bind", lib, "/lib_tools"]
    cmd += ["--setenv", "OMP_NUM_THREADS", "4", os.path.join(env, "bin", "python"), script, *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"
    return r.returncode, (r.stdout + r.stderr)[-20000:]


def accept_tools(ws, lib, tools, log):
    """Tools the agent wrote: each joins `lib` (kept across campaigns) only if its own test passes
    in the sandbox; a path outside `work/` is refused."""
    out = []
    for t in tools or []:
        path, test = t.get("path", ""), t.get("test", "")
        if not (path.startswith("work/") and test.startswith("work/")
                and os.path.exists(os.path.join(ws, path)) and os.path.exists(os.path.join(ws, test))):
            out.append({**t, "accepted": False, "why": "path and test must be files under work/"})
            continue
        code, txt = sandbox(ws, test, [], lib)
        if code == 0:
            os.makedirs(lib, exist_ok=True)
            for p in (path, test):
                shutil.copy(os.path.join(ws, p), os.path.join(lib, os.path.basename(p)))
        out.append({**t, "accepted": code == 0, "test_output": txt[-1500:]})
        log("tool", f"{t.get('name')}: {'accepted' if code == 0 else 'rejected'}")
    return out
