"""Both streams, one call: events.jsonl (machine) and run.log (human). Never raises."""
import json
import os
import time


class Log:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.events = os.path.join(log_dir, "events.jsonl")
        self.human = os.path.join(log_dir, "run.log")
        self.t0 = time.time()
        self.cost = 0.0        # every `cost_usd` that passes through, tallied here

    def __call__(self, kind, msg="", **fields):
        try:
            self.cost += float(fields.get("cost_usd") or 0.0)
        except (TypeError, ValueError):
            pass
        rec = {"t": round(time.time() - self.t0, 2), "kind": kind, "msg": msg, **fields}
        try:
            with open(self.events, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            extra = " ".join(f"{k}={v}" for k, v in fields.items()
                             if k not in ("traceback", "reply", "prompt", "trace"))
            with open(self.human, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} [{kind}] {msg}"
                        f"{'  ' + extra if extra else ''}\n")
            print(f"[{kind}] {msg}", flush=True)
        except Exception:                        # noqa: BLE001 — logging must not kill a run
            pass


def quiet(*a, **k):
    """A log that records nothing (tests, replay)."""
