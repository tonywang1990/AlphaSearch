"""The records a campaign keeps, and the one rule about them: the agent never sees a test result.

    Plan      a policy's reply, parsed and validated: at most 4 experiments, lessons, tools, stop
    Journal   iters.jsonl (one row per round, with the sealed test verdict) and ledger.jsonl (one
              row per experiment, plus the round's lessons); what a resume reads back. `visible()`
              is the ledger as the agent may read it: every `test*` field stripped"""
import json
import os
from dataclasses import dataclass, field

MAX_EXPERIMENTS = 4
LEDGER_KEYS = ("id", "action", "hypothesis", "name", "base", "twins", "target_cell", "members")


@dataclass
class Plan:
    experiments: list = field(default_factory=list)   # [{"id", "action", ...}] (actions.ACTIONS)
    lessons: list = field(default_factory=list)
    tools: list = field(default_factory=list)         # [{"name", "path", "test"}] the agent wrote
    stop: bool = False

    @classmethod
    def parse(cls, obj):
        """A reply object -> Plan; the first MAX_EXPERIMENTS experiments are kept."""
        obj = obj or {}
        return cls(experiments=list(obj.get("experiments") or [])[:MAX_EXPERIMENTS],
                   lessons=list(obj.get("lessons") or []), tools=list(obj.get("tools") or []),
                   stop=bool(obj.get("stop")))


def _jsonl(path):
    return [json.loads(x) for x in open(path)] if os.path.exists(path) else []


class Journal:
    def __init__(self, out):
        self.iters_path, self.ledger_path = os.path.join(out, "iters.jsonl"), os.path.join(out, "ledger.jsonl")
        self.iters, self.ledger = _jsonl(self.iters_path), _jsonl(self.ledger_path)

    @property
    def failed(self):
        return {r for d in self.iters for r in d.get("failed", [])}

    def visible(self):
        """The ledger as the agent reads it: no test field (an agent that saw what held on test
        steered by it, which invalidated that arm: 452 §8.2)."""
        return [{k: v for k, v in r.items() if not k.startswith("test")} for r in self.ledger]

    def add_round(self, rows, row):
        with open(self.ledger_path, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=float) + "\n")
                self.ledger.append(r)
        with open(self.iters_path, "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        self.iters.append(row)
