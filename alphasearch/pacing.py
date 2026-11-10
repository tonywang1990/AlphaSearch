"""How a live campaign paces itself (ADR 0472-research-api-v1-live). Training costs real minutes there, so the
round's model count comes from what one model costs (measured on this campaign's own jobs), the time left before
the deadline and the compute budget; and the search ends when it stops improving. A benchmark (universe)
campaign reveals pre-fitted models for free and is not paced."""
import statistics
import time


def arch_minutes(journal, by_set=False):
    """arch (or arch.feature_set) -> median wall minutes per trained model, from every round's `train_seconds`
    (0 = a cached reuse). The feature set matters as much as the arch: an all-store (1,939-column) linear takes
    over an hour where a 30-column one takes 20 minutes."""
    by = {}
    for r in journal.iters:
        for name, s in (r.get("train_seconds") or {}).items():
            if s > 0:
                p = name.split(".")
                by.setdefault(f"{p[0]}.{p[3]}" if by_set else p[0], []).append(s / 60)
    return {a: round(statistics.median(v), 1) for a, v in sorted(by.items())}


def pace(budget, journal, cfg, now=None):
    """-> (models this round may train, the state.json budget block, the reason to stop now or None).
    `minutes_left` is to the training deadline (the deadline less the final reserve): a job still running then
    is cancelled."""
    mins = arch_minutes(journal)
    per = statistics.median(mins.values()) if mins else cfg.model_minutes      # the prior until one is measured
    cheapest = min(mins.values()) if mins else cfg.model_minutes
    left = (budget.deadline - (now or time.time())) / 60 - cfg.final_reserve_min if budget.deadline else None
    par = max(1, budget.max_parallel)
    by_time = cfg.models if left is None else par * int(left // per)
    n = max(0, min(cfg.models, by_time, int(budget.seconds_left / 60 // per)))
    block = {"live": True, "minutes_left": None if left is None else round(left),
             "compute_minutes_left": round(budget.seconds_left / 60), "parallel": par,
             "minutes_per_model_by_arch": mins or {"prior (nothing measured yet)": cfg.model_minutes},
             "minutes_per_model_by_arch_and_feature_set": arch_minutes(journal, by_set=True),
             "unspent_is_saved": True}
    if left is not None and left < cheapest:
        return 0, block, f"deadline: {left:.0f} min left, the cheapest model takes {cheapest:.0f}"
    if budget.seconds_left / 60 < cheapest:
        return 0, block, (f"budget: {budget.seconds_left / 60:.0f} compute min left, "
                          f"the cheapest model takes {cheapest:.0f}")
    return max(n, 1), block, None


def train_until(budget, cfg):
    """The epoch training must end by (the deadline less the final reserve), or None without a deadline."""
    return budget.deadline - cfg.final_reserve_min * 60 if budget.deadline else None


def plateau(journal, cfg):
    """The last `plateau_rounds` rounds together raised F's val score by less than `plateau_gain` and none of
    them drew a `better` held-out verdict: more of the same is fitting the val days, not finding signal."""
    it, k = journal.iters, cfg.plateau_rounds
    if len(it) <= k:
        return None
    gain = it[-1]["F_after_val"] - it[-1 - k]["F_after_val"]
    if gain < cfg.plateau_gain and all(r["test_verdict"] != "better" for r in it[-k:]):
        return f"plateau: val {gain:+.4f} over the last {k} rounds, no better verdict"
    return None
