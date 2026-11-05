"""Every knob of a campaign, typed, with its default and the evidence behind it (PR 452,
dota/docs/agent-search.md). A campaign is `run(infra, Config(...), out)`; nothing else is read
from the environment."""
from dataclasses import dataclass, field

SWAP_T = 2.0       # the path-dependent loop's swap bar; the frontier search runs with 0 (452 §10.3)
ENSEMBLE_T = 2.0   # the agent's own ensemble must beat the frontier's pick at paired val t >= 2 (§10.4)


@dataclass(frozen=True)
class Config:
    policy: str = "agent"                  # who decides what to train: agent | surrogate | random
    rounds: int = 6                        # rounds per campaign (a live campaign: a safety cap, it paces itself)
    models: int = 16                       # models trained per round at most (carried models are free)
    seed: int = 0                          # the run's rng: baseline draws, Thompson, the fill
    # the fixed search
    select_objective: str = "ic"           # ic (fit label) | ic_drift (IC10/30/60: §10.2) | wf | infra
    wf_chunks: int = 4                     # wf: each period's aggregator days cut in time order; folds 2..n scored
    worst_weight: float = 0.5              # wf: a fold scores mean(cells) + worst_weight * min(cells)
    weight_shrink: float = 0.25            # wf: ridge toward equal weights (S + shrink * diag S)
    regime_cells: bool = False             # wf: + a cell per session on its least-reverting third of units
    keep_reference: bool = False           # every selected F keeps prod's legs (the search adds to the incumbent)
    cost_table: str | None = None          # JSON {member: [served features]}: the operator's complexity model
    cost_lambda: float = 0.0               # selection score lost per 100 served features (LU: ~1 µs per 100)
    seq_cost: int = 177                    # a sequence leg's recurrence beyond its NA inputs, in features (SC h64 GRU)
    # a live campaign's pacing (pacing.py): it ends on the agent's stop, a plateau, the deadline or the budget
    plateau_rounds: int = 2                # rounds a plateau is judged over
    plateau_gain: float = 0.002            # less val gain than this over them, and no better verdict = plateau
    final_reserve_min: float = 45          # minutes kept before the deadline for the final pick and report
    model_minutes: float = 40              # a model's cost before this campaign has measured one (SC: 35-40)
    reveal_all: bool = False               # a universe benchmark: every pre-fitted member is in the bank at start (free)
    swap_t: float = SWAP_T                 # the swap/path-switch gate; 0 in frontier mode (§10.3)
    frontier: bool = False                 # re-pick F from prod over the whole bank every round (§10.5)
    sur_target: str = "marginal"           # marginal | best_swap (what the surrogate learns: §10.7)
    corrector_w: float = 0.5               # host mode: every non-host member's fixed weight
    weights_block: str = "fit"             # fit | validation: the block ensemble weights are fitted on
    report: bool = False                   # once per campaign: the gate's Table 1 of the final F on the eval block
    # the agent
    prompt: str | None = None              # the brief (default prompts/agent.md, brief v15)
    model: str = "claude-opus-5-5"
    effort: str = "high"
    docs: tuple[str, ...] = ()             # reference docs copied into every workspace
    # memory
    experience: str | None = None          # this symbol's experience store (online memory; tools/ inside)
    memory: str | None = None              # a fixed cross-symbol memory.md (default: rendered from prod)
    memory_stores: dict = field(default_factory=dict)   # {symbol: experience store} summarized into it

    def __post_init__(self):
        assert self.policy in ("agent", "surrogate", "random"), self.policy
        assert self.sur_target in ("marginal", "best_swap"), self.sur_target
        assert self.weights_block in ("fit", "validation"), self.weights_block
