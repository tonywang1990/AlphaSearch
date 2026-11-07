"""What AlphaSearch needs from futs-infra: the research interface v1 (futs ADR
`research-api-v1`), as the client sees it. Every other module codes against this file;
only `infra/futs.py` (the real adapter) and `infra/fake.py` (tests, dry runs) implement it.

Nothing here carries a row or a timestamp: search-tier data is per-unit sums (`Moments`),
test days come back as a `Verdict` word, and oos is reachable only through `evaluate`."""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

API_VERSIONS = ("1.",)  # the interface majors this client speaks

ErrorCode = Literal["invalid", "not_permitted", "budget_exhausted", "busy", "unavailable", "not_found", "infra_failure"]


class ApiError(Exception):
    def __init__(self, code: ErrorCode, message: str, retryable: bool = False):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.retryable = code, message, retryable


@dataclass(frozen=True)
class Description:
    api_version: str
    campaign: str
    symbol: str
    axes: dict[str, list[str]]  # grammar axis -> its values, in recipe-name order
    objective: str  # the verdict/final metric, e.g. "ic", "ev_drift"
    k_max: int  # the most members an ensemble may hold
    reference: list[str]  # prod's members: the 1.0 of every score
    host: str | None = None  # host mode: this member is in every ensemble at weight 1
    columns: list[str] = field(default_factory=list[str])  # store columns (screen / pcorr)
    feature_sets: dict[str, list[str]] = field(default_factory=dict[str, list[str]])  # composed sets by name
    feature_set_rule: dict[str, Any] | None = None  # None: the campaign admits no new feature sets
    universe: bool = False  # benchmark: training reveals pre-fitted models
    labels: list[str] = field(default_factory=lambda: ["10s"])  # the horizons `moments` carries
    fit_label: str = "10s"  # the label weights are fitted on and IC selects on
    notes: dict[str, dict[str, str]] = field(
        default_factory=dict[str, dict[str, str]]
    )  # axis -> value -> what it means
    feature_set_defs: dict[str, Any] = field(default_factory=dict[str, Any])  # how each feature set is composed
    served: list[dict[str, Any]] = field(
        default_factory=list[dict[str, Any]]
    )  # what prod serves per slot this week (cross-symbol memory)
    report: bool = False  # a strictly held-out test block `test_report` reads (API 1.1)


@dataclass(frozen=True)
class Budget:
    seconds_left: float
    verdicts_left: int
    deadline: float | None
    max_parallel: int


@dataclass(frozen=True)
class Model:
    name: str  # the recipe's name (axis values joined by ".")
    recipe: dict[str, str]
    id: str  # infra's handle


@dataclass(frozen=True)
class Moments:
    """Per (day, session) unit sums over one block, members in `names` order. With z a
    member's per-day standardized prediction, m its validity (0/1) and y a label:
    n, sz, szz: (U, K) = Σm, Σz, Σz²; zz: (U, K, K) = ZᵀZ; zzm: (U, K, K) = Σ z_i² m_j;
    labels[l][k]: (U, K) for k in sy, syy, szy = Σm·y, Σm·y², Σz·y, one per `label_names`."""

    names: list[str]
    day: NDArray[np.int64]  # (U,) day index
    session: NDArray[np.int64]  # (U,) session index into `sessions`
    period: NDArray[np.int64]  # (U,) period index into `periods`
    periods: list[str]
    sessions: list[str]
    n: NDArray[np.float64]
    sz: NDArray[np.float64]
    szz: NDArray[np.float64]
    zz: NDArray[np.float64]
    zzm: NDArray[np.float64]
    labels: list[dict[str, NDArray[np.float64]]]
    reference: list[tuple[str, float]]  # prod's legs (name, signed weight)
    label_names: list[str] = field(default_factory=lambda: ["10s"])


@dataclass(frozen=True)
class Scored:
    value: float
    daily: dict[str, float]  # day -> value, for paired SEs on the client


Verdict = Literal["better", "worse", "flat", "not_scored"]


class Infra(Protocol):
    """One campaign on futs-infra. Methods block until their job is terminal."""

    def describe(self) -> Description: ...
    def budget(self) -> Budget: ...
    def recipes(self, where: dict[str, Any] | None = None) -> list[dict[str, str]]: ...
    def validate(self, recipes: list[dict[str, str]]) -> dict[str, str | None]: ...
    def train(self, recipes: list[dict[str, str]]) -> tuple[list[Model], dict[str, str]]: ...
    def models(self) -> list[Model]: ...
    def moments(self, block: Literal["fit", "validation"]) -> Moments: ...
    def score_many(self, sets: list[list[tuple[str, float]]], objective: str, cell: str | None = None) -> list[Scored]:
        """Each set on the validation days; `cell` ("<period>/<session>", session one of the moment
        sessions or "day") narrows them to one cell (infra API 1.2)."""
        ...

    def verdict(self, subject: list[tuple[str, float]], compare_to: list[tuple[str, float]]) -> Verdict: ...
    def test_report(
        self, subject: list[tuple[str, float]], compare_to: list[tuple[str, float]] | None = None
    ) -> dict[str, Any]:
        """The eval gate's Table 1 on the held-out test block: {"subject"|"compare_to": {period: {session:
        {"EVf@10s": value, ...}}}}, compare_to defaulting to the reference; a counted look."""
        ...

    def screen(self, columns: list[str], parent: list[tuple[str, float]]) -> dict[str, Any]: ...
    def pcorr(self, columns: list[str], parent: list[tuple[str, float]], cell: str | None = None) -> dict[str, Any]: ...
    def propose_feature_set(self, name: str, columns: list[str], base: str | None) -> dict[str, Any]: ...
    def close(self, outcome: Literal["no_candidate", "incomplete"], reason: str) -> None: ...


def check_version(d: Description) -> None:
    if not d.api_version.startswith(API_VERSIONS):
        raise ApiError("not_permitted", f"infra speaks {d.api_version}; this client {API_VERSIONS}")


def name_of(recipe: dict[str, str], axes: dict[str, list[str]]) -> str:
    """The recipe's name: its axis values in the grammar's axis order."""
    return ".".join(str(recipe[a]) for a in axes if a in recipe)
