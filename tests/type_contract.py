"""Pyright checks both implementations without constructing a live campaign."""

from alphasearch.infra.contract import Infra
from alphasearch.infra.fake import FakeInfra
from alphasearch.infra.futs import FutsInfra


def fake() -> Infra:
    return FakeInfra()


def installed(campaign: str) -> Infra:
    return FutsInfra(campaign)
