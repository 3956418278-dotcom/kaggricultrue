"""The only public output contract of the intraday planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .state import Position


@dataclass(frozen=True)
class TurnDecision:
    worker_actions: tuple[tuple[object, ...], ...]
    market_orders: tuple[tuple[object, ...], ...] = ()


@dataclass(frozen=True)
class Realization:
    placements: Mapping[str, Position] = field(default_factory=dict)
    turns: tuple[TurnDecision, ...] = ()


class PlanningFailure(RuntimeError):
    """No complete, exactly executable realization was found for the fixed Plan."""

    def __init__(self, reason: str, *, diagnostics: Mapping[str, object] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = dict(diagnostics or {})
