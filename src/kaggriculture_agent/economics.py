"""Six-dimensional economic commitment representation.

The dimensions deliberately retain heterogeneous game structure.  Profit and density
metrics are derived views used after feasibility, never substitutes for the record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .state import Position


@dataclass(frozen=True)
class TimedAmount:
    step: int
    item: str
    quantity: int


@dataclass(frozen=True)
class TimedCash:
    step: int
    purpose: str
    amount: int


@dataclass(frozen=True)
class CashDimension:
    upfront: int = 0
    scheduled: tuple[TimedCash, ...] = ()
    reserve_required: int = 0
    sunk_cost: int = 0


@dataclass(frozen=True)
class TimeDimension:
    start_step: int
    completion_step: int
    last_value_step: int
    deadlines: tuple[int, ...] = ()


@dataclass(frozen=True)
class OccupancyInterval:
    position: Position | None
    start_step: int
    end_step: int


@dataclass(frozen=True)
class LandDimension:
    intervals: tuple[OccupancyInterval, ...] = ()
    capacity_created: int = 0

    @property
    def tile_days(self) -> float:
        return sum(
            max(0, interval.end_step - interval.start_step + 1) / 24
            for interval in self.intervals
        )


@dataclass(frozen=True)
class WorkAmount:
    day: int
    kind: str
    actions: int
    travel_actions: int = 0
    position: Position | None = None
    deadline_step: int | None = None


@dataclass(frozen=True)
class ActionDimension:
    work: tuple[WorkAmount, ...] = ()
    capacity_supplied: Mapping[int, int] = field(default_factory=dict)

    @property
    def service_actions(self) -> int:
        return sum(item.actions for item in self.work)

    @property
    def travel_actions(self) -> int:
        return sum(item.travel_actions for item in self.work)

    @property
    def total_actions(self) -> int:
        return self.service_actions + self.travel_actions


@dataclass(frozen=True)
class PhysicalDimension:
    inputs: tuple[TimedAmount, ...] = ()
    outputs: tuple[TimedAmount, ...] = ()
    peak_shed_units: int = 0


@dataclass(frozen=True)
class RevenueDimension:
    realized_now: int = 0
    projected_sales: tuple[TimedAmount, ...] = ()
    projected_gross: int = 0
    terminal_cash_effect: int = 0
    terminal_unsold_units: int = 0


@dataclass(frozen=True)
class EconomicCommitment:
    identifier: str
    kind: str
    target: Position | None
    existing: bool
    cash: CashDimension
    time: TimeDimension
    land: LandDimension
    actions: ActionDimension
    physical: PhysicalDimension
    revenue: RevenueDimension
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def terminal_profit(self) -> int:
        # sunk_cost is documentary only: existing assets are evaluated marginally.
        scheduled_cost = sum(max(0, flow.amount) for flow in self.cash.scheduled)
        return (
            self.revenue.realized_now
            + self.revenue.projected_gross
            + self.revenue.terminal_cash_effect
            - self.cash.upfront
            - scheduled_cost
        )

    @property
    def profit_per_action(self) -> float:
        return self.terminal_profit / max(1, self.actions.total_actions)

    @property
    def profit_per_tile_day(self) -> float:
        return self.terminal_profit / max(1.0, self.land.tile_days)
