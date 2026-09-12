"""Frozen macro programme exchanged with the intraday executor."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from .state import Position

Action = tuple[object, ...]


@dataclass(frozen=True, order=True)
class ProgrammeEvent:
    step: int
    priority: int
    event_id: str
    kind: str
    tile: Position | None = None
    asset_id: str | None = None
    item: str | None = None
    quantity: int = 0
    action: Action = ("PASS",)
    mandatory: bool = True
    deadline: int | None = None
    prerequisite: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class AssetProgramme:
    asset_id: str
    asset_type: str
    tile: Position
    decision: str                    # KEEP / EXIT / NEW
    existing: bool
    purpose: str = "LONG"
    release_turn: int | None = None
    service_schedule: tuple[ProgrammeEvent, ...] = ()
    output_schedule: tuple[ProgrammeEvent, ...] = ()
    harvest_schedule: tuple[ProgrammeEvent, ...] = ()
    stock_schedule: tuple[ProgrammeEvent, ...] = ()
    sale_schedule: tuple[ProgrammeEvent, ...] = ()


@dataclass(frozen=True)
class LandProgramme:
    quadrant: str
    buy_step: int
    cost: int
    exact_tile: Position
    use: str


@dataclass(frozen=True)
class FertilizerProgramme:
    produced: Mapping[int, int] = field(default_factory=dict)
    bought: Mapping[int, int] = field(default_factory=dict)
    used: Mapping[int, int] = field(default_factory=dict)
    sold: Mapping[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkerRoute:
    worker: int
    day: int
    zone: str
    actions: Mapping[int, Action]


@dataclass(frozen=True)
class Programme:
    formed_step: int
    day: int
    shops: tuple[str, ...]
    assets: tuple[AssetProgramme, ...] = ()
    events: tuple[ProgrammeEvent, ...] = ()
    farm_output: Mapping[int, Mapping[str, int]] = field(default_factory=dict)
    field_stock: Mapping[int, Mapping[str, int]] = field(default_factory=dict)
    worker_stock: Mapping[int, Mapping[str, int]] = field(default_factory=dict)
    shed_stock: Mapping[int, Mapping[str, int]] = field(default_factory=dict)
    planned_sale: Mapping[int, Mapping[str, int]] = field(default_factory=dict)
    market_inventory: Mapping[int, Mapping[str, int]] = field(default_factory=dict)
    wheat_feed: Mapping[Position, tuple[int, ...]] = field(default_factory=dict)
    wheat_buffer: Mapping[Position, tuple[int, ...]] = field(default_factory=dict)
    carrot_buffer: Mapping[Position, tuple[int, ...]] = field(default_factory=dict)
    fertilizer: FertilizerProgramme = field(default_factory=FertilizerProgramme)
    land: tuple[LandProgramme, ...] = ()
    worker_count: int = 1
    inner: frozenset[Position] = frozenset()
    outer: frozenset[Position] = frozenset()
    placement_load: Mapping[Position, int] = field(default_factory=dict)
    return_mode: Mapping[str, str] = field(default_factory=dict)
    routes: tuple[WorkerRoute, ...] = ()
    terminal_cash: int = 0
    feasible: bool = False
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def with_events(self, *events: ProgrammeEvent) -> "Programme":
        return replace(self, events=tuple(sorted((*self.events, *events))))

    def events_at(self, step: int) -> tuple[ProgrammeEvent, ...]:
        return tuple(event for event in self.events if event.step == step)

    @property
    def keep(self) -> tuple[AssetProgramme, ...]:
        return tuple(a for a in self.assets if a.decision == "KEEP")

    @property
    def exits(self) -> tuple[AssetProgramme, ...]:
        return tuple(a for a in self.assets if a.decision == "EXIT")

    @property
    def additions(self) -> tuple[AssetProgramme, ...]:
        return tuple(a for a in self.assets if a.decision == "NEW")
