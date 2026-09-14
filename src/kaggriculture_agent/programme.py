"""Frozen macro programme exchanged with the intraday executor."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

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
class CurrentAssetState:
    """Observation-backed state for an asset that already exists.

    ``official`` is the complete transition-bearing tile record.  The two event
    collections are deliberately short: executable commitments for the current
    day and the next biologically relevant event only.
    """

    asset_id: str
    asset_type: str
    tile: Position
    official: Mapping[str, Any]
    held_product: str | None = None
    held_quantity: int = 0
    mode: str | None = None              # PRODUCE / MAINTAIN / EXIT / GROW
    daily_value: float = 0.0
    care_approved: bool = False
    input_gain: int = 0
    next_production_step: int | None = None
    next_harvest_step: int | None = None
    today_events: tuple[ProgrammeEvent, ...] = ()
    next_events: tuple[ProgrammeEvent, ...] = ()
    liquidation_step: int | None = None
    physical_release_step: int | None = None
    exit_order: int | None = None

    @property
    def animal_mode(self) -> str | None:
        """Compatibility spelling used by plan inspection code."""
        return self.mode


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
    wheat_buffer: Mapping[Position, tuple[int, ...]] = field(default_factory=dict)
    carrot_buffer: Mapping[Position, tuple[int, ...]] = field(default_factory=dict)
    fertilizer: FertilizerProgramme = field(default_factory=FertilizerProgramme)
    land: tuple[LandProgramme, ...] = ()
    worker_count: int = 1
    inner: frozenset[Position] = frozenset()
    outer: frozenset[Position] = frozenset()
    placement_load: Mapping[Position, int] = field(default_factory=dict)
    return_mode: Mapping[str, str] = field(default_factory=dict)
    must_return: Mapping[int, frozenset[Position]] = field(default_factory=dict)
    return_reason: Mapping[int, Mapping[Position, str]] = field(default_factory=dict)
    partition_template_id: str | None = None
    zone_assignment: Mapping[Position, str] = field(default_factory=dict)
    routes: tuple[WorkerRoute, ...] = ()
    terminal_cash: int = 0
    feasible: bool = False
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    current_assets: tuple[CurrentAssetState, ...] = ()

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
