"""Fixed daily must-return labels consumed by intraday route selection."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Mapping

from . import rules
from .midgame_config import DEFAULT_MIDGAME_PARAMETERS, MidgameParameters
from .programme import Programme
from .state import Position, State


@dataclass(frozen=True)
class ReturnRequirements:
    day: int
    output_units: Mapping[Position, int]
    must_return: frozenset[Position]
    eod: frozenset[Position]
    reason: Mapping[Position, str]
    eod_units: int
    eod_capacity: int


def _harvest_units(programme: Programme, day: int) -> dict[Position, int]:
    units: Counter[Position] = Counter()
    for event in programme.events:
        if event.tile is None or event.step // rules.TURNS_PER_DAY != day:
            continue
        operation = str(event.action[0]) if event.action else "PASS"
        if operation == "HARVEST":
            units[event.tile] += max(1, int(event.quantity))
        elif operation == "COLLECT_FERTILIZER":
            units[event.tile] += 1
    return dict(units)


def return_requirements(
    state: State,
    programme: Programme,
    *,
    day: int | None = None,
    eod_capacity: int | None = None,
    params: MidgameParameters = DEFAULT_MIDGAME_PARAMETERS,
) -> ReturnRequirements:
    """Choose the farthest indivisible tile bundles that may remain EOD.

    D1-D10 additionally force harvested output at shed distance <= 3 to return.
    Every other forced return is a shed-capacity consequence.  This function
    labels bundles only; a route still has to prove DROP by turn 22.
    """
    selected_day = state.day if day is None else day
    output = _harvest_units(programme, selected_day)
    if eod_capacity is None:
        opening = sum(max(0, int(quantity)) for quantity in state.shed.values())
        eod_capacity = max(0, rules.SHED_CAPACITY - opening)
    else:
        eod_capacity = max(0, min(rules.SHED_CAPACITY, int(eod_capacity)))

    reason: dict[Position, str] = {}
    forced = set()
    # Explicit same-day relays are physical dependencies, not an economic
    # guess: their product must reach the shed before its downstream pickup.
    for event in programme.events:
        if (event.tile in output
                and event.step // rules.TURNS_PER_DAY == selected_day
                and event.source in {"FEED_SUPPLY", "F_RELAY"}):
            forced.add(event.tile)
            reason[event.tile] = str(event.source)
    # There is no end-of-day refresh after the final actionable turn.
    if selected_day == rules.TERMINAL_ACTION_STEP // rules.TURNS_PER_DAY:
        for tile in output:
            forced.add(tile)
            reason[tile] = "TERMINAL_LIQUIDATION"
    if selected_day <= params.early_return_last_day:
        for tile in output:
            if rules.distance_to_shed(tile, state.board_size) \
                    <= params.early_return_distance:
                forced.add(tile)
                reason[tile] = "EARLY_NEAR_SHED"

    candidates = sorted(
        (tile for tile in output if tile not in forced),
        key=lambda tile: (
            -rules.distance_to_shed(tile, state.board_size),
            tile[1], tile[0]))
    eod: set[Position] = set()
    eod_units = 0
    capacity_reached = False
    for tile in candidates:
        quantity = output[tile]
        if not capacity_reached and eod_units + quantity <= eod_capacity:
            eod.add(tile)
            eod_units += quantity
        else:
            capacity_reached = True
            forced.add(tile)
            reason[tile] = "EOD_CAPACITY"

    return ReturnRequirements(
        selected_day, output, frozenset(forced), frozenset(eod), reason,
        eod_units, eod_capacity)


def annotate_programme_returns(
    state: State,
    programme: Programme,
    *,
    params: MidgameParameters = DEFAULT_MIDGAME_PARAMETERS,
) -> Programme:
    days = sorted({event.step // rules.TURNS_PER_DAY
                   for event in programme.events})
    labels = {day: return_requirements(
        state, programme, day=day, params=params) for day in days}
    return replace(
        programme,
        must_return={day: value.must_return for day, value in labels.items()},
        return_reason={day: value.reason for day, value in labels.items()},
    )


def mark_tile_workloads(workloads, labels: ReturnRequirements):
    """Copy fixed must-return labels onto zonal ``TileWorkload`` objects."""
    return {
        tile: replace(workload, must_return=tile in labels.must_return)
        for tile, workload in workloads.items()
    }
