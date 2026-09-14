"""Authored fixed-road templates for two-, three-, and four-land farms.

There is deliberately no partition generator here. Each zone is authored as
an entry tile followed by a literal movement string. That ordered simple path
is both its ownership map and normal traversal. Every layout has only a few
fixed operating versions whose return zones and return closures are known in
advance; there is no all-subset return mask.

At runtime inactive tiles are removed from the authored road. The remaining
task chains are joined by deterministic shortest Manhattan connectors without
revisiting a traversed cell. This connector never changes ownership or the
zone's fixed return closure. Exact movement, preload, tile actions, DROP,
turn-22 return, and 24-turn feasibility are checked before selection.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from . import rules
from .state import Position

LandMask = tuple[str, ...]
NW: LandMask = ("NW",)
NORTH: LandMask = ("NW", "NE")
THREE_LAND: LandMask = ("NW", "NE", "SW")
FULL: LandMask = ("NW", "NE", "SW", "SE")
_DELTAS = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}


@dataclass(frozen=True)
class TileWorkload:
    actions: int
    output_units: int = 0
    pickup_items: frozenset[str] = frozenset()
    must_return: bool = False


@dataclass(frozen=True)
class ZoneRoad:
    zone_id: str
    road: tuple[Position, ...]
    entry: Position
    shed_access: Position
    returns_to_shed: bool
    action_limit_by_tile: Mapping[Position, int]
    certified_return_workload: Mapping[Position, TileWorkload]


@dataclass(frozen=True)
class PartitionTemplate:
    template_id: str
    road_layout_id: str
    owned_land_mask: LandMask
    worker_count: int
    family: str
    workload_mode: str
    return_mode: str
    zones: Mapping[str, ZoneRoad]
    tile_to_zone: Mapping[Position, str]

    def render(self) -> str:
        return "\n".join("".join(
            self.tile_to_zone.get((x, y), ".")
            for x in range(rules.BOARD_SIZE))
            for y in range(rules.BOARD_SIZE))


@dataclass(frozen=True)
class CompiledZoneRoute:
    zone_id: str
    positions: tuple[Position, ...]
    task_order: tuple[Position, ...]
    movement_turns: int
    preload_turns: int
    task_turns: int
    drop_turn: int | None
    finish_turn: int
    returned_units: int


@dataclass(frozen=True)
class TemplateSolution:
    template: PartitionTemplate
    routes: Mapping[str, CompiledZoneRoute]
    total_movement: int
    max_finish_turn: int
    returned_units: int


@dataclass(frozen=True)
class _Zone:
    zone_id: str
    start: Position
    moves: str


@dataclass(frozen=True)
class _Layout:
    layout_id: str
    mask: LandMask
    worker_count: int
    family: str
    workload_mode: str
    action_limit: int
    zones: tuple[_Zone, ...]


def _walk(start: Position, moves: str) -> tuple[Position, ...]:
    """Materialize one literal road; digits repeat the preceding move."""
    result = [start]
    index = 0
    while index < len(moves):
        token = moves[index]
        if token.isspace():
            index += 1
            continue
        if token not in _DELTAS:
            raise ValueError(f"invalid road token {token!r}")
        index += 1
        digits = []
        while index < len(moves) and moves[index].isdigit():
            digits.append(moves[index]); index += 1
        count = int("".join(digits)) if digits else 1
        dx, dy = _DELTAS[token]
        for _ in range(count):
            x, y = result[-1]
            result.append((x + dx, y + dy))
    return tuple(result)


def _owned_tiles(mask: LandMask) -> frozenset[Position]:
    return frozenset((x, y) for y in range(rules.BOARD_SIZE)
                     for x in range(rules.BOARD_SIZE)
                     if rules.quadrant((x, y)) in mask)


def _nearest_access(entry: Position) -> Position:
    return min(rules.shed_access(), key=lambda p: (
        rules.manhattan(p, entry), p[1], p[0]))


_LAYOUT_RETURN_VARIANTS: Mapping[
    str, tuple[tuple[str, frozenset[str]], ...]
] = MappingProxyType({
    # Final actionable day: every harvested unit must reach the shed because
    # there is no subsequent end-of-day inventory refresh.
    "nw-w12-strips-v": (("TERMINAL_RETURN", frozenset("ABCDEFGHIJKL")),),
    "north-w12-qa": (("TERMINAL_RETURN", frozenset("ABCDEFGHIJKL")),),
    "three-w12-qa": (("TERMINAL_RETURN", frozenset("ABCDEFGHIJKL")),),
    # These are fixed workload-mode variants, not masks selected from today's
    # state.  The first separates the two near-side roads needed when eighteen
    # far-corner Melons would otherwise leave 108 units for EOD collection.
    "three-w12-qg": (("MELON_OVERFLOW_RETURN", frozenset(("G", "H"))),),
    "north-w12-cross-spine": (
        ("HOTSPOT_RETURN", frozenset("FGKL")),
        ("TERMINAL_RETURN", frozenset("ABCDEFGHIJKL"))),
    "north-w9-cross-spine": (
        ("TERMINAL_RETURN", frozenset("ABCDEFGHI")),),
    "north-w10-cross-spine": (
        ("TERMINAL_RETURN", frozenset("ABCDEFGHIJ")),),
    "north-w11-cross-spine": (
        ("TERMINAL_RETURN", frozenset("ABCDEFGHIJK")),),
    "three-w12-qd": (("HOTSPOT_RETURN", frozenset("HK")),),
    "three-w12-qi": (("HOTSPOT_RETURN", frozenset("GKL")),),
    "three-w12-qj": (("HOTSPOT_RETURN", frozenset("AEJK")),),
    "three-w12-qk": (("HOTSPOT_RETURN", frozenset("CFHIJ")),),
    "three-w12-ql": (("HOTSPOT_RETURN", frozenset("HIJ")),),
    "three-w12-qn": (("CORE_RETURN", frozenset("BCDFGHJKL")),),
    "three-w12-qo": (("CORE_RETURN", frozenset("DEFHIKL")),),
    "three-w12-qp": (
        ("CORE_RETURN", frozenset("BCGHIKL")),
        ("DENSE_CORE_RETURN", frozenset("HIK")),
    ),
    "three-w12-qq": (("CORE_RETURN", frozenset("BCEFJKL")),),
    # These two variants cover a denser 32-tile harvest wave while leaving the
    # farthest authored roads for EOD collection.
    "three-w12-qc": (("HARVEST_WAVE_RETURN",
                       frozenset("BCDFGHIJL")),),
    "full-w12-qc": (("HARVEST_WAVE_RETURN",
                      frozenset("BCEFGHIJKL")),),
})


def _make_layout(layout: _Layout) -> tuple[PartitionTemplate, ...]:
    """Validate one road layout and attach bounded fixed return versions."""
    paths = {zone.zone_id: _walk(zone.start, zone.moves)
             for zone in layout.zones}
    all_tiles = [tile for path in paths.values() for tile in path]
    if len(all_tiles) != len(set(all_tiles)):
        raise ValueError(f"{layout.layout_id}: zone roads overlap or self-repeat")
    expected = _owned_tiles(layout.mask)
    if set(all_tiles) != set(expected):
        raise ValueError(
            f"{layout.layout_id}: coverage mismatch "
            f"missing={sorted(expected-set(all_tiles))} "
            f"extra={sorted(set(all_tiles)-expected)}")
    for zone_id, road in paths.items():
        if any(rules.manhattan(a, b) != 1 for a, b in zip(road, road[1:])):
            raise ValueError(f"{layout.layout_id}:{zone_id}: road jumps")
        if rules.distance_to_shed(road[0]) != min(
                rules.distance_to_shed(tile) for tile in road):
            raise ValueError(
                f"{layout.layout_id}:{zone_id}: entry is not nearest to shed")

    ranked = tuple(sorted(paths, key=lambda zone_id: (
        min(rules.distance_to_shed(tile) for tile in paths[zone_id]),
        sum(rules.distance_to_shed(tile) for tile in paths[zone_id]),
        max(rules.distance_to_shed(tile) for tile in paths[zone_id]),
        zone_id,
    )))
    narrow = max(1, (layout.worker_count + 2) // 3)
    wide = max(narrow + 1, (2 * layout.worker_count + 2) // 3)
    proposed_modes = (
        ("EOD", frozenset()),
        ("NARROW_RETURN", frozenset(ranked[:narrow])),
        ("WIDE_RETURN", frozenset(ranked[:wide])),
        *_LAYOUT_RETURN_VARIANTS.get(layout.layout_id, ()),
    )
    modes = []
    seen_return_sets = set()
    for mode in proposed_modes:
        if mode[1] not in seen_return_sets:
            modes.append(mode)
            seen_return_sets.add(mode[1])
    templates = []
    tile_to_zone = MappingProxyType({
        tile: zone_id for zone_id, road in paths.items() for tile in road
    })
    for mode, return_zones in modes:
        zones = {}
        for zone_id, road in paths.items():
            certificate = {
                tile: TileWorkload(1, 6, must_return=True)
                for tile in road[:min(4, len(road))]
            }
            zones[zone_id] = ZoneRoad(
                zone_id, road, road[0], _nearest_access(road[0]),
                zone_id in return_zones,
                MappingProxyType({tile: layout.action_limit for tile in road}),
                MappingProxyType(certificate))
        templates.append(PartitionTemplate(
            f"{layout.layout_id}-{mode.lower()}", layout.layout_id,
            layout.mask, layout.worker_count, layout.family,
            layout.workload_mode, mode, MappingProxyType(zones), tile_to_zone))
    return tuple(templates)


# Literal movement strings are the maintained source of truth. Higher-worker
# layouts are deliberately the denser-work alternatives.
_LAYOUTS: tuple[_Layout, ...] = (
    # Two land: 10x5 northern farm.
    _Layout("north-w3-edge", NORTH, 3, "EDGE_BANDS", "STAGGERED", 2, (
        _Zone("A", (2, 4), "N4 W2 S E S W S E S W"),
        _Zone("B", (5, 4), "N4 W2 S E S W S E S W"),
        _Zone("C", (6, 4), "N4 E3 S W2 S E2 S2 W N W S"))),
    _Layout("north-w4-cross", NORTH, 4, "COMPACT_CROSS", "ORDINARY", 3, (
        _Zone("A", (4, 1), "N W S W N W2 S E"),
        _Zone("B", (5, 1), "N E S E N E2 S W"),
        _Zone("C", (4, 4), "N2 W4 S2 E N E2 S W"),
        _Zone("D", (5, 4), "N2 E4 S2 W N W2 S E"))),
    _Layout("north-w5-columns", NORTH, 5, "EDGE_COLUMNS", "DENSE", 4, (
        _Zone("A", (1, 4), "W N E N W N2 E S"),
        _Zone("B", (3, 4), "W N E N W N2 E S"),
        _Zone("C", (4, 4), "E N W N E N2 W S"),
        _Zone("D", (6, 4), "E N W N E N2 W S"),
        _Zone("E", (8, 4), "E N W N E N2 W S"))),
    _Layout("north-w6-blocks", NORTH, 6, "SHORT_BLOCKS", "PEAK", 4, (
        _Zone("A", (2, 1), "N W2 S E"),
        _Zone("B", (4, 1), "W N E2 S"),
        _Zone("C", (6, 1), "N E S E N E S"),
        _Zone("D", (2, 4), "N2 W2 S E S W"),
        _Zone("E", (5, 4), "N2 W2 S E S W"),
        _Zone("F", (6, 4), "N2 E3 S2 W N W S"))),
    # Whole-north line systems.  The central road crosses NW/NE; the side
    # roads are compact simple paths rather than independent 5x5 partitions.
    _Layout("north-w3-cross-spine", NORTH, 3, "CROSS_SPINE", "STAGGERED", 2, (
        _Zone("A", (3, 4), "W3 N E3 N W3 N E3 N W3"),
        _Zone("B", (4, 4), "N4 E S4"),
        _Zone("C", (6, 4), "E3 N W3 N E3 N W3 N E3"))),
    _Layout("north-w4-cross-spine", NORTH, 4, "CROSS_SPINE", "ORDINARY", 3, (
        _Zone("A", (3, 4), "W3 N E3 N W3 N E3 N W3"),
        _Zone("B", (4, 4), "N E S"),
        _Zone("C", (4, 2), "N2 E S2"),
        _Zone("D", (6, 4), "E3 N W3 N E3 N W3 N E3"))),
    _Layout("north-w5-cross-spine", NORTH, 5, "CROSS_SPINE", "ORDINARY", 3, (
        _Zone("A", (3, 2), "W3 N E3 N W3"),
        _Zone("B", (3, 4), "W3 N E3"),
        _Zone("C", (4, 4), "N4 E S4"),
        _Zone("D", (6, 2), "E3 N W3 N E3"),
        _Zone("E", (6, 4), "E3 N W3"))),
    _Layout("north-w6-cross-spine", NORTH, 6, "CROSS_SPINE", "DENSE", 4, (
        _Zone("A", (3, 2), "W3 N E3 N W3"),
        _Zone("B", (3, 4), "W3 N E3"),
        _Zone("C", (4, 4), "N E S"),
        _Zone("D", (4, 2), "N2 E S2"),
        _Zone("E", (6, 2), "E3 N W3 N E3"),
        _Zone("F", (6, 4), "E3 N W3"))),
    _Layout("north-w7-cross-spine", NORTH, 7, "CROSS_SPINE", "HIGH_VARIANCE", 5, (
        _Zone("A", (3, 1), "N W3 S E2"),
        _Zone("B", (1, 4), "W N E N W"),
        _Zone("C", (3, 4), "W N E N W"),
        _Zone("D", (4, 4), "N4 E S4"),
        _Zone("E", (6, 1), "N E3 S W2"),
        _Zone("F", (8, 4), "E N W N E"),
        _Zone("G", (6, 4), "E N W N E"))),
    _Layout("north-w8-cross-spine", NORTH, 8, "CROSS_SPINE", "HIGH_VARIANCE", 5, (
        _Zone("A", (3, 1), "N W3 S E2"),
        _Zone("B", (1, 4), "W N E N W"),
        _Zone("C", (3, 4), "W N E N W"),
        _Zone("D", (4, 4), "N E S"),
        _Zone("E", (4, 2), "N2 E S2"),
        _Zone("F", (6, 1), "N E3 S W2"),
        _Zone("G", (8, 4), "E N W N E"),
        _Zone("H", (6, 4), "E N W N E"))),
    _Layout("north-w9-cross-spine", NORTH, 9, "CROSS_SPINE", "HIGH_VARIANCE", 5, (
        _Zone("A", (0, 4), "N4"), _Zone("B", (1, 4), "N4"),
        _Zone("C", (2, 4), "N4"), _Zone("D", (3, 4), "N4"),
        _Zone("E", (4, 4), "N4 E S4"),
        _Zone("F", (6, 4), "N4"), _Zone("G", (7, 4), "N4"),
        _Zone("H", (8, 4), "N4"), _Zone("I", (9, 4), "N4"))),
    _Layout("north-w10-cross-spine", NORTH, 10, "CROSS_SPINE", "HIGH_VARIANCE", 5, (
        _Zone("A", (0, 4), "N4"), _Zone("B", (1, 4), "N4"),
        _Zone("C", (2, 4), "N4"), _Zone("D", (3, 4), "N4"),
        _Zone("E", (4, 4), "N E S"), _Zone("F", (4, 2), "N2 E S2"),
        _Zone("G", (6, 4), "N4"), _Zone("H", (7, 4), "N4"),
        _Zone("I", (8, 4), "N4"), _Zone("J", (9, 4), "N4"))),
    _Layout("north-w11-cross-spine", NORTH, 11, "CROSS_SPINE", "HIGH_VARIANCE", 5, (
        _Zone("A", (3, 0), "W3"), _Zone("B", (3, 1), "W3"),
        _Zone("C", (3, 2), "W3"), _Zone("D", (3, 3), "W3"),
        _Zone("E", (3, 4), "W3"), _Zone("F", (4, 4), "N4 E S4"),
        _Zone("G", (6, 0), "E3"), _Zone("H", (6, 1), "E3"),
        _Zone("I", (6, 2), "E3"), _Zone("J", (6, 3), "E3"),
        _Zone("K", (6, 4), "E3"))),
    _Layout("north-w12-cross-spine", NORTH, 12, "CROSS_SPINE", "HIGH_VARIANCE", 5, (
        _Zone("A", (3, 0), "W3"), _Zone("B", (3, 1), "W3"),
        _Zone("C", (3, 2), "W3"), _Zone("D", (3, 3), "W3"),
        _Zone("E", (3, 4), "W3"), _Zone("F", (4, 4), "N E S"),
        _Zone("G", (4, 2), "N2 E S2"), _Zone("H", (6, 0), "E3"),
        _Zone("I", (6, 1), "E3"), _Zone("J", (6, 2), "E3"),
        _Zone("K", (6, 3), "E3"), _Zone("L", (6, 4), "E3"))),

    # Three-land top-spine systems keep NW/NE as one authored line system and
    # attach compact southwest roads; they are not quadrant-by-quadrant maps.
    _Layout("three-w5-top-spine", THREE_LAND, 5, "TOP_CROSS_SPINE", "DENSE", 4, (
        _Zone("A", (3, 4), "W3 N E3 N W3 N E3 N W3"),
        _Zone("B", (4, 4), "N4 E S4"),
        _Zone("C", (6, 4), "E3 N W3 N E3 N W3 N E3"),
        _Zone("D", (4, 5), "W S E S W S E S W2 N2"),
        _Zone("E", (2, 5), "S W N W S2 E S W S E"))),
    _Layout("three-w6-top-spine", THREE_LAND, 6, "TOP_CROSS_SPINE", "DENSE", 4, (
        _Zone("A", (3, 4), "W3 N E3 N W3 N E3 N W3"),
        _Zone("B", (4, 4), "N E S"),
        _Zone("C", (4, 2), "N2 E S2"),
        _Zone("D", (6, 4), "E3 N W3 N E3 N W3 N E3"),
        _Zone("E", (4, 5), "W S E S W S E S W2 N2"),
        _Zone("F", (2, 5), "S W N W S2 E S W S E"))),
    _Layout("three-w7-top-spine", THREE_LAND, 7, "TOP_CROSS_SPINE", "HIGH_VARIANCE", 5, (
        _Zone("A", (3, 4), "W3 N E3 N W3 N E3 N W3"),
        _Zone("B", (4, 4), "N E S"),
        _Zone("C", (4, 2), "N2 E S2"),
        _Zone("D", (6, 4), "E3 N W3 N E3 N W3 N E3"),
        _Zone("E", (2, 7), "W2 S2 E2 N W"),
        _Zone("F", (4, 6), "W S E S W S E"),
        _Zone("G", (4, 5), "W4 S E2"))),
    _Layout("three-w8-top-spine", THREE_LAND, 8, "TOP_CROSS_SPINE", "HIGH_VARIANCE", 5, (
        _Zone("A", (3, 2), "W3 N E3 N W3"),
        _Zone("B", (3, 4), "W3 N E3"),
        _Zone("C", (4, 4), "N4 E S4"),
        _Zone("D", (6, 2), "E3 N W3 N E3"),
        _Zone("E", (6, 4), "E3 N W3"),
        _Zone("F", (2, 7), "W2 S2 E2 N W"),
        _Zone("G", (4, 6), "W S E S W S E"),
        _Zone("H", (4, 5), "W4 S E2"))),
    _Layout("three-w9-top-spine", THREE_LAND, 9, "TOP_CROSS_SPINE", "HIGH_VARIANCE", 5, (
        _Zone("A", (3, 2), "W3 N E3 N W3"),
        _Zone("B", (3, 4), "W3 N E3"),
        _Zone("C", (4, 4), "N E S"),
        _Zone("D", (4, 2), "N2 E S2"),
        _Zone("E", (6, 2), "E3 N W3 N E3"),
        _Zone("F", (6, 4), "E3 N W3"),
        _Zone("G", (4, 5), "W2 S E2 S W2"),
        _Zone("H", (4, 8), "S W3 N E2"),
        _Zone("I", (1, 5), "W S E S W S2"))),
    _Layout("three-w10-top-spine", THREE_LAND, 10, "TOP_CROSS_SPINE", "HIGH_VARIANCE", 5, (
        _Zone("A", (3, 2), "W3 N E3 N W3"),
        _Zone("B", (3, 4), "W3 N E3"),
        _Zone("C", (4, 4), "N E S"),
        _Zone("D", (4, 2), "N2 E S2"),
        _Zone("E", (6, 2), "E3 N W3 N E3"),
        _Zone("F", (6, 4), "E3 N W3"),
        _Zone("G", (2, 7), "W2 S2 E N E S"),
        _Zone("H", (4, 7), "W S E S W"),
        _Zone("I", (2, 5), "S W N W S"),
        _Zone("J", (4, 5), "W S E"))),
    _Layout("three-w11-top-spine", THREE_LAND, 11, "TOP_CROSS_SPINE", "HIGH_VARIANCE", 5, (
        _Zone("A", (3, 1), "N W3 S E2"),
        _Zone("B", (1, 4), "W N E N W"),
        _Zone("C", (3, 4), "W N E N W"),
        _Zone("D", (4, 4), "N4 E S4"),
        _Zone("E", (6, 1), "N E3 S W2"),
        _Zone("F", (8, 4), "E N W N E"),
        _Zone("G", (6, 4), "E N W N E"),
        _Zone("H", (2, 7), "W2 S2 E N E S"),
        _Zone("I", (4, 7), "W S E S W"),
        _Zone("J", (2, 5), "S W N W S"),
        _Zone("K", (4, 5), "W S E"))),
    _Layout("three-w12-top-spine", THREE_LAND, 12, "TOP_CROSS_SPINE", "HIGH_VARIANCE", 5, (
        _Zone("A", (3, 1), "N W3 S E2"),
        _Zone("B", (1, 4), "W N E N W"),
        _Zone("C", (3, 4), "W N E N W"),
        _Zone("D", (4, 4), "N E S"),
        _Zone("E", (4, 2), "N2 E S2"),
        _Zone("F", (6, 1), "N E3 S W2"),
        _Zone("G", (8, 4), "E N W N E"),
        _Zone("H", (6, 4), "E N W N E"),
        _Zone("I", (2, 7), "W2 S2 E N E S"),
        _Zone("J", (4, 7), "W S E S W"),
        _Zone("K", (2, 5), "S W N W S"),
        _Zone("L", (4, 5), "W S E"))),
    _Layout("three-w12-ne-cross", THREE_LAND, 12, "NE_HOT_CROSS", "HIGH_VARIANCE", 5, (
        _Zone("A", (3, 2), "W3 N E3 N W3"),
        _Zone("B", (3, 4), "W3 N E3"),
        _Zone("C", (4, 4), "N4 E S4"),
        _Zone("D", (6, 0), "E3"), _Zone("E", (6, 1), "E3"),
        _Zone("F", (6, 2), "E3"), _Zone("G", (6, 3), "E3"),
        _Zone("H", (6, 4), "E"), _Zone("I", (8, 4), "E"),
        _Zone("J", (2, 7), "W2 S2 E2 N W"),
        _Zone("K", (4, 6), "W S E S W S E"),
        _Zone("L", (4, 5), "W4 S E2"))),
    _Layout("three-w12-nw-cross", THREE_LAND, 12, "NW_HOT_CROSS", "HIGH_VARIANCE", 5, (
        _Zone("A", (3, 0), "W3"), _Zone("B", (3, 1), "W3"),
        _Zone("C", (3, 2), "W3"), _Zone("D", (3, 3), "W3"),
        _Zone("E", (1, 4), "W"), _Zone("F", (3, 4), "W"),
        _Zone("G", (4, 4), "N4 E S4"),
        _Zone("H", (6, 2), "E3 N W3 N E3"),
        _Zone("I", (6, 4), "E3 N W3"),
        _Zone("J", (2, 7), "W2 S2 E2 N W"),
        _Zone("K", (4, 6), "W S E S W S E"),
        _Zone("L", (4, 5), "W4 S E2"))),
    _Layout("three-w12-sw-cross", THREE_LAND, 12, "SW_HOT_CROSS", "HIGH_VARIANCE", 5, (
        _Zone("A", (3, 2), "W3 N E3 N W3"),
        _Zone("B", (3, 4), "W3 N E3"),
        _Zone("C", (4, 4), "N4 E S4"),
        _Zone("D", (6, 2), "E3 N W3 N E3"),
        _Zone("E", (6, 4), "E3 N W3"),
        _Zone("F", (1, 8), "W S E"),
        _Zone("G", (3, 8), "W S E"),
        _Zone("H", (4, 8), "S"),
        _Zone("I", (4, 7), "W4"),
        _Zone("J", (1, 5), "W S E"),
        _Zone("K", (3, 5), "W S E"),
        _Zone("L", (4, 5), "S"))),

    # Three land. The six-worker line is the explicit corner/edge/skeleton case.
    _Layout("three-w5-bands", THREE_LAND, 5, "EDGE_AND_WEST", "STAGGERED", 2, (
        _Zone("A", (2, 4), "N4 W2 S E S W S E S W"),
        _Zone("B", (5, 4), "N4 W2 S E S W S E S W"),
        _Zone("C", (6, 4), "N4 E3 S W2 S E2 S2 W N W S"),
        _Zone("D", (4, 5), "S W N W S W2 N E"),
        _Zone("E", (4, 7), "W4 S2 E N E S E N E S"))),
    _Layout("three-w6-corner", THREE_LAND, 6, "CORNER_EDGE_SKELETON", "ORDINARY", 3, (
        _Zone("A", (2, 2), "W2 N2 E2 S W"),
        _Zone("B", (4, 2), "W N2 E S E N E S2 W"),
        _Zone("C", (7, 4), "E2 N W2 N E2 N2 W2 S E"),
        _Zone("I", (4, 4), "E2 N W3 S2 E S W"),
        _Zone("D", (2, 4), "N W2 S E S W S E2 N"),
        _Zone("E", (4, 7), "W4 S2 E N E S E2 N W"))),
    _Layout("three-w7-refined", THREE_LAND, 7, "CORNER_REFINEMENT", "DENSE", 4, (
        _Zone("A", (1, 2), "W N2 E S"),
        _Zone("B", (4, 2), "N2 W2 S E S W"),
        _Zone("C", (5, 2), "N2 E2 S W S E"),
        _Zone("D", (8, 4), "E N W N E N2 W S"),
        _Zone("I", (4, 4), "E3 N W4 S2 E S W S E S W S E"),
        _Zone("E", (2, 4), "N W2 S E S W S E2 N"),
        _Zone("F", (2, 7), "W2 S2 E N E S"))),
    _Layout("three-w8-peak", THREE_LAND, 8, "SHORT_EDGE_BLOCKS", "PEAK", 4, (
        _Zone("A", (1, 2), "W N2 E S"),
        _Zone("B", (4, 2), "N2 W2 S E S W"),
        _Zone("C", (5, 2), "N2 E2 S W S E"),
        _Zone("D", (8, 4), "E N W N E N2 W S"),
        _Zone("E", (2, 4), "N W2 S E S W S E2 N"),
        _Zone("F", (2, 7), "W2 S2 E N E S"),
        _Zone("I", (4, 5), "W S E S W S2 E N"),
        _Zone("J", (4, 4), "W N E2 S E N E S"))),
    _Layout("three-w9-max", THREE_LAND, 9, "MAX_DENSITY", "PEAK", 4, (
        _Zone("A", (1, 2), "W N2 E S"),
        _Zone("B", (4, 2), "N2 W2 S E S W"),
        _Zone("C", (5, 2), "N2 E2 S W S E"),
        _Zone("D", (8, 4), "E N W N E N2 W S"),
        _Zone("E", (2, 4), "N W2 S E S W S E2 N"),
        _Zone("F", (2, 7), "W2 S2 E N E S"),
        _Zone("G", (4, 7), "W S2 E N"),
        _Zone("I", (4, 5), "W S E"),
        _Zone("J", (4, 4), "W N E2 S E N E S"))),

    # Four land: whole-map line systems from ordinary to maximum load.
    _Layout("full-w6-bands", FULL, 6, "WHOLE_FARM_BANDS", "STAGGERED", 2, (
        _Zone("A", (2, 4), "N4 W2 S E S W S E S W"),
        _Zone("B", (5, 4), "N4 W2 S E S W S E S W"),
        _Zone("C", (6, 4), "N4 E3 S W2 S E2 S2 W N W S"),
        _Zone("D", (2, 5), "W2 S4 E2 N W N2 E S"),
        _Zone("E", (5, 5), "S4 W2 N E N W N2 E S"),
        _Zone("F", (6, 5), "E3 S4 W N3 W2 S E S2 W N"))),
    _Layout("full-w7-offset", FULL, 7, "OFFSET_EDGE_BLOCKS", "ORDINARY", 3, (
        _Zone("A", (2, 4), "N4 W2 S E S W S E S W"),
        _Zone("B", (5, 4), "N4 W2 S E S W S E S W"),
        _Zone("C", (6, 4), "N4 E3 S W2 S E2 S2 W N W S"),
        _Zone("D", (1, 5), "W S E S W S2 E N"),
        _Zone("E", (4, 5), "W2 S4 E2 N W N2 E S"),
        _Zone("F", (5, 5), "E S W S E S2 W N"),
        _Zone("G", (7, 5), "E2 S4 W2 N E N2 W S"))),
    _Layout("full-w8-grid", FULL, 8, "EIGHT_COMPACT_ROADS", "DENSE", 3, (
        _Zone("A", (1, 4), "W N E N W N2 E S"),
        _Zone("B", (3, 4), "W N E N W N2 E S"),
        _Zone("C", (4, 4), "E N W N E N2 W S"),
        _Zone("D", (6, 4), "N4 E3 S W2 S E2 S2 W N W S"),
        _Zone("E", (1, 5), "W S E S W S2 E N"),
        _Zone("F", (3, 5), "W S E S W S2 E N"),
        _Zone("G", (4, 5), "E S W S E S2 W N"),
        _Zone("H", (6, 5), "E3 S4 W N3 W2 S E S2 W N"))),
    _Layout("full-w9-grid", FULL, 9, "NINE_SHORT_ROADS", "DENSE", 4, (
        _Zone("A", (2, 2), "N2 W2 S E S W"),
        _Zone("B", (5, 2), "N2 W2 S E S W"),
        _Zone("C", (6, 2), "N2 E3 S2 W N W S"),
        _Zone("D", (2, 5), "W2 N2 E2 S W"),
        _Zone("E", (4, 4), "N W S2 E2 N2"),
        _Zone("F", (6, 4), "N E3 S2 W N W S W"),
        _Zone("G", (2, 6), "W2 S3 E2 N W N E"),
        _Zone("H", (4, 6), "W S3 E2 N W N E N"),
        _Zone("I", (6, 6), "E3 S3 W N2 W2 S E S W"))),
    _Layout("full-w10-columns", FULL, 10, "TEN_COLUMNS", "PEAK", 4, (
        _Zone("A", (1, 4), "W N E N W N2 E S"),
        _Zone("B", (3, 4), "W N E N W N2 E S"),
        _Zone("C", (4, 4), "E N W N E N2 W S"),
        _Zone("D", (6, 4), "E N W N E N2 W S"),
        _Zone("E", (8, 4), "E N W N E N2 W S"),
        _Zone("F", (1, 5), "W S E S W S2 E N"),
        _Zone("G", (3, 5), "W S E S W S2 E N"),
        _Zone("H", (4, 5), "E S W S E S2 W N"),
        _Zone("I", (6, 5), "E S W S E S2 W N"),
        _Zone("J", (8, 5), "E S W S E S2 W N"))),
    _Layout("full-w11-quadrants", FULL, 11, "QUADRANT_REFINEMENT", "PEAK", 4, (
        _Zone("A", (4, 1), "N W S W N W2 S E"),
        _Zone("B", (1, 4), "W N2 E S"),
        _Zone("C", (4, 4), "N2 W2 S E S W"),
        _Zone("D", (5, 1), "N E S E N E2 S W"),
        _Zone("E", (5, 4), "E N2 W S"),
        _Zone("F", (7, 4), "N2 E2 S W S E"),
        _Zone("G", (1, 5), "W S2 E N"),
        _Zone("H", (4, 5), "W2 S2 E N E S"),
        _Zone("I", (4, 8), "S W N W S W2 N E"),
        _Zone("J", (5, 5), "S E N E S E2 N W"),
        _Zone("K", (5, 7), "E4 S2 W N W S W N W S"))),
)


_LOCAL_ROADS: Mapping[tuple[int, str], tuple[_Zone, ...]] = MappingProxyType({
    (1, "A"): (_Zone("A", (4, 4), "W4 N E4 N W4 N E4 N W4"),),
    (1, "B"): (_Zone("A", (4, 4), "N4 W S4 W N4 W S4 W N4"),),
    (2, "A"): (
        _Zone("A", (4, 2), "W4 N2 E S E N E S E N"),
        _Zone("B", (4, 4), "N W S W N W S W N")),
    (2, "B"): (
        _Zone("A", (2, 4), "W2 N E2 N W2 N2 E S E N"),
        _Zone("B", (4, 4), "W N E N W N E N W")),
    (2, "C"): (
        _Zone("A", (4, 4), "W N E N W N E N W2 S2"),
        _Zone("B", (2, 4), "N W S W N2 E N W N E")),
    (3, "A"): (
        _Zone("A", (4, 1), "N W S W N W S W N"),
        _Zone("B", (2, 4), "W2 N2 E S E N"),
        _Zone("C", (4, 4), "W N E N W")),
    (3, "B"): (
        _Zone("A", (1, 4), "W N E N W N E N W"),
        _Zone("B", (4, 1), "N W S W N"),
        _Zone("C", (4, 4), "W2 N2 E S E N")),
    (3, "C"): (
        _Zone("A", (1, 1), "W N E"),
        _Zone("B", (4, 1), "N W2 S E"),
        _Zone("C", (4, 4), "W4 N2 E S E N E S E N")),
    (3, "D"): (
        _Zone("A", (2, 2), "W2 N2 E S E N"),
        _Zone("B", (4, 3), "N W N2 E S"),
        _Zone("C", (4, 4), "W N W S W2 N E")),
    (3, "E"): (
        _Zone("A", (4, 4), "W2 N E2 N W2"),
        _Zone("B", (4, 1), "N W3 S E2"),
        _Zone("C", (1, 4), "W N E N W N2")),
    (3, "F"): (
        _Zone("A", (2, 2), "W2 N2 E2 S W"),
        _Zone("B", (4, 3), "W N E N W N E"),
        _Zone("C", (4, 4), "W4 N E2")),
    (4, "A"): (
        _Zone("A", (2, 2), "W2 N2 E S E N"),
        _Zone("B", (4, 2), "W N E N W"),
        _Zone("C", (2, 4), "N W S W N"),
        _Zone("D", (4, 4), "W N E")),
    (4, "B"): (
        _Zone("A", (2, 1), "N W S W N"),
        _Zone("B", (4, 2), "W N E N W"),
        _Zone("C", (1, 4), "W N E N W"),
        _Zone("D", (4, 4), "N W S W N2")),
    (4, "C"): (
        _Zone("A", (1, 1), "W N E"),
        _Zone("B", (3, 1), "W N E"),
        _Zone("C", (4, 4), "N4"),
        _Zone("D", (3, 4), "W3 N2 E S E N E S")),
    (4, "D"): (
        _Zone("A", (2, 2), "W2 N2 E2 S W"),
        _Zone("B", (4, 2), "W N E N W"),
        _Zone("C", (2, 4), "W2 N E2"),
        _Zone("D", (4, 4), "W N E")),
    (5, "A"): (
        _Zone("A", (1, 1), "W N E"),
        _Zone("B", (3, 1), "W N E"),
        _Zone("C", (4, 4), "N4"),
        _Zone("D", (1, 4), "W N E N W"),
        _Zone("E", (3, 4), "W N E N W")),
    (5, "B"): (
        _Zone("A", (2, 1), "N W S W N"),
        _Zone("B", (1, 4), "W N E N W"),
        _Zone("C", (4, 4), "W N E"),
        _Zone("D", (4, 2), "W N E N W"),
        _Zone("E", (2, 4), "N2")),
    (6, "A"): (
        _Zone("A", (1, 1), "W N E"),
        _Zone("B", (3, 1), "W N E"),
        _Zone("C", (4, 1), "N"),
        _Zone("D", (1, 4), "W N E N W"),
        _Zone("E", (3, 4), "W N E N W"),
        _Zone("F", (4, 4), "N2")),
    (6, "B"): (
        _Zone("A", (1, 1), "W N E"),
        _Zone("B", (1, 3), "W N E"),
        _Zone("C", (1, 4), "W"),
        _Zone("D", (4, 1), "N W S W N"),
        _Zone("E", (3, 4), "W N E N W"),
        _Zone("F", (4, 4), "N2")),
    (6, "C"): (
        _Zone("A", (0, 4), "N4"),
        _Zone("B", (1, 4), "N4"),
        _Zone("C", (2, 4), "N4"),
        _Zone("D", (3, 4), "N4"),
        _Zone("E", (4, 4), "N"),
        _Zone("F", (4, 2), "N2")),
    (6, "D"): (
        _Zone("A", (4, 0), "W4"),
        _Zone("B", (4, 1), "W4"),
        _Zone("C", (4, 2), "W4"),
        _Zone("D", (4, 3), "W4"),
        _Zone("E", (2, 4), "W2"),
        _Zone("F", (4, 4), "W")),
    (6, "E"): (
        _Zone("A", (2, 0), "W2"),
        _Zone("B", (2, 1), "W2"),
        _Zone("C", (2, 2), "W2"),
        _Zone("D", (4, 2), "W N E N W"),
        _Zone("E", (2, 4), "W2 N E2"),
        _Zone("F", (4, 4), "W N E")),
})


def _encode_road(path: tuple[Position, ...]) -> str:
    inverse = {delta: token for token, delta in _DELTAS.items()}
    return " ".join(inverse[(b[0] - a[0], b[1] - a[1])]
                    for a, b in zip(path, path[1:]))


def _transform_local(path: tuple[Position, ...], quadrant: str) -> tuple[Position, ...]:
    if quadrant == "NW":
        return path
    if quadrant == "NE":
        return tuple((9 - x, y) for x, y in path)
    if quadrant == "SW":
        return tuple((x, 9 - y) for x, y in path)
    if quadrant == "SE":
        return tuple((9 - x, 9 - y) for x, y in path)
    raise ValueError(f"unknown quadrant {quadrant}")


def _compose_quadrants(
    layout_id: str,
    mask: LandMask,
    allocations: tuple[int, ...],
    variant: str | tuple[str, ...],
) -> _Layout:
    """Materialize one fixed catalogue entry; no workload affects this step."""
    if len(mask) != len(allocations) or sum(allocations) > 12:
        raise ValueError(f"{layout_id}: invalid quadrant allocation")
    variants = ((variant,) * len(mask)
                if isinstance(variant, str) else variant)
    if len(variants) != len(mask):
        raise ValueError(f"{layout_id}: invalid local-road variants")
    zone_names = iter("ABCDEFGHIJKL")
    roads = []
    for quadrant, count, local_variant in zip(mask, allocations, variants):
        for local in _LOCAL_ROADS[(count, local_variant)]:
            path = _transform_local(_walk(local.start, local.moves), quadrant)
            roads.append(_Zone(next(zone_names), path[0], _encode_road(path)))
    return _Layout(
        layout_id, mask, sum(allocations),
        f"QUADRANT_ROADS_{''.join(variants)}", "HIGH_VARIANCE", 5,
        tuple(roads))


# These are fixed catalogue entries. Allocations are literal and never chosen
# from today's workload; A/B select two different authored 5x5 road systems.
_COMPOSITION_SPECS: tuple[
    tuple[str, LandMask, tuple[int, ...], str | tuple[str, ...]], ...
] = (
    ("nw-w1-qa", NW, (1,), "A"),
    ("nw-w1-qb", NW, (1,), "B"),
    ("nw-w2-qa", NW, (2,), "A"),
    ("nw-w2-qb", NW, (2,), "B"),
    ("nw-w3-qa", NW, (3,), "A"),
    ("nw-w3-qb", NW, (3,), "B"),
    ("nw-w4-qa", NW, (4,), "A"),
    ("nw-w4-qb", NW, (4,), "B"),
    ("nw-w5-qa", NW, (5,), "A"),
    ("nw-w5-qb", NW, (5,), "B"),
    ("nw-w6-qa", NW, (6,), "A"),
    ("nw-w6-qb", NW, (6,), "B"),
    ("north-w3-qa", NORTH, (2, 1), "A"),
    ("north-w3-qb", NORTH, (1, 2), "B"),
    ("north-w4-qa", NORTH, (2, 2), "A"),
    ("north-w4-qb", NORTH, (2, 2), "B"),
    ("north-w4-qc", NORTH, (2, 2), "C"),
    ("north-w5-qa", NORTH, (3, 2), "A"),
    ("north-w5-qb", NORTH, (2, 3), "B"),
    ("north-w5-qc", NORTH, (3, 2), ("E", "C")),
    ("north-w5-qd", NORTH, (2, 3), ("C", "F")),
    ("north-w6-qa", NORTH, (3, 3), "A"),
    ("north-w6-qb", NORTH, (3, 3), "B"),
    ("north-w6-qc", NORTH, (3, 3), ("E", "F")),
    ("north-w6-qd", NORTH, (3, 3), ("F", "E")),
    ("north-w7-qa", NORTH, (4, 3), "A"),
    ("north-w7-qb", NORTH, (3, 4), "B"),
    ("north-w8-qa", NORTH, (4, 4), "A"),
    ("north-w8-qb", NORTH, (4, 4), "B"),
    ("north-w9-qa", NORTH, (5, 4), "A"),
    ("north-w9-qb", NORTH, (4, 5), "B"),
    ("north-w10-qa", NORTH, (5, 5), "A"),
    ("north-w10-qb", NORTH, (5, 5), "B"),
    ("north-w11-qa", NORTH, (6, 5), "A"),
    ("north-w11-qb", NORTH, (5, 6), "B"),
    ("north-w12-qa", NORTH, (6, 6), "A"),
    ("north-w12-qb", NORTH, (6, 6), "B"),
    ("north-w12-qc", NORTH, (6, 6), ("C", "D")),
    ("north-w12-qd", NORTH, (6, 6), ("D", "C")),
    ("three-w5-qa", THREE_LAND, (2, 2, 1), "A"),
    ("three-w5-qb", THREE_LAND, (1, 2, 2), "B"),
    ("three-w6-qa", THREE_LAND, (2, 2, 2), "A"),
    ("three-w6-qb", THREE_LAND, (2, 2, 2), "B"),
    ("three-w7-qa", THREE_LAND, (3, 2, 2), "A"),
    ("three-w7-qb", THREE_LAND, (2, 3, 2), "B"),
    ("three-w8-qa", THREE_LAND, (3, 3, 2), "A"),
    ("three-w8-qb", THREE_LAND, (2, 3, 3), "B"),
    ("three-w8-qc", THREE_LAND, (3, 3, 2), ("E", "F", "C")),
    ("three-w8-qd", THREE_LAND, (2, 3, 3), ("C", "E", "F")),
    ("three-w9-qa", THREE_LAND, (3, 3, 3), "A"),
    ("three-w9-qb", THREE_LAND, (3, 3, 3), "B"),
    ("three-w9-qc", THREE_LAND, (3, 3, 3), "C"),
    ("three-w9-qd", THREE_LAND, (3, 3, 3), "D"),
    ("three-w9-qe", THREE_LAND, (3, 3, 3), ("E", "F", "E")),
    ("three-w9-qf", THREE_LAND, (3, 3, 3), ("F", "E", "F")),
    ("three-w10-qa", THREE_LAND, (4, 3, 3), "A"),
    ("three-w10-qb", THREE_LAND, (3, 4, 3), "B"),
    ("three-w10-qc", THREE_LAND, (4, 3, 3), ("A", "E", "F")),
    ("three-w10-qd", THREE_LAND, (3, 4, 3), ("E", "B", "F")),
    ("three-w11-qa", THREE_LAND, (4, 4, 3), "A"),
    ("three-w11-qb", THREE_LAND, (3, 4, 4), "B"),
    ("three-w12-qa", THREE_LAND, (4, 4, 4), "A"),
    ("three-w12-qb", THREE_LAND, (4, 4, 4), "B"),
    ("three-w12-qc", THREE_LAND, (4, 4, 4), "C"),
    ("three-w12-qd", THREE_LAND, (4, 4, 4), ("A", "B", "A")),
    ("three-w12-qe", THREE_LAND, (6, 3, 3), "A"),
    ("three-w12-qf", THREE_LAND, (3, 6, 3), "B"),
    ("three-w12-qg", THREE_LAND, (3, 3, 6), "A"),
    ("three-w12-qh", THREE_LAND, (6, 3, 3), ("C", "E", "F")),
    ("three-w12-qi", THREE_LAND, (3, 6, 3), ("E", "C", "F")),
    ("three-w12-qj", THREE_LAND, (3, 3, 6), ("E", "F", "C")),
    ("three-w12-qk", THREE_LAND, (6, 3, 3), ("D", "F", "E")),
    ("three-w12-ql", THREE_LAND, (3, 6, 3), ("F", "D", "E")),
    ("three-w12-qm", THREE_LAND, (3, 3, 6), ("F", "E", "D")),
    ("three-w12-qn", THREE_LAND, (4, 4, 4), "D"),
    ("three-w12-qo", THREE_LAND, (6, 3, 3), ("E", "F", "F")),
    ("three-w12-qp", THREE_LAND, (3, 6, 3), ("F", "E", "F")),
    ("three-w12-qq", THREE_LAND, (3, 3, 6), ("F", "F", "E")),
    ("full-w6-qa", FULL, (2, 2, 1, 1), "A"),
    ("full-w6-qb", FULL, (1, 2, 2, 1), "B"),
    ("full-w7-qa", FULL, (2, 2, 2, 1), "A"),
    ("full-w7-qb", FULL, (1, 2, 2, 2), "B"),
    ("full-w8-qa", FULL, (2, 2, 2, 2), "A"),
    ("full-w8-qb", FULL, (2, 2, 2, 2), "B"),
    ("full-w9-qa", FULL, (3, 2, 2, 2), "A"),
    ("full-w9-qb", FULL, (2, 3, 2, 2), "B"),
    ("full-w10-qa", FULL, (3, 3, 2, 2), "A"),
    ("full-w10-qb", FULL, (2, 3, 3, 2), "B"),
    ("full-w11-qa", FULL, (3, 3, 3, 2), "A"),
    ("full-w11-qb", FULL, (2, 3, 3, 3), "B"),
    ("full-w12-qa", FULL, (3, 3, 3, 3), "A"),
    ("full-w12-qb", FULL, (3, 3, 3, 3), "B"),
    ("full-w12-qc", FULL, (3, 3, 3, 3), "C"),
    ("full-w12-qd", FULL, (3, 3, 3, 3), "D"),
    ("full-w12-qe", FULL, (6, 2, 2, 2), "A"),
    ("full-w12-qf", FULL, (2, 6, 2, 2), "B"),
    ("full-w12-qg", FULL, (2, 2, 6, 2), "A"),
    ("full-w12-qh", FULL, (2, 2, 2, 6), "B"),
    ("full-w12-qi", FULL, (3, 4, 3, 2), "A"),
    ("full-w12-qj", FULL, (2, 3, 4, 3), "B"),
    ("full-w12-qk", FULL, (3, 2, 3, 4), "A"),
    ("full-w12-ql", FULL, (4, 3, 2, 3), "B"),
    ("full-w12-qm", FULL, (2, 4, 4, 2), "A"),
    ("full-w12-qn", FULL, (4, 2, 2, 4), "B"),
    ("full-w12-qo", FULL, (4, 3, 3, 2), ("C", "A", "A", "A")),
    ("full-w12-qp", FULL, (3, 4, 3, 2), ("A", "C", "A", "A")),
    ("full-w12-qq", FULL, (3, 2, 4, 3), ("A", "A", "C", "A")),
    ("full-w12-qr", FULL, (3, 3, 2, 4), ("A", "A", "A", "C")),
    ("full-w12-qs", FULL, (5, 5, 1, 1), "A"),
    ("full-w12-qt", FULL, (1, 1, 5, 5), "B"),
    ("full-w12-qu", FULL, (5, 1, 5, 1), "A"),
    ("full-w12-qv", FULL, (1, 5, 1, 5), "B"),
    ("full-w12-qw", FULL, (5, 1, 1, 5), "A"),
    ("full-w12-qx", FULL, (1, 5, 5, 1), "B"),
)


_COMPOSED_LAYOUTS = tuple(
    _compose_quadrants(layout_id, mask, allocations, variant)
    for layout_id, mask, allocations, variant in _COMPOSITION_SPECS)


def _fixed_nw_strips(layout_id: str, splits: tuple[int, ...], *,
                     horizontal: bool = False) -> _Layout:
    """Materialize one declared thick strip split; never reads workload."""
    if len(splits) != 5 or any(not 1 <= count <= 5 for count in splits):
        raise ValueError(f"{layout_id}: invalid strip declaration")
    lengths = {
        1: (5,), 2: (3, 2), 3: (2, 2, 1),
        4: (2, 1, 1, 1), 5: (1, 1, 1, 1, 1),
    }
    roads = []
    zone_names = iter("ABCDEFGHIJKL")
    for fixed, count in enumerate(splits):
        offset = 0
        for length in lengths[count]:
            varying = range(offset, offset + length)
            segment = tuple((value, fixed) if horizontal else (fixed, value)
                            for value in varying)
            offset += length
            if rules.distance_to_shed(segment[-1]) < rules.distance_to_shed(segment[0]):
                segment = tuple(reversed(segment))
            roads.append(_Zone(next(zone_names), segment[0], _encode_road(segment)))
    return _Layout(layout_id, NW, sum(splits), "NW_FIXED_SEGMENTS",
                   "HIGH_VARIANCE", 5, tuple(roads))


_NW_SEGMENT_LAYOUTS: tuple[_Layout, ...] = tuple(
    _fixed_nw_strips(layout_id, splits, horizontal=horizontal)
    for layout_id, splits, horizontal in (
        ("nw-w7-strips-v", (2, 1, 1, 1, 2), False),
        ("nw-w7-strips-h", (2, 1, 1, 1, 2), True),
        ("nw-w8-strips-v", (2, 1, 2, 1, 2), False),
        ("nw-w8-strips-h", (2, 1, 2, 1, 2), True),
        ("nw-w9-strips-v", (2, 2, 1, 2, 2), False),
        ("nw-w9-strips-h", (2, 2, 1, 2, 2), True),
        ("nw-w10-strips-v", (2, 2, 2, 2, 2), False),
        ("nw-w10-strips-h", (2, 2, 2, 2, 2), True),
        ("nw-w11-strips-v", (3, 2, 2, 2, 2), False),
        ("nw-w11-strips-h", (3, 2, 2, 2, 2), True),
        ("nw-w12-strips-v", (3, 2, 2, 2, 3), False),
        ("nw-w12-strips-h", (3, 2, 2, 2, 3), True),
    ))


HAND_AUTHORED_TEMPLATES: tuple[PartitionTemplate, ...] = tuple(
    template
    for layout in (*_LAYOUTS, *_COMPOSED_LAYOUTS, *_NW_SEGMENT_LAYOUTS)
    for template in _make_layout(layout))
TEMPLATE_LIBRARY: Mapping[tuple[LandMask, int], tuple[PartitionTemplate, ...]] = MappingProxyType({
    (mask, workers): tuple(t for t in HAND_AUTHORED_TEMPLATES
                           if t.owned_land_mask == mask and t.worker_count == workers)
    for mask in (NW, NORTH, THREE_LAND, FULL)
    for workers in range(1, 13)
    if any(t.owned_land_mask == mask and t.worker_count == workers
           for t in HAND_AUTHORED_TEMPLATES)})


def templates_for(mask: LandMask, worker_count: int) -> tuple[PartitionTemplate, ...]:
    return TEMPLATE_LIBRARY.get((tuple(mask), worker_count), ())


def _shortest_connector(start: Position, target: Position,
                        forbidden: frozenset[Position]) -> tuple[Position, ...] | None:
    if start == target:
        return (start,)
    queue = deque([start])
    prior: dict[Position, Position | None] = {start: None}
    while queue:
        current = queue.popleft()
        neighbours = []
        for dx, dy in _DELTAS.values():
            nxt = (current[0] + dx, current[1] + dy)
            if (0 <= nxt[0] < rules.BOARD_SIZE and 0 <= nxt[1] < rules.BOARD_SIZE
                    and (nxt not in forbidden or nxt == target)
                    and nxt not in prior):
                neighbours.append(nxt)
        neighbours.sort(key=lambda p: (rules.manhattan(p, target), p[1], p[0]))
        for nxt in neighbours:
            prior[nxt] = current
            if nxt == target:
                path = [nxt]
                while path[-1] != start:
                    path.append(prior[path[-1]])
                return tuple(reversed(path))
            queue.append(nxt)
    return None


def compile_zone_route(template: PartitionTemplate, zone_id: str,
                       workload: Mapping[Position, TileWorkload]
                       ) -> CompiledZoneRoute | None:
    zone = template.zones[zone_id]
    returns_to_shed = zone.returns_to_shed
    tasks = {tile: work for tile, work in workload.items()
             if template.tile_to_zone.get(tile) == zone_id and work.actions > 0}
    if any(work.actions > zone.action_limit_by_tile[tile]
           for tile, work in tasks.items()):
        return None
    if not tasks:
        return CompiledZoneRoute(zone_id, (), (), 0, 0, 0, None, -1, 0)
    ordered = tuple(tile for tile in zone.road if tile in tasks)
    preload = len(set().union(*(tasks[t].pickup_items for t in ordered)))
    positions = [zone.shed_access]
    visited: set[Position] = {zone.shed_access}

    def connect(target: Position, preserve: Iterable[Position] = ()) -> bool:
        forbidden = frozenset((visited | set(preserve))
                              - {positions[-1], target})
        path = _shortest_connector(positions[-1], target, forbidden)
        if path is None:
            return False
        for point in path[1:]:
            if point in visited and not (
                    returns_to_shed and point == zone.shed_access):
                return False
            visited.add(point)
            positions.append(point)
        return True

    if not connect(zone.entry, ordered):
        return None
    visited.add(zone.entry)
    for index, tile in enumerate(ordered):
        if not connect(tile, ordered[index + 1:]):
            return None
    task_turns = sum(tasks[t].actions for t in ordered)
    movement = len(positions) - 1
    operations = preload + movement + task_turns
    drop_turn = None
    if returns_to_shed:
        if not connect(zone.shed_access):
            return None
        movement = len(positions) - 1
        operations = preload + movement + task_turns + 1
        drop_turn = operations - 1
        if drop_turn > 22:
            return None
    if operations > rules.TURNS_PER_DAY:
        return None
    return CompiledZoneRoute(
        zone_id, tuple(positions), ordered, movement, preload, task_turns,
        drop_turn, operations - 1,
        sum(tasks[t].output_units for t in ordered) if returns_to_shed else 0)


def solve_template(template: PartitionTemplate,
                   workload: Mapping[Position, TileWorkload]
                   ) -> TemplateSolution | None:
    if any(tile not in template.tile_to_zone for tile, work in workload.items()
           if work.actions > 0):
        return None
    if any(work.must_return
           and not template.zones[template.tile_to_zone[tile]].returns_to_shed
           for tile, work in workload.items() if work.actions > 0):
        return None
    routes = {}
    for zone_id in template.zones:
        route = compile_zone_route(template, zone_id, workload)
        if route is None:
            return None
        routes[zone_id] = route
    returned = sum(route.returned_units for route in routes.values())
    return TemplateSolution(
        template, MappingProxyType(routes),
        sum(route.movement_turns for route in routes.values()),
        max((route.finish_turn for route in routes.values()), default=-1), returned)


def select_template(mask: LandMask, worker_count: int,
                    workload: Mapping[Position, TileWorkload]
                    ) -> TemplateSolution | None:
    candidates = tuple(filter(None, (
        solve_template(t, workload)
        for t in templates_for(mask, worker_count))))
    return min(candidates, key=lambda result: (
        result.max_finish_turn, result.total_movement,
        sum(zone.returns_to_shed for zone in result.template.zones.values()),
        result.template.template_id)
    ) if candidates else None


def select_minimum_workforce(mask: LandMask,
                             workload: Mapping[Position, TileWorkload], *,
                             minimum_workers: int = 1
                             ) -> TemplateSolution | None:
    for workers in range(max(1, minimum_workers), 13):
        result = select_template(mask, workers, workload)
        if result is not None:
            return result
    return None
