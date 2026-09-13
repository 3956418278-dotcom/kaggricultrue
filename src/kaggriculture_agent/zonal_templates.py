"""Hand-drawn fixed partition templates for the D11+ intraday executor.

The ASCII maps below are the source of truth.  This module does not search for,
grow, balance, clip, or otherwise generate partitions.  It only materializes
objective metadata (adjacency, boundary tiles, Manhattan distances and four
fixed scan orders) from the authored maps.

Coordinates use the official board convention: x grows right, y grows down.
``.`` is locked land.  Zone letters are local to one template.  ``I`` and
``J`` are conventionally used for INNER skeleton zones, but the authoritative
labels are the explicit ``inner_zones`` argument on each template.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from . import rules
from .state import Position


LandMask = tuple[str, ...]

NW: LandMask = ("NW",)
NORTH: LandMask = ("NW", "NE")
THREE_QUADRANTS: LandMask = ("NW", "NE", "SW")
FULL: LandMask = ("NW", "NE", "SW", "SE")


@dataclass(frozen=True)
class ZoneTemplate:
    zone_id: str
    behavior: str
    tiles: tuple[Position, ...]
    boundary_tiles: frozenset[Position]
    adjacent_zones: frozenset[str]
    shed_access: Position
    shed_distance: int
    tile_distances: tuple[tuple[int, ...], ...]
    cheap_sweeps: Mapping[str, tuple[Position, ...]]


@dataclass(frozen=True)
class PartitionTemplate:
    template_id: str
    owned_land_mask: LandMask
    worker_count: int
    family: str
    workload_phase: str
    rows: tuple[str, ...]
    rationale: str
    tile_to_zone: Mapping[Position, str]
    zones: Mapping[str, ZoneTemplate]

    def render(self) -> str:
        return "\n".join(self.rows)


def _owned_tiles(mask: LandMask) -> frozenset[Position]:
    return frozenset(
        (x, y)
        for y in range(rules.BOARD_SIZE)
        for x in range(rules.BOARD_SIZE)
        if rules.quadrant((x, y), rules.BOARD_SIZE) in mask
    )


def _row_snake(tiles: frozenset[Position], reverse: bool) -> tuple[Position, ...]:
    result: list[Position] = []
    ys = sorted({y for _, y in tiles}, reverse=reverse)
    for index, y in enumerate(ys):
        xs = sorted(x for x, yy in tiles if yy == y)
        if bool(index % 2) ^ reverse:
            xs.reverse()
        result.extend((x, y) for x in xs)
    return tuple(result)


def _column_snake(tiles: frozenset[Position], reverse: bool) -> tuple[Position, ...]:
    result: list[Position] = []
    xs = sorted({x for x, _ in tiles}, reverse=reverse)
    for index, x in enumerate(xs):
        ys = sorted(y for xx, y in tiles if xx == x)
        if bool(index % 2) ^ reverse:
            ys.reverse()
        result.extend((x, y) for y in ys)
    return tuple(result)


def _clockwise(tiles: frozenset[Position], reverse: bool) -> tuple[Position, ...]:
    """Fixed outside-in rectangular sweep, filtered to the authored zone."""
    remaining = set(tiles)
    result: list[Position] = []
    while remaining:
        min_x = min(x for x, _ in remaining)
        max_x = max(x for x, _ in remaining)
        min_y = min(y for _, y in remaining)
        max_y = max(y for _, y in remaining)
        ring = (
            [(x, min_y) for x in range(min_x, max_x + 1)]
            + [(max_x, y) for y in range(min_y + 1, max_y + 1)]
            + [(x, max_y) for x in range(max_x - 1, min_x - 1, -1)]
            + [(min_x, y) for y in range(max_y - 1, min_y, -1)]
        )
        ring = [tile for tile in ring if tile in remaining]
        if reverse:
            ring.reverse()
        for tile in ring:
            if tile in remaining:
                remaining.remove(tile)
                result.append(tile)
    return tuple(result)


def _materialize(
    template_id: str,
    mask: LandMask,
    worker_count: int,
    family: str,
    workload_phase: str,
    rows: tuple[str, ...],
    inner_zones: frozenset[str],
    rationale: str,
) -> PartitionTemplate:
    if len(rows) != rules.BOARD_SIZE or any(len(row) != rules.BOARD_SIZE for row in rows):
        raise ValueError(f"{template_id}: expected a 10x10 map")
    tile_to_zone = {
        (x, y): marker
        for y, row in enumerate(rows)
        for x, marker in enumerate(row)
        if marker != "."
    }
    expected = _owned_tiles(mask)
    if frozenset(tile_to_zone) != expected:
        missing = sorted(expected - tile_to_zone.keys())
        extra = sorted(tile_to_zone.keys() - expected)
        raise ValueError(f"{template_id}: land coverage mismatch; missing={missing}, extra={extra}")
    zone_ids = frozenset(tile_to_zone.values())
    if len(zone_ids) != worker_count:
        raise ValueError(f"{template_id}: {len(zone_ids)} zones for {worker_count} workers")
    if not inner_zones or not inner_zones <= zone_ids:
        raise ValueError(f"{template_id}: invalid INNER zone set {sorted(inner_zones)}")

    zones: dict[str, ZoneTemplate] = {}
    access_tiles = rules.shed_access(rules.BOARD_SIZE)
    for zone_id in sorted(zone_ids):
        zone_tiles = frozenset(tile for tile, owner in tile_to_zone.items() if owner == zone_id)
        start = min(zone_tiles)
        reached = {start}
        frontier = [start]
        while frontier:
            x, y = frontier.pop()
            for neighbour in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbour in zone_tiles and neighbour not in reached:
                    reached.add(neighbour)
                    frontier.append(neighbour)
        if reached != set(zone_tiles):
            raise ValueError(f"{template_id}: zone {zone_id} is disconnected")

        boundary: set[Position] = set()
        adjacent: set[str] = set()
        for x, y in zone_tiles:
            for neighbour in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                other = tile_to_zone.get(neighbour)
                if other is not None and other != zone_id:
                    boundary.add((x, y))
                    adjacent.add(other)
        ordered_tiles = tuple(sorted(zone_tiles, key=lambda p: (p[1], p[0])))
        access = min(
            access_tiles,
            key=lambda point: (
                min(rules.manhattan(point, tile) for tile in zone_tiles),
                point[1],
                point[0],
            ),
        )
        distances = tuple(
            tuple(rules.manhattan(a, b) for b in ordered_tiles)
            for a in ordered_tiles
        )
        sweeps = MappingProxyType({
            "clockwise": _clockwise(zone_tiles, False),
            "counter_clockwise": _clockwise(zone_tiles, True),
            "snake_a": _row_snake(zone_tiles, False),
            "snake_b": _column_snake(zone_tiles, False),
        })
        zones[zone_id] = ZoneTemplate(
            zone_id=zone_id,
            behavior="INNER" if zone_id in inner_zones else "OUTER",
            tiles=ordered_tiles,
            boundary_tiles=frozenset(boundary),
            adjacent_zones=frozenset(adjacent),
            shed_access=access,
            shed_distance=min(rules.manhattan(tile, access) for tile in zone_tiles),
            tile_distances=distances,
            cheap_sweeps=sweeps,
        )
    return PartitionTemplate(
        template_id=template_id,
        owned_land_mask=mask,
        worker_count=worker_count,
        family=family,
        workload_phase=workload_phase,
        rows=rows,
        rationale=rationale,
        tile_to_zone=MappingProxyType(tile_to_zone),
        zones=MappingProxyType(zones),
    )


# These maps are deliberately literal.  Variants are repeated explicitly so a
# reviewer can judge the complete farm topology without mentally applying a
# transform or trusting a partition generator.
HAND_DRAWN_TEMPLATES: tuple[PartitionTemplate, ...] = (
    # NW 5x5: ordinary templates preserve a corner 3x3 as one OUTER zone.
    _materialize("nw-w2-corner-l", NW, 2, "corner_block", "NORMAL", (
        "AAAII.....", "AAAII.....", "AAAII.....", "IIIII.....", "IIIII.....",
        "..........", "..........", "..........", "..........", "..........",
    ), frozenset("I"), "A is a 3x3 corner sweep; I is one thick L-shaped shed-facing skeleton."),
    _materialize("nw-w2-stair", NW, 2, "staircase", "PEAK", (
        "AAAII.....", "AAAII.....", "AAIII.....", "AIIII.....", "IIIII.....",
        "..........", "..........", "..........", "..........", "..........",
    ), frozenset("I"), "A short staircase gives I more of the high-flow centre without a thin corridor."),
    _materialize("nw-w3-top-chain", NW, 3, "edge_chain", "NORMAL", (
        "AAABB.....", "AAABB.....", "AAABB.....", "CCCBB.....", "CCCBB.....",
        "..........", "..........", "..........", "..........", "..........",
    ), frozenset("B"), "The 3x3 corner A continues into compact C on one edge and shed-facing B on the other."),
    _materialize("nw-w3-left-chain", NW, 3, "edge_chain_rotated", "NORMAL", (
        "AAACC.....", "AAACC.....", "AAACC.....", "BBBBB.....", "BBBBB.....",
        "..........", "..........", "..........", "..........", "..........",
    ), frozenset("B"), "Rotated edge-chain variant; template switching chooses which edge carries INNER flow."),
    _materialize("nw-w4-corner-refine", NW, 4, "corner_refinement", "PEAK", (
        "AAABB.....", "AAABB.....", "CCCBB.....", "CCCII.....", "CCCII.....",
        "..........", "..........", "..........", "..........", "..........",
    ), frozenset("I"), "Peak form splits the former 3x3 corner into compact 3x2 and 3x3 blocks."),
    _materialize("nw-w4-large-inner", NW, 4, "central_inner", "PEAK", (
        "AABBB.....", "AABBB.....", "AAIII.....", "CCIII.....", "CCIII.....",
        "..........", "..........", "..........", "..........", "..........",
    ), frozenset("I"), "A larger 3x3 INNER accepts dense return work while three short OUTER blocks remain compact."),
    _materialize("nw-w5-grid-core", NW, 5, "compact_grid", "PEAK", (
        "AABBC.....", "AABBC.....", "DIIIC.....", "DIIIC.....", "DIIIC.....",
        "..........", "..........", "..........", "..........", "..........",
    ), frozenset(("C", "I")), "Two joined INNER blocks form the shed-facing core; three edge OUTERs stay compact."),
    _materialize("nw-w5-offset-core", NW, 5, "compact_grid_rotated", "PEAK", (
        "AAABB.....", "AAABB.....", "CCDDB.....", "CCDII.....", "CCDII.....",
        "..........", "..........", "..........", "..........", "..........",
    ), frozenset("I"), "Offset compact blocks move the dense INNER area toward the shed corner."),

    # Two northern quadrants: the scarce INNER area is a central spine or core,
    # never a separate corridor for every OUTER block.
    _materialize("north-w3-straight-spine", NORTH, 3, "central_spine", "NORMAL", (
        "AAAAIIBBBB", "AAAAIIBBBB", "AAAAIIBBBB", "AAAAIIBBBB", "AAAAIIBBBB",
        "..........", "..........", "..........", "..........", "..........",
    ), frozenset("I"), "Two broad OUTER halves share one two-tile-wide central INNER spine."),
    _materialize("north-w3-flared-spine", NORTH, 3, "central_spine_flared", "NORMAL", (
        "AAAAIIBBBB", "AAAAIIBBBB", "AAAAIIBBBB", "AAAIIIIBBB", "AAAIIIIBBB",
        "..........", "..........", "..........", "..........", "..........",
    ), frozenset("I"), "The INNER spine widens only near shed access; both OUTER zones stay single-sweep shapes."),
    _materialize("north-w4-split-spine", NORTH, 4, "two_inner", "NORMAL", (
        "AAAAIIBBBB", "AAAAIIBBBB", "AAAAIIBBBB", "AAAAJJBBBB", "AAAAJJBBBB",
        "..........", "..........", "..........", "..........", "..........",
    ), frozenset(("I", "J")), "Two adjacent INNER blocks divide central traffic without cutting the edge OUTER blocks."),
    _materialize("north-w4-horizontal-blocks", NORTH, 4, "horizontal_blocks", "PEAK", (
        "AAAABBBBBB", "AAAABBBBBB", "CCCCDDDDDD", "CCCCDDDDDD", "CCCCDDDDDD",
        "..........", "..........", "..........", "..........", "..........",
    ), frozenset(("C", "D")), "Peak template uses four compact horizontal blocks; the lower pair owns shed traffic."),
    _materialize("north-w5-edge-blocks", NORTH, 5, "edge_blocks", "NORMAL", (
        "AAAAIIBBBB", "AAAAIIBBBB", "AAAAIIBBBB", "CCCCIIDDDD", "CCCCIIDDDD",
        "..........", "..........", "..........", "..........", "..........",
    ), frozenset("I"), "Four compact OUTER rectangles share a single central INNER spine."),
    _materialize("north-w5-wide-core", NORTH, 5, "wide_inner", "PEAK", (
        "AAAIIIBBBB", "AAAIIIBBBB", "AAAIIIBBBB", "DDDIIIEEEE", "DDDIIIEEEE",
        "..........", "..........", "..........", "..........", "..........",
    ), frozenset("I"), "A wider INNER core handles clustered return days; edge blocks remain thick rectangles."),
    _materialize("north-w6-double-spine", NORTH, 6, "two_inner", "PEAK", (
        "AAAIIBBCCC", "AAAIIBBCCC", "DDDIIBBEEE", "DDDIIBBEEE", "DDDIIBBEEE",
        "..........", "..........", "..........", "..........", "..........",
    ), frozenset(("I", "B")), "Two full-height INNER blocks absorb a high central workload; four OUTERs are compact."),
    _materialize("north-w6-one-spine", NORTH, 6, "one_inner_edge_chain", "PEAK", (
        "AAAIIBBCCC", "AAAIIBBCCC", "DDDIIBBCCC", "DDDIIEEEEE", "DDDIIEEEEE",
        "..........", "..........", "..........", "..........", "..........",
    ), frozenset("I"), "One INNER spine leaves room for a second compact OUTER in each edge chain."),

    # Three quadrants: w=5 uses broad rectangular phases.  w=6 introduces the
    # explicit 3x3 corner plus a compact edge continuation from the user example.
    _materialize("three-w5-vertical-blocks", THREE_QUADRANTS, 5, "rectangular_blocks", "NORMAL", (
        "AAAABBBBCC", "AAAABBBBCC", "AAAABBBBCC", "AAAABBBBCC", "AAAABBBBCC",
        "DDDII.....", "DDDII.....", "DDDII.....", "DDDII.....", "DDDII.....",
    ), frozenset(("B", "I")), "Two INNER rectangles meet at the shed; three OUTER edge blocks need no private corridors."),
    _materialize("three-w5-horizontal-blocks", THREE_QUADRANTS, 5, "rectangular_blocks_rotated", "NORMAL", (
        "AAAAADDDDD", "AAAAADDDDD", "AAAAADDDDD", "AAAAAIIIII", "BBBBBIIIII",
        "BBBBB.....", "BBBBB.....", "BBBBB.....", "CCCCC.....", "CCCCC.....",
    ), frozenset(("B", "I")), "Rotated block structure changes which outer edge is chained before entering INNER."),
    _materialize("three-w6-corner-top", THREE_QUADRANTS, 6, "corner_edge_chain", "NORMAL", (
        "AAABBBBCCC", "AAABBBBCCC", "AAABBBBCCC", "DDDIIIIICC", "DDDIIIIICC",
        "DDDII.....", "DDDII.....", "EEEII.....", "EEEII.....", "EEEII.....",
    ), frozenset("I"), "A 3x3 corner continues through top-edge B/C; only the other direction enters thick INNER I."),
    _materialize("three-w6-corner-left", THREE_QUADRANTS, 6, "corner_edge_chain_rotated", "NORMAL", (
        "AAADDDDEEE", "AAADDDDEEE", "AAADDDDEEE", "BBBIIIIIII", "BBBIIIIIII",
        "BBBII.....", "BBBII.....", "CCCII.....", "CCCCC.....", "CCCCC.....",
    ), frozenset("I"), "Rotated version chains compact OUTER blocks down the left edge instead of across the top."),
    _materialize("three-w7-corner-peak", THREE_QUADRANTS, 7, "corner_refinement", "PEAK", (
        "AABBBCCCDD", "AABBBCCCDD", "AABBBCCCDD", "EEEIIIIIDD", "EEEIIIIIDD",
        "EEEII.....", "EEEII.....", "FFFII.....", "FFFII.....", "FFFII.....",
    ), frozenset("I"), "Peak form refines the top edge and lower-left edge while retaining one thick INNER skeleton."),
    _materialize("three-w7-corner-peak-rotated", THREE_QUADRANTS, 7, "corner_refinement_rotated", "PEAK", (
        "AAAEEEEFFF", "AAAEEEEFFF", "BBBEEEEFFF", "BBBIIIIIII", "BBBIIIIIII",
        "CCCII.....", "CCCII.....", "CCCII.....", "DDDDD.....", "DDDDD.....",
    ), frozenset("I"), "Rotated peak refinement moves the extra split from the top chain to the left chain."),
    _materialize("three-w8-two-inner", THREE_QUADRANTS, 8, "two_inner_peak", "PEAK", (
        "AABBBCCCDD", "AABBBCCCDD", "AABBBCCCDD", "EEEJJJJJDD", "EEEJJJJJDD",
        "EEEII.....", "EEEII.....", "FFFII.....", "FFFII.....", "FFFII.....",
    ), frozenset(("I", "J")), "The high-flow skeleton itself splits into north and south INNER zones for dense peaks."),
    _materialize("three-w8-two-inner-rotated", THREE_QUADRANTS, 8, "two_inner_peak_rotated", "PEAK", (
        "AAAEEEEFFF", "AAAEEEEFFF", "BBBEEEEFFF", "BBBJJIIIII", "BBBJJIIIII",
        "CCCJJ.....", "CCCJJ.....", "CCCJJ.....", "DDDDD.....", "DDDDD.....",
    ), frozenset(("I", "J")), "Rotated dense-peak form splits the skeleton while preserving compact edge blocks."),

    # Full farm: a single cross/T skeleton for ordinary days, two thick joined
    # INNER zones for denser days, then compact edge refinement for peaks.
    _materialize("full-w5-central-core", FULL, 5, "central_core", "NORMAL", (
        "AAAAABBBBB", "AAAAABBBBB", "AAAAABBBBB", "AAAIIIIBBB", "AAAIIIIBBB",
        "CCCIIIIDDD", "CCCIIIIDDD", "CCCCCDDDDD", "CCCCCDDDDD", "CCCCCDDDDD",
    ), frozenset("I"), "One compact 4x4 INNER core joins four thick corner OUTER blocks."),
    _materialize("full-w5-horizontal-t", FULL, 5, "central_t", "NORMAL", (
        "AAAAABBBBB", "AAAAABBBBB", "AAAAABBBBB", "AAAAABBBBB", "IIIIIIIIII",
        "IIIIIIIIII", "CCCCIIDDDD", "CCCCIIDDDD", "CCCCIIDDDD", "CCCCIIDDDD",
    ), frozenset("I"), "A horizontal high-flow bone with one short central stem favors east-west workload phases."),
    _materialize("full-w6-diagonal-inner", FULL, 6, "two_inner", "NORMAL", (
        "AAAAIIBBBB", "AAAAIIBBBB", "AAAAIIBBBB", "AAAAIIBBBB", "IIIIIJJJJJ",
        "IIIIIJJJJJ", "CCCCJJDDDD", "CCCCJJDDDD", "CCCCJJDDDD", "CCCCJJDDDD",
    ), frozenset(("I", "J")), "Two thick staircase INNER zones split central flow; four corner OUTERs remain rectangular."),
    _materialize("full-w6-diagonal-inner-rotated", FULL, 6, "two_inner_rotated", "NORMAL", (
        "AAAAIICCCC", "AAAAIICCCC", "AAAAIICCCC", "AAAAIICCCC", "IIIIIIJJJJ",
        "IIIIJJJJJJ", "BBBBJJDDDD", "BBBBJJDDDD", "BBBBJJDDDD", "BBBBJJDDDD",
    ), frozenset(("I", "J")), "Rotated diagonal ownership changes which current worker positions reach each core cheaply."),
    _materialize("full-w7-edge-refine", FULL, 7, "edge_refinement", "PEAK", (
        "AAACCIIBBB", "AAACCIIBBB", "AAACCIIBBB", "AAACCIIBBB", "IIIIIIJJJJ",
        "IIIIIIJJJJ", "DDDDJJJEEE", "DDDDJJJEEE", "DDDDJJJEEE", "DDDDJJJEEE",
    ), frozenset(("I", "J")), "One top corner band is refined while two joined INNER zones keep cross-farm flow local."),
    _materialize("full-w7-edge-refine-rotated", FULL, 7, "edge_refinement_rotated", "PEAK", (
        "AAAAIIDDDD", "AAAAIIDDDD", "AAAAIIDDDD", "CCCCIIDDDD", "CCCCIIJJJJ",
        "IIIIIIJJJJ", "IIIIJJJJJJ", "BBBBJJEEEE", "BBBBJJEEEE", "BBBBJJEEEE",
    ), frozenset(("I", "J")), "Rotated edge refinement offers a different compact-block orientation at the same workforce."),
    _materialize("full-w8-both-edges", FULL, 8, "two_inner_edge_refinement", "PEAK", (
        "AAACCIIBBB", "AAACCIIBBB", "AAACCIIBBB", "AAACCIIBBB", "IIIIIIJJJJ",
        "IIIIIIJJJJ", "DDDEEJJFFF", "DDDEEJJFFF", "DDDEEJJFFF", "DDDEEJJFFF",
    ), frozenset(("I", "J")), "Both outer bands split into compact blocks for synchronized harvest/fertilize peaks."),
    _materialize("full-w8-both-edges-rotated", FULL, 8, "two_inner_edge_refinement_rotated", "PEAK", (
        "AAAAIIDDDD", "AAAAIIDDDD", "AAAAIIDDDD", "CCCCIIEEEE", "CCCCIIEEEE",
        "IIIIIIJJJJ", "IIIIJJJJJJ", "BBBBJJFFFF", "BBBBJJFFFF", "BBBBJJFFFF",
    ), frozenset(("I", "J")), "Rotated peak structure shifts compact splits between vertical edge chains."),
)


TEMPLATE_LIBRARY: Mapping[tuple[LandMask, int], tuple[PartitionTemplate, ...]] = MappingProxyType({
    (mask, workers): tuple(
        template for template in HAND_DRAWN_TEMPLATES
        if template.owned_land_mask == mask and template.worker_count == workers
    )
    for mask in (NW, NORTH, THREE_QUADRANTS, FULL)
    for workers in range(1, 12)
    if any(
        template.owned_land_mask == mask and template.worker_count == workers
        for template in HAND_DRAWN_TEMPLATES
    )
})


def templates_for(mask: LandMask, worker_count: int) -> tuple[PartitionTemplate, ...]:
    """Return authored templates only; absence never triggers generation."""
    return TEMPLATE_LIBRARY.get((tuple(mask), worker_count), ())
