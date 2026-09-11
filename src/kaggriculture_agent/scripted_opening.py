"""One fixed four-day opening for visual inspection.

This is deliberately not a planner or a reusable opening policy.  Layout,
staffing, worker routes, purchases, and sales are literal data for one opening.
After the fourth game day every unit passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from . import rules


Position = tuple[int, int]
Action = tuple[object, ...]

SEED = 17
DAY1_HANDS = 5
DAY3_HANDS = 4
DAY4_HANDS = 8

ABANDONED_TILES: frozenset[Position] = frozenset({
    (0, 0), (1, 0), (8, 0), (9, 0),
    (0, 1), (9, 1),
})

MELON_TILES: tuple[Position, ...] = (
    (4, 2), (5, 2),
    (3, 3), (4, 3), (5, 3), (6, 3),
    (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4),
)
INNER_MELONS: tuple[Position, ...] = (
    (4, 4), (5, 4), (6, 4), (5, 3), (4, 3), (3, 4),
)
OUTER_MELONS: tuple[Position, ...] = tuple(
    position for position in MELON_TILES if position not in INNER_MELONS
)
_OUTER_MELON_SWEEP: tuple[Position, ...] = (
    (2, 4), (3, 3), (4, 2), (5, 2), (6, 3), (7, 4),
)

# Literal 10x5 layout (rows y=0..4):
#   ..CCWWCC..
#   .CCWWWWCC.
#   CCWGMMCCCC
#   CWWMMMMWWC
#   WWMMMMMMWW
WHEAT_TILES: tuple[Position, ...] = (
    (4, 0), (5, 0),
    (3, 1), (4, 1), (5, 1), (6, 1),
    (2, 2),
    (1, 3), (2, 3), (7, 3), (8, 3),
    (0, 4), (1, 4), (8, 4), (9, 4),
)
WHEAT_DAY3_TILE: Position = (3, 1)
DAY4_PASTURES: tuple[Position, ...] = ((2, 3), (4, 1), (5, 1))

CARROT_TILES: tuple[Position, ...] = (
    (2, 0), (3, 0), (6, 0), (7, 0),
    (1, 1), (2, 1), (7, 1), (8, 1),
    (0, 2), (1, 2), (6, 2), (7, 2), (8, 2), (9, 2),
    (0, 3), (9, 3),
)

GOOSE_TILE: Position = (3, 2)
_SHED_NW: Position = (4, 4)
_DAY1_STARTS: tuple[Position, ...] = (
    # Worker 0 moves before the hour-0 hires resolve, leaving all four shed
    # access cells empty for the official min-occupancy spawn rule.
    (4, 4), (4, 4), (5, 4), (4, 5), (5, 5), (4, 4),
)
_DAY3_STARTS: tuple[Position, ...] = (
    (4, 4), (5, 4), (4, 5), (5, 5), (4, 4),
)
_DAY4_STARTS: tuple[Position, ...] = (
    (4, 4), (5, 4), (4, 5), (5, 5), (4, 4), (5, 4),
    (4, 5), (5, 5), (4, 4),
)


class ScriptedOpeningError(RuntimeError):
    """The fixed opening no longer matches the observed game contract."""


@dataclass(frozen=True)
class _Visit:
    position: Position
    actions: tuple[Action, ...]


def choose_day4_animals(obs: Any) -> list[str]:
    """The sole future strategy hook; intentionally ignores the shop."""

    del obs
    return ["COW", "COW", "SHEEP"]


def _move(origin: Position, target: Position) -> tuple[Action, ...]:
    actions: list[Action] = []
    x, y = origin
    while x != target[0]:
        operation = "EAST" if x < target[0] else "WEST"
        actions.append((operation,))
        x += 1 if operation == "EAST" else -1
    while y != target[1]:
        operation = "SOUTH" if y < target[1] else "NORTH"
        actions.append((operation,))
        y += 1 if operation == "SOUTH" else -1
    return tuple(actions)


def _route(start: Position, visits: tuple[_Visit, ...]) -> tuple[Action, ...]:
    result: list[Action] = []
    current = start
    for visit in visits:
        result.extend(_move(current, visit.position))
        result.extend(visit.actions)
        current = visit.position
    return tuple(result)


def _crop_visits(
    crop: str, positions: tuple[Position, ...], *, harvest: bool = False
) -> tuple[_Visit, ...]:
    actions: tuple[Action, ...] = (
        (("WATER",), ("HARVEST",))
        if harvest else (("PLANT", crop), ("WATER",))
    )
    return tuple(_Visit(position, actions) for position in positions)


def _water_visits(positions: tuple[Position, ...]) -> tuple[_Visit, ...]:
    return tuple(_Visit(position, (("WATER",),)) for position in positions)


def _drop(position: Position = _SHED_NW) -> _Visit:
    return _Visit(position, (("DROP",),))


def _day1_routes() -> tuple[tuple[Action, ...], ...]:
    visits = (
        (*_crop_visits("MELON", ((3, 4), (2, 4))),
         *_crop_visits("WHEAT", ((2, 3), (1, 3), (1, 4), (0, 4))),
         *_crop_visits("CARROT", ((0, 3), (0, 2)))),
        (_Visit(_SHED_NW, (("PICKUP", "GOOSE", 1),)),
         *_crop_visits("MELON", ((3, 3),)),
         _Visit(GOOSE_TILE, (("BUILD_COOP",), ("PLACE", "GOOSE"))),
         *_crop_visits("WHEAT", ((2, 2),)),
         *_crop_visits("CARROT", ((1, 2), (1, 1), (2, 1), (2, 0)))),
        _crop_visits(
            "CARROT",
            ((6, 2), (7, 2), (7, 1), (8, 1), (8, 2), (9, 2), (9, 3)),
        ),
        (*_crop_visits("MELON", ((4, 3), (4, 2))),
         *_crop_visits("WHEAT", ((4, 1), (3, 1))),
         *_crop_visits("CARROT", ((3, 0),)),
         *_crop_visits("WHEAT", ((4, 0), (5, 0)))),
        (*_crop_visits("MELON", ((7, 4), (6, 4), (6, 3))),
         *_crop_visits("WHEAT", ((7, 3), (8, 3), (8, 4), (9, 4)))),
        (*_crop_visits("MELON", ((4, 4), (5, 4), (5, 3), (5, 2))),
         *_crop_visits("WHEAT", ((5, 1), (6, 1))),
         *_crop_visits("CARROT", ((6, 0), (7, 0)))),
    )
    return tuple(_route(start, route) for start, route in zip(_DAY1_STARTS, visits))


def _day2_routes() -> tuple[tuple[Action, ...], ...]:
    visits = (
        _Visit(_SHED_NW, (("PICKUP", "WHEAT", 1),)),
        *_water_visits(((2, 4), (3, 3))),
        _Visit(GOOSE_TILE, (
            ("FEED",), ("CARE",), ("COLLECT_FERTILIZER",),
        )),
        *_water_visits(((4, 2), (5, 2), (6, 3), (7, 4))),
    )
    return (_route(_SHED_NW, visits),)


def _day3_routes() -> tuple[tuple[Action, ...], ...]:
    left_carrots = ((0, 3), (0, 2), (1, 2), (1, 1),
                    (2, 1), (2, 0), (3, 0), (6, 0))
    right_carrots = ((9, 3), (9, 2), (8, 2), (8, 1),
                     (7, 1), (7, 0), (7, 2), (6, 2))
    left_wheat = ((2, 3), (1, 3), (1, 4), (0, 4),
                  (2, 2), (4, 1), (4, 0))
    right_wheat = ((8, 4), (9, 4), (8, 3), (7, 3),
                   (6, 1), (5, 1), (5, 0))
    main = _route(_DAY3_STARTS[0], (
        *_water_visits(INNER_MELONS),
        _Visit(WHEAT_DAY3_TILE, (("WATER",), ("HARVEST",))),
        _Visit(GOOSE_TILE, (
            ("FEED",), ("CARE",), ("COLLECT_FERTILIZER",),
        )),
    ))
    return (
        main,
        _route(_DAY3_STARTS[1], _water_visits(right_wheat)),
        _route(_DAY3_STARTS[2], _water_visits(left_carrots)),
        _route(_DAY3_STARTS[3], _water_visits(right_carrots)),
        _route(_DAY3_STARTS[4], _water_visits(left_wheat)),
    )


def _day4_routes() -> tuple[tuple[Action, ...], ...]:
    # Literal first phase supplied for the financing batch and fixed pastures.
    # Keep the order: its exact hour-15 deposits fund the animal purchase.
    early = (
        (
            ("PICKUP", "WHEAT", 1),
            ("WEST",), ("WEST",), ("NORTH",),
            ("WATER",), ("HARVEST",), ("BUILD_PASTURE",),
            ("EAST",), ("NORTH",),
            ("FEED",), ("COLLECT_FERTILIZER",), ("CARE",),
            ("EAST",), ("SOUTH",), ("SOUTH",), ("DROP",),
        ),
        (
            ("EAST",), ("EAST",), ("NORTH",), ("NORTH",),
            ("WATER",), ("HARVEST",), ("EAST",),
            ("WATER",), ("HARVEST",),
            ("WEST",), ("WEST",), ("WEST",),
            ("SOUTH",), ("SOUTH",), ("DROP",),
        ),
        (
            ("WEST",), ("WEST",), ("NORTH",), ("WATER",),
            ("WEST",), ("NORTH",), ("NORTH",),
            ("WATER",), ("HARVEST",), ("WEST",),
            ("WATER",), ("HARVEST",), ("SOUTH",),
            ("WATER",), ("HARVEST",),
        ),
        (
            ("EAST",), ("NORTH",), ("NORTH",), ("WATER",),
            ("NORTH",), ("NORTH",), ("NORTH",),
            ("WATER",), ("HARVEST",),
            ("WEST",), ("SOUTH",), ("SOUTH",), ("SOUTH",), ("SOUTH",),
            ("DROP",),
        ),
        (
            ("NORTH",), ("NORTH",), ("NORTH",),
            ("WATER",), ("HARVEST",), ("BUILD_PASTURE",),
            ("EAST",),
            ("WATER",), ("HARVEST",), ("BUILD_PASTURE",),
            ("SOUTH",), ("WATER",), ("SOUTH",), ("SOUTH",), ("DROP",),
        ),
        (
            ("EAST",), ("EAST",), ("NORTH",), ("NORTH",), ("NORTH",),
            ("WATER",), ("HARVEST",), ("WEST",), ("SOUTH",),
            ("WATER",), ("HARVEST",),
            ("WEST",), ("SOUTH",), ("SOUTH",), ("DROP",),
        ),
        (
            ("NORTH",), ("NORTH",), ("NORTH",), ("WATER",),
            ("WEST",), ("WEST",), ("NORTH",),
            ("WATER",), ("HARVEST",),
            ("EAST",), ("EAST",), ("SOUTH",), ("SOUTH",), ("SOUTH",),
            ("DROP",),
        ),
        (
            ("EAST",), ("EAST",), ("NORTH",), ("WATER",),
            ("EAST",), ("EAST",), ("NORTH",),
            ("WATER",), ("HARVEST",),
            ("WEST",), ("WEST",), ("WEST",), ("WEST",), ("SOUTH",),
            ("DROP",),
        ),
        (
            ("WEST",), ("NORTH",), ("WATER",),
            ("NORTH",), ("NORTH",), ("NORTH",),
            ("WATER",), ("HARVEST",),
            ("EAST",), ("SOUTH",), ("SOUTH",), ("SOUTH",), ("SOUTH",),
            ("DROP",),
        ),
    )
    schedules = [list(route) for route in early]

    def append_at(worker: int, hour: int, actions: tuple[Action, ...]) -> None:
        index = hour if worker == 0 else hour - 1
        if len(schedules[worker]) > index:
            raise ScriptedOpeningError(
                f"Day 4 worker {worker} overlaps hour {hour}"
            )
        schedules[worker].extend(
            (("PASS",),) * (index - len(schedules[worker]))
        )
        schedules[worker].extend(actions)

    # Animals bought during hour 15 become available at hour 16.  Hand 8
    # reserves Wheat at hour 15; main and hand 1 take the other two at hour 16.
    append_at(0, 16, (
        ("PICKUP", "WHEAT", 1), ("PICKUP", "SHEEP", 1),
        ("WEST",), ("WEST",), ("NORTH",),
        ("PLACE", "SHEEP"), ("FEED",), ("CARE",),
    ))
    append_at(1, 16, (
        ("PICKUP", "WHEAT", 1), ("PICKUP", "COW", 1),
        ("NORTH",), ("NORTH",), ("NORTH",),
        ("PLACE", "COW"), ("FEED",), ("CARE",),
    ))
    append_at(8, 15, (
        ("PICKUP", "WHEAT", 1), ("PICKUP", "COW", 1),
        ("NORTH",), ("NORTH",), ("NORTH",),
        ("PLACE", "COW"), ("FEED",), ("CARE",),
    ))

    # Five carrots remain outside the financing batch.  Complete them in
    # place; daily refresh deposits their output, so no return trip is added.
    append_at(2, 16, _route((0, 3), (
        _Visit((1, 1), (("WATER",), ("HARVEST",))),
    )))
    append_at(3, 16, _route((5, 4), (
        _Visit((7, 0), (("WATER",), ("HARVEST",))),
    )))
    append_at(6, 16, _route((4, 4), (
        _Visit((2, 0), (("WATER",), ("HARVEST",))),
    )))
    append_at(5, 16, _route((5, 4), (
        _Visit((8, 1), (("WATER",), ("HARVEST",))),
    )))
    append_at(7, 16, _route((5, 4), (
        _Visit((9, 2), (("WATER",), ("HARVEST",))),
    )))
    return tuple(tuple(schedule) for schedule in schedules)


_ROUTES = {
    0: _day1_routes(),
    1: _day2_routes(),
    2: _day3_routes(),
    3: _day4_routes(),
}
def _check_routes() -> None:
    expected_workers = {0: 6, 1: 1, 2: 5, 3: 9}
    for day, routes in _ROUTES.items():
        if len(routes) != expected_workers[day]:
            raise ScriptedOpeningError(
                f"Day {day + 1}: expected {expected_workers[day]} fixed routes, "
                f"got {len(routes)}"
            )
        for worker, route in enumerate(routes):
            start = 0 if worker == 0 else 1
            if start + len(route) > 24:
                raise ScriptedOpeningError(
                    f"Day {day + 1} worker {worker} exceeds the day by "
                    f"{start + len(route) - 24} turn(s): "
                    f"start={start}, actions={len(route)}"
                )
    # With only four hands, 89 services plus the minimum movement between 44
    # distinct service cells already exceed the five workers' action slots.
    fewer_worker_lower_bound = 89 + (44 - DAY1_HANDS)
    fewer_worker_capacity = 24 + (DAY1_HANDS - 1) * 23
    if fewer_worker_lower_bound <= fewer_worker_capacity:
        raise ScriptedOpeningError(
            "Day 1 minimum-hand certificate no longer proves five hands are "
            "necessary"
        )


_check_routes()


def day1_required_hands() -> int:
    """Return the checked hand count for these literal Day-1 routes."""

    return DAY1_HANDS


def route_lengths() -> Mapping[int, tuple[int, ...]]:
    """Expose fixed route lengths for the watcher's diagnostic header."""

    return {day: tuple(len(route) for route in routes)
            for day, routes in _ROUTES.items()}


def _market_orders(day: int, hour: int, obs: Any) -> tuple[Action, ...]:
    if day == 0 and hour == 0:
        return (
            ("BUY_LAND",),
            ("BUY_SEED", "MELON", 12),
            ("BUY_SEED", "CARROT", 16),
            ("BUY_SEED", "WHEAT", 15),
            ("BUY_ANIMAL", "GOOSE", 1),
            *(("HIRE",) for _ in range(DAY1_HANDS)),
        )
    if day == 0 and hour == 1:
        return (("BUY_PRODUCT", "WHEAT", 1),)
    if day == 2 and hour == 0:
        return tuple(("HIRE",) for _ in range(DAY3_HANDS))
    if day == 3 and hour == 0:
        animals = choose_day4_animals(obs)
        if animals != ["COW", "COW", "SHEEP"]:
            raise ScriptedOpeningError(
                f"Day 4 animal hook returned unsupported fixed choice: {animals!r}"
            )
        return tuple(("HIRE",) for _ in range(DAY4_HANDS))
    if day == 3 and hour == 1:
        return (("SELL", "FERTILIZER", 2),)
    if day == 3 and hour == 15:
        # By this market phase the fixed unit actions have made exactly the
        # financing batch available and reserved three Wheat for animal feed.
        return (
            ("SELL", "CARROT", 24),
            ("SELL", "WHEAT", 6),
            ("SELL", "FERTILIZER", 1),
            ("BUY_ANIMAL", "COW", 2),
            ("BUY_ANIMAL", "SHEEP", 1),
        )
    return ()


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def scripted_opening_action(obs: Any) -> dict[str, list[Any]]:
    """Return the literal action for this observation's day and hour."""

    day = int(_value(obs, "day", 0))
    hour = int(_value(obs, "hour", 0))
    farms = _value(obs, "farms", ()) or ()
    player = int(_value(obs, "player", 0))
    hand_count = len(_value(farms[player], "hands", ()) or ())
    worker_count = hand_count + 1

    if day >= 4:
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in range(hand_count)],
            "market": [],
        }

    routes = _ROUTES[day]
    expected = len(routes)
    if worker_count != expected and not (hour == 0 and day in (0, 2, 3)):
        raise ScriptedOpeningError(
            f"Day {day + 1} turn {hour}: expected {expected} workers, "
            f"observed {worker_count}"
        )
    actions: list[Action] = []
    for worker in range(worker_count):
        start = 0 if worker == 0 else 1
        index = hour - start
        route = routes[worker]
        actions.append(route[index] if 0 <= index < len(route) else ("PASS",))
    orders = _market_orders(day, hour, obs)
    return {
        "farmer": list(actions[0]),
        "hands": [list(action) for action in actions[1:]],
        "market": [list(order) for order in orders],
    }


def agent(obs: Any) -> dict[str, list[Any]]:
    """Kaggle agent entry for the fixed opening machine."""

    return scripted_opening_action(obs)
