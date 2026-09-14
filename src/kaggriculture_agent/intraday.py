"""Production intraday executor backed only by fixed zonal road templates.

Macro choices are frozen before this module runs. The executor groups same-tile
events, selects one authored tile-to-zone road template, crops inactive cells,
and materializes exact worker actions. There is no whole-farm assignment, pair
repair, nearest-neighbour route, or legacy fallback path.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from types import MappingProxyType
from typing import Iterable, Mapping

from . import rules
from . import zonal_templates as zonal
from .programme import Programme, ProgrammeEvent, WorkerRoute
from .state import Position, State


class PlanningFailure(RuntimeError):
    pass


_INTRADAY_CACHE: dict[tuple[object, ...], Programme | None] = {}


def clear_route_cache() -> None:
    _INTRADAY_CACHE.clear()


_ACTION_ORDER = {
    "FERTILIZE": 0, "WATER": 1, "FEED": 1, "CARE": 2,
    "HARVEST": 3, "COLLECT_FERTILIZER": 4, "DIG": 5,
    "BUILD_COOP": 6, "BUILD_PASTURE": 6, "PLACE": 7, "PLANT": 8,
}
_MOVES = {
    (0, -1): ("NORTH",), (0, 1): ("SOUTH",),
    (-1, 0): ("WEST",), (1, 0): ("EAST",),
}


@dataclass(frozen=True)
class _Bundle:
    tile: Position
    day: int
    release: int
    deadline: int
    events: tuple[ProgrammeEvent, ...]


def placement_load(programme: Programme) -> dict[Position, int]:
    load: Counter[Position] = Counter()
    assets = {asset.asset_id: asset for asset in programme.assets}
    for event in programme.events:
        if event.tile is not None and event.action and event.action[0] != "PASS":
            load[event.tile] += 1
        if event.asset_id in assets and event.kind in {"HARVEST", "DROP"}:
            load[assets[event.asset_id].tile] += 1
    return dict(load)


def _bundles(state: State, programme: Programme) -> tuple[_Bundle, ...]:
    grouped: dict[tuple[int, Position], list[ProgrammeEvent]] = defaultdict(list)
    for event in programme.events:
        if event.tile is None or not event.action or event.action[0] == "PASS":
            continue
        grouped[(event.step // rules.TURNS_PER_DAY, event.tile)].append(event)
    result = []
    for (day, tile), events in sorted(grouped.items()):
        events.sort(key=lambda event: (
            event.step, _ACTION_ORDER.get(str(event.action[0]), 99),
            event.event_id))
        first = state.step if day == state.day else day * rules.TURNS_PER_DAY
        result.append(_Bundle(
            tile, day,
            max(first, min(event.step for event in events)),
            min(event.deadline if event.deadline is not None
                else (day + 1) * rules.TURNS_PER_DAY - 1 for event in events),
            tuple(events)))
    return tuple(result)


def _needed(events: Iterable[ProgrammeEvent]) -> Counter[str]:
    needed: Counter[str] = Counter()
    for event in events:
        action = event.action
        operation = str(action[0]) if action else "PASS"
        if operation == "FEED":
            needed["WHEAT"] += 1
        elif operation == "FERTILIZE":
            needed["FERTILIZER"] += 1
        elif operation == "PLACE" and len(action) > 1:
            needed[str(action[1])] += 1
    return needed


def _preload_needed(events: Iterable[ProgrammeEvent]) -> Counter[str]:
    """Minimum opening carry after crediting earlier same-route production."""
    balance: Counter[str] = Counter()
    required: Counter[str] = Counter()
    for event in events:
        operation = str(event.action[0]) if event.action else "PASS"
        if operation == "FEED":
            item, delta = "WHEAT", -1
        elif operation == "FERTILIZE":
            item, delta = "FERTILIZER", -1
        elif operation == "PLACE":
            item, delta = str(event.action[1]), -1
        elif operation == "HARVEST":
            item, delta = event.item or "UNKNOWN", max(1, event.quantity)
        elif operation == "COLLECT_FERTILIZER":
            item, delta = "FERTILIZER", 1
        else:
            continue
        balance[item] += delta
        required[item] = max(required[item], -balance[item])
    return +required


def _output_units(events: Iterable[ProgrammeEvent]) -> int:
    return sum(max(1, int(event.quantity)) if event.action[0] == "HARVEST" else 1
               for event in events if event.action and event.action[0] in {
                   "HARVEST", "COLLECT_FERTILIZER"})


def _workload(programme: Programme, bundles: Iterable[_Bundle], day: int
              ) -> Mapping[Position, zonal.TileWorkload]:
    forced = programme.must_return.get(day, frozenset())
    return MappingProxyType({bundle.tile: zonal.TileWorkload(
        actions=len(bundle.events), output_units=_output_units(bundle.events),
        pickup_items=frozenset(_needed(bundle.events)),
        must_return=bundle.tile in forced)
        for bundle in bundles})


def _effective_land_mask(state: State, programme: Programme) -> zonal.LandMask:
    owned = set(state.own.owned_land)
    owned.update(land.quadrant for land in programme.land)
    land_buys = sum(1 for event in programme.events
                    if event.kind == "BUY_LAND" and event.item not in owned)
    while land_buys > 0 and len(owned) < 4:
        owned.add(rules.LAND_ORDER[len(owned) - 1])
        land_buys -= 1
    ordered = tuple(q for q in ("NW", "NE", "SW", "SE") if q in owned)
    if ordered == ("NW",):
        return zonal.NW
    if ordered in {zonal.NORTH, zonal.THREE_LAND}:
        return ordered
    if ordered == zonal.FULL:
        raise PlanningFailure("four-land zonal runtime is not enabled")
    raise PlanningFailure(f"no fixed-road family for land mask {ordered!r}")


class _ResourceCalendar:
    """Shared shed arrivals/reservations; worker carry remains private."""

    def __init__(self, state: State, programme: Programme,
                 drops: Mapping[int, Mapping[str, int]] | None):
        self.arrivals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for item, quantity in state.shed.items():
            if quantity > 0:
                self.arrivals[item].append((state.step, int(quantity)))
        for event in programme.events:
            if (event.kind in {"BUY_PRODUCT", "BUY_ANIMAL"}
                    and event.item and event.quantity > 0):
                self.arrivals[event.item].append((event.step + 1, event.quantity))
        for step, amounts in (drops or {}).items():
            for item, quantity in amounts.items():
                if quantity > 0:
                    # Conservatively expose a cross-worker DROP on the next
                    # turn. Same-turn reuse depends on worker action order;
                    # delaying one turn is always legal and deterministic.
                    self.arrivals[item].append((step + 1, int(quantity)))
        for timeline in self.arrivals.values():
            timeline.sort()
        self.reserved: Counter[str] = Counter()

    def available_step(self, item: str, quantity: int) -> int | None:
        target, running = self.reserved[item] + quantity, 0
        for step, amount in self.arrivals.get(item, ()):
            running += amount
            if running >= target:
                return step
        return None

    def reserve(self, item: str, quantity: int) -> None:
        self.reserved[item] += quantity


def _worker_starts(state: State, day: int, workforce: int) -> tuple[Position, ...]:
    access = rules.shed_access(state.board_size)
    positions = ([worker.position for worker in state.workers]
                 if day == state.day else [access[0]])
    while len(positions) < workforce:
        positions.append(min(access, key=lambda point: (
            positions.count(point), access.index(point))))
    return tuple(positions[:workforce])


def _assign_workers(starts: tuple[Position, ...], solution: zonal.TemplateSolution,
                    required_return_workers: frozenset[int] = frozenset()
                    ) -> Mapping[str, int]:
    """Exact small worker-to-zone assignment with no persistent ownership."""
    zone_ids = tuple(sorted(solution.template.zones))
    count = len(zone_ids)
    costs = tuple(tuple(
        (10_000 if worker in required_return_workers
         and not solution.template.zones[zone_id].returns_to_shed else 0)
        + rules.manhattan(
            starts[worker], solution.template.zones[zone_id].shed_access)
        for worker in range(count)) for zone_id in zone_ids)

    @lru_cache(maxsize=None)
    def best(index: int, used: int) -> tuple[int, tuple[int, ...]]:
        if index == count:
            return 0, ()
        return min((costs[index][worker] + best(
            index + 1, used | (1 << worker))[0],
            (worker, *best(index + 1, used | (1 << worker))[1]))
            for worker in range(count) if not used & (1 << worker))

    return MappingProxyType(dict(zip(zone_ids, best(0, 0)[1])))


def _path_actions(path: tuple[Position, ...]) -> tuple[tuple[str], ...]:
    actions = []
    for left, right in zip(path, path[1:]):
        delta = (right[0] - left[0], right[1] - left[1])
        if delta not in _MOVES:
            raise PlanningFailure(f"zonal road contains a jump: {left}->{right}")
        actions.append(_MOVES[delta])
    return tuple(actions)


def _move_actions(origin: Position, target: Position) -> tuple[tuple[str], ...]:
    x, y = origin
    points = [(x, y)]
    while x != target[0]:
        x += 1 if x < target[0] else -1
        points.append((x, y))
    while y != target[1]:
        y += 1 if y < target[1] else -1
        points.append((x, y))
    return _path_actions(tuple(points))


def _materialize_zone(state: State, day: int, worker: int,
                      zone: zonal.ZoneRoad,
                      compiled: zonal.CompiledZoneRoute,
                      bundles: Mapping[Position, _Bundle],
                      calendar: _ResourceCalendar,
                      workforce: int) -> WorkerRoute | None:
    day_first = max(state.step, day * rules.TURNS_PER_DAY)
    day_last = min((day + 1) * rules.TURNS_PER_DAY - 1,
                   rules.TERMINAL_ACTION_STEP)
    start = _worker_starts(state, day, workforce)[worker]
    # When hiring, every existing worker passes on the hire turn. This keeps
    # the shed-access occupancy used by the pinned spawn rule equal to the
    # assignment prediction; hired workers become actionable next turn.
    hiring = (workforce > len(state.workers) if day == state.day
              else workforce > 1)
    cursor = day_first + int(hiring)
    actions: dict[int, tuple[object, ...]] = {}
    inventory = Counter(state.workers[worker].inventory
                        if day == state.day and worker < len(state.workers) else {})

    def emit(action: tuple[object, ...]) -> bool:
        nonlocal cursor
        if cursor > day_last:
            return False
        actions[cursor] = action
        cursor += 1
        return True

    for action in _move_actions(start, zone.shed_access):
        if not emit(action):
            return None
    all_events = tuple(event for tile in compiled.task_order
                       for event in bundles[tile].events)
    missing = _preload_needed(all_events) - inventory
    for item in sorted(missing):
        quantity = missing[item]
        available = calendar.available_step(item, quantity)
        if available is None:
            return None
        cursor = max(cursor, available)
        if not emit(("PICKUP", item, quantity)):
            return None
        inventory[item] += quantity
        calendar.reserve(item, quantity)

    task_tiles, executed = set(compiled.task_order), set()

    def execute_tile(current: Position) -> bool:
        nonlocal cursor
        if current not in task_tiles or current in executed:
            return True
        bundle = bundles[current]
        cursor = max(cursor, bundle.release)
        for event in bundle.events:
            cursor = max(cursor, event.step)
            deadline = event.deadline if event.deadline is not None else bundle.deadline
            if cursor > min(deadline, day_last) or not emit(tuple(event.action)):
                return False
            operation = str(event.action[0])
            if operation == "FEED":
                inventory["WHEAT"] -= 1
            elif operation == "FERTILIZE":
                inventory["FERTILIZER"] -= 1
            elif operation == "PLACE":
                inventory[str(event.action[1])] -= 1
            elif operation == "HARVEST":
                inventory[event.item or "UNKNOWN"] += max(1, event.quantity)
            elif operation == "COLLECT_FERTILIZER":
                inventory["FERTILIZER"] += 1
            if min(inventory.values(), default=0) < 0:
                return False
        executed.add(current)
        return True

    positions = compiled.positions or (zone.shed_access,)
    current = zone.shed_access
    if not execute_tile(current):
        return None
    for point, movement in zip(positions[1:], _path_actions(positions)):
        if not emit(movement):
            return None
        current = point
        if not execute_tile(current):
            return None
    if executed != task_tiles:
        return None
    if zone.returns_to_shed:
        if current != zone.shed_access or not emit(("DROP",)):
            return None
        if cursor - 1 > day * rules.TURNS_PER_DAY + 22:
            return None
    return WorkerRoute(worker, day,
                       f"{'INNER' if zone.returns_to_shed else 'OUTER'}:{zone.zone_id}",
                       actions)


def _candidate_solutions(mask: zonal.LandMask, workforce: int,
                         workload: Mapping[Position, zonal.TileWorkload]
                         ) -> tuple[zonal.TemplateSolution, ...]:
    candidates = tuple(filter(None, (zonal.solve_template(template, workload)
        for template in zonal.templates_for(mask, workforce))))
    return tuple(sorted(candidates, key=lambda result: (
        result.max_finish_turn, result.total_movement,
        sum(zone.returns_to_shed for zone in result.template.zones.values()),
        result.template.template_id)))


def _compile_day(state: State, programme: Programme, mask: zonal.LandMask,
                 day: int, bundles: tuple[_Bundle, ...], workforce: int,
                 drops: Mapping[int, Mapping[str, int]] | None
                 ) -> tuple[tuple[WorkerRoute, ...], zonal.TemplateSolution] | None:
    workload = _workload(programme, bundles, day)
    by_tile = {bundle.tile: bundle for bundle in bundles}
    starts = _worker_starts(state, day, workforce)
    terminal_day = rules.TERMINAL_ACTION_STEP // rules.TURNS_PER_DAY
    required_return_workers = frozenset(
        worker.index for worker in state.workers if worker.inventory
    ) if day == state.day == terminal_day else frozenset()
    for solution in _candidate_solutions(mask, workforce, workload):
        if sum(zone.returns_to_shed for zone in solution.template.zones.values()) \
                < len(required_return_workers):
            continue
        assignment = _assign_workers(starts, solution, required_return_workers)
        if any(not solution.template.zones[zone_id].returns_to_shed
               for zone_id, worker in assignment.items()
               if worker in required_return_workers):
            continue
        calendar = _ResourceCalendar(state, programme, drops)
        routes = []
        order = sorted(solution.template.zones, key=lambda zone_id: (
            min((by_tile[tile].deadline for tile in
                 solution.routes[zone_id].task_order),
                default=(day + 1) * rules.TURNS_PER_DAY), zone_id))
        for zone_id in order:
            route = _materialize_zone(
                state, day, assignment[zone_id], solution.template.zones[zone_id],
                solution.routes[zone_id], by_tile, calendar, workforce)
            if route is None:
                break
            routes.append(route)
        else:
            return tuple(routes), solution
    return None


def _compile(state: State, programme: Programme, mask: zonal.LandMask,
             bundles: tuple[_Bundle, ...], workforce: int,
             drops: Mapping[int, Mapping[str, int]] | None
             ) -> tuple[tuple[WorkerRoute, ...], tuple[zonal.TemplateSolution, ...]] | None:
    routes, solutions = [], []
    for day in sorted({bundle.day for bundle in bundles}) or [state.day]:
        result = _compile_day(state, programme, mask, day,
                              tuple(b for b in bundles if b.day == day),
                              workforce, drops)
        if result is None:
            return None
        day_routes, solution = result
        routes.extend(day_routes)
        solutions.append(solution)
    return tuple(routes), tuple(solutions)


def _cache_key(state: State, programme: Programme, complete_actions: bool
               ) -> tuple[object, ...]:
    return (state.step, tuple(state.own.owned_land),
        tuple((worker.position, tuple(sorted(worker.inventory.items())))
              for worker in state.workers),
        tuple(sorted((event.step, event.priority, event.event_id, event.kind,
                      event.tile, event.asset_id, event.item, event.quantity,
                      event.action, event.deadline, event.source)
                     for event in programme.events if event.kind != "HIRE")),
        tuple(sorted((day, tuple(sorted(tiles)))
                     for day, tiles in programme.must_return.items())),
        tuple((land.quadrant, land.buy_step) for land in programme.land),
        complete_actions)


def solve_intraday(state: State, programme: Programme, *,
                   complete_actions: bool = True) -> Programme:
    key = _cache_key(state, programme, complete_actions)
    if key in _INTRADAY_CACHE:
        cached = _INTRADAY_CACHE[key]
        if cached is None:
            raise PlanningFailure("no fixed zonal template realizes the programme")
        return cached
    mask, bundles = _effective_land_mask(state, programme), _bundles(state, programme)
    from .simulation import drop_arrivals

    seed_drops: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for event in programme.events:
        if (event.tile is not None and event.source in {"FEED_SUPPLY", "F_RELAY"}
                and event.action and event.action[0] in {
                    "HARVEST", "COLLECT_FERTILIZER"}):
            seed_drops[event.step][event.item or "FERTILIZER"] += max(1, event.quantity)
    initial_drops = {step: dict(amounts) for step, amounts in seed_drops.items()}

    for workforce in range(len(state.workers), 13):
        drops: Mapping[int, Mapping[str, int]] = initial_drops
        compiled = None
        stable = False
        for _ in range(3 if complete_actions else 1):
            compiled = _compile(state, programme, mask, bundles, workforce, drops)
            if compiled is None:
                break
            routes, solutions = compiled
            actual = drop_arrivals(state, replace(
                programme, routes=routes, worker_count=workforce))
            if actual == drops:
                stable = True
                break
            drops = actual
        if compiled is None or not stable:
            continue
        routes, solutions = compiled
        if complete_actions:
            complete: dict[tuple[int, int], dict[int, tuple[object, ...]]] = {}
            zone_by_worker = {}
            for route in routes:
                complete[(route.day, route.worker)] = dict(route.actions)
                zone_by_worker[(route.day, route.worker)] = route.zone
            for day in sorted({bundle.day for bundle in bundles}) or [state.day]:
                first = max(state.step, day * 24)
                last = min((day + 1) * 24 - 1, rules.TERMINAL_ACTION_STEP)
                for worker in range(workforce):
                    actions = complete.setdefault((day, worker), {})
                    for step in range(first, last + 1):
                        actions.setdefault(step, ("PASS",))
            routes = tuple(WorkerRoute(
                worker, day, zone_by_worker.get((day, worker), "OUTER:IDLE"),
                dict(sorted(actions.items())))
                for (day, worker), actions in sorted(complete.items()))

        hires = max(0, workforce - len(state.workers))
        hire_events = (() if not hires else (ProgrammeEvent(
            state.step, -100, f"hire:{state.day}", "HIRE", quantity=hires),))
        events = tuple(sorted({event.event_id: event for event in (
            *(event for event in programme.events if event.kind != "HIRE"),
            *hire_events)}.values()))
        selected = solutions[0]
        inner = frozenset(tile for tile, zone_id in selected.template.tile_to_zone.items()
                          if selected.template.zones[zone_id].returns_to_shed)
        result = replace(programme, events=events, worker_count=workforce,
            inner=inner, outer=frozenset(selected.template.tile_to_zone) - inner,
            placement_load=placement_load(programme),
            return_mode={zone_id: ("MIDDAY_RETURN" if zone.returns_to_shed
                                   else "EOD_RETURN")
                         for zone_id, zone in selected.template.zones.items()},
            partition_template_id=selected.template.template_id,
            zone_assignment=selected.template.tile_to_zone, routes=routes)
        _INTRADAY_CACHE[key] = result
        return result
    _INTRADAY_CACHE[key] = None
    raise PlanningFailure("no fixed zonal template realizes every mandatory task")
