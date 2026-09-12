"""Deterministic executor for a frozen Programme.

It assigns workers, orders fixed-tile bundles, and materializes exact pickup,
movement, drop and market-checkpoint routes. It never changes macro choices.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Iterable

from . import rules
from .programme import Programme, ProgrammeEvent, WorkerRoute
from .state import Position, State


class PlanningFailure(RuntimeError):
    pass


_DAY_ROUTE_CACHE = {}
_INTRADAY_CACHE = {}


def clear_route_cache():
    _DAY_ROUTE_CACHE.clear()
    _INTRADAY_CACHE.clear()


_ACTION_ORDER = {
    "FERTILIZE": 0, "WATER": 1, "FEED": 1, "CARE": 2,
    "HARVEST": 3, "COLLECT_FERTILIZER": 4, "DIG": 5,
    "BUILD_COOP": 6, "BUILD_PASTURE": 6, "PLACE": 7, "PLANT": 8,
}


@dataclass(frozen=True)
class _Bundle:
    tile: Position
    day: int
    release: int
    deadline: int
    events: tuple[ProgrammeEvent, ...]
    zone: str
    return_required: bool


def _move_path(origin: Position, target: Position):
    x, y = origin
    while x != target[0]:
        if x < target[0]:
            x += 1; yield ("EAST",)
        else:
            x -= 1; yield ("WEST",)
    while y != target[1]:
        if y < target[1]:
            y += 1; yield ("SOUTH",)
        else:
            y -= 1; yield ("NORTH",)


def _nearest_access(position: Position, size: int) -> Position:
    return min(rules.shed_access(size), key=lambda p: (rules.manhattan(position, p), p[1], p[0]))


def placement_load(programme: Programme):
    load = Counter()
    assets = {asset.asset_id: asset for asset in programme.assets}
    for event in programme.events:
        if event.tile is not None and event.kind not in {"OUTPUT", "FIELD_STOCK", "SALE"}:
            load[event.tile] += 1
            if event.action and event.action[0] in {"FEED", "FERTILIZE", "PLACE"}:
                load[event.tile] += 1
        if event.asset_id in assets and event.kind in {"HARVEST", "DROP"}:
            load[assets[event.asset_id].tile] += 1
    return dict(load)


def return_modes(programme: Programme):
    sales = programme.planned_sale
    result = {}
    for asset in programme.assets:
        mode = "EOD_RETURN"
        events = asset.service_schedule + asset.harvest_schedule
        for event in events:
            same_day_sale = any(
                step // rules.TURNS_PER_DAY == event.step // rules.TURNS_PER_DAY
                and amounts.get(event.item or "", 0) > 0
                for step, amounts in sales.items()
            )
            needs_shed = event.action and event.action[0] in {"FEED", "FERTILIZE", "PLACE"}
            if (event.action and event.action[0] == "HARVEST" and same_day_sale) or needs_shed:
                mode = "MIDDAY_RETURN"
                break
        result[asset.asset_id] = mode
    return result


def zones(state: State, programme: Programme, modes):
    programme_tiles = set(state.own.usable_tiles)
    programme_tiles.update(asset.tile for asset in programme.assets)
    ordered = sorted(programme_tiles,
                     key=lambda p: (rules.distance_to_shed(p, state.board_size), p[1], p[0]))
    midday = [asset.tile for asset in programme.assets if modes.get(asset.asset_id) == "MIDDAY_RETURN"]
    if not midday:
        return frozenset(), frozenset(ordered)
    ranks = [ordered.index(tile) for tile in midday if tile in ordered]
    if not ranks:
        return frozenset(), frozenset(ordered)
    rank = max(ranks)
    inner = frozenset(ordered[:rank + 1])
    return inner, frozenset(position for position in ordered if position not in inner)


def _bundles(state: State, programme: Programme, inner, modes):
    grouped = defaultdict(list)
    for event in programme.events:
        if event.tile is None or not event.action or event.action[0] == "PASS":
            continue
        day = event.step // rules.TURNS_PER_DAY
        grouped[day, event.tile].append(event)
    result = []
    for (day, tile), events in grouped.items():
        events.sort(key=lambda e: (e.step, _ACTION_ORDER.get(str(e.action[0]), 99), e.event_id))
        release = max(min(e.step for e in events), state.step if day == state.day else day * 24)
        deadline = min(e.deadline if e.deadline is not None else (day + 1) * 24 - 1 for e in events)
        asset_ids = {e.asset_id for e in events if e.asset_id}
        return_required = tile in inner or any(modes.get(a) == "MIDDAY_RETURN" for a in asset_ids)
        result.append(_Bundle(tile, day, release, deadline, tuple(events), "INNER" if tile in inner else "OUTER", return_required))
    return result


def _outer_order(bundles: list[_Bundle], start: Position):
    remaining, ordered, current = list(bundles), [], start
    while remaining:
        chosen = min(remaining, key=lambda b: (rules.manhattan(current, b.tile), b.tile[1], b.tile[0]))
        remaining.remove(chosen); ordered.append(chosen); current = chosen.tile
    # Deterministic 2-opt; accept only strict distance reductions. Bundle
    # deadlines are rechecked by the final scheduler.
    improved = True
    while improved:
        improved = False
        for i in range(len(ordered) - 1):
            for j in range(i + 2, len(ordered)):
                before = start if i == 0 else ordered[i - 1].tile
                after = ordered[j + 1].tile if j + 1 < len(ordered) else None
                old = rules.manhattan(before, ordered[i].tile)
                new = rules.manhattan(before, ordered[j].tile)
                if after is not None:
                    old += rules.manhattan(ordered[j].tile, after)
                    new += rules.manhattan(ordered[i].tile, after)
                if new < old:
                    ordered[i:j + 1] = reversed(ordered[i:j + 1]); improved = True
                    break
            if improved: break
    return ordered


def _needed(events: Iterable[ProgrammeEvent]):
    need = Counter()
    for event in events:
        op = str(event.action[0]) if event.action else "PASS"
        if op == "FEED": need["WHEAT"] += 1
        elif op == "FERTILIZE": need["FERTILIZER"] += 1
        elif op == "PLACE" and len(event.action) > 1: need[str(event.action[1])] += 1
    return need


class ResourceAvailability:
    def __init__(self, state: State, programme: Programme, drops=None):
        self.arrivals = defaultdict(list)
        for item, qty in state.shed.items():
            if qty > 0:
                self.arrivals[item].append((state.step, qty))
        for event in programme.events:
            if event.kind in {"BUY_PRODUCT", "BUY_ANIMAL", "BUY_SEED"} and event.item and event.quantity > 0:
                self.arrivals[event.item].append((event.step + 1, event.quantity))
        if drops is not None:
            for step, amounts in drops.items():
                for item, qty in amounts.items():
                    if qty > 0:
                        self.arrivals[item].append((step, qty))
        for item in self.arrivals:
            self.arrivals[item].sort(key=lambda x: x[0])
        self.consumed = Counter()

    def signature(self):
        return tuple(sorted((item, tuple(timeline)) for item, timeline in self.arrivals.items()))

    def get_available_step(self, item: str, quantity_needed: int) -> int | None:
        already = self.consumed[item]
        target = already + quantity_needed
        accum = 0
        for step, qty in self.arrivals[item]:
            accum += qty
            if accum >= target:
                return step
        return None

    def record_pickup(self, item: str, quantity: int):
        self.consumed[item] += quantity


def _compile_worker(state: State, day: int, worker: int, bundles: list[_Bundle], start: Position,
                    available_step: int, resource_avail: ResourceAvailability):
    actions, current, cursor = {}, start, available_step
    inventory = Counter(state.workers[worker].inventory if day == state.day and worker < len(state.workers) else {})
    for bundle in bundles:
        cursor = max(cursor, bundle.release)
        need = _needed(bundle.events)
        missing = need - inventory
        if missing:
            avail_steps = []
            for item, qty in missing.items():
                st = resource_avail.get_available_step(item, qty)
                if st is None:
                    return None
                avail_steps.append(st)
            cursor = max(cursor, max(avail_steps, default=cursor))
            access = _nearest_access(current, state.board_size)
            for action in _move_path(current, access): actions[cursor] = action; cursor += 1
            current = access
            for item in sorted(missing):
                actions[cursor] = ("PICKUP", item, missing[item]); cursor += 1
                inventory[item] += missing[item]
                resource_avail.record_pickup(item, missing[item])
        for action in _move_path(current, bundle.tile): actions[cursor] = action; cursor += 1
        current = bundle.tile
        for event in bundle.events:
            cursor = max(cursor, event.step)
            if cursor > (event.deadline if event.deadline is not None else bundle.deadline):
                return None
            actions[cursor] = tuple(event.action); cursor += 1
            op = str(event.action[0])
            if op == "FEED": inventory["WHEAT"] -= 1
            elif op == "FERTILIZE": inventory["FERTILIZER"] -= 1
            elif op == "PLACE" and len(event.action) > 1: inventory[str(event.action[1])] -= 1
            elif op == "HARVEST": inventory[event.item or "UNKNOWN"] += max(1, event.quantity)
            elif op == "COLLECT_FERTILIZER": inventory["FERTILIZER"] += 1
        if bundle.return_required:
            access = _nearest_access(current, state.board_size)
            for action in _move_path(current, access): actions[cursor] = action; cursor += 1
            current = access
            if sum(inventory.values()) > 0:
                actions[cursor] = ("DROP",); cursor += 1; inventory.clear()
    day_end = min((day + 1) * 24 - 1, rules.TERMINAL_ACTION_STEP)
    if cursor - 1 > day_end:
        return None
    return actions


def _day_signature(state: State, day: int, bundles, workforce: int, resource_avail: ResourceAvailability):
    return (
        state.step if day == state.day else day * rules.TURNS_PER_DAY,
        workforce,
        tuple((worker.position, tuple(sorted(worker.inventory.items())))
              for worker in state.workers) if day == state.day else (),
        resource_avail.signature(),
        tuple((bundle.tile, bundle.day, bundle.release, bundle.deadline, bundle.zone,
               bundle.return_required,
               tuple((event.step, event.event_id, event.action, event.item,
                      event.quantity, event.deadline) for event in bundle.events))
              for bundle in bundles),
    )


def _compile_day(state: State, day: int, bundles, workforce: int, resource_avail: ResourceAvailability):
    key = _day_signature(state, day, bundles, workforce, resource_avail)
    if key in _DAY_ROUTE_CACHE:
        return _DAY_ROUTE_CACHE[key]
    routes = []
    starts = []
    access = rules.shed_access(state.board_size)
    for worker in range(workforce):
        if day == state.day and worker < len(state.workers):
            starts.append(state.workers[worker].position)
        else:
            starts.append(access[worker % len(access)])
    assigned = [[] for _ in range(workforce)]
    inner = sorted((bundle for bundle in bundles if bundle.zone == "INNER"),
                   key=lambda bundle: (bundle.deadline, bundle.tile[1], bundle.tile[0]))
    outer = _outer_order([bundle for bundle in bundles if bundle.zone == "OUTER"], access[0])
    for bundle in (*inner, *outer):
        choice = min(range(workforce), key=lambda worker: (
            sum(len(assigned_bundle.events) + 2 * rules.manhattan(
                starts[worker], assigned_bundle.tile)
                for assigned_bundle in assigned[worker]), worker))
        assigned[choice].append(bundle)
    for worker, work in enumerate(assigned):
        if not work:
            continue
        available = max(state.step, day * rules.TURNS_PER_DAY)
        if not (day == state.day and worker < len(state.workers)):
            available = day * rules.TURNS_PER_DAY + 1
        actions = _compile_worker(state, day, worker, work, starts[worker], available,
                                  resource_avail)
        if actions is None:
            _DAY_ROUTE_CACHE[key] = None
            return None
        routes.append(WorkerRoute(worker, day, "MIXED", actions))
    result = tuple(routes)
    _DAY_ROUTE_CACHE[key] = result
    return result


def _compile(state: State, programme: Programme, bundles, workforce: int, drops):
    routes = []
    resource_avail = ResourceAvailability(state, programme, drops)
    for day in range(state.day, rules.TERMINAL_ACTION_STEP // 24 + 1):
        day_bundles = [b for b in bundles if b.day == day]
        if not day_bundles:
            continue
        day_routes = _compile_day(state, day, day_bundles, workforce, resource_avail)
        if day_routes is None:
            return None
        routes.extend(day_routes)
    return tuple(routes)


def solve_intraday(state: State, programme: Programme, *, complete_actions: bool = True) -> Programme:
    key = (
        state.step,
        tuple((w.position, tuple(sorted(w.inventory.items()))) for w in state.workers),
        tuple(sorted((a.asset_id, a.tile, a.decision,
                      tuple((e.step, e.action, e.deadline) for e in a.service_schedule))
                     for a in programme.assets)),
        tuple(sorted((e.step, e.kind, e.item, e.quantity, e.tile, e.action, e.deadline)
                     for e in programme.events if e.kind != "HIRE")),
        complete_actions,
    )
    if key in _INTRADAY_CACHE:
        cached = _INTRADAY_CACHE[key]
        if cached is None:
            raise PlanningFailure("no worker count can realize every mandatory programme task")
        return cached

    load = placement_load(programme)
    modes = return_modes(programme)
    inner, outer = zones(state, programme, modes)
    bundles = _bundles(state, programme, inner, modes)
    minimum = len(state.workers)
    maximum = max(minimum, min(minimum + 10, 11))

    optimistic_drops = defaultdict(lambda: defaultdict(int))
    for asset in programme.assets:
        for ev in asset.harvest_schedule:
            item = ev.item or (asset.asset_type if asset.asset_type in rules.CROPS else rules.ANIMALS[asset.asset_type].product)
            optimistic_drops[ev.step][item] += ev.quantity
    optimistic_drops = {s: dict(a) for s, a in optimistic_drops.items()}

    from .simulation import drop_arrivals

    for workforce in range(minimum, maximum + 1):
        current_drops = optimistic_drops
        last_routes = None
        max_iters = 3 if complete_actions else 1
        for _ in range(max_iters):
            routes = _compile(state, programme, bundles, workforce, current_drops)
            if routes is None:
                break
            temp_prog = replace(programme, routes=routes)
            new_drops = drop_arrivals(state, temp_prog)
            if new_drops == current_drops:
                last_routes = routes
                break
            current_drops = new_drops
            
        if last_routes is None:
            continue
        routes = last_routes
        if complete_actions:
            complete = {}
            for route in routes:
                complete.setdefault((route.day, route.worker), {}).update(route.actions)
            for day in range(state.day, rules.TERMINAL_ACTION_STEP // 24 + 1):
                first=max(state.step,day*24); last=min((day+1)*24-1,rules.TERMINAL_ACTION_STEP)
                for worker in range(workforce):
                    actions=complete.setdefault((day,worker),{})
                    for step in range(first,last+1): actions.setdefault(step,("PASS",))
            routes=tuple(WorkerRoute(worker,day,"INNER" if any(
                asset.tile in inner for asset in programme.assets) else "OUTER",dict(sorted(actions.items())))
                for (day,worker),actions in sorted(complete.items()))
        hire_events = []
        for day in range(state.day, rules.TERMINAL_ACTION_STEP // 24 + 1):
            existing = len(state.workers) if day == state.day else 1
            quantity = max(0, workforce - existing)
            if quantity:
                step = max(state.step, day * 24)
                hire_events.append(ProgrammeEvent(step, -100, f"hire:{day}", "HIRE", quantity=quantity))
        base_events=tuple(event for event in programme.events if event.kind!="HIRE")
        unique={event.event_id:event for event in (*base_events,*hire_events)}
        res = replace(programme, events=tuple(sorted(unique.values())),
                      worker_count=workforce, inner=inner, outer=outer,
                      placement_load=load, return_mode=modes, routes=routes)
        _INTRADAY_CACHE[key] = res
        return res
    _INTRADAY_CACHE[key] = None
    raise PlanningFailure("no worker count can realize every mandatory programme task")
