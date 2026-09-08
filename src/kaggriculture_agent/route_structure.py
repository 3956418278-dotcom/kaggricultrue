"""Event-route representation for one fixed Daily Plan.

This module owns combinatorial realization choices.  It deliberately contains
no turn-by-turn movement expansion: that belongs to ``route_compiler``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Mapping

from . import rules
from .intent import EntityWork, IntentProblem, ServiceEvent, compile_intent


@dataclass(frozen=True)
class ResourceLink:
    """One physical input's chosen source.

    ``kind`` is CARRY, SHED, PURCHASE, or EVENT.  EVENT links name a producing
    service goal and state whether the item crosses the shed between workers.
    Seeds use SHED/PURCHASE to mean the global seed store and need no pickup.
    """

    consumer: str
    item: str
    quantity: int
    kind: str
    producer: str | None = None
    source_worker: int | None = None
    via_shed: bool = False


@dataclass(frozen=True)
class TileLease:
    """An entity's occupancy interval expressed through semantic events."""

    entity: str
    position: tuple[int, int]
    available_after: str | None = None
    released_by: str | None = None


@dataclass(frozen=True)
class SyncBundle:
    """Goals intended for one turn in ascending physical worker order."""

    goals: tuple[str, ...]


@dataclass(frozen=True)
class LogisticsEvent:
    """An explicit shed operation inside a worker route."""

    identifier: str
    operation: str
    item: str | None = None
    quantity: int = 0
    requires: tuple[str, ...] = ()
    consumers: tuple[str, ...] = ()
    purpose: str = "input"


@dataclass(frozen=True)
class RouteSkeleton:
    """All choices that can materially alter the executable realization."""

    workforce: int
    routes: tuple[tuple[str, ...], ...]
    path_choices: Mapping[str, int]
    placements: Mapping[str, tuple[int, int]]
    resources: tuple[ResourceLink, ...]
    leases: tuple[TileLease, ...]
    synchronizations: tuple[SyncBundle, ...] = ()
    spawn_preferences: tuple[tuple[int, int], ...] = ()
    market_priority: tuple[str, ...] = ()
    logistics: Mapping[str, LogisticsEvent] = field(default_factory=dict)
    acquisitions: Mapping[str, tuple] = field(default_factory=dict)
    required_entry_caps: tuple[int, ...] = ()
    hire_caps: tuple[int, ...] = ()


@dataclass(frozen=True)
class RouteProblem:
    intent: IntentProblem
    complete_paths: Mapping[str, tuple[tuple[ServiceEvent, ...], ...]]
    goal_entity: Mapping[str, str]
    goals: Mapping[str, object]


def build_route_problem(state, plan, progress=None) -> RouteProblem:
    intent = compile_intent(state, plan, progress)
    paths = {}
    goal_entity, goals = {}, {}
    for entity in intent.entities:
        complete = tuple(path for path in entity.paths if len(path) == len(entity.goals))
        paths[entity.identifier] = complete
        for goal in entity.goals:
            goal_entity[goal.identifier] = entity.identifier
            goals[goal.identifier] = goal
    return RouteProblem(intent, paths, goal_entity, goals)


def selected_paths(problem: RouteProblem, skeleton: RouteSkeleton):
    result = {}
    for entity in problem.intent.entities:
        options = problem.complete_paths[entity.identifier]
        choice = skeleton.path_choices.get(entity.identifier, 0)
        result[entity.identifier] = options[choice] if options and 0 <= choice < len(options) else ()
    return result


def selected_events(problem: RouteProblem, skeleton: RouteSkeleton) -> dict[str, ServiceEvent]:
    return {event.goal: event for path in selected_paths(problem, skeleton).values() for event in path}


def event_predecessors(problem: RouteProblem, skeleton: RouteSkeleton) -> dict[str, frozenset[str]]:
    predecessors = {}
    for path in selected_paths(problem, skeleton).values():
        prior = []
        for event in path:
            predecessors[event.goal] = frozenset(prior)
            prior.append(event.goal)
    for lease in skeleton.leases:
        if lease.available_after:
            first = next((event.goal for event in selected_paths(problem, skeleton).get(lease.entity, ())), None)
            if first:
                predecessors[first] = predecessors.get(first, frozenset()) | {lease.available_after}
    return predecessors


def event_positions(problem: RouteProblem, skeleton: RouteSkeleton) -> dict[str, tuple[int, int]]:
    return {event.goal: skeleton.placements[entity]
            for entity, path in selected_paths(problem, skeleton).items()
            for event in path if entity in skeleton.placements}


def _released_by(path: tuple[ServiceEvent, ...]) -> str | None:
    """Return the first event after which the tile is genuinely unoccupied."""
    for event in path:
        if event.after is None or event.after == "LOCKED":
            return event.goal
    return None


def _opening_compatible(entity: EntityWork, position, state) -> bool:
    raw = state.tile_at(position).raw
    if entity.existing:
        return len(entity.positions) == 1 and position == entity.positions[0]
    if raw == "LOCKED":
        return True
    return raw == entity.opening


def initial_placements(problem: RouteProblem, path_choices: Mapping[str, int]):
    """Choose legal temporal leases without assuming whole-day exclusion."""
    state = problem.intent.state
    chosen = RouteSkeleton(0, (), path_choices, {}, (), ())
    paths = selected_paths(problem, chosen)
    placements = {}
    leases = []
    tails: dict[tuple[int, int], str | None] = {}
    permanently_occupied = set()

    # Existing entities own their real opening location.  A clearing event can
    # release it for a later lifecycle; persistent assets cannot be overlapped.
    for entity in problem.intent.entities:
        if not entity.existing or not entity.positions:
            continue
        position = entity.positions[0]
        path = paths[entity.identifier]
        release = _released_by(path)
        placements[entity.identifier] = position
        leases.append(TileLease(entity.identifier, position, None, release))
        if release:
            tails[position] = release
        else:
            permanently_occupied.add(position)

    # Long-lived work receives the scarce near-shed cells first.  A released
    # opening tile is a real candidate, not a forbidden collision.
    new = [entity for entity in problem.intent.entities if not entity.existing]
    new.sort(key=lambda e: (-e.future_service_days, -len(e.goals), e.identifier))
    for entity in new:
        candidates = []
        release = _released_by(paths[entity.identifier])
        for position in entity.positions:
            if position in permanently_occupied or not _opening_compatible(entity, position, state):
                # A nonmatching opening is allowed only after a known clear.
                if position not in tails:
                    continue
            predecessor = tails.get(position)
            score = (rules.distance_to_shed(position) * max(1, entity.future_service_days),
                     int(predecessor is not None), position[1], position[0])
            candidates.append((score, position, predecessor))
        if not candidates:
            continue
        _, position, predecessor = min(candidates)
        placements[entity.identifier] = position
        leases.append(TileLease(entity.identifier, position, predecessor, release))
        if release:
            tails[position] = release
        else:
            permanently_occupied.add(position)
            tails.pop(position, None)
    return placements, tuple(leases)


def _route_cost(start, route, positions):
    cost, here = 0, start
    for goal in route:
        there = positions[goal]
        cost += rules.manhattan(here, there) + 1
        here = there
    return cost


def _starts(state, workforce):
    access = rules.shed_access(state.board_size)
    starts = [w.position for w in state.workers]
    opening = len(starts)
    starts.extend(access[(i-opening) % len(access)] for i in range(opening, workforce))
    return tuple(starts)


def realizable_opening_cash(state):
    """Upper bound available from current cash and currently stored products."""
    cash = state.money
    for item in rules.SELLABLE_PRODUCTS:
        quantity = state.shed.get(item, 0)
        if quantity:
            cash += rules.projected_sale_revenue(
                item, quantity, state.market_inventory.get(item, rules.MARKET_I0),
                state.step, state.step, state.unlocked_shops, 0)
    return cash


def initial_routes(problem: RouteProblem, skeleton: RouteSkeleton, workforce: int):
    """Build spatially coherent feasible routes from causal visit blocks.

    The block is only a construction device: returned routes still contain
    atomic events and later search may split them for same-turn coordination.
    Starting from coherent visits avoids manufacturing cross-worker waits before
    the joint search has any evidence that a split is useful.
    """
    paths = selected_paths(problem, skeleton)
    positions = event_positions(problem, skeleton)
    block_routes = [[] for _ in range(workforce)]
    starts = _starts(problem.intent.state, workforce)
    lease_predecessor = {lease.entity: lease.available_after for lease in skeleton.leases}
    pending = {entity: tuple(event.goal for event in path) for entity, path in paths.items()
               if path and entity in skeleton.placements}
    placed_goals = set()
    horizon = min(problem.intent.state.turns_left_today, problem.intent.state.turns_left)
    events = selected_events(problem, skeleton)
    access = rules.shed_access(problem.intent.state.board_size)

    def flattened(blocks):
        return [goal for block in blocks for goal in block]

    def route_cost(worker, goals):
        """Service turns plus logistics implied by this assignment."""
        service = _route_cost(starts[worker], goals, positions)
        carried = Counter(problem.intent.state.workers[worker].inventory
                          if worker < len(problem.intent.state.workers) else {})
        pickups = set()
        for goal in goals:
            event = events[goal]
            for item, quantity in _needed(event).items():
                if item.endswith("_SEED"):
                    continue
                available = min(quantity, carried[item])
                carried[item] -= available
                if available < quantity:
                    pickups.add(item)
            carried.update(_produced(event))
        if not pickups:
            return service
        first = positions[goals[0]] if goals else starts[worker]
        via = min(rules.manhattan(starts[worker], shed)
                  + rules.manhattan(shed, first) for shed in access)
        direct = rules.manhattan(starts[worker], first)
        return service+len(pickups)+max(0, via-direct)

    def capacity(worker):
        if worker < len(problem.intent.state.workers):
            return max(1, horizon)
        hire_rank = worker-len(problem.intent.state.workers)
        return max(1, horizon-1-hire_rank//rules.MAX_MARKET_ORDERS)

    while pending:
        ready = [(min(problem.goals[goal].deadline for goal in block), -len(block), entity, block)
                 for entity, block in pending.items()
                 if lease_predecessor.get(entity) is None
                 or lease_predecessor[entity] in placed_goals]
        if not ready:
            ready = [(min(problem.goals[goal].deadline for goal in block), -len(block), entity, block)
                     for entity, block in pending.items()]
        _, _, entity, block = min(ready)
        best = None
        lease_requirement = lease_predecessor.get(entity)
        current_costs = [route_cost(w, flattened(block_route))
                         for w, block_route in enumerate(block_routes)]
        for worker in range(workforce):
            for at in range(len(block_routes[worker])+1):
                if lease_requirement and any(lease_requirement in prior
                                             for prior in block_routes[worker][at:]):
                    continue
                trial_blocks = block_routes[worker][:at]+[block]+block_routes[worker][at:]
                trial_cost = route_cost(worker, flattened(trial_blocks))
                costs = list(current_costs); costs[worker] = trial_cost
                overruns = [max(0, cost-capacity(w)) for w, cost in enumerate(costs)]
                # Capacity violations are infeasibility, but among feasible
                # insertions total route cost owns the decision.  Peak load is
                # deliberately late: using it first fragments natural spatial
                # districts merely to make route lengths look alike.
                candidate = (int(any(overruns)), sum(overruns), sum(costs),
                             trial_cost-current_costs[worker], max(costs),
                             current_costs[worker], worker, at)
                if best is None or candidate < best[0]:
                    best = candidate, worker, at
        _, worker, at = best
        block_routes[worker].insert(at, block)
        placed_goals.update(block)
        del pending[entity]
    return tuple(tuple(flattened(route)) for route in block_routes)


def _needed(event: ServiceEvent):
    return Counter({item: -quantity for item, quantity in event.delta.items() if quantity < 0})


def _produced(event: ServiceEvent):
    return Counter({item: quantity for item, quantity in event.delta.items() if quantity > 0})


def initial_resources(problem: RouteProblem, skeleton: RouteSkeleton):
    """Allocate finite opening stock, then route-local output, then purchases."""
    state = problem.intent.state
    events = selected_events(problem, skeleton)
    route_of = {goal: worker for worker, route in enumerate(skeleton.routes) for goal in route}
    order = {goal: n for route in skeleton.routes for n, goal in enumerate(route) if goal in events}
    positions = event_positions(problem, skeleton)
    starts = _starts(state, skeleton.workforce)
    rough_time = {}
    for worker, route in enumerate(skeleton.routes):
        elapsed, here = (0 if worker < len(state.workers) else 1), starts[worker]
        for token in route:
            if token not in events or token not in positions:
                continue
            there = positions[token]
            elapsed += rules.manhattan(here, there)+1
            rough_time[token] = elapsed; here = there
    carry = {w.index: Counter(w.inventory) for w in state.workers}
    shed = Counter(state.shed)
    seeds = Counter({f"{crop}_SEED": n for crop, n in state.seeds.items()})
    unused_outputs = defaultdict(list)
    for goal, event in events.items():
        for item, quantity in _produced(event).items():
            unused_outputs[item].extend([goal] * quantity)
    links = []
    for worker, route in enumerate(skeleton.routes):
        for goal in route:
            if goal not in events:
                continue
            for item, quantity in _needed(events[goal]).items():
                left = quantity
                if worker in carry:
                    n = min(left, carry[worker][item])
                    if n:
                        links.append(ResourceLink(goal, item, n, "CARRY", source_worker=worker))
                        carry[worker][item] -= n; left -= n
                store = seeds if item.endswith("_SEED") else shed
                n = min(left, store[item])
                if n:
                    links.append(ResourceLink(goal, item, n, "SHED"))
                    store[item] -= n; left -= n
                # Prefer an earlier output on the same route; it needs no depot.
                for producer in list(unused_outputs[item]):
                    if not left:
                        break
                    if route_of.get(producer) == worker and order.get(producer, 10**9) < order[goal]:
                        links.append(ResourceLink(goal, item, 1, "EVENT", producer=producer))
                        unused_outputs[item].remove(producer); left -= 1
                # A cross-route output is explicit and shed-mediated.
                eligible = sorted((producer for producer in unused_outputs[item]
                                   if route_of.get(producer) != worker
                                   and rough_time.get(producer, 10**9) < rough_time.get(goal, -1)),
                                  key=lambda producer: (rough_time[producer], producer))
                while left and eligible:
                    producer = eligible.pop(0)
                    unused_outputs[item].remove(producer)
                    links.append(ResourceLink(goal, item, 1, "EVENT", producer=producer, via_shed=True))
                    left -= 1
                if left:
                    links.append(ResourceLink(goal, item, left, "PURCHASE"))
    return tuple(links)


def with_initial_logistics(problem: RouteProblem, skeleton: RouteSkeleton) -> RouteSkeleton:
    """Create explicit batched pickups and cross-worker shed transfers."""
    events = selected_events(problem, skeleton)
    route_of = {token: worker for worker, route in enumerate(skeleton.routes) for token in route}
    routes = [list(route) for route in skeleton.routes]
    logistics = {}
    pickups = defaultdict(list)
    transfers = defaultdict(list)
    for link in skeleton.resources:
        if link.item.endswith("_SEED") or link.kind == "CARRY":
            continue
        worker = route_of.get(link.consumer)
        if worker is None:
            continue
        if link.kind in {"SHED", "PURCHASE"}:
            pickups[worker, link.item].append(link)
        elif link.kind == "EVENT" and link.via_shed:
            producer_worker = route_of.get(link.producer)
            if producer_worker is not None:
                transfers[producer_worker, worker, link.item, link.producer].append(link)

    def insert_before(worker, token, consumers):
        indices = [routes[worker].index(goal) for goal in consumers if goal in routes[worker]]
        routes[worker].insert(min(indices) if indices else 0, token)

    for (worker, item), links in sorted(pickups.items()):
        consumers = tuple(link.consumer for link in links)
        identifier = f"pickup:{worker}:{item}"
        logistics[identifier] = LogisticsEvent(identifier, "PICKUP", item,
            sum(link.quantity for link in links), consumers=consumers)
        insert_before(worker, identifier, consumers)
    for (producer_worker, consumer_worker, item, producer), links in sorted(transfers.items()):
        quantity = sum(link.quantity for link in links)
        deposit = f"transfer-out:{producer_worker}:{consumer_worker}:{item}:{producer}"
        pickup = f"transfer-in:{producer_worker}:{consumer_worker}:{item}:{producer}"
        logistics[deposit] = LogisticsEvent(deposit, "PLACE", item, quantity,
            requires=(producer,), purpose="transfer")
        logistics[pickup] = LogisticsEvent(pickup, "PICKUP", item, quantity,
            requires=(deposit,), consumers=tuple(link.consumer for link in links), purpose="transfer")
        at = routes[producer_worker].index(producer)+1
        routes[producer_worker].insert(at, deposit)
        insert_before(consumer_worker, pickup, logistics[pickup].consumers)

    if problem.intent.state.day == 29:
        for worker in range(skeleton.workforce):
            identifier = f"terminal-drop:{worker}"
            logistics[identifier] = LogisticsEvent(identifier, "DROP", purpose="terminal-liquidation")
            routes[worker].append(identifier)
    return replace(skeleton, routes=tuple(tuple(route) for route in routes), logistics=logistics)


def _required_operating_cash(problem: RouteProblem, skeleton: RouteSkeleton):
    state = problem.intent.state
    cost = rules.hire_expenditure(
        state.hires_today, max(0, skeleton.workforce-len(state.workers)))
    for order in skeleton.acquisitions.values():
        operation, item, quantity = order
        if operation == "BUY_SEED":
            cost += rules.CROPS[item].seed_cost*quantity
        elif operation == "BUY_ANIMAL":
            cost += rules.ANIMALS[item].cost*quantity
        elif operation == "BUY_PRODUCT":
            inventory = state.market_inventory.get(item, rules.MARKET_I0)
            cost += sum(rules.market_price(item, inventory-1-n) for n in range(quantity))
    for offset, _ in enumerate(q for q in problem.intent.land if q not in state.unlocked_quadrants):
        index = len(state.unlocked_quadrants)-1+offset
        if index < len(rules.LAND_PRICES): cost += rules.LAND_PRICES[index]
    return cost


def with_financing_logistics(problem: RouteProblem, skeleton: RouteSkeleton) -> RouteSkeleton:
    """Add only deposits that can close a material realization cash gap.

    These are execution-side liquidity operations.  They are not Plan goals and
    do not prescribe sales; the market owner decides whether the deposited
    product is actually sold from the then-current market state.
    """
    state = problem.intent.state
    reserved = Counter()
    for link in skeleton.resources:
        if link.kind == "SHED" and not link.item.endswith("_SEED"):
            reserved[link.item] += link.quantity
    available = state.money
    for item in rules.SELLABLE_PRODUCTS:
        quantity = max(0, state.shed.get(item, 0)-reserved[item])
        if quantity:
            available += rules.projected_sale_revenue(
                item, quantity, state.market_inventory.get(item, rules.MARKET_I0),
                state.step, state.step, state.unlocked_shops, 0)
    shortfall = _required_operating_cash(problem, skeleton)-available
    if shortfall <= 0:
        return skeleton

    events = selected_events(problem, skeleton)
    route_of = {token: worker for worker, route in enumerate(skeleton.routes) for token in route}
    committed_output = Counter()
    for link in skeleton.resources:
        if link.kind == "EVENT" and link.producer:
            committed_output[link.producer, link.item] += link.quantity
    candidates = []
    for goal, event in events.items():
        worker = route_of.get(goal)
        if worker is None: continue
        for item, quantity in _produced(event).items():
            quantity -= committed_output[goal, item]
            if quantity <= 0 or item not in rules.SELLABLE_PRODUCTS: continue
            unit_value = rules.market_price(item, state.market_inventory.get(item, rules.MARKET_I0))
            candidates.append((-unit_value, worker, goal, item, quantity))
    routes = [list(route) for route in skeleton.routes]
    logistics = dict(skeleton.logistics)
    for negative_value, worker, producer, item, quantity in sorted(candidates):
        if shortfall <= 0: break
        value = -negative_value
        amount = min(quantity, max(1, (shortfall+value-1)//value))
        identifier = f"liquidity:{worker}:{producer}:{item}"
        logistics[identifier] = LogisticsEvent(identifier, "PLACE", item, amount,
            requires=(producer,), purpose="market-liquidity")
        routes[worker].insert(routes[worker].index(producer)+1, identifier)
        shortfall -= value*amount
    return replace(skeleton, routes=tuple(tuple(route) for route in routes), logistics=logistics)


def initial_skeleton(problem: RouteProblem, workforce: int | None = None) -> RouteSkeleton:
    state = problem.intent.state
    path_choices = {entity.identifier: 0 for entity in problem.intent.entities}
    placements, leases = initial_placements(problem, path_choices)
    event_count = sum(len(problem.complete_paths[e.identifier][0])
                      for e in problem.intent.entities if problem.complete_paths[e.identifier] and e.identifier in placements)
    horizon = max(1, min(state.turns_left_today, state.turns_left) - (1 if len(state.workers) == 1 else 0))
    if workforce is None:
        # Service plus a conservative two moves per affected entity. This is an
        # initializer only; search can split/merge routes. The upper bound comes
        # from real cash, useful work, horizon, and market entries—not a fixed
        # hand cap.
        useful_work = event_count + 2*len(problem.intent.entities)
        workforce = max(len(state.workers), (useful_work+horizon-1)//horizon)
        cash, affordable = realizable_opening_cash(state), len(state.workers)
        while affordable < max(len(state.workers), event_count):
            cost = rules.fibonacci_hire_cost(state.hires_today+affordable-len(state.workers))
            if cost > cash: break
            cash -= cost; affordable += 1
        market_bound = len(state.workers) + rules.MAX_MARKET_ORDERS*max(0, horizon-1)
        workforce = min(workforce, affordable, market_bound, max(len(state.workers), event_count))
    shell = RouteSkeleton(workforce, tuple(() for _ in range(workforce)), path_choices, placements, (), leases,
                          spawn_preferences=_starts(state, workforce)[len(state.workers):])
    routes = initial_routes(problem, shell, workforce)
    shell = replace(shell, routes=routes)
    shell = replace(shell, resources=initial_resources(problem, shell))
    shell = with_initial_logistics(problem, shell)
    from .route_compiler import purchase_orders_for_links
    acquisitions = purchase_orders_for_links(shell.resources)
    shell = replace(shell, acquisitions=acquisitions)
    purchases = sorted(acquisitions)
    priority = tuple([*(f"HIRE:{n}" for n in range(max(0, workforce-len(state.workers)))),
                      *(f"BUY:{item}" for item in purchases),
                      *(f"LAND:{quadrant}" for quadrant in problem.intent.land)])
    hire_count = max(0, workforce-len(state.workers))
    hire_cost = rules.hire_expenditure(state.hires_today, hire_count)
    sellable = any(state.shed.get(item, 0) for item in rules.SELLABLE_PRODUCTS)
    # Selling remains market-owned. Reserve one entry only when opening cash
    # cannot finance the selected staffing and an actual sale can change that.
    caps = (rules.MAX_MARKET_ORDERS-1,) if hire_cost > state.money and sellable else ()
    shell = replace(shell, market_priority=priority, required_entry_caps=caps)
    return with_financing_logistics(problem, shell)


def validate_skeleton(problem: RouteProblem, skeleton: RouteSkeleton):
    errors = []
    if skeleton.workforce != len(skeleton.routes) or skeleton.workforce < len(problem.intent.state.workers):
        errors.append("workforce/routes mismatch")
    events = selected_events(problem, skeleton)
    routed = [goal for route in skeleton.routes for goal in route]
    if len(routed) != len(set(routed)):
        errors.append("route event routed more than once")
    unknown = set(routed) - set(events) - set(skeleton.logistics)
    if unknown:
        errors.append(f"unknown routed events: {len(unknown)}")
    entities = {e.identifier: e for e in problem.intent.entities}
    for entity, position in skeleton.placements.items():
        if entity not in entities or position not in entities[entity].positions:
            errors.append(f"placement outside Plan domain: {entity}")
    leases = {lease.entity: lease for lease in skeleton.leases}
    for entity in skeleton.placements:
        if entity not in leases or leases[entity].position != skeleton.placements[entity]:
            errors.append(f"missing or inconsistent tile lease: {entity}")
    needs = Counter()
    for goal, event in events.items():
        if goal not in routed:
            continue
        for item, quantity in _needed(event).items():
            needs[goal, item] += quantity
    supplied = Counter()
    for link in skeleton.resources:
        supplied[link.consumer, link.item] += link.quantity
    for key, quantity in needs.items():
        if supplied[key] != quantity:
            errors.append(f"resource links do not cover {key}: {supplied[key]}/{quantity}")
    for link in skeleton.resources:
        if link.kind not in {"CARRY", "SHED", "PURCHASE", "EVENT"}:
            errors.append(f"unknown resource source: {link.kind}")
        if link.kind == "EVENT" and link.producer not in events:
            errors.append(f"unknown resource producer: {link.producer}")
    predecessors = event_predecessors(problem, skeleton)
    for bundle in skeleton.synchronizations:
        if len(bundle.goals) != len(set(bundle.goals)) or any(g not in events for g in bundle.goals):
            errors.append("invalid synchronization bundle")
        if any(a not in predecessors.get(b, ()) for a, b in zip(bundle.goals, bundle.goals[1:])):
            errors.append("synchronization does not follow local/lease precedence")
    for identifier, event in skeleton.logistics.items():
        if identifier != event.identifier or identifier not in routed:
            errors.append(f"unrouted or inconsistent logistics event: {identifier}")
        if event.operation not in {"PICKUP", "PLACE", "DROP", "PASS", "NORTH", "SOUTH", "EAST", "WEST"}:
            errors.append(f"invalid logistics operation: {identifier}")
        if event.operation in {"PICKUP", "PLACE"} and (not event.item or event.quantity <= 0):
            errors.append(f"invalid logistics quantity: {identifier}")
        if any(requirement not in routed for requirement in event.requires):
            errors.append(f"unknown logistics dependency: {identifier}")
    for item, order in skeleton.acquisitions.items():
        if not order or not str(order[0]).startswith("BUY_") or int(order[2]) <= 0:
            errors.append(f"invalid acquisition: {item}")
    if not route_precedence_feasible(problem, skeleton):
        errors.append("route order contains a causal cycle")
    if any(cap < 0 or cap > rules.MAX_MARKET_ORDERS for cap in skeleton.required_entry_caps):
        errors.append("invalid required market-entry cap")
    if any(cap < 0 or cap > rules.MAX_MARKET_ORDERS for cap in skeleton.hire_caps):
        errors.append("invalid hire-wave cap")
    if errors:
        raise ValueError("; ".join(errors))
    return True


def route_precedence_edges(problem: RouteProblem, skeleton: RouteSkeleton):
    """Return semantic precedence edges for the selected realization.

    Worker-route order itself is added by :func:`route_precedence_feasible`.
    Keeping this graph with the representation gives constructors, mutations,
    validation and compilation one causal contract.
    """
    routed = {token for route in skeleton.routes for token in route}
    edges = set()
    for successor, predecessors in event_predecessors(problem, skeleton).items():
        if successor not in routed:
            continue
        edges.update((predecessor, successor) for predecessor in predecessors
                     if predecessor in routed)
    for link in skeleton.resources:
        if (link.kind == "EVENT" and link.producer in routed
                and link.consumer in routed):
            edges.add((link.producer, link.consumer))
    for identifier, event in skeleton.logistics.items():
        if identifier not in routed:
            continue
        edges.update((required, identifier) for required in event.requires
                     if required in routed)
        edges.update((identifier, consumer) for consumer in event.consumers
                     if consumer in routed)
    return frozenset(edges)


def route_precedence_feasible(problem: RouteProblem, skeleton: RouteSkeleton):
    """Whether route order and semantic dependencies form one causal DAG."""
    nodes = {token for route in skeleton.routes for token in route}
    successor = defaultdict(set)
    indegree = Counter({token: 0 for token in nodes})

    def add_edge(before, after):
        if before == after or before not in nodes or after not in nodes:
            return
        if after not in successor[before]:
            successor[before].add(after)
            indegree[after] += 1

    for route in skeleton.routes:
        for before, after in zip(route, route[1:]):
            add_edge(before, after)
    for before, after in route_precedence_edges(problem, skeleton):
        add_edge(before, after)

    ready = [token for token in nodes if not indegree[token]]
    visited = 0
    while ready:
        token = ready.pop()
        visited += 1
        for following in successor[token]:
            indegree[following] -= 1
            if not indegree[following]:
                ready.append(following)
    return visited == len(nodes)
