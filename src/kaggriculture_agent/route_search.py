"""Deterministic large-neighborhood search over event-route realizations."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from random import Random
from time import perf_counter

from . import rules
from .intraday import EndValue
from .route_compiler import compile_skeleton
from .route_structure import (
    ResourceLink, RouteSkeleton, SyncBundle, TileLease, build_route_problem,
    event_positions, event_predecessors, initial_resources, initial_routes,
    initial_skeleton,
    realizable_opening_cash, route_precedence_feasible, selected_events,
    selected_paths, validate_skeleton, with_initial_logistics,
    with_financing_logistics,
)


@dataclass(frozen=True)
class RouteSearchConfig:
    # ``iterations`` counts cheap structural proposals, not exact simulations.
    iterations: int = 300
    population: int = 48
    exact_candidates: int = 18
    refinement_candidates: int = 4
    repair_rounds: int = 6
    max_exact_evaluations: int = 96
    compression_rounds: int = 2
    compression_candidates: int = 4
    ruin_probability: float = .35
    random_seed: int = 0


def _released_by(path):
    return next((event.goal for event in path if event.after is None or event.after == "LOCKED"), None)


def _route_rank(skeleton, goal):
    for worker, route in enumerate(skeleton.routes):
        if goal in route: return worker, route.index(goal)
    return len(skeleton.routes), 10**9


def rebuild_leases(problem, skeleton):
    """Derive temporal occupancy from placements and routed lifecycle order."""
    paths = selected_paths(problem, skeleton)
    entities = {e.identifier: e for e in problem.intent.entities}
    grouped = defaultdict(list)
    for entity, position in skeleton.placements.items(): grouped[position].append(entity)
    leases = []
    for position, occupants in grouped.items():
        occupants.sort(key=lambda entity: (
            0 if entities[entity].existing else 1,
            min((_route_rank(skeleton, e.goal) for e in paths[entity]), default=(10**9, 10**9)),
            entity))
        available = None
        for entity in occupants:
            release = _released_by(paths[entity])
            leases.append(TileLease(entity, position, available, release))
            available = release
            if release is None and entity != occupants[-1]:
                # Preserve the conflicting choice. Exact compilation will
                # expose its unfulfilled work; do not invent another tile.
                available = f"blocked:{entity}"
    return tuple(sorted(leases, key=lambda lease: lease.entity))


def _market_priority(problem, skeleton):
    purchases = sorted(skeleton.acquisitions)
    state = problem.intent.state
    return tuple([*(f"HIRE:{n}" for n in range(max(0, skeleton.workforce-len(state.workers)))),
                  *(f"BUY:{item}" for item in purchases),
                  *(f"LAND:{q}" for q in problem.intent.land)])


def normalize(problem, skeleton):
    """Refresh only derived tile leases; preserve realization choices verbatim.

    Resource sources, acquisition batches, logistics placement and market order
    are searched decisions.  Reconstructing those fields here made an unrelated
    mutation evaluate a different candidate.  Route-changing operators use the
    explicit ``_rebuild_route_support`` boundary below when their edit really
    invalidates dependent execution structure.
    """
    return replace(skeleton, leases=rebuild_leases(problem, skeleton))


def _acquisition_item(order):
    operation, item, _ = order
    return f"{item}_SEED" if operation == "BUY_SEED" else item


def _reconcile_acquisitions(resources, acquisitions):
    """Preserve purchase batching/order while matching current resource links."""
    from .route_compiler import purchase_orders_for_links

    required = purchase_orders_for_links(resources)
    remaining = {item: int(order[2]) for item, order in required.items()}
    result = {}
    for key, order in acquisitions.items():
        item = _acquisition_item(order)
        quantity = min(int(order[2]), remaining.get(item, 0))
        if quantity <= 0:
            continue
        result[key] = (order[0], order[1], quantity)
        remaining[item] -= quantity
    for item, quantity in remaining.items():
        if quantity <= 0:
            continue
        operation, product, _ = required[item]
        key = item
        suffix = 1
        while key in result:
            key = f"{item}:remainder:{suffix}"
            suffix += 1
        result[key] = (operation, product, quantity)
    return result


def _reconcile_market_priority(problem, before, skeleton):
    """Keep the searched relative order and append only newly required tokens."""
    state = problem.intent.state
    required = tuple([
        *(f"HIRE:{n}" for n in range(max(0, skeleton.workforce-len(state.workers)))),
        *(f"BUY:{key}" for key in skeleton.acquisitions),
        *(f"LAND:{quadrant}" for quadrant in problem.intent.land),
    ])
    needed = set(required)
    retained = [token for token in before.market_priority if token in needed]
    retained.extend(token for token in required if token not in retained)
    return tuple(retained)


def _resource_choices_compatible(problem, before, skeleton):
    """Whether all previous source choices remain meaningful after a route edit."""
    events = selected_events(problem, skeleton)
    routed = {token for route in skeleton.routes for token in route}
    route_of = {token: worker for worker, route in enumerate(skeleton.routes)
                for token in route}
    order = {token: at for route in skeleton.routes for at, token in enumerate(route)}
    needs = Counter()
    for goal, event in events.items():
        if goal not in routed:
            continue
        for item, quantity in event.delta.items():
            if quantity < 0:
                needs[goal, item] += -quantity
    supplied = Counter()
    for link in before.resources:
        if link.consumer not in routed or link.consumer not in events:
            return False
        supplied[link.consumer, link.item] += link.quantity
        if link.kind == "CARRY" and link.source_worker != route_of[link.consumer]:
            return False
        if link.kind == "EVENT":
            producer = events.get(link.producer)
            if producer is None or producer.delta.get(link.item, 0) <= 0:
                return False
            if (not link.via_shed
                    and (route_of.get(link.producer) != route_of[link.consumer]
                         or order.get(link.producer, 10**9) >= order[link.consumer])):
                return False
    if supplied != needs:
        return False
    trial = replace(skeleton, resources=before.resources)
    return route_precedence_feasible(problem, trial)


def _restore_compatible_logistics(problem, before, rebuilt):
    """Retain explicit logistics ordering when its semantic event is unchanged."""
    common = {token for token, event in rebuilt.logistics.items()
              if before.logistics.get(token) == event}
    if not common:
        return rebuilt
    old_worker = {token: worker for worker, route in enumerate(before.routes)
                  for token in route}
    routes = [list(route) for route in rebuilt.routes]
    new_worker = {token: worker for worker, route in enumerate(routes) for token in route}
    for token in sorted(common, key=lambda item: (
            old_worker.get(item, 10**9),
            before.routes[old_worker[item]].index(item) if item in old_worker else 10**9,
            item)):
        worker = new_worker.get(token)
        if worker is None or worker != old_worker.get(token):
            continue
        routes[worker].remove(token)
        old_route = before.routes[worker]
        old_at = old_route.index(token)
        following = next((item for item in old_route[old_at+1:]
                          if item in routes[worker] and item not in before.logistics), None)
        prior = next((item for item in reversed(old_route[:old_at])
                      if item in routes[worker] and item not in before.logistics), None)
        if following is not None:
            at = routes[worker].index(following)
        elif prior is not None:
            at = routes[worker].index(prior)+1
        else:
            at = len(routes[worker])
        routes[worker].insert(at, token)
        new_worker[token] = worker
    candidate = replace(rebuilt, routes=tuple(tuple(route) for route in routes))
    return candidate if route_precedence_feasible(problem, candidate) else rebuilt


def _rebuild_route_support(problem, before, candidate):
    """Reconcile only choices that depend on a changed route/path/workforce."""
    service_routes = tuple(tuple(token for token in route
                                 if token not in before.logistics)
                           for route in candidate.routes)
    shell = replace(candidate, routes=service_routes, logistics={})
    shell = replace(shell, leases=rebuild_leases(problem, shell))
    resources = (before.resources if _resource_choices_compatible(problem, before, shell)
                 else initial_resources(problem, shell))
    shell = replace(shell, resources=resources)
    shell = with_initial_logistics(problem, shell)
    acquisitions = _reconcile_acquisitions(resources, before.acquisitions)
    shell = replace(shell, acquisitions=acquisitions)
    shell = replace(shell, market_priority=_reconcile_market_priority(problem, before, shell))
    shell = with_financing_logistics(problem, shell)
    return _restore_compatible_logistics(problem, before, shell)


def skeleton_key(skeleton):
    return (skeleton.workforce, skeleton.routes,
            tuple(sorted(skeleton.path_choices.items())), tuple(sorted(skeleton.placements.items())),
            tuple(skeleton.resources), tuple(skeleton.leases), tuple(skeleton.synchronizations),
            skeleton.spawn_preferences, skeleton.market_priority,
            tuple(sorted(skeleton.logistics.items())), tuple(sorted(skeleton.acquisitions.items())),
            skeleton.required_entry_caps, skeleton.hire_caps)


def _liquid_value(state):
    value = state.money
    prior = Counter()
    for item in rules.SELLABLE_PRODUCTS:
        quantity = state.owned_total(item)
        if quantity:
            value += rules.projected_sale_revenue(item, quantity,
                state.market_inventory.get(item, rules.MARKET_I0), state.step, state.step,
                state.unlocked_shops, prior[item])
            prior[item] += quantity
    return value


def resulting_state_value(opening_state, resulting_state):
    """Maintained same-day economic value of a reached owned state.

    Hire, input and land expenditure already reduce reached cash.  Their ledger
    fields are therefore diagnostics and must never be subtracted again here.
    ``EndValue`` also retains quoted inventory and surviving-asset option value,
    unlike the older liquid-stock-only helper above.
    """
    return EndValue(opening_state)(resulting_state)[0]


def result_score(result, opening_state=None):
    d = result.diagnostics
    economic = (_liquid_value(result.final_state) if opening_state is None else
                resulting_state_value(opening_state, result.final_state))
    return (len(result.completed), economic, -d["used_worker_turns"],
            -d["movement"], -d["logistics"])


def _abstract_schedule(problem, skeleton):
    """Cheap fixed-route schedule used only to prune structural proposals.

    It retains worker order, travel, service duration, local/cross-worker
    precedence, logistics dependencies, hire availability, and deadlines.  It
    deliberately does not award the final result: shortlisted structures are
    always replayed by ``compile_skeleton`` through the real rule transitions.
    """
    events = selected_events(problem, skeleton)
    positions = event_positions(problem, skeleton)
    predecessors = {goal: set(required)
                    for goal, required in event_predecessors(problem, skeleton).items()}
    for link in skeleton.resources:
        if link.kind == "EVENT" and link.producer:
            predecessors.setdefault(link.consumer, set()).add(link.producer)
    route_of = {token: worker for worker, route in enumerate(skeleton.routes) for token in route}
    for token, logistic in skeleton.logistics.items():
        predecessors[token] = set(logistic.requires)
    starts = [worker.position for worker in problem.intent.state.workers]
    access = rules.shed_access(problem.intent.state.board_size)
    for worker in range(len(starts), skeleton.workforce):
        preference = worker-len(problem.intent.state.workers)
        starts.append(skeleton.spawn_preferences[preference]
                      if preference < len(skeleton.spawn_preferences)
                      else access[preference % len(access)])

    node_position, route_edges, source_ready = {}, {}, {}
    total_movement = logistics_count = 0
    for worker, route in enumerate(skeleton.routes):
        previous, here = None, starts[worker]
        # At most ten hands can be created per market turn.  Existing workers
        # act at offset zero; newly hired workers first act on a later turn.
        hire_rank = max(0, worker-len(problem.intent.state.workers))
        available = 0 if worker < len(problem.intent.state.workers) else 1+hire_rank//rules.MAX_MARKET_ORDERS
        for token in route:
            if token in events:
                there = positions.get(token)
            else:
                logistic = skeleton.logistics.get(token)
                if logistic is None:
                    continue
                logistics_count += int(logistic.operation in {"PICKUP", "PLACE", "DROP"})
                if logistic.operation in {"PICKUP", "PLACE", "DROP"}:
                    there = min(access, key=lambda p: (rules.manhattan(here, p), p))
                else:
                    there = here
            if there is None:
                continue
            travel = rules.manhattan(here, there)
            total_movement += travel
            node_position[token] = there
            route_edges[token] = (previous, travel+1)
            if previous is None:
                source_ready[token] = available+travel
            previous, here = token, there

    finish = {}
    pending = {token for route in skeleton.routes for token in route if token in node_position}
    cycle = False
    while pending:
        progressed = False
        for token in tuple(pending):
            previous, duration = route_edges[token]
            required = predecessors.get(token, set())
            if (previous is not None and previous not in finish) or any(dep not in finish for dep in required):
                continue
            ready = source_ready.get(token, 0) if previous is None else finish[previous]+duration
            worker = route_of[token]
            for dep in required:
                # Ordered unit execution permits a lower-index worker's effect
                # to feed a higher-index worker in the same turn.
                delay = int(route_of.get(dep, worker) >= worker)
                ready = max(ready, finish[dep]+delay)
            finish[token] = ready
            pending.remove(token); progressed = True
        if not progressed:
            cycle = True
            break

    horizon = min(problem.intent.state.turns_left_today, problem.intent.state.turns_left)
    completed = 0
    lateness = 0
    for goal in events:
        if goal not in finish:
            continue
        deadline = min(horizon-1, max(0, problem.goals[goal].deadline-problem.intent.state.step))
        if finish[goal] <= deadline:
            completed += 1
        else:
            lateness += finish[goal]-deadline
    completed += sum(q in problem.intent.state.unlocked_quadrants for q in problem.intent.land)
    makespan = max(finish.values(), default=0)
    # Lateness/overrun describe unrealized Plan feasibility, so they remain
    # part of the primary completion tier. Economic ranking belongs to exact
    # reached states; transaction-ledger fields are not proxy objectives here.
    return (-int(cycle), completed, -lateness, -max(0, makespan-(horizon-1)),
            -total_movement, -logistics_count)


def _relocate_event(problem, skeleton, rng):
    nonempty = [w for w, route in enumerate(skeleton.routes)
                if any(token not in skeleton.logistics for token in route)]
    if not nonempty: return skeleton
    source = rng.choice(nonempty); route = list(skeleton.routes[source])
    choices = [n for n, token in enumerate(route) if token not in skeleton.logistics]
    goal = route.pop(rng.choice(choices))
    routes = [list(r) for r in skeleton.routes]
    routes[source] = route
    target = rng.randrange(skeleton.workforce)
    routes[target].insert(rng.randrange(len(routes[target])+1), goal)
    return replace(skeleton, routes=tuple(tuple(r) for r in routes))


def _relocate_entity(problem, skeleton, rng):
    paths = selected_paths(problem, skeleton)
    candidates = [entity for entity, path in paths.items() if path]
    if not candidates: return skeleton
    entity = rng.choice(candidates); goals = [e.goal for e in paths[entity]]
    routes = [[g for g in route if g not in goals] for route in skeleton.routes]
    target = rng.randrange(skeleton.workforce); at = rng.randrange(len(routes[target])+1)
    routes[target][at:at] = goals
    return replace(skeleton, routes=tuple(tuple(r) for r in routes))


def _cross_exchange(problem, skeleton, rng):
    if skeleton.workforce < 2: return skeleton
    a, b = rng.sample(range(skeleton.workforce), 2)
    ra, rb = list(skeleton.routes[a]), list(skeleton.routes[b])
    ia, ib = rng.randrange(len(ra)+1), rng.randrange(len(rb)+1)
    ra[ia:], rb[ib:] = rb[ib:], ra[ia:]
    routes = list(skeleton.routes); routes[a], routes[b] = tuple(ra), tuple(rb)
    return replace(skeleton, routes=tuple(routes))


def _change_path(problem, skeleton, rng):
    candidates = [entity for entity, paths in problem.complete_paths.items() if len(paths) > 1]
    if not candidates: return skeleton
    entity = rng.choice(candidates); old = skeleton.path_choices.get(entity, 0)
    choices = [n for n in range(len(problem.complete_paths[entity])) if n != old]
    selected = rng.choice(choices)
    old_goals = {event.goal for event in problem.complete_paths[entity][old]}
    new_goals = tuple(event.goal for event in problem.complete_paths[entity][selected])
    routes = [[token for token in route
               if token not in old_goals and token not in skeleton.logistics]
              for route in skeleton.routes]
    old_workers = [worker for worker, route in enumerate(skeleton.routes)
                   if any(goal in route for goal in old_goals)]
    target = old_workers[0] if old_workers else rng.randrange(skeleton.workforce)
    routes[target].extend(new_goals)
    path_choices = dict(skeleton.path_choices); path_choices[entity] = selected
    synchronizations = tuple(bundle for bundle in skeleton.synchronizations
                             if not (set(bundle.goals) & old_goals))
    return replace(skeleton, routes=tuple(tuple(route) for route in routes),
                   path_choices=path_choices, synchronizations=synchronizations,
                   logistics={})


def _change_placement(problem, skeleton, rng):
    entities = {e.identifier: e for e in problem.intent.entities}
    candidates = [entity for entity in skeleton.placements
                  if not entities[entity].existing and len(entities[entity].positions) > 1]
    if not candidates: return skeleton
    entity = rng.choice(candidates); current = skeleton.placements[entity]
    positions = [p for p in entities[entity].positions if p != current]
    placements = dict(skeleton.placements); placements[entity] = rng.choice(positions)
    return replace(skeleton, placements=placements)


def _resize_workforce(problem, skeleton, rng):
    state = problem.intent.state
    if rng.random() < .5 and skeleton.workforce > len(state.workers):
        remove = rng.randrange(len(state.workers), skeleton.workforce)
        routes = [list(r) for r in skeleton.routes]
        abandoned = routes.pop(remove)
        for goal in abandoned:
            target = min(range(len(routes)), key=lambda w: (len(routes[w]), w))
            routes[target].insert(rng.randrange(len(routes[target])+1), goal)
        preference = remove-len(state.workers)
        preferences = list(skeleton.spawn_preferences)
        if 0 <= preference < len(preferences):
            preferences.pop(preference)
        return replace(skeleton, workforce=skeleton.workforce-1,
                       routes=tuple(tuple(r) for r in routes),
                       spawn_preferences=tuple(preferences))
    event_count = sum(map(len, skeleton.routes))
    if skeleton.workforce >= max(len(state.workers), event_count): return skeleton
    hires = skeleton.workforce-len(state.workers)
    spent = rules.hire_expenditure(state.hires_today, max(0, hires+1))
    if spent > realizable_opening_cash(state): return skeleton
    routes = [list(r) for r in skeleton.routes] + [[]]
    source = max(range(len(routes)-1), key=lambda w: len(routes[w]))
    if routes[source]: routes[-1].append(routes[source].pop(rng.randrange(len(routes[source]))))
    access = rules.shed_access(state.board_size)
    preferences = (*skeleton.spawn_preferences, access[hires % len(access)])
    return replace(skeleton, workforce=skeleton.workforce+1,
                   routes=tuple(tuple(r) for r in routes), spawn_preferences=preferences)


def _change_sync(problem, skeleton, rng):
    paths = selected_paths(problem, skeleton)
    route_of = {goal: worker for worker, route in enumerate(skeleton.routes) for goal in route}
    existing = {bundle.goals for bundle in skeleton.synchronizations}
    possible = []
    for path in paths.values():
        for a, b in zip(path, path[1:]):
            if route_of.get(a.goal) != route_of.get(b.goal):
                pair = (a.goal, b.goal)
                if route_of.get(a.goal, 10**9) < route_of.get(b.goal, -1): possible.append(pair)
    bundles = list(skeleton.synchronizations)
    if bundles and (not possible or rng.random() < .5):
        bundles.pop(rng.randrange(len(bundles)))
    elif possible:
        pair = rng.choice(possible)
        if pair not in existing: bundles.append(SyncBundle(pair))
    return replace(skeleton, synchronizations=tuple(bundles))


def _change_market_priority(problem, skeleton, rng):
    tokens = list(skeleton.market_priority)
    if len(tokens) < 2: return skeleton
    a, b = rng.sample(range(len(tokens)), 2); tokens[a], tokens[b] = tokens[b], tokens[a]
    return replace(skeleton, market_priority=tuple(tokens))


def _change_spawn_preference(problem, skeleton, rng):
    if not skeleton.spawn_preferences: return skeleton
    preferences = list(skeleton.spawn_preferences)
    index = rng.randrange(len(preferences))
    access = rules.shed_access(problem.intent.state.board_size)
    alternatives = [position for position in access if position != preferences[index]]
    if alternatives: preferences[index] = rng.choice(alternatives)
    return replace(skeleton, spawn_preferences=tuple(preferences))


def _change_market_cap(problem, skeleton, rng):
    state = problem.intent.state
    hire_count = max(0, skeleton.workforce-len(state.workers))
    hire_cost = rules.hire_expenditure(state.hires_today, hire_count)
    if hire_cost <= state.money or not any(state.shed.get(item, 0) for item in rules.SELLABLE_PRODUCTS):
        return skeleton
    current = skeleton.required_entry_caps[0] if skeleton.required_entry_caps else rules.MAX_MARKET_ORDERS
    alternatives = [cap for cap in (8, 9, 10) if cap != current]
    cap = rng.choice(alternatives)
    return replace(skeleton, required_entry_caps=(() if cap == 10 else (cap,)))


def _change_resource_source(problem, skeleton, rng):
    links = list(skeleton.resources)
    if not links: return skeleton
    index = rng.randrange(len(links)); link = links[index]
    events = selected_events(problem, skeleton)
    route_of = {goal: worker for worker, route in enumerate(skeleton.routes) for goal in route}
    order = {goal: n for route in skeleton.routes for n, goal in enumerate(route)}
    producers = [goal for goal, event in events.items() if event.delta.get(link.item, 0) > 0
                 and event.delta.get(link.item, 0) >= link.quantity
                 and goal != link.consumer]
    if link.kind == "PURCHASE" and producers:
        producer = rng.choice(producers)
        direct = route_of.get(producer) == route_of.get(link.consumer) and order.get(producer, 10**9) < order.get(link.consumer, -1)
        links[index] = replace(link, kind="EVENT", producer=producer, via_shed=not direct, source_worker=None)
    elif link.kind == "EVENT":
        links[index] = replace(link, kind="PURCHASE", producer=None, via_shed=False, source_worker=None)
    routes = tuple(tuple(token for token in route if token not in skeleton.logistics)
                   for route in skeleton.routes)
    shell = replace(skeleton, routes=routes, logistics={}, resources=tuple(links))
    shell = with_initial_logistics(problem, shell)
    shell = replace(shell, acquisitions=_reconcile_acquisitions(
        links, skeleton.acquisitions))
    shell = replace(shell, market_priority=_reconcile_market_priority(
        problem, skeleton, shell))
    shell = with_financing_logistics(problem, shell)
    shell = _restore_compatible_logistics(problem, skeleton, shell)
    return shell if route_precedence_feasible(problem, shell) else skeleton


def _move_logistics(problem, skeleton, rng):
    workers = [worker for worker, route in enumerate(skeleton.routes)
               if any(token in skeleton.logistics for token in route)]
    if not workers: return skeleton
    worker = rng.choice(workers); route = list(skeleton.routes[worker])
    choices = [n for n, token in enumerate(route) if token in skeleton.logistics]
    token = route.pop(rng.choice(choices))
    alternatives = list(range(len(route)+1)); rng.shuffle(alternatives)
    for at in alternatives:
        trial_route = list(route); trial_route.insert(at, token)
        routes = list(skeleton.routes); routes[worker] = tuple(trial_route)
        trial = replace(skeleton, routes=tuple(routes))
        if route_precedence_feasible(problem, trial):
            return trial
    return skeleton


def _relocate_segment(problem, skeleton, rng):
    candidates = [w for w, route in enumerate(skeleton.routes) if route]
    if not candidates: return skeleton
    source = rng.choice(candidates); route = list(skeleton.routes[source])
    begin = rng.randrange(len(route)); end = min(len(route), begin+rng.randint(1, min(6, len(route)-begin)))
    segment = route[begin:end]; del route[begin:end]
    target = rng.randrange(skeleton.workforce)
    routes = [list(r) for r in skeleton.routes]; routes[source] = route
    at = rng.randrange(len(routes[target])+1); routes[target][at:at] = segment
    return replace(skeleton, routes=tuple(tuple(r) for r in routes))


def _reverse_segment(problem, skeleton, rng):
    candidates = [w for w, route in enumerate(skeleton.routes) if len(route) >= 2]
    if not candidates: return skeleton
    worker = rng.choice(candidates); route = list(skeleton.routes[worker])
    a, b = sorted(rng.sample(range(len(route)+1), 2))
    if b-a < 2: return skeleton
    route[a:b] = reversed(route[a:b])
    routes = list(skeleton.routes); routes[worker] = tuple(route)
    return replace(skeleton, routes=tuple(routes))


def _route_service_cost(start, route, positions):
    """Exact worker turns for a service-only route under shortest movement."""
    here, cost = start, 0
    for goal in route:
        if goal not in positions:
            continue
        there = positions[goal]
        cost += rules.manhattan(here, there) + 1
        here = there
    return cost


def _ruin_recreate(problem, skeleton, rng, focus_goals=(), atomic=None):
    """Remove a related set of work and jointly rebuild its assignment/order.

    The ordinary mutations change one event, one entity, or two suffixes.  That
    is a poor neighborhood when two or more tight routes must exchange several
    visits at once.  This neighborhood selects work from the real event graph,
    removes it from every worker, and uses capacity-aware regret insertion to
    reconstruct a different reachable route structure.  Some rounds keep an
    entity's local causal path together; others expose its atomic events so
    cross-worker ordered effects remain searchable.
    """
    paths = selected_paths(problem, skeleton)
    positions = event_positions(problem, skeleton)
    entities = [entity for entity, path in paths.items()
                if path and entity in skeleton.placements]
    if len(entities) < 2:
        return skeleton

    route_of = {goal: worker for worker, route in enumerate(skeleton.routes)
                for goal in route if goal in positions}
    entity_workers = {
        entity: {route_of[event.goal] for event in path if event.goal in route_of}
        for entity, path in paths.items()
    }
    entity_position = {entity: skeleton.placements[entity] for entity in entities}
    focus_entities = {problem.goal_entity[goal] for goal in focus_goals
                      if goal in problem.goal_entity
                      and problem.goal_entity[goal] in entity_position}
    count = min(len(entities), max(len(focus_entities)+2,
        rng.randint(max(2, len(entities)//10), max(3, len(entities)//4))))
    mode = rng.randrange(4)
    anchor = rng.choice(entities)
    if mode == 0:  # A spatial district, crossing whatever routes currently serve it.
        ax, ay = entity_position[anchor]
        ranked = sorted(entities, key=lambda entity: (
            rules.manhattan((ax, ay), entity_position[entity]), rng.random()))
    elif mode == 1:  # Several fragments from the most burdened route and its neighbors.
        service_routes = [[goal for goal in route if goal in positions]
                          for route in skeleton.routes]
        starts = _worker_starts(problem, skeleton)
        burdened = max(range(skeleton.workforce),
                       key=lambda worker: (_route_service_cost(starts[worker],
                           service_routes[worker], positions), len(service_routes[worker])))
        ranked = sorted(entities, key=lambda entity: (
            int(burdened not in entity_workers.get(entity, set())),
            rules.manhattan(entity_position[anchor], entity_position[entity]),
            rng.random()))
    elif mode == 2:  # Work spanning a pair of worker regions.
        occupied = sorted({worker for workers in entity_workers.values() for worker in workers})
        selected_workers = set(rng.sample(occupied, min(2, len(occupied)))) if occupied else set()
        ranked = sorted(entities, key=lambda entity: (
            int(not (entity_workers.get(entity, set()) & selected_workers)), rng.random()))
    else:  # Unbiased disruption prevents the relation metric becoming a template.
        ranked = list(entities); rng.shuffle(ranked)
    if focus_entities:
        focus_positions = [entity_position[entity] for entity in focus_entities]
        ranked = sorted((entity for entity in entities if entity not in focus_entities),
            key=lambda entity: (
                int(not any(entity_workers.get(entity, set()) & entity_workers.get(focus, set())
                            for focus in focus_entities)),
                min(rules.manhattan(entity_position[entity], position)
                    for position in focus_positions),
                rng.random()))
        ruined_entities = set(focus_entities)
        ruined_entities.update(ranked[:max(0, count-len(ruined_entities))])
    else:
        ruined_entities = set(ranked[:count])
    ruined_goals = {event.goal for entity in ruined_entities for event in paths[entity]}
    unpositioned = [[goal for goal in route
                     if goal not in skeleton.logistics and goal not in positions]
                    for route in skeleton.routes]
    routes = [[goal for goal in route
               if goal not in skeleton.logistics and goal in positions
               and goal not in ruined_goals]
              for route in skeleton.routes]

    # Entity blocks preserve cheap coherent visits; atomic rounds deliberately
    # admit split service chains and same-turn multiworker realization.
    atomic = rng.random() < .4 if atomic is None else atomic
    if atomic:
        units = [(event.goal,) for entity in sorted(ruined_entities) for event in paths[entity]]
    else:
        units = [tuple(event.goal for event in paths[entity]) for entity in sorted(ruined_entities)]
    if not units:
        return skeleton

    starts = _worker_starts(problem, skeleton)
    horizon = min(problem.intent.state.turns_left_today, problem.intent.state.turns_left)

    def capacity(worker):
        support_margin = max(1, horizon//8)
        if worker < len(problem.intent.state.workers):
            return max(1, horizon-support_margin)
        hire_rank = worker-len(problem.intent.state.workers)
        return max(0, horizon-1-hire_rank//rules.MAX_MARKET_ORDERS-support_margin)

    predecessor = event_predecessors(problem, skeleton)
    successors = defaultdict(set)
    for goal, required in predecessor.items():
        for prior in required:
            successors[prior].add(goal)

    def insertion_options(unit, current):
        options = []
        current_overruns = [max(0, cost-capacity(w)) for w, cost in enumerate(current)]
        total_cost = sum(current)
        for worker in range(skeleton.workforce):
            route = routes[worker]
            indices = {goal: n for n, goal in enumerate(route)}
            other_overrun_max = max((value for w, value in enumerate(current_overruns) if w != worker),
                                    default=0)
            other_overrun_sum = sum(current_overruns)-current_overruns[worker]
            for at in range(len(routes[worker])+1):
                before = starts[worker] if at == 0 else positions[route[at-1]]
                after = positions[route[at]] if at < len(route) else None
                inserted = rules.manhattan(before, positions[unit[0]])+1
                inserted += sum(rules.manhattan(positions[a], positions[b])+1
                                for a, b in zip(unit, unit[1:]))
                if after is not None:
                    inserted += rules.manhattan(positions[unit[-1]], after)
                    removed = rules.manhattan(before, after)
                else:
                    removed = 0
                trial_cost = current[worker]+inserted-removed
                overrun = max(0, trial_cost-capacity(worker))
                # Same-route inversions are avoidable causal waits. Cross-route
                # precedence remains legal and is timed by the abstract model.
                inversions = sum(1 for goal in unit for required in predecessor.get(goal, ())
                                 if required in indices and indices[required] >= at)
                inversions += sum(1 for goal in unit for following in successors.get(goal, ())
                                  if following in indices and indices[following] < at)
                if inversions:
                    continue
                latest = max((problem.goals[goal].deadline for goal in unit),
                             default=problem.intent.state.step+horizon-1)
                finish = problem.intent.state.step + trial_cost - 1
                lateness = max(0, finish-latest)
                revised_total = total_cost-current[worker]+trial_cost
                revised_max = max((cost for w, cost in enumerate(current)
                                   if w != worker), default=0)
                revised_max = max(revised_max, trial_cost)
                # Feasibility and deadlines come first. Among equally feasible
                # insertions, total route cost—not artificial load balance—owns
                # assignment; peak load is only a late tie-break.
                key = (int(bool(max(other_overrun_max, overrun))),
                       other_overrun_sum+overrun, lateness, revised_total,
                       trial_cost-current[worker], revised_max, worker, at)
                options.append((key, worker, at))
        options.sort()
        return options

    while units:
        current = [_route_service_cost(starts[w], route, positions)
                   for w, route in enumerate(routes)]
        choices = []
        for unit in units:
            options = insertion_options(unit, current)
            if not options:
                return skeleton
            best = options[0][0]
            feasible = sum(option[0][:3] == best[:3] for option in options)
            second = options[1][0] if len(options) > 1 else best
            # Most constrained and highest-regret work is committed first.
            regret = (second[3]-best[3], second[4]-best[4], second[5]-best[5])
            deadline = min(problem.goals[goal].deadline for goal in unit)
            choices.append(((feasible, deadline, tuple(-value for value in regret),
                             -len(unit), unit), unit, options))
        _, unit, options = min(choices, key=lambda row: row[0])
        # Mostly steepest insertion, with deterministic seeded breadth among
        # structurally close choices. Exact abstract/replay scoring follows.
        near = [option for option in options
                if option[0][:3] == options[0][0][:3]][:4]
        selected = near[0] if len(near) == 1 or rng.random() < .8 else rng.choice(near[1:])
        _, worker, at = selected
        routes[worker][at:at] = unit
        units.remove(unit)

    return replace(skeleton, routes=tuple(
        tuple([*route, *unpositioned[worker]]) for worker, route in enumerate(routes)),
        logistics={})


def _worker_starts(problem, skeleton):
    state = problem.intent.state
    access = rules.shed_access(state.board_size)
    starts = [worker.position for worker in state.workers]
    starts.extend(skeleton.spawn_preferences)
    while len(starts) < skeleton.workforce:
        starts.append(access[(len(starts)-len(state.workers)) % len(access)])
    return tuple(starts[:skeleton.workforce])


def _reconstruct_open_placement_matching(problem, skeleton, rng):
    """Find a complete matching for coupled open placement domains.

    Greedily moving one asset cannot repair nested domains: the desired tile
    may be occupied by another new asset which itself has alternatives. This
    augmenting-path reconstruction considers every domain edge and displaces
    earlier assignments when necessary. Existing assets that are not released
    by today's selected path remain fixed.
    """
    paths = selected_paths(problem, skeleton)
    entities = {entity.identifier: entity for entity in problem.intent.entities}
    new = [entity for entity in problem.intent.entities
           if not entity.existing and paths[entity.identifier]]
    fixed = set()
    for entity in problem.intent.entities:
        if not entity.existing or entity.identifier not in skeleton.placements:
            continue
        if _released_by(paths[entity.identifier]) is None:
            fixed.add(skeleton.placements[entity.identifier])

    candidates = {}
    for entity in new:
        positions = [position for position in entity.positions if position not in fixed]
        positions.sort(key=lambda position: (
            int(position != skeleton.placements.get(entity.identifier)),
            rules.distance_to_shed(position)*max(1, entity.future_service_days),
            rng.random(), position))
        candidates[entity.identifier] = positions
    order = sorted((entity for entity in new
                    if entity.identifier not in skeleton.placements), key=lambda entity: (
        len(candidates[entity.identifier]), -entity.future_service_days,
        -len(entity.goals), entity.identifier))
    occupant = {position: entity.identifier for entity in new
                for position in (skeleton.placements.get(entity.identifier),)
                if position is not None and position not in fixed}

    def assign(identifier, seen):
        for position in candidates[identifier]:
            if position in seen:
                continue
            seen.add(position)
            other = occupant.get(position)
            if other is None or assign(other, seen):
                occupant[position] = identifier
                return True
        return False

    for entity in order:
        if not assign(entity.identifier, set()):
            return skeleton
    placements = dict(skeleton.placements)
    placements.update({identifier: position for position, identifier in occupant.items()})
    routes = tuple(tuple(token for token in route if token not in skeleton.logistics)
                   for route in skeleton.routes)
    return replace(skeleton, placements=placements, routes=routes, logistics={})


def _compress_workforce_frontier(problem, skeleton, limit):
    """Reconstruct hired-worker routes after removing one paid hand.

    A full realization used to be terminal for exact repair, so excess hands
    survived even when their work could be absorbed by a different assignment
    and ordering.  Each candidate here removes a real hired worker and lets the
    same regret-based ruin/recreate machinery jointly rebuild that worker's
    entities with neighboring work. Exact transition replay decides whether
    Plan quality was preserved.
    """
    state = problem.intent.state
    if skeleton.workforce <= len(state.workers) or not limit:
        return []
    proposals = {}
    hired = range(len(state.workers), skeleton.workforce)
    # Least loaded hands are the obvious compression basins, but every hired
    # worker remains reachable when the limit permits it.
    ranked_hands = sorted(hired, key=lambda worker: (
        sum(token in problem.goals for token in skeleton.routes[worker]),
        len(skeleton.routes[worker]), worker))[:max(4, limit*2)]
    for remove in ranked_hands:
        removed_goals = tuple(token for token in skeleton.routes[remove]
                              if token in problem.goals)
        routes = [tuple(token for token in route if token not in skeleton.logistics)
                  for route in skeleton.routes]
        routes.pop(remove)
        preferences = list(skeleton.spawn_preferences)
        preference = remove-len(state.workers)
        if 0 <= preference < len(preferences):
            preferences.pop(preference)
        reduced = replace(skeleton, workforce=skeleton.workforce-1,
                          routes=tuple(routes), spawn_preferences=tuple(preferences),
                          logistics={})
        repair_seed = (state.step*1_000_003 + remove*1009
                       + sum(sum(map(ord, goal)) for goal in removed_goals))
        # Distinct regret tie-break basins matter: on dense development days
        # the first successful ejection chains occur beyond offsets 0/1.
        trials = 4
        for offset in range(trials):
            try:
                trial = _ruin_recreate(problem, reduced,
                    Random(repair_seed+offset), focus_goals=removed_goals)
                trial = _construct_route_candidate(problem, skeleton, trial)
                if trial is None:
                    continue
                proposals[skeleton_key(trial)] = (_abstract_schedule(problem, trial), trial)
            except (ValueError, IndexError):
                continue
    return [trial for _, trial in sorted(proposals.values(), key=lambda row: row[0], reverse=True)[:limit]]


def _project_adjacent_workforce(problem, skeleton, rng):
    """Migrate assignment/order structure into the adjacent lower basin.

    This is a search proposal, not post-hoc staffing compression. The removed
    hand's causal work is jointly ruin/recreated with neighboring work, after
    which the target workforce population can continue changing assignments,
    order, resources, batching, placement and logistics independently.
    """
    state = problem.intent.state
    if skeleton.workforce <= len(state.workers):
        return skeleton
    hired = list(range(len(state.workers), skeleton.workforce))
    ranked = sorted(hired, key=lambda worker: (
        sum(token in problem.goals for token in skeleton.routes[worker]),
        len(skeleton.routes[worker]), worker))
    remove = rng.choice(ranked[:min(4, len(ranked))])
    abandoned = tuple(token for token in skeleton.routes[remove]
                      if token in problem.goals)
    routes = [tuple(token for token in route if token not in skeleton.logistics)
              for route in skeleton.routes]
    routes.pop(remove)
    preferences = list(skeleton.spawn_preferences)
    preference = remove-len(state.workers)
    if 0 <= preference < len(preferences):
        preferences.pop(preference)
    reduced = replace(skeleton, workforce=skeleton.workforce-1,
                      routes=tuple(routes), spawn_preferences=tuple(preferences),
                      logistics={})
    rebuilt = _ruin_recreate(problem, reduced, rng, focus_goals=abandoned)
    candidate = _construct_route_candidate(problem, skeleton, rebuilt)
    if candidate is None:
        raise ValueError("adjacent workforce reconstruction is not causal")
    return candidate


def _reconcile_synchronizations(problem, skeleton):
    events = selected_events(problem, skeleton)
    route_of = {goal: worker for worker, route in enumerate(skeleton.routes)
                for goal in route}
    predecessors = event_predecessors(problem, skeleton)
    retained = []
    for bundle in skeleton.synchronizations:
        workers = [route_of.get(goal) for goal in bundle.goals]
        if (all(goal in events for goal in bundle.goals)
                and None not in workers and len(set(workers)) == len(workers)
                and workers == sorted(workers)
                and all(before in predecessors.get(after, ())
                        for before, after in zip(bundle.goals, bundle.goals[1:]))):
            retained.append(bundle)
    return replace(skeleton, synchronizations=tuple(retained))


def _construct_route_candidate(problem, before, candidate):
    """Make a route edit complete and causal before it enters search."""
    old_logistics = set(before.logistics)
    routes = tuple(tuple(token for token in route if token not in old_logistics)
                   for route in candidate.routes)
    shell = replace(candidate, routes=routes, logistics={}, resources=())
    shell = replace(shell, leases=rebuild_leases(problem, shell))
    if not route_precedence_feasible(problem, shell):
        return None
    shell = _reconcile_synchronizations(problem, shell)
    shell = _rebuild_route_support(problem, before, shell)
    if not route_precedence_feasible(problem, shell):
        return None
    validate_skeleton(problem, shell)
    return shell


MUTATIONS = (_relocate_event, _relocate_entity, _cross_exchange, _change_path,
             _change_placement, _resize_workforce, _change_sync,
             _change_market_priority, _change_resource_source, _move_logistics,
             _relocate_segment, _reverse_segment, _change_spawn_preference,
             _change_market_cap)
WITHIN_WORKFORCE_MUTATIONS = tuple(mutation for mutation in MUTATIONS
                                   if mutation is not _resize_workforce)


def _mutate(problem, skeleton, rng, ruin_probability=.10, fixed_workforce=False):
    # A substantial fraction of proposals must cross the assignment/order
    # valleys that single-event moves cannot cross.
    choices = WITHIN_WORKFORCE_MUTATIONS if fixed_workforce else MUTATIONS
    mutation = _ruin_recreate if rng.random() < ruin_probability else rng.choice(choices)
    route_mutations = {_relocate_event, _relocate_entity, _cross_exchange,
                       _change_path, _resize_workforce, _relocate_segment,
                       _reverse_segment, _ruin_recreate}
    preserving_mutations = {_change_sync, _change_market_priority,
                            _change_spawn_preference, _change_market_cap,
                            _change_resource_source, _move_logistics}
    # Sampling retries are internal to candidate construction: malformed
    # lifecycle orders never consume a population/evaluation proposal.
    for _ in range(8):
        trial = mutation(problem, skeleton, rng)
        if mutation in route_mutations:
            trial = _construct_route_candidate(problem, skeleton, trial)
            if trial is None:
                continue
        elif mutation is _change_placement:
            trial = normalize(problem, trial)
            if len(trial.placements) < len(
                    [entity for entity in problem.intent.entities
                     if problem.complete_paths[entity.identifier]]):
                trial = normalize(problem, _reconstruct_open_placement_matching(
                    problem, trial, rng))
        elif mutation not in preserving_mutations:
            trial = normalize(problem, trial)
        if route_precedence_feasible(problem, trial):
            validate_skeleton(problem, trial)
            return trial
    return skeleton


def _exact_repair_frontier(problem, skeleton, result, limit):
    """Relocate compiler-observed unfinished causal tails to spare workers."""
    events = selected_events(problem, skeleton)
    unfinished_plan = set(result.unfulfilled) & set(problem.goals)
    unfinished = unfinished_plan & set(events)
    if (not unfinished_plan and not result.diagnostics.get("unbought_inputs")) or not limit:
        return []
    used = [0] * skeleton.workforce
    available = [0] * skeleton.workforce
    for execution in result.executions:
        for worker, action in enumerate(execution.worker_actions):
            available[worker] += 1
            if action[0] != "PASS": used[worker] += 1
    targets = sorted(range(skeleton.workforce),
                     key=lambda worker: (-(available[worker]-used[worker]),
                                         -available[worker], worker))
    paths = selected_paths(problem, skeleton)
    entities = {work.identifier: work for work in problem.intent.entities}
    proposals = {}
    placement_proposals = {}
    market_repairs = []
    for purchase, missing_quantity in result.diagnostics.get("unbought_inputs", {}).items():
        token = f"BUY:{purchase}"
        if token not in skeleton.market_priority: continue
        priority = [entry for entry in skeleton.market_priority if entry != token]
        priority.insert(0, token)
        market_repairs.append(replace(skeleton, market_priority=tuple(priority)))
        order = skeleton.acquisitions.get(purchase)
        if order and int(order[2]) > int(missing_quantity) > 0:
            early, deferred = f"{purchase}#critical", f"{purchase}#deferred"
            acquisitions = dict(skeleton.acquisitions); del acquisitions[purchase]
            acquisitions[early] = (order[0], order[1], int(missing_quantity))
            acquisitions[deferred] = (order[0], order[1], int(order[2])-int(missing_quantity))
            original = list(skeleton.market_priority)
            old_at = original.index(token); original[old_at:old_at+1] = [f"BUY:{deferred}"]
            # Every insertion frontier is a real batching/order alternative;
            # exact replay decides how many early hires it is worth displacing.
            for at in range(old_at+1):
                split_priority = list(original); split_priority.insert(at, f"BUY:{early}")
                market_repairs.append(replace(skeleton, acquisitions=acquisitions,
                                              market_priority=tuple(split_priority)))
    if result.diagnostics.get("unbought_inputs"):
        buys = [token for token in skeleton.market_priority if token.startswith("BUY:")]
        rest = [token for token in skeleton.market_priority if not token.startswith("BUY:")]
        market_repairs.append(replace(skeleton, market_priority=tuple([*buys, *rest])))
    unfinished_blocks = []
    for entity, path in paths.items():
        if not any(event.goal in unfinished_plan for event in path):
            continue
        # Rebuild the whole causal visit, including effects that happened in
        # the failed trajectory. Moving only the unfinished suffix can strand
        # it behind a producer on another tight route; compilation starts from
        # day-start state, so every selected effect must remain represented.
        block = [event.goal for event in path]
        unfinished_blocks.append(tuple(block))
        work = entities[entity]
        source = next((worker for worker, route in enumerate(skeleton.routes)
                       if any(goal in route for goal in block)), None)
        # An open lifecycle can be absent from both placement and routes when
        # nested placement domains defeated a greedy binding. Reconstruct the
        # whole open-asset matching, then jointly rebuild the missing lifecycle
        # and its neighboring assignments. This searches coupled displacements
        # instead of repeatedly colliding with the same occupied tile.
        if not work.existing and entity not in skeleton.placements:
            placement_seed = (problem.intent.state.step*1_000_003
                              + sum(map(ord, entity)))
            for offset in range(max(4, limit*2)):
                placement_rng = Random(placement_seed+offset)
                try:
                    placed = _reconstruct_open_placement_matching(
                        problem, skeleton, placement_rng)
                    trial = _ruin_recreate(
                        problem, placed, placement_rng, focus_goals=block)
                    trial = _construct_route_candidate(problem, skeleton, trial)
                    if trial is None:
                        continue
                    key = skeleton_key(trial)
                    row = (_abstract_schedule(problem, trial), trial)
                    proposals[key] = row
                    placement_proposals[key] = row
                except (ValueError, IndexError):
                    continue
        elif not work.existing and len(work.positions) > 1:
            placement_choices = sorted(work.positions, key=lambda position: (
                rules.distance_to_shed(position)*max(1, work.future_service_days),
                position))[:max(4, limit)]
            for position in placement_choices:
                if position == skeleton.placements.get(entity):
                    continue
                placements = dict(skeleton.placements)
                placements[entity] = position
                try:
                    trial = normalize(problem, replace(
                        skeleton, placements=placements))
                    if not route_precedence_feasible(problem, trial):
                        continue
                    validate_skeleton(problem, trial)
                    key = skeleton_key(trial)
                    proposals[key] = (_abstract_schedule(problem, trial), trial)
                except (ValueError, IndexError):
                    continue
        if source is None:
            continue
        base_routes = [[token for token in route
                        if token not in skeleton.logistics and token not in block]
                       for route in skeleton.routes]
        repair_targets = [source, *(worker for worker in targets if worker != source)][:min(7, len(targets))]
        starts = _worker_starts(problem, skeleton)
        positions = event_positions(problem, skeleton)
        relocation_choices = []
        for target in repair_targets:
            for at in range(len(base_routes[target])+1):
                routes = [list(route) for route in base_routes]
                routes[target][at:at] = block
                costs = tuple(_route_service_cost(starts[worker], route, positions)
                              for worker, route in enumerate(routes))
                relocation_choices.append(
                    ((sum(costs), max(costs), costs[target], target, at), routes))
        # Keep the real insertion domain, but only normalize its strongest
        # capacity/route-cost representatives. Exhaustively rebuilding resource
        # links for every insertion point overwhelmed the bounded search.
        for _, routes in sorted(relocation_choices, key=lambda row: row[0])[:max(4, limit*2)]:
            try:
                trial = _construct_route_candidate(problem, skeleton, replace(
                    skeleton, routes=tuple(tuple(route) for route in routes), logistics={}))
                if trial is None:
                    continue
                key = skeleton_key(trial)
                proposals[key] = (_abstract_schedule(problem, trial), trial)
            except (ValueError, IndexError):
                continue

        ejection_choices = []
        for target in repair_targets:
            # Exchange the late tail with one complete causal block from the
            # target route. This is the smallest ejection chain that can repair
            # two simultaneously tight routes without adding staff.
            if target == source:
                continue
            target_entities = []
            for token in base_routes[target]:
                other = problem.goal_entity.get(token)
                if other is not None and other != entity and other not in target_entities:
                    target_entities.append(other)
            source_at = min((n for n, token in enumerate(base_routes[source])
                             if problem.goal_entity.get(token) == entity),
                            default=len(base_routes[source]))
            for other in target_entities:
                other_block = [event.goal for event in paths[other]
                               if event.goal in base_routes[target]]
                if not other_block: continue
                target_at = min(base_routes[target].index(goal) for goal in other_block)
                routes = [list(route) for route in base_routes]
                routes[target] = [goal for goal in routes[target] if goal not in other_block]
                routes[source][source_at:source_at] = other_block
                routes[target][target_at:target_at] = block
                costs = tuple(_route_service_cost(starts[worker], route, positions)
                              for worker, route in enumerate(routes))
                ejection_choices.append(((sum(costs), max(costs), target, other), routes))
        for _, routes in sorted(ejection_choices, key=lambda row: row[0])[:max(4, limit*2)]:
            try:
                trial = _construct_route_candidate(problem, skeleton, replace(
                    skeleton, routes=tuple(tuple(route) for route in routes), logistics={}))
                if trial is None:
                    continue
                key = skeleton_key(trial)
                proposals[key] = (_abstract_schedule(problem, trial), trial)
            except (ValueError, IndexError):
                continue
    # Compiler feedback identifies a basin, not a single event to relocate.
    # Rebuild a related multi-route district around every unfinished goal so a
    # repair can exchange several assignments/orders in one move.
    if unfinished_plan:
        repair_rng = Random(problem.intent.state.step*1_000_003
                            + sum(sum(map(ord, goal)) for goal in sorted(unfinished_plan)))
        for _ in range(max(8, limit*3)):
            try:
                trial = _construct_route_candidate(problem, skeleton, _ruin_recreate(
                    problem, skeleton, repair_rng, focus_goals=unfinished_plan))
                if trial is None:
                    continue
                key = skeleton_key(trial)
                proposals[key] = (_abstract_schedule(problem, trial), trial)
            except (ValueError, IndexError):
                continue
    ranked = sorted(proposals.values(), key=lambda row: row[0], reverse=True)

    # A coupled neighborhood redistributes every unfinished block together.
    # This handles cases where repairing one tail merely makes another route
    # the bottleneck.
    redistributed = None
    if unfinished_blocks:
        positions = event_positions(problem, skeleton)
        state = problem.intent.state
        access = rules.shed_access(state.board_size)
        starts = [worker.position for worker in state.workers]
        starts.extend(skeleton.spawn_preferences)
        while len(starts) < skeleton.workforce:
            starts.append(access[(len(starts)-len(state.workers)) % len(access)])
        remove = {goal for block in unfinished_blocks for goal in block}
        routes = [[token for token in route
                   if token not in skeleton.logistics and token not in remove]
                  for route in skeleton.routes]

        def route_cost(worker, route):
            here, cost = starts[worker], 0
            for goal in route:
                if goal not in positions: continue
                there = positions[goal]
                cost += rules.manhattan(here, there)+1; here = there
            return cost

        for block in sorted(unfinished_blocks,
                            key=lambda goals: min(problem.goals[g].deadline for g in goals)):
            costs = [route_cost(worker, route) for worker, route in enumerate(routes)]
            best = None
            for worker in targets:
                for at in range(len(routes[worker])+1):
                    trial = routes[worker][:at]+list(block)+routes[worker][at:]
                    cost = route_cost(worker, trial)
                    revised = costs[:worker]+[cost]+costs[worker+1:]
                    candidate = (sum(revised), cost-costs[worker],
                                 max(revised), -(available[worker]-used[worker]),
                                 worker, at)
                    if best is None or candidate < best[0]: best = candidate, worker, at
            _, worker, at = best
            routes[worker][at:at] = block
        try:
            redistributed = _construct_route_candidate(problem, skeleton, replace(
                skeleton, routes=tuple(tuple(route) for route in routes), logistics={}))
        except (ValueError, IndexError):
            redistributed = None
    # A financing/input correction cannot consume the whole repair frontier:
    # observed deficits are primarily route tails, and those alternatives must
    # still receive exact transition evaluation.
    result_frontier = list(market_repairs[:min(2, max(1, limit//3))])
    placement_ranked = sorted(placement_proposals.values(),
                              key=lambda row: row[0], reverse=True)
    result_frontier.extend(trial for _, trial in placement_ranked[:limit]
                           if skeleton_key(trial) not in {
                               skeleton_key(item) for item in result_frontier})
    if redistributed is not None and len(result_frontier) < limit:
        result_frontier.append(redistributed)
    present = {skeleton_key(trial) for trial in result_frontier}
    result_frontier.extend(trial for _, trial in ranked
                           if skeleton_key(trial) not in present)
    return result_frontier[:limit]


def _complete_initial_placement(problem, skeleton, seed):
    """Ensure every placeable Plan entity enters routes before search begins."""
    missing = [entity.identifier for entity in problem.intent.entities
               if problem.complete_paths[entity.identifier]
               and entity.identifier not in skeleton.placements]
    if not missing:
        return skeleton
    placed = _reconstruct_open_placement_matching(problem, skeleton, Random(seed))
    still_missing = [identifier for identifier in missing
                     if identifier not in placed.placements]
    if still_missing:
        # Preserve genuine placement infeasibility for exact reporting.
        return skeleton
    # The old initializer omitted events whose entity lacked a greedy placement.
    # Rebuild all route blocks once from the complete matching so search never
    # optimizes or benchmarks a silently reduced Plan.
    routes = initial_routes(problem, placed, placed.workforce)
    rebuilt = _construct_route_candidate(problem, skeleton, replace(
        placed, routes=routes, logistics={}))
    return rebuilt or skeleton


def _workforce_levels(problem, base, maximum):
    """Return every capacity-feasible staffing level as an independent basin.

    The lower bound counts mandatory service actions only and is therefore a
    proof of insufficiency below it, not an estimate of route efficiency.
    Travel, pickups, batching and synchronization remain choices searched
    jointly inside each retained workforce level.
    """
    state = problem.intent.state
    opening = len(state.workers)
    horizon = min(state.turns_left_today, state.turns_left)
    event_count = sum(len(path[0]) for path in problem.complete_paths.values() if path)

    def available_turns(workforce):
        turns = opening*horizon
        for hire_rank in range(workforce-opening):
            first_action = 1+hire_rank//rules.MAX_MARKET_ORDERS
            turns += max(0, horizon-first_action)
        return turns

    lower = next((workforce for workforce in range(opening, maximum+1)
                  if available_turns(workforce) >= event_count), maximum)
    # A zero-work Plan still owns the actual opening workforce basin.
    lower = min(lower, base.workforce)
    return tuple(range(lower, maximum+1))


def solve_routes(state, plan, config=None, progress=None):
    config = config or RouteSearchConfig()
    started = perf_counter()
    problem = build_route_problem(state, plan, progress)
    seed = config.random_seed + state.step*1_000_003 + state.player*97
    base = initial_skeleton(problem)
    base = _complete_initial_placement(problem, base, seed+base.workforce)
    event_count = sum(len(path[0]) for path in problem.complete_paths.values() if path)

    cash, maximum = realizable_opening_cash(state), len(state.workers)
    while maximum < max(len(state.workers), event_count):
        cost = rules.fibonacci_hire_cost(state.hires_today+maximum-len(state.workers))
        if cost > cash:
            break
        cash -= cost
        maximum += 1
    horizon = min(state.turns_left_today, state.turns_left)
    maximum = min(maximum,
                  len(state.workers)+rules.MAX_MARKET_ORDERS*max(0, horizon-1),
                  max(len(state.workers), event_count))

    starts = {}
    for workforce in _workforce_levels(problem, base, maximum):
        try:
            skeleton = initial_skeleton(problem, workforce)
            skeleton = _complete_initial_placement(
                problem, skeleton, seed+workforce)
            starts[workforce] = (_abstract_schedule(problem, skeleton),
                                 skeleton_key(skeleton), skeleton)
        except ValueError:
            continue
    if not starts:
        starts[base.workforce] = (_abstract_schedule(problem, base),
                                  skeleton_key(base), base)

    # Each workforce is an independent search basin. The total proposal budget
    # is unchanged and split deterministically; no high-capacity population can
    # evict a lower-capacity assignment/order/resource-routing hypothesis.
    workforces = tuple(sorted(starts))
    basin_capacity = max(4, config.population//len(workforces))
    populations = {workforce: [starts[workforce]] for workforce in workforces}
    basin_rng = {workforce: Random(seed+104_729*workforce) for workforce in workforces}
    basin_iterations = Counter()
    basin_generated = Counter()
    seen = {row[1] for row in starts.values()}
    generated = accepted = invalid = 0
    global_iterations = min(config.iterations, config.iterations*2//3)
    global_population = sorted(starts.values(), key=lambda row: row[0], reverse=True)
    global_population = global_population[:config.population]
    global_rng = Random(seed)
    for _ in range(global_iterations):
        if global_rng.random() < .35:
            workforce = global_rng.choice(workforces)
            source_population = populations[workforce]
            parent = source_population[
                global_rng.randrange(max(1, (len(source_population)+1)//2))][2]
        else:
            parent = global_population[
                global_rng.randrange(max(1, min(len(global_population),
                                                config.population//2)))][2]
        try:
            candidate = _mutate(problem, parent, global_rng,
                                min(.10, config.ruin_probability), fixed_workforce=False)
            key = skeleton_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            generated += 1
            abstract = _abstract_schedule(problem, candidate)
            row = (abstract, key, candidate)
        except (ValueError, IndexError):
            invalid += 1
            continue
        global_population.append(row)
        global_population.sort(key=lambda item: item[0], reverse=True)
        del global_population[config.population:]
        if candidate.workforce in populations:
            population = populations[candidate.workforce]
            basin_generated[candidate.workforce] += 1
            if len(population) < basin_capacity or row[0] > population[-1][0]:
                population.append(row)
                population.sort(key=lambda item: (item[0], repr(item[1])), reverse=True)
                del population[basin_capacity:]
                accepted += 1

    local_iterations = config.iterations-global_iterations
    # Four protected proposals, an exact witness and later reconstruction keep
    # every capacity-feasible level searchable without discarding the proven
    # shared topology population.
    # Remaining proposals go to the completion neighborhood exposed by the
    # current farm/Plan, rather than to levels whose own best schedule is far
    # below feasible. This allocation is derived before exact outcomes and does
    # not use reference staffing.
    minimum_quota = min(4, local_iterations//len(workforces))
    opening_best = max(row[0][1] for row in starts.values())
    competitive_floor = opening_best-max(8, event_count//5)
    competitive = tuple(workforce for workforce in workforces
                        if starts[workforce][0][1] >= competitive_floor) or workforces
    workforce_schedule = [workforce for workforce in workforces
                          for _ in range(minimum_quota)]
    workforce_schedule.extend(competitive[n % len(competitive)]
                              for n in range(local_iterations-len(workforce_schedule)))
    for workforce in workforce_schedule:
        basin_iterations[workforce] += 1
        population = populations[workforce]
        rng = basin_rng[workforce]
        try:
            # Adjacent-island migration transfers a coherent realization
            # hypothesis, then reconstructs it under this basin's capacity.
            # It competes for the same proposal budget as every other move.
            if (workforce+1 in populations and rng.random() < .30):
                source_population = populations[workforce+1]
                source = source_population[
                    rng.randrange(max(1, (len(source_population)+1)//2))][2]
                candidate = _project_adjacent_workforce(problem, source, rng)
            else:
                parent = population[rng.randrange(max(1, (len(population)+1)//2))][2]
                candidate = _mutate(problem, parent, rng, config.ruin_probability,
                                    fixed_workforce=True)
            if candidate.workforce != workforce:
                raise AssertionError("workforce-conditioned mutation changed basin")
            key = skeleton_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            generated += 1
            basin_generated[workforce] += 1
            row = (_abstract_schedule(problem, candidate), key, candidate)
        except (ValueError, IndexError):
            invalid += 1
            continue
        if len(population) < basin_capacity or row[0] > population[-1][0]:
            population.append(row)
            population.sort(key=lambda item: (item[0], repr(item[1])), reverse=True)
            del population[basin_capacity:]
            accepted += 1

    # Exact replay first admits the best structure from every workforce basin,
    # then spends remaining breadth round-robin. This is deliberately not a
    # single approximate frontier with token workforce diversity.
    exact_pool = []
    exact_keys = set()

    def admit(skeleton):
        key = skeleton_key(skeleton)
        if key in exact_keys:
            return
        exact_keys.add(key)
        exact_pool.append(skeleton)

    for workforce in workforces:
        admit(populations[workforce][0][2])
    admit(base)
    rank = 1
    while len(exact_pool) < max(config.exact_candidates, len(workforces)):
        added = False
        for workforce in workforces:
            population = populations[workforce]
            if rank < len(population):
                before = len(exact_pool)
                admit(population[rank][2])
                added |= len(exact_pool) != before
                if len(exact_pool) >= max(config.exact_candidates, len(workforces)):
                    break
        if not added:
            break
        rank += 1

    exact_rows = []
    exact_evaluated = set()
    exact_compilations = 0
    basin_exact = Counter()

    def compile_exact(skeleton):
        nonlocal exact_compilations, invalid
        key = skeleton_key(skeleton)
        if key in exact_evaluated or exact_compilations >= config.max_exact_evaluations:
            return None
        exact_evaluated.add(key)
        exact_compilations += 1
        basin_exact[skeleton.workforce] += 1
        try:
            result = compile_skeleton(problem, skeleton)
        except ValueError:
            invalid += 1
            return None
        return (result_score(result, state), key, skeleton, result)

    for skeleton in exact_pool:
        row = compile_exact(skeleton)
        if row is not None:
            exact_rows.append(row)
    if not exact_rows:
        result = compile_skeleton(problem, base)
        exact_compilations += 1
        basin_exact[base.workforce] += 1
        exact_rows = [(result_score(result, state), skeleton_key(base), base, result)]
    exact_rows.sort(key=lambda row: (row[0], repr(row[1])), reverse=True)
    base_rows = [row for row in exact_rows if row[1] == skeleton_key(base)]
    initial_exact_score = base_rows[0][0] if base_rows else exact_rows[0][0]

    # Preserve exact-replay capacity for the realization that survives basin
    # selection and workforce reconstruction. Early broad feedback is useful,
    # but spending the cap before the final structure exists can leave a known
    # causal tail unrepaired. This reallocates, rather than increases, budget.
    final_feedback_reserve = min(
        config.max_exact_evaluations,
        max(8, config.refinement_candidates*3))
    search_phase_limit = max(
        exact_compilations,
        config.max_exact_evaluations-final_feedback_reserve)

    # A multilevel neighborhood lets a coherent high-capacity structure seed an
    # adjacent workforce island. The target island reconstructs the removed
    # hand's causal district and competes through exact reached states; staffing
    # itself is never a score. This is part of route search, before repair and
    # final selection, rather than a post-hoc compression of the winner.
    cross_basin = []
    cross_basin_attempts = 0
    descent_source = exact_rows[0]
    descent_budget = min(search_phase_limit,
                         exact_compilations+max(8, config.refinement_candidates*10))
    while (descent_source[2].workforce-1 in populations
           and exact_compilations < descent_budget):
        target_workforce = descent_source[2].workforce-1
        target_added = []
        matched = 0
        migrations = _compress_workforce_frontier(
            problem, descent_source[2], max(4, config.refinement_candidates*2))
        for candidate in migrations:
            if exact_compilations >= descent_budget:
                break
            try:
                key = skeleton_key(candidate)
                if key in exact_evaluated:
                    continue
                cross_basin_attempts += 1
                row = compile_exact(candidate)
                if row is None:
                    continue
                target_added.append(row)
                same_quality = (
                    row[3].completed == descent_source[3].completed
                    and row[3].diagnostics.get("unlocked_land_missing")
                    == descent_source[3].diagnostics.get("unlocked_land_missing"))
                matched += int(same_quality)
                if matched >= 2:
                    break
            except (ValueError, IndexError):
                invalid += 1
        if target_added:
            cross_basin.extend(target_added)
            exact_rows.extend(target_added)
            exact_rows.sort(key=lambda row: (row[0], repr(row[1])), reverse=True)
        feasible_target = [row for row in exact_rows
                           if row[2].workforce == target_workforce
                           and row[3].completed == descent_source[3].completed
                           and row[3].diagnostics.get("unlocked_land_missing")
                               == descent_source[3].diagnostics.get("unlocked_land_missing")]
        if not feasible_target:
            break
        descent_source = max(feasible_target,
                             key=lambda row: (row[0], repr(row[1])))

    # Exact failure feedback is round-robin across workforce basins. Finding a
    # full high-workforce route does not terminate repairs in lower basins.
    refinement = []
    initial_best_completion = max(len(row[3].completed) for row in exact_rows)
    repair_floor = initial_best_completion-max(8, event_count//5)
    repair_workforces = [workforce for workforce in workforces
                         if any(row[2].workforce == workforce
                                and len(row[3].completed) >= repair_floor
                                for row in exact_rows)]
    # Bound expensive exact-feedback construction while retaining a spread of
    # near-feasible workforce levels. Every basin already received structural
    # search and exact replay; this frontier is for diagnosed causal tails.
    if len(repair_workforces) > 8:
        indices = {round(n*(len(repair_workforces)-1)/7) for n in range(8)}
        repair_workforces = [workforce for n, workforce in enumerate(repair_workforces)
                             if n in indices]

    for _ in range(config.repair_rounds):
        added = []
        for workforce in repair_workforces:
            if exact_compilations >= search_phase_limit:
                break
            basin_rows = [row for row in exact_rows if row[2].workforce == workforce]
            if not basin_rows:
                continue
            basin = max(basin_rows, key=lambda row: (row[0], repr(row[1])))
            candidates = _exact_repair_frontier(
                problem, basin[2], basin[3], config.refinement_candidates)
            candidate = next((trial for trial in candidates
                              if skeleton_key(trial) not in exact_evaluated), None)
            if candidate is None:
                continue
            seen.add(skeleton_key(candidate))
            row = compile_exact(candidate)
            if row is not None:
                added.append(row)
        if not added:
            break
        refinement.extend(added)
        exact_rows.extend(added)
        exact_rows.sort(key=lambda row: (row[0], repr(row[1])), reverse=True)

    # Compiler feedback may improve the source topology after the first
    # multilevel wave. Re-seed adjacent workforce basins once from that improved
    # source, within the same hard exact-evaluation cap, before any cleanup.
    reconstruction_limit = max(
        exact_compilations,
        search_phase_limit-config.compression_rounds*config.compression_candidates)
    descent_source = exact_rows[0]
    while (descent_source[2].workforce-1 in populations
           and exact_compilations < reconstruction_limit):
        target_workforce = descent_source[2].workforce-1
        target_added = []
        matched = 0
        for candidate in _compress_workforce_frontier(
                problem, descent_source[2], max(4, config.refinement_candidates*2)):
            if exact_compilations >= reconstruction_limit:
                break
            key = skeleton_key(candidate)
            if key in exact_evaluated:
                continue
            cross_basin_attempts += 1
            row = compile_exact(candidate)
            if row is None:
                continue
            target_added.append(row)
            same_quality = (
                row[3].completed == descent_source[3].completed
                and row[3].diagnostics.get("unlocked_land_missing")
                    == descent_source[3].diagnostics.get("unlocked_land_missing"))
            matched += int(same_quality)
            if matched >= 2:
                break
        if target_added:
            cross_basin.extend(target_added)
            exact_rows.extend(target_added)
            exact_rows.sort(key=lambda row: (row[0], repr(row[1])), reverse=True)
        feasible_target = [row for row in exact_rows
                           if row[2].workforce == target_workforce
                           and row[3].completed == descent_source[3].completed
                           and row[3].diagnostics.get("unlocked_land_missing")
                               == descent_source[3].diagnostics.get("unlocked_land_missing")]
        if not feasible_target:
            break
        descent_source = max(feasible_target,
                             key=lambda row: (row[0], repr(row[1])))

    # Compression remains a bounded cleanup around the best reached state. It
    # cannot alter achieved goals/land and shares the exact-compilation cap; it
    # is no longer the mechanism by which low-workforce basins are discovered.
    compression = []
    compression_attempted = 0
    compression_rounds_completed = 0
    direct_incumbent = exact_rows[0]
    incumbent = exact_rows[0]
    for _ in range(config.compression_rounds):
        if exact_compilations >= search_phase_limit:
            break
        _, _, incumbent_skeleton, incumbent_result = incumbent
        added = []
        for candidate in _compress_workforce_frontier(
                problem, incumbent_skeleton, config.compression_candidates):
            if exact_compilations >= search_phase_limit:
                break
            key = skeleton_key(candidate)
            if key in exact_evaluated:
                continue
            seen.add(key)
            compression_attempted += 1
            row = compile_exact(candidate)
            if row is None:
                continue
            compressed = row[3]
            if (compressed.completed != incumbent_result.completed
                    or compressed.diagnostics.get("unlocked_land_missing")
                       != incumbent_result.diagnostics.get("unlocked_land_missing")):
                continue
            if row[0] > incumbent[0]:
                added.append(row)
        if not added:
            break
        incumbent = max(added, key=lambda row: (row[0], repr(row[1])))
        compression.extend(added)
        exact_rows.extend(added)
        exact_rows.sort(key=lambda row: (row[0], repr(row[1])), reverse=True)
        compression_rounds_completed += 1

    # Workforce reconstruction can expose a new causal tail even when it
    # preserves the incumbent's previously achieved set. Give that resulting
    # basin one ordinary compiler-feedback neighborhood before final selection;
    # this uses the existing exact cap and the general repair mechanism.
    final_feedback = []
    for _ in range(config.repair_rounds):
        incumbent = exact_rows[0]
        remaining_exact = config.max_exact_evaluations-exact_compilations
        if not incumbent[3].unfulfilled or remaining_exact <= 0:
            break
        added = []
        for candidate in _exact_repair_frontier(
                problem, incumbent[2], incumbent[3],
                min(config.refinement_candidates, remaining_exact)):
            row = compile_exact(candidate)
            if row is not None:
                added.append(row)
        if not added:
            break
        final_feedback.extend(added)
        exact_rows.extend(added)
        exact_rows.sort(key=lambda row: (row[0], repr(row[1])), reverse=True)

    score, _, skeleton, result = exact_rows[0]
    best_exact_by_workforce = {}
    for row in exact_rows:
        best_exact_by_workforce.setdefault(row[2].workforce, row)
    basin_diagnostics = tuple({
        "workforce": workforce,
        "structural_iterations": basin_iterations[workforce],
        "generated": basin_generated[workforce],
        "retained": len(populations[workforce]),
        "best_abstract_score": populations[workforce][0][0],
        "exact_compilations": basin_exact[workforce],
        "best_completed": (len(best_exact_by_workforce[workforce][3].completed)
                           if workforce in best_exact_by_workforce else None),
        "best_economic_state_value": (best_exact_by_workforce[workforce][0][1]
                                      if workforce in best_exact_by_workforce else None),
    } for workforce in workforces)
    diagnostics = {**result.diagnostics,
        "search_iterations": config.iterations, "search_generated": generated,
        "search_global_iterations": global_iterations,
        "search_workforce_conditioned_iterations": local_iterations,
        "search_accepted": accepted, "search_invalid": invalid,
        "search_exact_evaluations": exact_compilations,
        "search_max_exact_evaluations": config.max_exact_evaluations,
        "search_final_feedback_reserve": final_feedback_reserve,
        "search_refinement_scores": tuple(row[0] for row in refinement),
        "search_final_feedback_scores": tuple(row[0] for row in final_feedback),
        "cross_basin_reconstruction_evaluations": len(cross_basin),
        "cross_basin_reconstruction_attempts": cross_basin_attempts,
        "cross_basin_reconstruction_workforces": tuple(sorted(
            {row[2].workforce for row in cross_basin})),
        "search_workforce_basins": basin_diagnostics,
        "staffing_compression_evaluations": len(compression),
        "staffing_compression_attempts": compression_attempted,
        "staffing_compression_rounds": compression_rounds_completed,
        "direct_search_workforce": direct_incumbent[2].workforce,
        "direct_search_completed": len(direct_incumbent[3].completed),
        "direct_search_economic_state_value": direct_incumbent[0][1],
        "search_seconds": perf_counter()-started,
        "search_improved_start": score > initial_exact_score,
        "search_score": score,
        "search_economic_state_value": score[1],
        "search_abstract_score": _abstract_schedule(problem, skeleton),
        "skeleton": {"routes": skeleton.routes, "path_choices": dict(skeleton.path_choices),
                     "placements": dict(skeleton.placements), "resources": tuple(link.__dict__ for link in skeleton.resources),
                     "leases": tuple(lease.__dict__ for lease in skeleton.leases),
                     "synchronizations": tuple(bundle.goals for bundle in skeleton.synchronizations),
                     "spawn_preferences": skeleton.spawn_preferences,
                     "market_priority": skeleton.market_priority,
                     "logistics": {key: value.__dict__ for key, value in skeleton.logistics.items()},
                     "acquisitions": dict(skeleton.acquisitions),
                     "required_entry_caps": skeleton.required_entry_caps,
                     "hire_caps": skeleton.hire_caps}}
    return replace(result, diagnostics=diagnostics)
