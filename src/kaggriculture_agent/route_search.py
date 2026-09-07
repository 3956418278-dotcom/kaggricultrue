"""Deterministic large-neighborhood search over event-route realizations."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from random import Random
from time import perf_counter

from . import rules
from .route_compiler import compile_skeleton
from .route_structure import (
    ResourceLink, RouteSkeleton, SyncBundle, TileLease, build_route_problem,
    event_positions, event_predecessors, initial_resources, initial_skeleton,
    realizable_opening_cash, selected_events, selected_paths, with_initial_logistics,
    with_financing_logistics,
)


@dataclass(frozen=True)
class RouteSearchConfig:
    # ``iterations`` counts cheap structural proposals, not exact simulations.
    iterations: int = 1600
    population: int = 48
    exact_candidates: int = 18
    refinement_candidates: int = 6
    repair_rounds: int = 3
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
    """Recompute consequences of changed routes without changing their choices."""
    routes = tuple(tuple(token for token in route if token not in skeleton.logistics)
                   for route in skeleton.routes)
    shell = replace(skeleton, routes=routes, logistics={})
    leases = rebuild_leases(problem, shell)
    shell = replace(shell, leases=leases, resources=())
    resources = initial_resources(problem, shell)
    shell = replace(shell, resources=resources)
    shell = with_initial_logistics(problem, shell)
    from .route_compiler import purchase_orders_for_links
    shell = replace(shell, acquisitions=purchase_orders_for_links(resources))
    shell = replace(shell, market_priority=_market_priority(problem, shell))
    return with_financing_logistics(problem, shell)


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


def result_score(result):
    d = result.diagnostics
    return (len(result.completed), _liquid_value(result.final_state),
            -d["used_worker_turns"], -d["movement"], -d["logistics"], -d["workforce"])


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
    return (-int(cycle), completed, -lateness, -max(0, makespan-(horizon-1)),
            -total_movement, -logistics_count, -skeleton.workforce)


def _relocate_event(skeleton, rng):
    nonempty = [w for w, route in enumerate(skeleton.routes)
                if any(token not in skeleton.logistics for token in route)]
    if not nonempty: return skeleton
    source = rng.choice(nonempty); route = list(skeleton.routes[source])
    choices = [n for n, token in enumerate(route) if token not in skeleton.logistics]
    goal = route.pop(rng.choice(choices))
    target = rng.randrange(skeleton.workforce); other = list(skeleton.routes[target])
    other.insert(rng.randrange(len(other)+1), goal)
    routes = [list(r) for r in skeleton.routes]
    routes[source], routes[target] = route, other
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


def _cross_exchange(skeleton, rng):
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
    path_choices = dict(skeleton.path_choices); path_choices[entity] = selected
    return replace(skeleton, path_choices=path_choices)


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
        return replace(skeleton, workforce=skeleton.workforce-1,
                       routes=tuple(tuple(r) for r in routes),
                       spawn_preferences=skeleton.spawn_preferences[:-1])
    event_count = sum(map(len, skeleton.routes))
    if skeleton.workforce >= max(len(state.workers), event_count): return skeleton
    hires = skeleton.workforce-len(state.workers)
    spent = sum(rules.fibonacci_hire_cost(state.hires_today+n) for n in range(max(0, hires+1)))
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


def _change_market_priority(skeleton, rng):
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
    hire_cost = sum(rules.fibonacci_hire_cost(state.hires_today+n) for n in range(hire_count))
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
    from .route_compiler import purchase_orders_for_links
    shell = replace(shell, acquisitions=purchase_orders_for_links(links))
    shell = replace(shell, market_priority=_market_priority(problem, shell))
    return with_financing_logistics(problem, shell)


def _move_logistics(skeleton, rng):
    workers = [worker for worker, route in enumerate(skeleton.routes)
               if any(token in skeleton.logistics for token in route)]
    if not workers: return skeleton
    worker = rng.choice(workers); route = list(skeleton.routes[worker])
    choices = [n for n, token in enumerate(route) if token in skeleton.logistics]
    token = route.pop(rng.choice(choices)); route.insert(rng.randrange(len(route)+1), token)
    routes = list(skeleton.routes); routes[worker] = tuple(route)
    return replace(skeleton, routes=tuple(routes))


def _relocate_segment(skeleton, rng):
    candidates = [w for w, route in enumerate(skeleton.routes) if route]
    if not candidates: return skeleton
    source = rng.choice(candidates); route = list(skeleton.routes[source])
    begin = rng.randrange(len(route)); end = min(len(route), begin+rng.randint(1, min(6, len(route)-begin)))
    segment = route[begin:end]; del route[begin:end]
    target = rng.randrange(skeleton.workforce)
    routes = [list(r) for r in skeleton.routes]; routes[source] = route
    at = rng.randrange(len(routes[target])+1); routes[target][at:at] = segment
    return replace(skeleton, routes=tuple(tuple(r) for r in routes))


def _reverse_segment(skeleton, rng):
    candidates = [w for w, route in enumerate(skeleton.routes) if len(route) >= 2]
    if not candidates: return skeleton
    worker = rng.choice(candidates); route = list(skeleton.routes[worker])
    a, b = sorted(rng.sample(range(len(route)+1), 2))
    if b-a < 2: return skeleton
    route[a:b] = reversed(route[a:b])
    routes = list(skeleton.routes); routes[worker] = tuple(route)
    return replace(skeleton, routes=tuple(routes))


MUTATIONS = (_relocate_event, _relocate_entity, _cross_exchange, _change_path,
             _change_placement, _resize_workforce, _change_sync,
             _change_market_priority, _change_resource_source, _move_logistics,
             _relocate_segment, _reverse_segment, _change_spawn_preference,
             _change_market_cap)


def _mutate(problem, skeleton, rng):
    mutation = rng.choice(MUTATIONS)
    trial = mutation(skeleton, rng) if mutation in (_relocate_event, _cross_exchange,
        _change_market_priority, _move_logistics, _relocate_segment,
        _reverse_segment) else mutation(problem, skeleton, rng)
    # Explicit resource-source mutations must survive normalization. Other
    # structural changes re-solve finite stock/output allocation conditionally.
    if mutation in (_change_resource_source, _move_logistics, _change_sync):
        return trial
    normalized = normalize(problem, trial)
    return replace(normalized, market_priority=trial.market_priority) if mutation is _change_market_priority else normalized


def _exact_repair_frontier(problem, skeleton, result, limit):
    """Relocate compiler-observed unfinished causal tails to spare workers."""
    events = selected_events(problem, skeleton)
    unfinished = set(result.unfulfilled) & set(events)
    if (not unfinished and not result.diagnostics.get("unbought_inputs")) or not limit:
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
        block = [event.goal for event in path if event.goal in unfinished]
        if not block: continue
        unfinished_blocks.append(tuple(block))
        source = next((worker for worker, route in enumerate(skeleton.routes)
                       if any(goal in route for goal in block)), None)
        if source is None: continue
        work = entities[entity]
        if not work.existing and len(work.positions) > 1:
            for position in work.positions:
                if position == skeleton.placements.get(entity): continue
                placements = dict(skeleton.placements); placements[entity] = position
                try:
                    trial = normalize(problem, replace(
                        skeleton, placements=placements,
                        routes=tuple(tuple(token for token in route
                                           if token not in skeleton.logistics)
                                     for route in skeleton.routes), logistics={}))
                    key = skeleton_key(trial)
                    proposals[key] = (_abstract_schedule(problem, trial), trial)
                except (ValueError, IndexError):
                    continue
        base_routes = [[token for token in route
                        if token not in skeleton.logistics and token not in block]
                       for route in skeleton.routes]
        repair_targets = [source, *(worker for worker in targets if worker != source)][:min(7, len(targets))]
        for target in repair_targets:
            for at in range(len(base_routes[target])+1):
                routes = [list(route) for route in base_routes]
                routes[target][at:at] = block
                try:
                    trial = normalize(problem, replace(
                        skeleton, routes=tuple(tuple(route) for route in routes), logistics={}))
                    key = skeleton_key(trial)
                    proposals[key] = (_abstract_schedule(problem, trial), trial)
                except (ValueError, IndexError):
                    continue
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
                try:
                    trial = normalize(problem, replace(
                        skeleton, routes=tuple(tuple(route) for route in routes), logistics={}))
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
                    candidate = (max(costs[:worker]+[cost]+costs[worker+1:]),
                                 cost, -(available[worker]-used[worker]), worker, at)
                    if best is None or candidate < best[0]: best = candidate, worker, at
            _, worker, at = best
            routes[worker][at:at] = block
        try:
            redistributed = normalize(problem, replace(
                skeleton, routes=tuple(tuple(route) for route in routes), logistics={}))
        except (ValueError, IndexError):
            redistributed = None
    result_frontier = list(market_repairs)
    if redistributed is not None: result_frontier.append(redistributed)
    result_frontier.extend(trial for _, trial in ranked
                           if redistributed is None or skeleton_key(trial) != skeleton_key(redistributed))
    return result_frontier[:limit]


def solve_routes(state, plan, config=None, progress=None):
    config = config or RouteSearchConfig()
    started = perf_counter()
    problem = build_route_problem(state, plan, progress)
    seed = config.random_seed + state.step*1_000_003 + state.player*97
    rng = Random(seed)
    starts = []
    base = initial_skeleton(problem)
    # Staffing is part of this one model. Determine the complete feasible
    # domain, then seed structurally distinct counts without exact-simulating
    # every integer in a potentially large interval.
    event_count = sum(len(path[0]) for path in problem.complete_paths.values() if path)
    cash, maximum = realizable_opening_cash(state), len(state.workers)
    while maximum < max(len(state.workers), event_count):
        cost = rules.fibonacci_hire_cost(state.hires_today+maximum-len(state.workers))
        if cost > cash: break
        cash -= cost; maximum += 1
    horizon = min(state.turns_left_today, state.turns_left)
    maximum = min(maximum, len(state.workers)+rules.MAX_MARKET_ORDERS*max(0, horizon-1),
                  max(len(state.workers), event_count))
    estimated = base.workforce
    workforce_seeds = {len(state.workers), maximum, estimated}
    workforce_seeds.update(range(max(len(state.workers), estimated-3), min(maximum, estimated+3)+1))
    span = maximum-len(state.workers)
    workforce_seeds.update(len(state.workers)+(span*n)//4 for n in range(1, 4))
    for workforce in sorted(workforce_seeds):
        try:
            skeleton = initial_skeleton(problem, workforce)
            starts.append((_abstract_schedule(problem, skeleton), skeleton_key(skeleton), skeleton))
        except ValueError:
            continue
    if not starts:
        starts = [(_abstract_schedule(problem, base), skeleton_key(base), base)]
    population = sorted(starts, key=lambda row: row[0], reverse=True)[:config.population]
    seen = {row[1] for row in population}
    generated = accepted = invalid = 0
    for _ in range(config.iterations):
        parent = population[rng.randrange(min(len(population), max(1, config.population//2)))][2]
        try:
            candidate = _mutate(problem, parent, rng)
            key = skeleton_key(candidate)
            if key in seen: continue
            seen.add(key); generated += 1
            abstract = _abstract_schedule(problem, candidate)
        except (ValueError, IndexError):
            invalid += 1
            continue
        row = (abstract, key, candidate)
        if len(population) < config.population or row[0] > population[-1][0]:
            population.append(row); population.sort(key=lambda item: item[0], reverse=True)
            del population[config.population:]; accepted += 1

    # Exact transition replay is the authority. Select a diverse structural
    # frontier rather than trusting the approximation's single top answer.
    exact_pool = [base]
    exact_keys = {skeleton_key(base)}
    workforce_seen = {base.workforce}
    for row in population:
        if len(exact_pool) >= config.exact_candidates:
            break
        if row[2].workforce not in workforce_seen or len(exact_pool) < max(4, config.exact_candidates//2):
            if row[1] not in exact_keys:
                exact_pool.append(row[2]); exact_keys.add(row[1]); workforce_seen.add(row[2].workforce)
    for row in starts:
        if len(exact_pool) >= config.exact_candidates: break
        if row[1] not in exact_keys:
            exact_pool.append(row[2]); exact_keys.add(row[1])
    exact_rows = []
    for skeleton in exact_pool:
        try:
            result = compile_skeleton(problem, skeleton)
            exact_rows.append((result_score(result), skeleton_key(skeleton), skeleton, result))
        except ValueError:
            invalid += 1
    if not exact_rows:
        result = compile_skeleton(problem, base)
        exact_rows = [(result_score(result), skeleton_key(base), base, result)]
    exact_rows.sort(key=lambda row: row[0], reverse=True)
    initial_exact_score = next(row[0] for row in exact_rows if row[1] == skeleton_key(base))

    # One exact-feedback neighborhood lets approximation errors be repaired
    # without returning to compile-every-mutation search.
    refinement = []
    for _ in range(config.repair_rounds):
        added = []
        for candidate in _exact_repair_frontier(
                problem, exact_rows[0][2], exact_rows[0][3], config.refinement_candidates):
            try:
                key = skeleton_key(candidate)
                if key in seen: continue
                seen.add(key)
                repaired = compile_skeleton(problem, candidate)
                added.append((result_score(repaired), key, candidate, repaired))
            except (ValueError, IndexError):
                invalid += 1
        if not added: break
        refinement.extend(added); exact_rows.extend(added)
        exact_rows.sort(key=lambda row: row[0], reverse=True)
    score, _, skeleton, result = exact_rows[0]
    diagnostics = {**result.diagnostics,
        "search_iterations": config.iterations, "search_generated": generated,
        "search_accepted": accepted, "search_invalid": invalid,
        "search_exact_evaluations": len(exact_pool)+len(refinement),
        "search_refinement_scores": tuple(row[0] for row in refinement),
        "search_seconds": perf_counter()-started,
        "search_improved_start": score > initial_exact_score,
        "search_score": score,
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
