"""Fixed-workforce event-based LNS with exact conditional evaluation.

There is one solving path: the outer controller opens independent workforce
basins; each basin searches only assignment, route order and open placement;
each proposal is realized by the conditional support network; and the exact
compiler supplies fulfillment and the reached state.  This module contains no
abstract population, exact shortlist, or post-search repair cascade.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from random import Random
from time import perf_counter

from . import rules
from .intraday import EndValue
from .route_structure import RouteProblem, RouteStructure, build_route_problem
from .route_support import (
    SupportConfig, SupportResult, rebuild_leases, solve_support, structure_key,
)


@dataclass(frozen=True)
class RouteSearchConfig:
    """Budgets count complete structural evaluations, not cheap mutations."""

    structural_evaluations: int = 96
    basin_probe_evaluations: int = 16
    basin_refinement_evaluations: int = 32
    max_workforce_challengers: int = 3
    destroy_fraction: float = 0.30
    support_time_limit_seconds: float = 0.20
    random_seed: int = 0
    allow_partial_fallback: bool = True


def resulting_state_value(opening_state, resulting_state):
    """Economic value already includes hire, input and land expenditure."""
    return EndValue(opening_state)(resulting_state)[0]


def result_score(result, opening_state):
    """The sole result order: Plan fulfillment, then reached-state value."""
    economic = resulting_state_value(opening_state, result.final_state)
    return len(result.completed), economic


def _support_score(support: SupportResult | None, opening_state):
    if support is None or support.compilation is None:
        return -1, -10**30
    return result_score(support.compilation, opening_state)


def _worker_starts(state, workforce):
    access = rules.shed_access(state.board_size)
    starts = [worker.position for worker in state.workers]
    starts.extend(access[(index-len(starts)) % len(access)]
                  for index in range(len(starts), workforce))
    return tuple(starts)


def _goal_positions(problem: RouteProblem, placements):
    return {goal: placements[entity]
            for goal, entity in problem.goal_entity.items()
            if entity in placements}


def _route_cost(start, route, positions):
    cost, here = 0, start
    for goal in route:
        if goal not in positions:
            continue
        there = positions[goal]
        cost += rules.manhattan(here, there) + 1
        here = there
    return cost


def _placement_seed(problem: RouteProblem):
    """Choose a legal initial point; placement remains open to LNS."""
    placements = {}
    occupied = set()
    for entity in problem.intent.entities:
        if entity.existing and entity.positions:
            placements[entity.identifier] = entity.positions[0]
            occupied.add(entity.positions[0])
    open_entities = [entity for entity in problem.intent.entities
                     if not entity.existing and entity.goals]
    open_entities.sort(key=lambda entity: (
        -entity.future_service_days, -len(entity.goals), entity.identifier))
    for entity in open_entities:
        candidates = [position for position in entity.positions
                      if position not in occupied]
        if not candidates:
            # Temporal reuse is legal and is checked by derived leases.
            candidates = list(entity.positions)
        if not candidates:
            continue
        position = min(candidates, key=lambda point: (
            rules.distance_to_shed(point), point[1], point[0]))
        placements[entity.identifier] = position
        occupied.add(position)
    return placements


def _entity_blocks(problem: RouteProblem):
    """Use one legal local mode only to seed precedence-compatible routes."""
    blocks = []
    for entity in problem.intent.entities:
        paths = problem.complete_paths[entity.identifier]
        if paths:
            blocks.append(tuple(event.goal for event in paths[0]))
        elif entity.goals:
            blocks.append(tuple(goal.identifier for goal in entity.goals))
    return tuple(blocks)


def _best_block_insertion(routes, block, starts, positions, horizon):
    best = None
    for worker, route in enumerate(routes):
        old_cost = _route_cost(starts[worker], route, positions)
        for at in range(len(route)+1):
            trial = route[:at] + list(block) + route[at:]
            # Service count is only a necessary bound. Full capacity comes from
            # the compiled support realization, with no fixed support margin.
            overflow = max(0, len(trial)-horizon[worker])
            delta = _route_cost(starts[worker], trial, positions)-old_cost
            key = (int(overflow > 0), overflow, delta, worker, at)
            if best is None or key < best[0]:
                best = key, worker, at
    return best


def _initial_structure(problem: RouteProblem, workforce: int):
    placements = _placement_seed(problem)
    positions = _goal_positions(problem, placements)
    state = problem.intent.state
    starts = _worker_starts(state, workforce)
    turns = min(state.turns_left_today, state.turns_left)
    horizon = [max(0, turns-(0 if worker < len(state.workers) else 1))
               for worker in range(workforce)]
    routes = [[] for _ in range(workforce)]
    blocks = sorted(_entity_blocks(problem), key=lambda block: (
        min((problem.goals[goal].deadline for goal in block), default=10**9),
        -len(block), block,
    ))
    for block in blocks:
        insertion = _best_block_insertion(
            routes, block, starts, positions, horizon)
        if insertion is None:
            continue
        _, worker, at = insertion
        routes[worker][at:at] = block
    return RouteStructure(workforce, tuple(tuple(route) for route in routes),
                          placements)


def _destroy_count(structure: RouteStructure, fraction):
    total = sum(map(len, structure.routes))
    if not total:
        return 0
    return min(total, max(2 if total > 1 else 1, round(total*fraction)))


def _related_destroy(problem: RouteProblem, structure: RouteStructure,
                     rng: Random, count: int, forced=()):
    all_goals = [goal for route in structure.routes for goal in route]
    if not all_goals:
        return set()
    positions = _goal_positions(problem, structure.placements)
    seed = next((goal for goal in forced if goal in positions), rng.choice(all_goals))
    seed_entity = problem.goal_entity[seed]
    seed_position = positions.get(seed, (0, 0))
    owner = {goal: worker for worker, route in enumerate(structure.routes)
             for goal in route}
    ranked = sorted(all_goals, key=lambda goal: (
        0 if goal in forced else 1,
        0 if problem.goal_entity[goal] == seed_entity else 1,
        rules.manhattan(seed_position, positions.get(goal, seed_position)),
        0 if owner[goal] != owner[seed] else 1,
        rng.random(),
    ))
    return set(ranked[:count]) | set(forced)


def _segment_destroy(structure: RouteStructure, rng: Random, count: int,
                     forced=()):
    removed = set(forced)
    candidates = [worker for worker, route in enumerate(structure.routes) if route]
    while candidates and len(removed) < count:
        worker = rng.choice(candidates)
        route = structure.routes[worker]
        length = min(len(route), max(1, (count-len(removed)+1)//2))
        start = rng.randrange(len(route)-length+1)
        removed.update(route[start:start+length])
        candidates.remove(worker)
    return removed


def _block_exchange(structure: RouteStructure, rng: Random):
    nonempty = [worker for worker, route in enumerate(structure.routes) if route]
    if len(nonempty) < 2:
        return None
    left, right = rng.sample(nonempty, 2)
    routes = [list(route) for route in structure.routes]
    a, b = routes[left], routes[right]
    a0 = rng.randrange(len(a)); a1 = rng.randrange(a0+1, len(a)+1)
    b0 = rng.randrange(len(b)); b1 = rng.randrange(b0+1, len(b)+1)
    a_block, b_block = a[a0:a1], b[b0:b1]
    a[a0:a1], b[b0:b1] = b_block, a_block
    return RouteStructure(structure.workforce,
                          tuple(tuple(route) for route in routes),
                          dict(structure.placements))


def _routes_admit_complete_modes(problem: RouteProblem, routes):
    owner = {goal: worker for worker, route in enumerate(routes) for goal in route}
    order = {goal: index for route in routes
             for index, goal in enumerate(route)}
    for entity in problem.intent.entities:
        routed = {goal.identifier for goal in entity.goals if goal.identifier in owner}
        if not routed:
            continue
        admitted = False
        for path in problem.complete_paths[entity.identifier]:
            sequence = [event.goal for event in path if event.goal in routed]
            if all(owner[before] != owner[after]
                   or order[before] < order[after]
                   for left, before in enumerate(sequence)
                   for after in sequence[left+1:]):
                admitted = True
                break
        if not admitted:
            return False
    return True


def _insertion_delta(route, at, goal, start, positions):
    before = start if at == 0 else positions[route[at-1]]
    after = None if at == len(route) else positions[route[at]]
    point = positions[goal]
    added = rules.manhattan(before, point) + 1
    if after is not None:
        added += rules.manhattan(point, after)-rules.manhattan(before, after)
    return added


def _regret_recreate(problem: RouteProblem, structure: RouteStructure,
                     removed, rng: Random):
    """Standard regret-2 reinsertion across all workers and positions."""
    routes = [[goal for goal in route if goal not in removed]
              for route in structure.routes]
    placements = dict(structure.placements)
    starts = _worker_starts(problem.intent.state, structure.workforce)
    entities = {entity.identifier: entity for entity in problem.intent.entities}
    touched = {problem.goal_entity[goal] for goal in removed}
    for identifier in sorted(touched):
        entity = entities[identifier]
        if entity.existing or not entity.positions:
            continue

        def placement_cost(position):
            nearest = min((rules.manhattan(
                position, placements[problem.goal_entity[goal]])
                for route in routes for goal in route
                if problem.goal_entity[goal] != identifier
                and problem.goal_entity[goal] in placements), default=0)
            return nearest + rules.distance_to_shed(position), rng.random()

        placements[identifier] = min(entity.positions, key=placement_cost)

    positions = _goal_positions(problem, placements)
    pending = set(removed)
    while pending:
        selected = None
        for goal in sorted(pending):
            if goal not in positions:
                return None
            options = []
            for worker, route in enumerate(routes):
                for at in range(len(route)+1):
                    trial_routes = [list(candidate) for candidate in routes]
                    trial_routes[worker].insert(at, goal)
                    if not _routes_admit_complete_modes(problem, trial_routes):
                        continue
                    options.append((_insertion_delta(
                        route, at, goal, starts[worker], positions),
                        rng.random(), worker, at))
            if not options:
                return None
            options.sort()
            best = options[0]
            second = options[1] if len(options) > 1 else best
            key = (second[0]-best[0], -best[0], rng.random())
            if selected is None or key > selected[0]:
                selected = key, goal, best
        _, goal, (_, _, worker, at) = selected
        routes[worker].insert(at, goal)
        pending.remove(goal)
    return RouteStructure(structure.workforce,
                          tuple(tuple(route) for route in routes), placements)


def _neighbor(problem: RouteProblem, current: RouteStructure, rng: Random,
              fraction: float, conflict=None, iteration=0):
    forced = tuple(conflict.goals if conflict else ())
    if iteration % 5 == 4:
        exchanged = _block_exchange(current, rng)
        if (exchanged is not None
                and _routes_admit_complete_modes(problem, exchanged.routes)):
            return exchanged, "block-exchange"
    count = _destroy_count(current, fraction)
    if iteration % 2:
        removed = _segment_destroy(current, rng, count, forced)
        operator = "multi-route-segment-ruin/regret-2"
    else:
        removed = _related_destroy(problem, current, rng, count, forced)
        operator = "related-ruin/regret-2"
    return _regret_recreate(problem, current, removed, rng), operator


@dataclass
class _Basin:
    workforce: int
    current_structure: RouteStructure
    current_support: SupportResult
    best_support: SupportResult
    evaluations: int = 1
    feasible_evaluations: int = 0
    improvements: int = 0
    conflicts: Counter | None = None
    operators: Counter | None = None

    def __post_init__(self):
        self.conflicts = self.conflicts or Counter()
        self.operators = self.operators or Counter()
        self.feasible_evaluations = int(self.current_support.feasible)


def _run_basin(problem: RouteProblem, basin: _Basin, budget: int,
               config: RouteSearchConfig, rng: Random):
    support_config = SupportConfig(config.support_time_limit_seconds)
    opening = problem.intent.state
    attempted = 0
    while attempted < max(0, budget):
        attempted += 1
        conflict = basin.current_support.conflict
        candidate, operator = _neighbor(
            problem, basin.current_structure, rng, config.destroy_fraction,
            conflict, basin.evaluations)
        if candidate is None or structure_key(candidate) == structure_key(
                basin.current_structure):
            continue
        support = solve_support(problem, candidate, support_config)
        basin.evaluations += 1
        basin.operators[operator] += 1
        basin.feasible_evaluations += int(support.feasible)
        if support.conflict:
            basin.conflicts[support.conflict.kind] += 1

        # Both search state and incumbent use exact compiler outcomes only.
        # Partial outcomes can guide search only because the upper problem
        # explicitly permits a fallback; support never selected their goals.
        if _support_score(support, opening) >= _support_score(
                basin.current_support, opening):
            basin.current_structure = candidate
            basin.current_support = support
        if _support_score(support, opening) > _support_score(
                basin.best_support, opening):
            basin.best_support = support
            basin.improvements += 1
    return basin


def _workforce_bound(problem: RouteProblem):
    """Physical outer bound; economics is decided only by exact realization."""
    state = problem.intent.state
    event_count = len(problem.goals)
    horizon = min(state.turns_left_today, state.turns_left)
    market_bound = (len(state.workers)
                    + rules.MAX_MARKET_ORDERS*max(0, horizon-1))
    return min(market_bound,
               max(len(state.workers), event_count))


def _first_workforce(problem: RouteProblem, maximum: int):
    state = problem.intent.state
    horizon = max(1, min(state.turns_left_today, state.turns_left))
    necessary = (len(problem.goals)+horizon-1)//horizon
    return min(maximum, max(len(state.workers), necessary))


def _open_basin(problem: RouteProblem, workforce: int,
                config: RouteSearchConfig):
    structure = _initial_structure(problem, workforce)
    support = solve_support(
        problem, structure, SupportConfig(config.support_time_limit_seconds))
    return _Basin(workforce, structure, support, support)


def _attach_search_diagnostics(result, opening_state, started, basins,
                               evaluations, selected_support):
    diagnostics = dict(result.diagnostics)
    diagnostics.update({
        "solver": "fixed-workforce-event-lns+conditional-resource-flow",
        "selection_objective": "(Plan fulfillment, V(S_end))",
        "structural_evaluations": evaluations,
        "selected_workforce": selected_support.structure.workforce,
        "conditional_support": dict(selected_support.diagnostics or {}),
        "workforce_basins": {
            basin.workforce: {
                "evaluations": basin.evaluations,
                "feasible_evaluations": basin.feasible_evaluations,
                "improvements": basin.improvements,
                "best_score": _support_score(basin.best_support, opening_state),
                "conflicts": dict(basin.conflicts),
                "operators": dict(basin.operators),
            }
            for basin in basins
        },
        "elapsed_seconds": perf_counter()-started,
    })
    return replace(result, diagnostics=diagnostics)


def solve_routes(state, plan, config=None, progress=None):
    """Solve a fixed Plan without a surrogate population or repair cascade."""
    config = config or RouteSearchConfig()
    started = perf_counter()
    problem = build_route_problem(state, plan, progress)
    maximum = _workforce_bound(problem)
    workforce = _first_workforce(problem, maximum)
    rng = Random(config.random_seed + state.step*1_000_003 + state.player*97)
    basins = []
    evaluations = 0
    best = None
    nonimproving_complete = 0
    found_complete = False
    economic_challengers = 0

    # Workforce sizes are opened sequentially from the necessary lower bound.
    # Each is an independent basin. Exact feasibility/economics alone decides
    # refinement and whether a larger challenger is worth opening.
    while (workforce <= maximum
           and evaluations < max(1, config.structural_evaluations)
           and (not found_complete
                or economic_challengers < config.max_workforce_challengers)):
        if found_complete:
            economic_challengers += 1
        basin = _open_basin(problem, workforce, config)
        evaluations += 1
        probe = min(config.basin_probe_evaluations,
                    max(0, config.structural_evaluations-evaluations))
        before = basin.evaluations
        _run_basin(problem, basin, probe, config,
                   Random(rng.randrange(2**31)))
        evaluations += basin.evaluations-before
        basins.append(basin)

        prior_score = _support_score(best, state)
        basin_score = _support_score(basin.best_support, state)
        if basin_score > prior_score:
            best = basin.best_support
            if basin.best_support.feasible:
                nonimproving_complete = 0
                found_complete = True
        elif basin.best_support.feasible:
            nonimproving_complete += 1
            found_complete = True

        if basin.best_support.feasible and evaluations < config.structural_evaluations:
            refine = min(config.basin_refinement_evaluations,
                         config.structural_evaluations-evaluations)
            before = basin.evaluations
            _run_basin(problem, basin, refine, config,
                       Random(rng.randrange(2**31)))
            evaluations += basin.evaluations-before
            if _support_score(basin.best_support, state) > _support_score(best, state):
                best = basin.best_support
                nonimproving_complete = 0

        # A complete larger basin that loses economically is direct evidence
        # to stop; an infeasible one opens the next actual capacity level.
        if basin.best_support.feasible and nonimproving_complete >= 1:
            break
        if evaluations >= config.structural_evaluations:
            break
        workforce += 1

    if best is None or best.compilation is None:
        kinds = Counter(
            basin.best_support.conflict.kind
            for basin in basins if basin.best_support.conflict)
        raise RuntimeError(
            "no structural candidate reached deterministic compilation; "
            f"support conflicts={dict(kinds)}")
    if not best.feasible and not config.allow_partial_fallback:
        raise RuntimeError(
            "no workforce basin could realize every mandatory Plan goal")
    return _attach_search_diagnostics(
        best.compilation, state, started, basins, evaluations,
        best)


__all__ = [
    "RouteSearchConfig", "rebuild_leases", "result_score",
    "resulting_state_value", "solve_routes",
]
