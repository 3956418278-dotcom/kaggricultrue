"""Conditional material-flow realization for one fixed route structure.

The structural search owns only workforce, event assignment/order and open
placement.  This module keeps every remaining Plan goal mandatory and solves
the induced support decisions.  Its material constraints are a compact
source-to-demand network; integer variables are introduced only for selected
event modes and fixed-charge/cross-route coupling.  There is deliberately no
worker-by-turn-by-item lattice here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Mapping

from . import rules
from .intraday import EndValue
from .route_compiler import compile_skeleton, purchase_orders_for_links
from .route_structure import (
    LogisticsEvent, ResourceLink, RouteProblem, RouteSkeleton, RouteStructure,
    TileLease, selected_paths, with_initial_logistics,
)


@dataclass(frozen=True)
class SupportConfig:
    """Limits for the cheap conditional solve used at every LNS proposal."""

    time_limit_seconds: float = 0.20


@dataclass(frozen=True)
class SupportConflict:
    """Machine-readable reason a complete structural candidate failed."""

    kind: str
    detail: str
    goals: tuple[str, ...] = ()
    workers: tuple[int, ...] = ()
    items: tuple[str, ...] = ()
    compiler: Mapping[str, object] | None = None


@dataclass(frozen=True)
class SupportResult:
    feasible: bool
    structure: RouteStructure
    skeleton: RouteSkeleton | None = None
    compilation: object | None = None
    conflict: SupportConflict | None = None
    diagnostics: Mapping[str, object] | None = None


@dataclass(frozen=True)
class _Arc:
    kind: str
    consumer: str
    item: str
    capacity: int
    source_worker: int | None = None
    producer: str | None = None
    via_shed: bool = False


def structure_key(structure: RouteStructure):
    return (structure.workforce, structure.routes,
            tuple(sorted(structure.placements.items())))


def _require_ortools():
    try:
        from ortools.linear_solver import pywraplp
    except ImportError as exc:  # pragma: no cover - planning extra is optional
        raise RuntimeError(
            "route solving requires the pinned planning dependencies; "
            "install requirements-planning.txt"
        ) from exc
    return pywraplp


def _route_owner(structure: RouteStructure):
    return {goal: worker for worker, route in enumerate(structure.routes)
            for goal in route}


def _route_order(structure: RouteStructure):
    return {goal: index for route in structure.routes
            for index, goal in enumerate(route)}


def _compatible_modes(problem: RouteProblem, structure: RouteStructure):
    """Return complete local modes compatible with fixed within-route order."""
    owner, order = _route_owner(structure), _route_order(structure)
    choices = {}
    for entity in problem.intent.entities:
        options = problem.complete_paths[entity.identifier]
        if not options:
            return None, SupportConflict(
                "service-mode", "no complete legal service path",
                tuple(goal.identifier for goal in entity.goals))
        compatible = []
        for index, path in enumerate(options):
            goals = [event.goal for event in path]
            incompatible = any(
                owner.get(before) == owner.get(after)
                and order.get(before, -1) > order.get(after, -1)
                for left, before in enumerate(goals)
                for after in goals[left + 1:]
            )
            if not incompatible:
                compatible.append(index)
        if not compatible:
            return None, SupportConflict(
                "service-mode", "fixed route order admits no complete local mode",
                tuple(goal.identifier for goal in entity.goals))
        choices[entity.identifier] = tuple(compatible)
    return choices, None


def _released_by(path):
    return next((event.goal for event in path
                 if event.after is None or event.after == "LOCKED"), None)


def rebuild_leases(problem: RouteProblem, skeleton: RouteSkeleton):
    """Derive temporal tile leases from the selected modes and route order."""
    paths = selected_paths(problem, skeleton)
    entities = {entity.identifier: entity for entity in problem.intent.entities}
    route_rank = {goal: (worker, index)
                  for worker, route in enumerate(skeleton.routes)
                  for index, goal in enumerate(route)}
    grouped = defaultdict(list)
    for entity, position in skeleton.placements.items():
        grouped[position].append(entity)
    leases = []
    for position, occupants in grouped.items():
        occupants.sort(key=lambda entity: (
            0 if entities[entity].existing else 1,
            min((route_rank.get(event.goal, (10**9, 10**9))
                 for event in paths[entity]), default=(10**9, 10**9)),
            entity,
        ))
        available = None
        for entity in occupants:
            release = _released_by(paths[entity])
            leases.append(TileLease(entity, position, available, release))
            available = release if release is not None else f"blocked:{entity}"
    return tuple(sorted(leases, key=lambda lease: lease.entity))


def _worker_starts(state, workforce):
    access = rules.shed_access(state.board_size)
    starts = [worker.position for worker in state.workers]
    starts.extend(access[(index-len(starts)) % len(access)]
                  for index in range(len(starts), workforce))
    return tuple(starts)


def _source_unit_cost(state, kind, item, ordinal=0):
    """Economic opportunity/purchase cost used by the conditional network."""
    if kind == "PURCHASE":
        if item.endswith("_SEED"):
            crop = item[:-5]
            return rules.CROPS[crop].seed_cost if crop in rules.CROPS else 10**6
        if item in rules.ANIMALS:
            return rules.ANIMALS[item].cost
        inventory = state.market_inventory.get(item, rules.MARKET_I0)
        return rules.market_price(item, inventory-1-ordinal)
    if item.endswith("_SEED"):
        crop = item[:-5]
        return rules.CROPS[crop].seed_cost if crop in rules.CROPS else 0
    if item in rules.ANIMALS:
        return rules.ANIMALS[item].cost
    if item in rules.SELLABLE_PRODUCTS:
        return rules.market_price(item, state.market_inventory.get(item,
                                                                   rules.MARKET_I0))
    return 0


def _mode_material(problem: RouteProblem, compatible):
    """Material coefficients indexed by entity, mode, goal and item."""
    needs, outputs = Counter(), Counter()
    max_need, max_output = Counter(), Counter()
    for entity, indices in compatible.items():
        for index in indices:
            for event in problem.complete_paths[entity][index]:
                for item, quantity in event.delta.items():
                    if quantity < 0:
                        needs[entity, index, event.goal, item] = -quantity
                        max_need[event.goal, item] = max(
                            max_need[event.goal, item], -quantity)
                    elif quantity > 0:
                        outputs[entity, index, event.goal, item] = quantity
                        max_output[event.goal, item] = max(
                            max_output[event.goal, item], quantity)
    return needs, outputs, max_need, max_output


def _make_arcs(problem: RouteProblem, structure: RouteStructure,
               max_need, max_output):
    state = problem.intent.state
    owner, order = _route_owner(structure), _route_order(structure)
    carry = Counter()
    for worker in state.workers:
        for item, quantity in worker.inventory.items():
            carry[worker.index, item] += quantity
    shed = Counter(state.shed)
    shed.update({f"{crop}_SEED": quantity for crop, quantity in state.seeds.items()})
    arcs = []
    for (consumer, item), quantity in sorted(max_need.items()):
        worker = owner[consumer]
        if carry[worker, item]:
            arcs.append(_Arc("CARRY", consumer, item, carry[worker, item],
                             source_worker=worker))
        if shed[item]:
            arcs.append(_Arc("SHED", consumer, item, shed[item]))
        arcs.append(_Arc("PURCHASE", consumer, item, quantity))
        for (producer, produced_item), capacity in sorted(max_output.items()):
            if produced_item != item or producer == consumer:
                continue
            producer_worker = owner[producer]
            if producer_worker == worker and order[producer] >= order[consumer]:
                continue
            arcs.append(_Arc("EVENT", consumer, item, capacity,
                             producer=producer,
                             source_worker=producer_worker,
                             via_shed=producer_worker != worker))
    return carry, shed, tuple(arcs)


def _solve_material_network(problem: RouteProblem, structure: RouteStructure,
                            compatible,
                            time_limit: float):
    """Solve the compact fixed-charge material network.

    Without the activation rows this is a totally unimodular source-demand
    flow model.  Binaries occur only for shared pickup batches and cross-worker
    event transfers and complete service modes; topological ranks enforce the
    timing coupling selected by those modes and transfers.  There is no binary
    variable capable of dropping a Plan goal.
    """
    pywraplp = _require_ortools()
    needs, outputs, max_need, max_output = _mode_material(problem, compatible)
    carry, shed, arcs = _make_arcs(problem, structure, max_need, max_output)
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:  # pragma: no cover - pinned wheel supplies SCIP
        raise RuntimeError("OR-Tools SCIP backend is unavailable")
    solver.SetTimeLimit(max(10, int(time_limit * 1000)))

    modes = {}
    for entity, indices in compatible.items():
        for index in indices:
            modes[entity, index] = solver.BoolVar(f"mode:{entity}:{index}")
        solver.Add(sum(modes[entity, index] for index in indices) == 1)

    flow = [solver.IntVar(0, arc.capacity, f"f:{index}")
            for index, arc in enumerate(arcs)]
    by_demand, by_source = defaultdict(list), defaultdict(list)
    for index, arc in enumerate(arcs):
        by_demand[arc.consumer, arc.item].append(index)
        if arc.kind == "CARRY":
            by_source["CARRY", arc.source_worker, arc.item].append(index)
        elif arc.kind == "SHED":
            by_source["SHED", arc.item].append(index)
        elif arc.kind == "EVENT":
            by_source["EVENT", arc.producer, arc.item].append(index)
    for (goal, item), _ in max_need.items():
        entity = problem.goal_entity[goal]
        required = sum(
            needs[entity, index, goal, item] * modes[entity, index]
            for index in compatible[entity])
        solver.Add(sum(flow[index] for index in by_demand[goal, item])
                   == required)
    for (worker, item), quantity in carry.items():
        indices = by_source.get(("CARRY", worker, item), ())
        if indices:
            solver.Add(sum(flow[index] for index in indices) <= quantity)
    for item, quantity in shed.items():
        indices = by_source.get(("SHED", item), ())
        if indices:
            solver.Add(sum(flow[index] for index in indices) <= quantity)
    available_output = {}
    for (producer, item), _ in max_output.items():
        indices = by_source.get(("EVENT", producer, item), ())
        entity = problem.goal_entity[producer]
        available = sum(
            outputs[entity, index, producer, item] * modes[entity, index]
            for index in compatible[entity])
        available_output[producer, item] = available
        if indices:
            solver.Add(sum(flow[index] for index in indices) <= available)

    owner = _route_owner(structure)
    pickup_groups, purchase_groups = defaultdict(list), defaultdict(list)
    for index, arc in enumerate(arcs):
        if arc.kind in {"SHED", "PURCHASE"} and not arc.item.endswith("_SEED"):
            pickup_groups[owner[arc.consumer], arc.item].append(index)
        if arc.kind == "PURCHASE":
            purchase_groups[arc.item].append(index)

    activations = []
    for key, indices in sorted(pickup_groups.items()):
        capacity = sum(arcs[index].capacity for index in indices)
        active = solver.BoolVar(f"pickup:{key[0]}:{key[1]}")
        solver.Add(sum(flow[index] for index in indices) <= capacity * active)
        solver.Add(sum(flow[index] for index in indices) >= active)
        activations.append(active)

    purchase_active = {}
    for item, indices in sorted(purchase_groups.items()):
        capacity = sum(arcs[index].capacity for index in indices)
        active = solver.BoolVar(f"purchase-batch:{item}")
        solver.Add(sum(flow[index] for index in indices) <= capacity * active)
        solver.Add(sum(flow[index] for index in indices) >= active)
        purchase_active[item] = active
        activations.append(active)

    goals = tuple(sorted(problem.goals))
    rank = {goal: solver.IntVar(0, max(0, len(goals)-1), f"rank:{goal}")
            for goal in goals}
    for route in structure.routes:
        for before, after in zip(route, route[1:]):
            solver.Add(rank[after] >= rank[before] + 1)
    big_m = max(1, len(goals))
    for entity, indices in compatible.items():
        for index in indices:
            path = problem.complete_paths[entity][index]
            for before, after in zip(path, path[1:]):
                solver.Add(rank[after.goal] >= rank[before.goal] + 1
                           - big_m * (1-modes[entity, index]))
    # Transfer timing is edge-specific even when logistics later batches
    # multiple selected edges into one shed operation.
    for index, arc in enumerate(arcs):
        if arc.kind != "EVENT" or not arc.via_shed:
            continue
        active = solver.BoolVar(
            f"transfer-edge:{arc.producer}:{arc.consumer}:{arc.item}")
        solver.Add(flow[index] <= arc.capacity * active)
        solver.Add(flow[index] >= active)
        solver.Add(rank[arc.consumer] >= rank[arc.producer] + 1
                   - big_m * (1-active))
        activations.append(active)

    # Cash is a shared resource. These aggregate rows are necessary economic
    # conditions; exact per-turn market and shed timing stays with the compiler.
    state = problem.intent.state
    shed_flow = defaultdict(list)
    for index, arc in enumerate(arcs):
        if arc.kind == "SHED" and not arc.item.endswith("_SEED"):
            shed_flow[arc.item].append(index)
    sale_active = {}
    opening_sales = {}
    liquidity = {}
    revenue_terms = []
    for item in rules.SELLABLE_PRODUCTS:
        opening = int(state.shed.get(item, 0))
        output_keys = [(producer, produced_item)
                       for producer, produced_item in max_output
                       if produced_item == item]
        carry_keys = [(worker, carried_item, capacity)
                      for (worker, carried_item), capacity in carry.items()
                      if carried_item == item and capacity > 0]
        if not opening and not output_keys and not carry_keys:
            continue
        active = solver.BoolVar(f"sale-entry:{item}")
        sale_active[item] = active
        activations.append(active)
        sold = solver.IntVar(0, opening, f"opening-sale:{item}")
        opening_sales[item] = sold
        if shed_flow[item]:
            solver.Add(sold + sum(flow[index] for index in shed_flow[item])
                       <= opening)
        solver.Add(sold <= opening * active)
        revenue_terms.append(_source_unit_cost(state, "EVENT", item) * sold)
        item_liquidity = []
        for producer, _ in output_keys:
            capacity = max_output[producer, item]
            amount = solver.IntVar(0, capacity,
                                   f"liquidity:{producer}:{item}")
            outgoing = by_source.get(("EVENT", producer, item), ())
            solver.Add(amount + sum(flow[index] for index in outgoing)
                       <= available_output[producer, item])
            deposit = solver.BoolVar(f"liquidity-deposit:{producer}:{item}")
            solver.Add(amount <= capacity * deposit)
            solver.Add(amount >= deposit)
            solver.Add(deposit <= active)
            liquidity["EVENT", producer, item] = amount
            item_liquidity.append(amount)
            activations.append(deposit)
            revenue_terms.append(_source_unit_cost(state, "EVENT", item)
                                 * amount)
        for worker, _, capacity in carry_keys:
            amount = solver.IntVar(0, capacity,
                                   f"carry-liquidity:{worker}:{item}")
            outgoing = by_source.get(("CARRY", worker, item), ())
            solver.Add(amount + sum(flow[index] for index in outgoing)
                       <= capacity)
            deposit = solver.BoolVar(f"carry-deposit:{worker}:{item}")
            solver.Add(amount <= capacity * deposit)
            solver.Add(amount >= deposit)
            solver.Add(deposit <= active)
            liquidity["CARRY", worker, item] = amount
            item_liquidity.append(amount)
            activations.append(deposit)
            revenue_terms.append(_source_unit_cost(state, "EVENT", item)
                                 * amount)
        solver.Add(sold + sum(item_liquidity) >= active)

    purchase_cost_terms = [
        _source_unit_cost(state, "PURCHASE", arc.item) * flow[index]
        for index, arc in enumerate(arcs) if arc.kind == "PURCHASE"
    ]
    hires = max(0, structure.workforce-len(state.workers))
    fixed_cost = rules.hire_expenditure(state.hires_today, hires)
    locked_land = [quadrant for quadrant in problem.intent.land
                   if quadrant not in state.unlocked_quadrants]
    for offset, _ in enumerate(locked_land):
        price_index = len(state.unlocked_quadrants)-1+offset
        if price_index < len(rules.LAND_PRICES):
            fixed_cost += rules.LAND_PRICES[price_index]
    if purchase_cost_terms or revenue_terms or fixed_cost > state.money:
        solver.Add(sum(purchase_cost_terms) + fixed_cost
                   <= state.money + sum(revenue_terms))
    horizon = max(0, min(state.turns_left_today, state.turns_left))
    if hires or locked_land or purchase_active or sale_active:
        solver.Add(hires + len(locked_land) + sum(purchase_active.values())
                   + sum(sale_active.values())
                   <= rules.MAX_MARKET_ORDERS * horizon)

    # Pure economic coefficients own support selection.  A small stable arc
    # rank resolves mathematically equal realizations; logistics is not a
    # separate optimization objective.
    objective = solver.Objective()
    mode_count = sum(len(indices) for indices in compatible.values())
    tie_bound = (sum((index+1)*arc.capacity
                     for index, arc in enumerate(arcs))
                 + mode_count*(mode_count+1)//2
                 + sum(state.shed.values()) + sum(carry.values())
                 + sum(max_output.values()))
    scale = tie_bound + 1
    purchase_ordinal = Counter()
    for index, arc in enumerate(arcs):
        ordinal = purchase_ordinal[arc.item]
        if arc.kind == "PURCHASE":
            purchase_ordinal[arc.item] += arc.capacity
        unit = _source_unit_cost(problem.intent.state, arc.kind, arc.item, ordinal)
        objective.SetCoefficient(flow[index], unit * scale + index)
    mode_rank = 0
    for entity, indices in sorted(compatible.items()):
        for index in indices:
            mode_rank += 1
            produced_value = sum(
                quantity * _source_unit_cost(
                    problem.intent.state, "EVENT", item)
                for (owner_entity, owner_index, _, item), quantity
                in outputs.items()
                if owner_entity == entity and owner_index == index)
            objective.SetCoefficient(
                modes[entity, index], -produced_value*scale + mode_rank)
    for variable in opening_sales.values():
        objective.SetCoefficient(variable, 1)
    for variable in liquidity.values():
        objective.SetCoefficient(variable, 1)
    objective.SetMinimization()
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        blocked_items = tuple(sorted({item for _, item in max_need}))
        return None, None, None, {}, SupportConflict(
            "resource-flow", "mandatory resource-flow network is infeasible",
            tuple(sorted({goal for goal, _ in max_need})), items=blocked_items)

    path_choices = {
        entity: next(index for index in indices
                     if modes[entity, index].solution_value() > 0.5)
        for entity, indices in compatible.items()
    }
    links = []
    for variable, arc in zip(flow, arcs):
        quantity = int(round(variable.solution_value()))
        if quantity:
            links.append(ResourceLink(
                arc.consumer, arc.item, quantity, arc.kind,
                producer=arc.producer, source_worker=arc.source_worker,
                via_shed=arc.via_shed,
            ))
    diagnostics = {
        "flow_variables": len(flow),
        "binary_variables": len(activations) + len(modes),
        "mode_variables": len(modes),
        "rank_variables": len(rank),
        "material_objective": objective.Value(),
        "purchase_batches": sum(variable.solution_value() > 0.5
                                for variable in purchase_active.values()),
    }
    selected_liquidity = {
        key: int(round(variable.solution_value()))
        for key, variable in liquidity.items()
        if variable.solution_value() > 0.5
    }
    diagnostics["liquidity_deposits"] = len(selected_liquidity)
    return (path_choices, tuple(links), selected_liquidity,
            diagnostics, None)


def _market_priority(problem: RouteProblem, skeleton: RouteSkeleton):
    state = problem.intent.state
    return tuple([
        *(f"HIRE:{index}" for index in range(
            max(0, skeleton.workforce-len(state.workers)))),
        *(f"BUY:{key}" for key in sorted(skeleton.acquisitions)),
        *(f"LAND:{quadrant}" for quadrant in problem.intent.land),
    ])


def _with_liquidity_logistics(skeleton: RouteSkeleton, selected):
    """Materialize only producer surplus selected by the cash-flow model."""
    if not selected:
        return skeleton
    routes = [list(route) for route in skeleton.routes]
    logistics = dict(skeleton.logistics)
    owner = {token: worker for worker, route in enumerate(routes)
             for token in route}
    for (kind, source, item), quantity in sorted(selected.items()):
        if kind == "EVENT":
            worker, requires = owner[source], (source,)
        else:
            worker, requires = source, ()
        identifier = f"liquidity:{kind.lower()}:{worker}:{source}:{item}"
        logistics[identifier] = LogisticsEvent(
            identifier, "PLACE", item, quantity,
            requires=requires, purpose="market-liquidity",
        )
        at = routes[worker].index(source)+1 if kind == "EVENT" else 0
        routes[worker].insert(at, identifier)
    return replace(skeleton, routes=tuple(tuple(route) for route in routes),
                   logistics=logistics)


def _compiler_conflict(problem: RouteProblem, compilation):
    diagnostics = compilation.diagnostics
    missing = tuple(sorted(compilation.unfulfilled))
    blockers = diagnostics.get("blockers", ())
    items = set()
    workers = set()
    for worker, blocker in enumerate(blockers):
        if not blocker:
            continue
        workers.add(worker)
        items.update(blocker.get("missing", ()))
    kind = "capacity-or-timing"
    if diagnostics.get("unbought_inputs"):
        kind = "market-cash-timing"
        items.update(diagnostics["unbought_inputs"])
    elif diagnostics.get("unlocked_land_missing"):
        kind = "land-timing"
    return SupportConflict(
        kind, "exact compiler could not realize every mandatory Plan goal",
        missing, tuple(sorted(workers)), tuple(sorted(items)), diagnostics)


def solve_support(problem: RouteProblem, structure: RouteStructure,
                  config: SupportConfig | None = None) -> SupportResult:
    """Return the best complete support realization for one fixed structure.

    Feasibility means exact completion of the entire residual Plan, including
    land commitments.  The function never changes workforce, assignment,
    route order, placement, or the requested goal set.
    """
    config = config or SupportConfig()
    if structure.workforce != len(structure.routes):
        conflict = SupportConflict("structure", "workforce/routes mismatch")
        return SupportResult(False, structure, conflict=conflict)
    routed = [goal for route in structure.routes for goal in route]
    expected = set(problem.goals)
    if len(routed) != len(set(routed)) or set(routed) != expected:
        missing = tuple(sorted(expected-set(routed)))
        detail = "structure must route every residual Plan goal exactly once"
        return SupportResult(False, structure, conflict=SupportConflict(
            "structure", detail, missing))
    required_entities = {problem.goal_entity[goal] for goal in expected}
    missing_entities = tuple(sorted(required_entities-set(structure.placements)))
    if missing_entities:
        goals = tuple(sorted(goal for goal in expected
                             if problem.goal_entity[goal] in missing_entities))
        return SupportResult(False, structure, conflict=SupportConflict(
            "placement", "mandatory entity has no selected placement", goals))

    compatible, conflict = _compatible_modes(problem, structure)
    if conflict:
        return SupportResult(False, structure, conflict=conflict)
    state = problem.intent.state
    spawn = _worker_starts(state, structure.workforce)[len(state.workers):]
    (path_choices, links, liquidity, flow_diagnostics,
     conflict) = _solve_material_network(
         problem, structure, compatible, config.time_limit_seconds)
    if conflict:
        return SupportResult(False, structure, conflict=conflict,
                             diagnostics=flow_diagnostics)
    shell = RouteSkeleton(
        structure.workforce, structure.routes, path_choices,
        dict(structure.placements), links, (), spawn_preferences=spawn,
    )
    shell = RouteSkeleton(
        shell.workforce, shell.routes, shell.path_choices, shell.placements,
        shell.resources, rebuild_leases(problem, shell),
        spawn_preferences=shell.spawn_preferences,
    )
    shell = with_initial_logistics(problem, shell)
    shell = _with_liquidity_logistics(shell, liquidity)
    acquisitions = purchase_orders_for_links(shell.resources)
    shell = RouteSkeleton(
        shell.workforce, shell.routes, shell.path_choices, shell.placements,
        shell.resources, shell.leases, shell.synchronizations,
        shell.spawn_preferences, (), shell.logistics, acquisitions,
    )
    shell = RouteSkeleton(
        shell.workforce, shell.routes, shell.path_choices, shell.placements,
        shell.resources, shell.leases, shell.synchronizations,
        shell.spawn_preferences, _market_priority(problem, shell),
        shell.logistics, shell.acquisitions,
    )
    try:
        compilation = compile_skeleton(problem, shell)
    except ValueError as exc:
        conflict = SupportConflict("invalid-realization", str(exc))
        return SupportResult(False, structure, shell, conflict=conflict,
                             diagnostics=flow_diagnostics)
    if compilation.unfulfilled:
        return SupportResult(False, structure, shell, compilation,
                             _compiler_conflict(problem, compilation),
                             flow_diagnostics)
    diagnostics = dict(flow_diagnostics)
    diagnostics.update({
        "support_model": "compact-fixed-charge-resource-flow",
        "economic_value": EndValue(state)(compilation.final_state)[0],
        "completed": len(compilation.completed),
    })
    return SupportResult(True, structure, shell, compilation,
                         diagnostics=diagnostics)
