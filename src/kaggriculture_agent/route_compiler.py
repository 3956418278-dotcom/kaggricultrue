"""Compile an event-route skeleton into validated primitive Kaggriculture turns."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping

from . import rules
from .execution import Execution
from .intent import matches, unit_event
from .market import realization_orders
from .route_structure import (
    ResourceLink, RouteProblem, RouteSkeleton, event_positions,
    event_predecessors, selected_events, validate_skeleton,
)


@dataclass(frozen=True)
class CompileFailure:
    kind: str
    detail: str
    goal: str | None = None
    worker: int | None = None
    step: int | None = None


@dataclass(frozen=True)
class RouteCompileResult:
    executions: tuple[Execution, ...]
    final_state: object
    placements: Mapping[str, tuple[int, int]]
    completed: frozenset[str]
    unfulfilled: frozenset[str]
    diagnostics: Mapping[str, object]
    expected: tuple[object, ...] = ()
    remaining_inputs: tuple[Mapping[str, int], ...] = ()


def _move_options(origin, target):
    options = []
    if origin[0] < target[0]: options.append(["EAST"])
    if origin[0] > target[0]: options.append(["WEST"])
    if origin[1] < target[1]: options.append(["SOUTH"])
    if origin[1] > target[1]: options.append(["NORTH"])
    return options or [["PASS"]]


def _physical_key(state):
    def inv(raw): return tuple((k, v) for k, v in raw.items() if v)
    return (tuple(tuple(sorted(t.raw.items())) if isinstance(t.raw, dict) else t.raw for t in state.tiles),
            tuple((w.position, inv(w.inventory)) for w in state.workers),
            inv(state.shed), tuple(sorted((k, v) for k, v in state.seeds.items() if v)),
            state.unlocked_quadrants, state.hires_today)


def _remaining_input_reserve(skeleton, completed_logistics):
    """Quantities that still have an explicit future pickup from the shed.

    This is intentionally source-aware. Carried output linked directly to a
    later task cannot release unrelated shed stock for sale.
    """
    reserve = Counter()
    for identifier, event in skeleton.logistics.items():
        if (identifier not in completed_logistics and event.operation == "PICKUP"
                and event.item and not event.item.endswith("_SEED")):
            reserve[event.item] += event.quantity
    return dict(reserve)


def purchase_orders_for_links(links):
    quantities = Counter()
    for link in links:
        if link.kind == "PURCHASE":
            quantities[link.item] += link.quantity
    orders = {}
    for item, quantity in quantities.items():
        if item.endswith("_SEED"):
            orders[item] = ("BUY_SEED", item[:-5], quantity)
        elif item in rules.ANIMALS:
            orders[item] = ("BUY_ANIMAL", item, quantity)
        else:
            orders[item] = ("BUY_PRODUCT", item, quantity)
    return orders


def _market_batch(state, skeleton, outstanding_purchases, outstanding_land,
                  max_required_entries=rules.MAX_MARKET_ORDERS,
                  max_hires=rules.MAX_MARKET_ORDERS):
    """Take the selected required decisions, preserving optional sale slots.

    ``max_required_entries`` is a realization choice, not a game limit.  It is
    normally ten, but can be lower on a financing-sensitive turn so the market
    module has room to sell before later required orders are attempted.
    """
    max_required_entries = max(0, min(rules.MAX_MARKET_ORDERS,
                                      int(max_required_entries)))
    max_hires = max(0, min(rules.MAX_MARKET_ORDERS, int(max_hires)))
    missing_hires = max(0, skeleton.workforce-len(state.workers))
    tokens = skeleton.market_priority or tuple(
        [*(f"BUY:{item}" for item in sorted(outstanding_purchases)),
         *(f"LAND:{q}" for q in outstanding_land),
         *(f"HIRE:{n}" for n in range(missing_hires))])
    buys, land, hires, used, sequence, selected_buys = [], 0, 0, set(), [], []
    for token in tokens:
        if len(buys)+land+hires >= max_required_entries:
            break
        if token.startswith("BUY:"):
            purchase = token[4:]
            if purchase in outstanding_purchases and purchase not in used:
                order = outstanding_purchases[purchase]
                buys.append(order); sequence.append(order); selected_buys.append(purchase); used.add(purchase)
        elif token.startswith("LAND:"):
            quadrant = token[5:]
            if quadrant in outstanding_land and quadrant not in used:
                land += 1; sequence.append(["BUY_LAND"]); used.add(quadrant)
        elif token.startswith("HIRE:") and hires < min(missing_hires, max_hires):
            hires += 1; sequence.append(["HIRE"])
    # A mutated priority may omit a still-needed token; append it rather than
    # silently deleting the fixed realization requirement.
    for purchase in sorted(outstanding_purchases):
        if len(buys)+land+hires >= max_required_entries: break
        if purchase not in used:
            order = outstanding_purchases[purchase]
            buys.append(order); sequence.append(order); selected_buys.append(purchase); used.add(purchase)
    for quadrant in outstanding_land:
        if len(buys)+land+hires >= max_required_entries: break
        if quadrant not in used:
            land += 1; sequence.append(["BUY_LAND"]); used.add(quadrant)
    extra_hires = min(missing_hires-hires, max_hires-hires,
                      max_required_entries-len(buys)-land-hires)
    hires += extra_hires; sequence.extend([["HIRE"] for _ in range(extra_hires)])
    return buys, hires, land, sequence, selected_buys


def _spawn_adjusted(state, actions, targets, hires, preferences, assignments,
                    continuity):
    """Compile a selected spawn layout into this hire turn's occupancy moves.

    A desired hand spawn is the combinatorial choice.  The exact move used to
    vacate/occupy shed access is an execution detail, so an otherwise idle or
    logistical worker may defer its next route token for one turn.  A service
    event already selected for this turn is never rewritten.
    """
    if not hires or not preferences:
        return actions
    indices, alternatives = [], []
    access = rules.shed_access(state.board_size)
    for worker, (action, target) in enumerate(zip(actions, targets)):
        if worker in assignments:
            continue
        position = state.workers[worker].position
        if target is not None and action[0] in ("NORTH", "SOUTH", "EAST", "WEST"):
            options = _move_options(position, target)
            if position in access:
                options = [*options, ["PASS"]]
        else:
            # Nonmovement actions keep the worker on the same square and are
            # therefore valid occupancy realizations too. Prefer retaining the
            # route action when it yields the requested spawn.
            options = [list(action), ["PASS"]]
            for operation, (dx, dy) in (("NORTH", (0, -1)), ("SOUTH", (0, 1)),
                                         ("WEST", (-1, 0)), ("EAST", (1, 0))):
                candidate = (position[0]+dx, position[1]+dy)
                if 0 <= candidate[0] < state.board_size and 0 <= candidate[1] < state.board_size:
                    options.append([operation])
        options = [option for n, option in enumerate(options) if option not in options[:n]]
        if action in options:
            options.remove(action); options.insert(0, action)
        if len(options) > 1:
            indices.append(worker); alternatives.append(options)
    if not indices:
        return actions
    def spawned_from_counts(counts):
        counts = list(counts)
        result = []
        for _ in range(hires):
            index = min(range(len(access)), key=lambda n: (counts[n], n))
            counts[index] += 1; result.append(access[index])
        return result

    moves = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
    def result_position(worker, action):
        position = state.workers[worker].position
        if action[0] in moves:
            dx, dy = moves[action[0]]
            return position[0]+dx, position[1]+dy
        return position

    mutable = set(indices)
    fixed_counts = [0] * len(access)
    for worker, action in enumerate(actions):
        if worker in mutable:
            continue
        position = result_position(worker, action)
        if position in access:
            fixed_counts[access.index(position)] += 1

    # Exact dynamic programming over the only state that influences spawning:
    # occupancy counts on the four shed-access cells. This avoids pruning a
    # globally useful combination of individually ordinary route moves.
    base = [list(action) for action in actions]
    frontier = {tuple(fixed_counts): ((0, 0, 0, 0, ()), base)}
    for worker, options in zip(indices, alternatives):
        following = {}
        for counts, (prior_cost, trial) in frontier.items():
            for option in options:
                position = result_position(worker, option)
                updated = list(counts)
                if position in access:
                    updated[access.index(position)] += 1
                disrupted = int(actions[worker][0] not in {"PASS", "NORTH", "SOUTH", "EAST", "WEST"}
                                and option != actions[worker])
                continuity_cost = continuity[worker]*int(option != actions[worker])
                route_distance = (rules.manhattan(position, targets[worker])
                                  if targets[worker] is not None else 0)
                changed = int(option != actions[worker])
                action_key = (*prior_cost[4], tuple(option))
                cost = (prior_cost[0]+disrupted,
                        prior_cost[1]+continuity_cost,
                        prior_cost[2]+route_distance,
                        prior_cost[3]+changed, action_key)
                candidate = [list(action) for action in trial]; candidate[worker] = option
                key = tuple(updated)
                if key not in following or cost < following[key][0]:
                    following[key] = (cost, candidate)
        frontier = following
    best = None
    for counts, (cost, trial) in frontier.items():
        spawned = spawned_from_counts(counts)
        mismatch = sum(rules.manhattan(a, b) for a, b in zip(spawned, preferences[:hires]))
        score = (mismatch, *cost)
        if best is None or score < best[0]:
            best = score, trial
    return best[1] if best else actions


def compile_skeleton(problem: RouteProblem, skeleton: RouteSkeleton) -> RouteCompileResult:
    """Earliest-feasible compilation; all consequential choices are explicit.

    The compiler chooses canonical shortest-path steps and batches quantities for
    already selected resource links. It does not change routes, placements,
    sources, staffing, market priority, synchronization, or tile leases.
    """
    validate_skeleton(problem, skeleton)
    state = problem.intent.state
    events = selected_events(problem, skeleton)
    positions = event_positions(problem, skeleton)
    predecessors = event_predecessors(problem, skeleton)
    route_of = {goal: worker for worker, route in enumerate(skeleton.routes) for goal in route}
    links_by_consumer = defaultdict(list)
    for index, link in enumerate(skeleton.resources):
        links_by_consumer[link.consumer].append((index, link))
    bundle_by_goal = {goal: bundle for bundle in skeleton.synchronizations for goal in bundle.goals}
    lease_by_entity = {lease.entity: lease for lease in skeleton.leases}
    pointers = [0] * skeleton.workforce
    completed, completed_logistics, executor = set(), set(), {}
    executions, expected, reserves, failures = [], [state], [], []
    purchase_template = ({item: list(order) for item, order in skeleton.acquisitions.items()}
                         or purchase_orders_for_links(skeleton.resources))
    purchase_remaining = {item: list(order) for item, order in purchase_template.items()}
    outstanding_land = [q for q in problem.intent.land if q not in state.unlocked_quadrants]
    used_turns = movement = logistics_actions = 0
    transaction_totals = Counter()
    transaction_breakdowns = defaultdict(Counter)
    hire_timing = []
    opening_workforce = len(state.workers)
    peak_workforce = opening_workforce
    horizon = min(state.turns_left_today, state.turns_left)

    def current_token(worker):
        if worker >= len(skeleton.routes): return None
        route = skeleton.routes[worker]
        while pointers[worker] < len(route) and (route[pointers[worker]] in completed
                                                 or route[pointers[worker]] in completed_logistics):
            pointers[worker] += 1
        return route[pointers[worker]] if pointers[worker] < len(route) else None

    def link_ready(link, worker):
        if link.kind == "EVENT":
            if link.producer not in completed: return False
            if link.via_shed: return True
            return executor.get(link.producer) == worker
        if link.kind == "CARRY": return link.source_worker == worker
        return True

    for turn_offset in range(horizon):
        if state.step > rules.TERMINAL_ACTION_STEP: break
        reserve = _remaining_input_reserve(skeleton, completed_logistics)
        reserves.append(reserve)
        required_cap = (skeleton.required_entry_caps[turn_offset]
                        if turn_offset < len(skeleton.required_entry_caps)
                        else rules.MAX_MARKET_ORDERS)
        hire_cap = (skeleton.hire_caps[turn_offset]
                    if turn_offset < len(skeleton.hire_caps)
                    else rules.MAX_MARKET_ORDERS)
        buys, hires, land, required_sequence, selected_purchase_keys = _market_batch(
            state, skeleton, purchase_remaining, outstanding_land, required_cap, hire_cap)
        micro = state
        actions = [["PASS"] for _ in state.workers]
        assignments = {}
        targets = [None for _ in state.workers]
        seed_budget = Counter(state.seeds)
        ready_bundles = set()
        for bundle in skeleton.synchronizations:
            workers = [route_of.get(g) for g in bundle.goals]
            internal = set(bundle.goals)
            if (not any(w is None for w in workers) and workers == sorted(workers)
                    and len(set(workers)) == len(workers)
                    and all(w < len(state.workers) and current_token(w) == g
                            and state.workers[w].position == positions[g]
                            for w, g in zip(workers, bundle.goals))
                    and all(not (predecessors.get(g, frozenset())-internal-completed)
                            for g in bundle.goals)):
                ready_bundles.add(bundle.goals)

        def sync_ready(goal):
            bundle = bundle_by_goal.get(goal)
            return bundle is None or bundle.goals in ready_bundles

        for worker in range(len(state.workers)):
            token = current_token(worker)
            pos = micro.workers[worker].position
            action = ["PASS"]
            logistics = skeleton.logistics.get(token)
            goal = token if token in events else None
            if logistics is not None:
                dependencies = set(logistics.requires)-completed-completed_logistics
                if dependencies:
                    action = ["PASS"]
                elif logistics.operation in {"NORTH", "SOUTH", "EAST", "WEST", "PASS"}:
                    # Hire-turn occupancy and other synchronization guards are
                    # explicit route events.  The compiler does not replace
                    # them with a locally attractive service or shed visit.
                    action = [logistics.operation]
                    if logistics.operation == "PASS":
                        completed_logistics.add(token); pointers[worker] += 1
                else:
                    target = min(rules.shed_access(state.board_size),
                                 key=lambda p: (rules.manhattan(pos, p), p))
                    targets[worker] = target
                    if pos not in rules.shed_access(state.board_size):
                        action = _move_options(pos, target)[0]
                    elif logistics.operation == "PICKUP":
                        if micro.shed.get(logistics.item, 0) >= logistics.quantity:
                            action = ["PICKUP", logistics.item, logistics.quantity]
                    elif logistics.operation == "PLACE":
                        if (micro.workers[worker].inventory.get(logistics.item, 0) >= logistics.quantity
                                and micro.shed_used+logistics.quantity <= rules.SHED_CAPACITY):
                            action = ["PLACE", logistics.item, logistics.quantity]
                    elif logistics.operation == "DROP":
                        if micro.workers[worker].carried:
                            action = ["DROP"]
                        else:
                            completed_logistics.add(token); pointers[worker] += 1
            elif goal is not None:
                event = events[goal]
                entity = problem.goal_entity[goal]
                lease = lease_by_entity.get(entity)
                target = positions.get(goal)
                targets[worker] = target
                unmet_predecessors = predecessors.get(goal, frozenset()) - completed
                resource_links = links_by_consumer.get(goal, ())
                missing = Counter()
                blocked_source = False
                for _, link in resource_links:
                    if not link_ready(link, worker): blocked_source = True
                    if not link.item.endswith("_SEED"):
                        missing[link.item] += link.quantity
                for item in list(missing):
                    missing[item] = max(0, missing[item]-micro.workers[worker].inventory.get(item, 0))
                    if not missing[item]: del missing[item]
                if lease and lease.available_after and lease.available_after not in completed:
                    unmet_predecessors = unmet_predecessors | {lease.available_after}

                if missing:
                    # Pickup timing and quantity are skeleton decisions. A
                    # missing item is structural infeasibility, not permission
                    # for the compiler to insert an unplanned shed trip.
                    action = ["PASS"]
                elif target is None:
                    failures.append(CompileFailure("missing-placement", "entity has no Plan-permitted placement", goal, worker, state.step))
                elif micro.tile_at(target).raw == "LOCKED":
                    action = ["PASS"]
                elif pos != target:
                    action = _move_options(pos, target)[0]
                elif blocked_source or unmet_predecessors or not sync_ready(goal):
                    action = ["PASS"]
                else:
                    item_seed = next((item for item, n in event.delta.items() if n < 0 and item.endswith("_SEED")), None)
                    if item_seed and seed_budget[item_seed[:-5]] < -event.delta[item_seed]:
                        action = ["PASS"]
                    else:
                        following, before, after, delta = unit_event(micro, worker, event.action)
                        if matches(problem.goals[goal], before, after, delta, event.action):
                            action = list(event.action)
                            micro = following
                            assignments[worker] = goal
                            completed.add(goal)
                            completed.update(problem.goals[goal].aliases)
                            executor[goal] = worker
                            pointers[worker] += 1
                            if item_seed: seed_budget[item_seed[:-5]] += event.delta[item_seed]
                        else:
                            failures.append(CompileFailure("service-noop", f"{event.action} did not realize its effect", goal, worker, state.step))
            if action == ["PASS"] or worker in assignments:
                actions[worker] = action
                continue
            actions[worker] = action
            before_action = micro
            micro = rules.advance_owned(micro,
                tuple(action if i == worker else ["PASS"] for i in range(len(micro.workers))), unit_only=True)
            if logistics is not None:
                moved_as_requested = (logistics.operation in {"NORTH", "SOUTH", "EAST", "WEST"}
                    and action[0] == logistics.operation
                    and micro.workers[worker].position != before_action.workers[worker].position)
                realized = (moved_as_requested or action[0] == "DROP" or
                    (action[0] == "PICKUP" and micro.workers[worker].inventory.get(logistics.item, 0)
                     - before_action.workers[worker].inventory.get(logistics.item, 0) >= logistics.quantity) or
                    (action[0] == "PLACE" and micro.shed.get(logistics.item, 0)
                     - before_action.shed.get(logistics.item, 0) >= logistics.quantity))
                if realized:
                    completed_logistics.add(token); pointers[worker] += 1

        original_actions = [list(action) for action in actions]
        actions = _spawn_adjusted(state, actions, targets, hires,
                                  skeleton.spawn_preferences[len(state.workers)-len(problem.intent.state.workers):],
                                  assignments,
                                  [len(skeleton.routes[worker])-pointers[worker]
                                   for worker in range(len(state.workers))])
        # A deferred logistics token was tentatively applied while unit actions
        # were assembled.  Undo only its route bookkeeping; the physical state
        # is recomputed below from the chosen simultaneous action tuple.
        for worker, (before, after) in enumerate(zip(original_actions, actions)):
            if before == after:
                continue
            token = skeleton.routes[worker][pointers[worker]-1] if pointers[worker] else None
            if token in skeleton.logistics and token in completed_logistics:
                completed_logistics.remove(token); pointers[worker] -= 1
        orders = realization_orders(state, tuple(actions), buys, hires, land, reserve,
                                    reserve_is_shed=True,
                                    required_sequence=required_sequence)
        after_units = rules.advance_owned(state, tuple(actions), unit_only=True)
        if _physical_key(after_units) != _physical_key(micro):
            # Spawn-tie movement changes only positions; recompute the exact
            # unit state used as the transition oracle instead of trusting the
            # incremental construction.
            micro = after_units
        turn_ledger = rules.solo_market_ledger(after_units, orders)
        following = rules.advance_owned(state, tuple(actions), orders)
        if turn_ledger["ending_cash"] != following.money:
            raise AssertionError("solo market ledger cash parity mismatch")
        for key in ("hire_expenditure", "input_expenditure", "land_expenditure",
                    "sale_revenue", "executed_hires", "executed_land_purchases"):
            transaction_totals[key] += turn_ledger[key]
        for key in ("input_expenditure_by_kind", "input_expenditure_by_item",
                    "purchased_quantity_by_item", "sale_revenue_by_item",
                    "sold_quantity_by_item"):
            transaction_breakdowns[key].update(turn_ledger[key])
        if turn_ledger["executed_hires"]:
            hire_timing.append({"step": state.step,
                                "count": turn_ledger["executed_hires"],
                                "expenditure": turn_ledger["hire_expenditure"]})
        peak_workforce = max(peak_workforce,
                             len(state.workers)+turn_ledger["executed_hires"])

        # Market requirements remain outstanding when cash/capacity rejected
        # them. Compare against the post-unit state, before any later retry.
        sold = Counter()
        for order in orders:
            if order[0] == "SELL": sold[order[1]] += int(order[2])
        gained_by_item = Counter()
        for purchase in selected_purchase_keys:
            order = purchase_remaining[purchase]
            item = order[1]
            if item not in gained_by_item:
                if order[0] == "BUY_SEED":
                    gained_by_item[item] = max(0, following.seeds.get(item, 0)-after_units.seeds.get(item, 0))
                else:
                    gained_by_item[item] = max(0, following.owned_total(item)
                        - after_units.owned_total(item)+sold[item])
            acquired = min(order[2], gained_by_item[item])
            order[2] -= acquired; gained_by_item[item] -= acquired
            if not order[2]: del purchase_remaining[purchase]
        outstanding_land = [q for q in outstanding_land if q not in following.unlocked_quadrants]

        for action in actions:
            if action[0] != "PASS": used_turns += 1
            if action[0] in ("NORTH", "SOUTH", "EAST", "WEST"): movement += 1
            if action[0] in ("PICKUP", "DROP") or (action[0] == "PLACE" and len(action) > 2): logistics_actions += 1
        executions.append(Execution(tuple(actions), (), assignments, tuple(orders)))
        state = following
        expected.append(state)

    all_goals = set(problem.goals)
    completed_goals = all_goals & completed
    completed_land = {f"land:{q}" for q in problem.intent.land if q in state.unlocked_quadrants}
    all_goal_ids = all_goals | {f"land:{q}" for q in problem.intent.land}
    achieved = completed_goals | completed_land
    blockers = []
    for worker in range(skeleton.workforce):
        token = current_token(worker)
        if token is None:
            blockers.append(None); continue
        if token in skeleton.logistics:
            event = skeleton.logistics[token]
            blockers.append({"token": token, "kind": "logistics",
                "operation": event.operation,
                "dependencies": tuple(sorted(set(event.requires)-completed-completed_logistics)),
                "shed_quantity": state.shed.get(event.item, 0) if event.item else None,
                "carried_quantity": (state.workers[worker].inventory.get(event.item, 0)
                                     if worker < len(state.workers) and event.item else None)})
            continue
        missing = Counter()
        for _, link in links_by_consumer.get(token, ()):
            if not link.item.endswith("_SEED"):
                missing[link.item] += link.quantity
        if worker < len(state.workers):
            for item in tuple(missing):
                missing[item] = max(0, missing[item]-state.workers[worker].inventory.get(item, 0))
                if not missing[item]: del missing[item]
        blockers.append({"token": token, "kind": "service",
            "position": state.workers[worker].position if worker < len(state.workers) else None,
            "target": positions.get(token), "missing": dict(missing),
            "predecessors": tuple(sorted(predecessors.get(token, frozenset())-completed))})
    diagnostics = {
        "model": "event-route-skeleton",
        "completed": len(achieved),
        "goal_count": len(all_goal_ids),
        "used_worker_turns": used_turns,
        "movement": movement,
        "logistics": logistics_actions,
        "workforce": skeleton.workforce,
        "opening_workforce": opening_workforce,
        "peak_workforce": peak_workforce,
        "executed_hires": transaction_totals["executed_hires"],
        "hire_timing": tuple(hire_timing),
        "hire_expenditure": transaction_totals["hire_expenditure"],
        "input_expenditure": transaction_totals["input_expenditure"],
        "land_expenditure": transaction_totals["land_expenditure"],
        "sale_revenue": transaction_totals["sale_revenue"],
        "ending_cash": state.money,
        **{key: dict(value) for key, value in transaction_breakdowns.items()},
        "compile_failures": tuple(failure.__dict__ for failure in failures),
        "unbought_inputs": {purchase: order[2] for purchase, order in purchase_remaining.items()},
        "unlocked_land_missing": tuple(outstanding_land),
        "remaining_route_heads": tuple(
            skeleton.routes[worker][pointers[worker]] if pointers[worker] < len(skeleton.routes[worker]) else None
            for worker in range(skeleton.workforce)),
        "completed_logistics": len(completed_logistics),
        "remaining_route_blockers": tuple(blockers),
    }
    return RouteCompileResult(tuple(executions), state, dict(skeleton.placements),
        frozenset(achieved), frozenset(all_goal_ids-achieved), diagnostics,
        tuple(expected), tuple(reserves))
