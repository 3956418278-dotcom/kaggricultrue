"""Bounded joint-state beam search over the entire remaining operating day.

The beam branches on joint task assignments, pickup batches, worker divisions,
staffing and spatial bindings. Routes persist between decision events. Every
edge is an actual one-turn farm transition; every returned leaf reaches refresh
or the episode cash boundary. Proposal heuristics are NOT the leaf objective.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

from . import rules
from .execution import (Execution, WorkTask, _task_target, _worker_can_do,
                        execute, generate_tasks)
from .market import build_market_orders
from .planner import Plan, _animal_commitment, _existing_crop_obligation
from .realization import ExecutionChoices, legacy_choices


def farm_key(state):
    """Exact owned dynamics, excluding reactive prices/cash and unseen RNG."""
    def inventory(inv):
        # Inventory insertion order matters to shed overflow.
        return tuple((k, v) for k, v in inv.items() if v)
    return (state.step, tuple(tuple(sorted(t.raw.items())) if isinstance(t.raw, dict) else t.raw for t in state.tiles),
            tuple((w.position, inventory(w.inventory)) for w in state.workers),
            inventory(state.shed), inventory(state.seeds), state.hires_today,
            state.unlocked_quadrants)


@dataclass(frozen=True)
class SearchConfig:
    beam_width: int = 3
    joint_branches: int = 4
    placement_variants: int = 3
    rollout_budget: int = 64  # deterministic effort, never a wall-clock cutoff


@dataclass(frozen=True)
class Trajectory:
    choices: ExecutionChoices
    executions: tuple[Execution, ...]
    expected: tuple[object, ...]
    final_state: object
    value: tuple
    diagnostics: dict


class EndValue:
    """Cash plus marginal asset proceeds under the daily model's assumptions.

    Prefixes are settled with no more work to the deterministic refresh. This
    prices lost survival/production, carry overflow and today's care correctly.
    Future maintenance is optimistic, as in the economic model. Future service
    burden and logistics break economic ties, not an arbitrary six-axis score.
    """
    def __init__(self, initial):
        self.initial = initial
        self.cache = {}

    def __call__(self, state):
        if state.step > rules.TERMINAL_ACTION_STEP:
            return (state.money, 0, 0)
        settled = state
        if state.day == self.initial.day:
            if state.day == 29:
                # No automatic drop at step 719. Only banked cash counts.
                return (state.money, 0, 0)
            settled = rules.advance_owned(replace(state, hour=23, step=state.day * 24 + 23), ())
        value = settled.money
        for item in rules.PRODUCTS:
            value += rules.projected_sale_revenue(item, settled.owned_total(item),
                settled.market_inventory[item], settled.step, settled.step, ())
        # Unsown seeds/unplaced animals retain their acquisition value as a
        # conservative option value, not as terminal cash.
        value += sum(q * rules.CROPS[c].seed_cost for c, q in settled.seeds.items())
        value += sum(settled.owned_total(a) * r.cost for a, r in rules.ANIMALS.items())
        debt, servicing = 0, 0
        for tile in settled.tiles:
            if tile.kind != "PLANT" and not tile.animal:
                continue
            key = (settled.day, tile.position, repr(tile.raw))
            if key not in self.cache:
                # Reference prices held fixed within a search; changing the
                # opponent's market is not incorrectly valued as our output.
                valued = replace(settled, market_inventory=self.initial.market_inventory,
                                 market_prices=self.initial.market_prices, unlocked_shops=self.initial.unlocked_shops)
                if tile.animal:
                    p = _animal_commitment(valued, tile.animal, tile.position, existing=True,
                        placed_day=tile.raw.get("placed_day", settled.day),
                        current_yield=tile.raw.get("yield_units", 0),
                        fertilizer_available=tile.raw.get("fertilizer_available", False),
                        pending_care_bonus=tile.raw.get("pending_care_bonus", 0))
                else:
                    p = _existing_crop_obligation(valued, tile)
                marginal = max(0, p.terminal_profit)
                if tile.kind == "PLANT" and tile.raw.get("max_lifespan_step", -1) >= 0:
                    # A crop left to rot before a worker can reach it tomorrow
                    # is not inventory at face value. Decay follows unit actions.
                    arrival = settled.step + rules.distance_to_shed(tile.position)
                    lifespan = tile.raw["max_lifespan_step"]
                    if arrival >= lifespan:
                        lost = max(0, (arrival - 1 - lifespan) // 2 + 1)
                        realizable = max(0, tile.raw.get("yield_units", 0) - lost)
                        marginal = min(marginal, rules.projected_sale_revenue(tile.raw["crop"],
                            realizable, valued.market_inventory[tile.raw["crop"]], arrival, settled.step, settled.unlocked_shops))
                self.cache[key] = (marginal, len({w.day for w in p.actions.work}))
            marginal, work = self.cache[key]
            value += marginal
            debt += tile.raw.get("consecutive_unwatered", tile.raw.get("consecutive_unfed", 0))
            servicing += work * rules.distance_to_shed(tile.position)
        return (value, -debt, -servicing)


def search_tasks(state, plan, choices):
    """Execution choices include carrying chains the benchmark cannot express."""
    tasks = list(generate_tasks(state, plan, choices))
    if state.step >= 707:
        for tile in state.animal_tiles():
            if tile.raw.get("fertilizer_available", False):
                tasks.append(WorkTask(f"terminal-collect:{tile.position}", "COLLECT_FERTILIZER",
                    ("COLLECT_FERTILIZER",), 4, 718, tile.position,
                    followup_actions=rules.distance_to_shed(tile.position) + 1))
    # Remove pickups for completed fertilizer work (the fixed daily intent stays).
    pending = sum(state.tile_at(p).kind == "PLANT" and
        state.tile_at(p).raw.get("fertilized_until_day", -1) < state.day + 2
        for p in plan.fertilize_targets)
    tasks = [t for t in tasks if t.identifier != "pickup:fertilizer"]
    if pending > state.carried_total("FERTILIZER") and state.shed.get("FERTILIZER", 0):
        tasks.append(WorkTask("pickup:fertilizer", "PICKUP",
            ("PICKUP", "FERTILIZER", min(pending, state.shed["FERTILIZER"])), 9,
            min(718, (state.day + 1) * 24 - 1), at_shed=True))
    # A single item can justify same-day sale. Search decides whether its trip
    # pays; at ordinary refresh keeping it carried is also a reachable option.
    for w in state.workers:
        if w.carried and not any(t.identifier == f"drop:{w.index}" for t in tasks):
            tasks.append(WorkTask(f"drop:{w.index}", "DROP", ("DROP",), 32,
                min(718, (state.day + 1) * 24 - 1), at_shed=True, eligible_worker=w.index))
    return tuple(tasks)


def joint_options(state, plan, choices, routes, count):
    tasks = search_tasks(state, plan, choices)
    by_id = {t.identifier: t for t in tasks}
    options = []
    # Multiple complete matchings, with route-continuity and interrupt variants.
    # Assignment is joint: tasks, atomic seeds and pickup stock are reserved.
    for variant in range(count):
        actions = [["PASS"] for _ in state.workers]
        assignments, used, stock, seeds = {}, set(), dict(state.shed), dict(state.seeds)
        pending = set(range(len(state.workers)))
        while pending:
            pairs = []
            for i in sorted(pending):
                worker = state.workers[i]
                for task in tasks:
                    if task.identifier in used or not _worker_can_do(worker, task):
                        continue
                    target = _task_target(task, worker, state.board_size)
                    distance = rules.manhattan(worker.position, target)
                    if distance + 1 + task.followup_actions > min(state.turns_left_today, state.turns_left):
                        continue
                    if task.kind == "PLANT" and seeds.get(str(task.action[1]), 0) <= 0:
                        continue
                    if task.kind == "PICKUP" and stock.get(str(task.action[1]), 0) <= 0:
                        continue
                    continuing = i < len(routes) and routes[i] == task.identifier
                    cluster = sum(t.target == target and _worker_can_do(worker, t) for t in tasks)
                    # Keep a pickup/use chain intact in completion rollouts.
                    # DROP remains an alternative (variant 3), but zero-distance
                    # shed access must not create an endless pickup/drop cycle.
                    input_drop = task.kind == "DROP" and any(
                        t.required_item and _worker_can_do(worker, t) for t in tasks)
                    if variant == 0:
                        rank = (not continuing, input_drop, distance, task.priority)
                    elif variant == 1:
                        rank = (not continuing, input_drop, distance / max(1, cluster), task.priority)
                    elif variant == 2:
                        rank = (not continuing, input_drop, task.deadline_step - state.step - distance - task.followup_actions,
                                distance, task.priority)
                    else:
                        rank = (distance, not bool(task.required_item), task.priority)
                    pairs.append((rank, task.identifier, i, task, target))
            if not pairs:
                break
            _, _, i, task, target = min(pairs, key=lambda p: p[:3])
            w = state.workers[i]
            action = list(task.action) if w.position == target else rules.move_toward(w.position, target)
            if action[0] == "PLANT":
                seeds[str(action[1])] -= 1
            if action[0] == "PICKUP":
                item = str(action[1])
                # Batch and split alternatives allow workers to divide feed
                # routes instead of forcing one worker to monopolize the stock.
                quantity = min(stock[item], int(action[2]))
                if variant % 2:
                    quantity = min(quantity, max(1, (quantity + len(pending) - 1) // len(pending)))
                action[2] = quantity
                stock[item] -= quantity
            actions[i], assignments[i] = action, task.identifier
            used.add(task.identifier)
            pending.remove(i)
        encoded = tuple(tuple(a) for a in actions)
        if all(encoded != tuple(tuple(a) for a in previous.worker_actions) for previous in options):
            options.append(Execution(tuple(actions), tasks, assignments, ()))
    # Wait is a real option: stored output does not decay, and town demand may
    # make delaying a sale better. Also prevents worthless weed-clearing tails.
    options.append(Execution(tuple(["PASS"] for _ in state.workers), tasks, {}, ()))
    return options


@dataclass(frozen=True)
class _Node:
    state: object
    trace: tuple = ()
    expected: tuple = ()
    routes: tuple = ()
    moves: int = 0
    logistics: int = 0
    idle: int = 0
    states: tuple = ()


def _edge(node, execution, plan, choices, greedy=False):
    orders = build_market_orders(node.state, plan, execution.worker_actions, choices)
    execution = replace(execution, market_orders=orders, benchmark_orders=greedy)
    following = rules.advance_owned(node.state, execution.worker_actions, orders)
    ops = [a[0] for a in execution.worker_actions]
    return _Node(following, node.trace + (execution,), node.expected + (farm_key(node.state),),
        tuple(execution.assignments.get(i) for i in range(len(following.workers))),
        node.moves + sum(op in ("NORTH", "SOUTH", "EAST", "WEST") for op in ops),
        node.logistics + sum(op in ("PICKUP", "DROP", "PLACE") for op in ops),
        node.idle + ops.count("PASS"), node.states + (node.state,))


def search_day(state, daily: Plan, config: SearchConfig | None = None, *, previous=None):
    started = perf_counter()
    config = config or SearchConfig()
    value = EndValue(state)
    horizon = min(state.turns_left_today, state.turns_left)
    roots = []
    staffing = (list(range(state.hires_today, max(state.hires_today, daily.max_hands) + 1))
                if state.hour <= 2 else [max(state.hires_today, previous.hire_count if previous else state.hires_today)])
    for hands in staffing:
        for placement in range(config.placement_variants if previous is None else 1):
            choices = legacy_choices(state, daily, placement, hands, previous)
            if choices not in roots:
                roots.append(choices)
    expanded, rollouts = 0, 0
    def rollout(node, choices, mode=0):
        nonlocal expanded
        while len(node.trace) < horizon:
            if mode == -1:
                action = execute(node.state, daily, choices)
            else:
                options = joint_options(node.state, daily, choices, node.routes, config.joint_branches)
                action = options[min(mode, len(options) - 2)]
            node = _edge(node, action, daily, choices, greedy=mode == -1)
            expanded += 1
        return node

    def final_rank(pair):
        node, _ = pair
        economic, debt, service = value(node.state)
        return (economic, debt, service - node.moves - node.logistics, -node.moves - node.logistics)
    # Beam entries are COMPLETE reachable trajectories, not shallow prefixes.
    # This avoids pruning a pickup/travel/use chain merely because its first
    # several actions spend cash/time before delivering any economic benefit.
    finalists = [(rollout(_Node(state), choices, mode), choices)
                  for choices in roots for mode in (-1, 0, 1)]
    beam = sorted(finalists, key=final_rank, reverse=True)[:config.beam_width]
    visited = set()
    # Spread the bounded neighborhood search across the horizon, not just its
    # first turns. Repeated passes can combine several beneficial route changes.
    times = sorted(range(horizon), key=lambda n: (int(f"{n:05b}"[::-1], 2), n))
    for _ in range(2):
        for index in times:
            for complete, choices in tuple(beam):
                prefix_ops = [a[0] for e in complete.trace[:index] for a in e.worker_actions]
                current = complete.states[index]
                previous = complete.trace[index - 1].assignments if index else {}
                prefix = _Node(current, complete.trace[:index], complete.expected[:index],
                    tuple(previous.get(i) for i in range(len(current.workers))),
                    sum(op in ("NORTH", "SOUTH", "EAST", "WEST") for op in prefix_ops),
                    sum(op in ("PICKUP", "DROP", "PLACE") for op in prefix_ops),
                    prefix_ops.count("PASS"), complete.states[:index])
                for action in joint_options(current, daily, choices, prefix.routes, config.joint_branches):
                    key = (farm_key(current), current.money, repr(choices.placements),
                           choices.hire_count, tuple(tuple(a) for a in action.worker_actions))
                    if key in visited or (action.worker_actions == complete.trace[index].worker_actions):
                        continue
                    visited.add(key)
                    if rollouts >= config.rollout_budget:
                        break
                    candidate = (rollout(_edge(prefix, action, daily, choices), choices), choices)
                    expanded += 1
                    rollouts += 1
                    # Dominance across complete reachable states and stable
                    # trajectory deduplication keep equivalent worker routes
                    # from occupying the entire beam.
                    pool = sorted((*beam, candidate), key=final_rank, reverse=True)
                    unique = {}
                    for pair in pool:
                        n, p = pair
                        signature = (farm_key(n.state), n.state.money, n.moves + n.logistics,
                                     tuple(tuple(tuple(a) for a in e.worker_actions) for e in n.trace))
                        unique.setdefault(signature, pair)
                    beam = list(unique.values())[:config.beam_width]
            if rollouts >= config.rollout_budget:
                break
    best, choices = max(beam, key=final_rank)
    unfinished = [p.identifier for p in (*daily.selected, *daily.obligations)
        if p.kind in ("CROP", "ANIMAL", "ANIMAL_PLACEMENT") and
        (choices.target(p) is None or not (
            best.state.tile_at(choices.target(p)).kind == "PLANT"
            and best.state.tile_at(choices.target(p)).raw.get("crop") == p.metadata.get("crop")
            if p.kind == "CROP" else
            choices.target(p) is not None and best.state.tile_at(choices.target(p)).animal == p.metadata.get("animal")))]
    return Trajectory(choices, best.trace, best.expected, best.state, final_rank((best, choices)),
        {"seconds": perf_counter() - started, "expanded": expanded, "roots": len(roots), "rollouts": rollouts,
         "horizon": horizon, "movement": best.moves, "logistics": best.logistics,
         "idle": best.idle, "hands": choices.hire_count, "value": final_rank((best, choices)),
         "unfulfilled_production": unfinished,
         "maintenance_debt": -value(best.state)[1]})


class IntradaySession:
    """Retain complete trajectories, repair own-state divergence, sell reactively."""
    def __init__(self):
        self.trajectories = {}
        self.diagnostics = []

    def execution_for(self, state, daily):
        entry = self.trajectories.get(state.player)
        trajectory, origin, identity = entry if entry else (None, -1, None)
        offset = state.step - origin
        valid_day = identity is daily and 0 <= offset < len(trajectory.executions) if trajectory else False
        def live_execution(trajectory, index):
            execution = trajectory.executions[index]
            orders = build_market_orders(state, daily, execution.worker_actions, trajectory.choices)
            return replace(execution, market_orders=orders)

        diverged = valid_day and farm_key(state) != trajectory.expected[offset]
        if valid_day and not diverged:
            execution = live_execution(trajectory, offset)
            predicted = rules.advance_owned(state, execution.worker_actions, execution.market_orders)
            expected = (trajectory.expected[offset + 1] if offset + 1 < len(trajectory.expected)
                        else farm_key(trajectory.final_state))
            # Changed trade prices/cash matter only when they change attainable
            # inputs or worker capacity. Repair before emitting the failed buy.
            diverged = farm_key(predicted) != expected
        if not valid_day or diverged:
            previous = trajectory.choices if valid_day else None
            trajectory = search_day(state, daily, previous=previous)
            origin, offset = state.step, 0
            self.trajectories[state.player] = (trajectory, origin, daily)
            self.diagnostics.append({"step": state.step, "reason": "divergence" if valid_day else "daily", **trajectory.diagnostics})
        return live_execution(trajectory, offset)
