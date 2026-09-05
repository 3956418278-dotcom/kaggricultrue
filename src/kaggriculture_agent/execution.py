"""Turn commitments into inspectable work and legal Kaggriculture actions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

from . import rules
from .planner import Plan
from .state import OwnedState, Position, WorkerState


@dataclass(frozen=True)
class WorkTask:
    identifier: str
    kind: str
    action: tuple[object, ...]
    priority: int
    deadline_step: int
    target: Position | None = None
    at_shed: bool = False
    required_item: str | None = None
    shared_item: str | None = None
    eligible_worker: int | None = None
    dependency: str | None = None
    capacity: int = 1
    followup_actions: int = 0


@dataclass(frozen=True)
class Execution:
    worker_actions: tuple[list[object], ...]
    tasks: tuple[WorkTask, ...]
    assignments: Mapping[int, str]
    market_orders: tuple[list[object], ...]
    defer_sales: bool = False
    benchmark_orders: bool = False


def _task_target(task: WorkTask, worker: WorkerState, board_size: int) -> Position:
    if task.at_shed:
        return min(
            rules.shed_access(board_size),
            key=lambda position: (rules.manhattan(worker.position, position), position[1], position[0]),
        )
    return worker.position if task.target is None else task.target


def _inventory_drop_tasks(state: OwnedState, terminal: bool) -> list[WorkTask]:
    tasks: list[WorkTask] = []
    for worker in state.workers:
        if not worker.carried:
            continue
        # Daily refresh drops every inventory into the shed and returns the
        # farmer to spawn. Do not spend late-day travel merely to preserve carry;
        # ordinary drops exist to expose a useful batch for intraday sale.
        if terminal or worker.carried >= 3:
            tasks.append(
                WorkTask(
                    f"drop:{worker.index}", "DROP", ("DROP",), 1 if terminal else 32,
                    rules.TERMINAL_ACTION_STEP if terminal else (state.day + 1) * 24 - 1,
                    at_shed=True, eligible_worker=worker.index,
                )
            )
    return tasks


def generate_tasks(state: OwnedState, plan: Plan) -> tuple[WorkTask, ...]:
    """Generate current concrete work; future work stays in commitments."""
    from .realization import Realization, bind_plan
    if not isinstance(plan, Realization):
        plan = bind_plan(state, plan)
    terminal = state.step >= 707
    tasks = _inventory_drop_tasks(state, terminal)
    end_of_day = min(rules.TERMINAL_ACTION_STEP, (state.day + 1) * 24 - 1)

    if terminal:
        for tile in state.crop_tiles():
            raw = tile.raw
            quantity = int(raw.get("yield_units", 0) or 0)
            rule = rules.CROPS[str(raw["crop"])]
            age = state.day - int(raw.get("planted_day", state.day))
            if quantity <= 0 or age < rule.first_yield_day:
                continue
            water_gain = rules.one_time_water_gain(
                str(raw["crop"]),
                planted_day=int(raw.get("planted_day", state.day)),
                day=state.day,
                yield_units=quantity,
                fertilized_until_day=int(raw.get("fertilized_until_day", -1)),
                watered_today=bool(raw.get("watered_today", False)),
            )
            return_actions = rules.distance_to_shed(tile.position) + 1
            if water_gain:
                tasks.append(
                    WorkTask(
                        f"terminal-water:{tile.position}",
                        "WATER",
                        ("WATER",),
                        2,
                        rules.TERMINAL_ACTION_STEP,
                        tile.position,
                        followup_actions=1 + return_actions,
                    )
                )
            else:
                tasks.append(
                    WorkTask(
                        f"terminal-harvest:{tile.position}",
                        "HARVEST",
                        ("HARVEST",),
                        3,
                        rules.TERMINAL_ACTION_STEP,
                        tile.position,
                        followup_actions=return_actions,
                    )
                )
        for tile in state.animal_tiles():
            if int(tile.raw.get("yield_units", 0) or 0) <= 0:
                continue
            tasks.append(
                WorkTask(
                    f"terminal-animal-harvest:{tile.position}",
                    "HARVEST",
                    ("HARVEST",),
                    3,
                    rules.TERMINAL_ACTION_STEP,
                    tile.position,
                    followup_actions=rules.distance_to_shed(tile.position) + 1,
                )
            )

    if not terminal:
        for tile in state.crop_tiles():
            raw = tile.raw
            rule = rules.CROPS[str(raw["crop"])]
            first_day = int(raw.get("planted_day", state.day)) + rule.first_yield_day
            age = state.day - int(raw.get("planted_day", state.day))
            harvest_ready = rule.ongoing and state.day >= first_day
            final_water_gain = rules.one_time_water_gain(
                str(raw["crop"]),
                planted_day=int(raw.get("planted_day", state.day)),
                day=state.day,
                yield_units=int(raw.get("yield_units", 0) or 0),
                fertilized_until_day=int(raw.get("fertilized_until_day", -1)),
                watered_today=bool(raw.get("watered_today", False)),
            )
            harvest_ready = harvest_ready or (
                not rule.ongoing
                and age >= rule.max_yield_day
                and final_water_gain == 0
            )
            if int(raw.get("yield_units", 0) or 0) > 0 and harvest_ready:
                tasks.append(WorkTask(f"harvest:{tile.position}", "HARVEST", ("HARVEST",), 7, end_of_day, tile.position))
            awaiting_fertilizer = (
                tile.position in plan.fertilize_targets
                and int(raw.get("fertilized_until_day", -1)) < state.day + 2
            )
            if not bool(raw.get("watered_today", False)) and not awaiting_fertilizer:
                tasks.append(WorkTask(f"water:{tile.position}", "WATER", ("WATER",), 12, end_of_day, tile.position))
            if tile.position in plan.fertilize_targets and int(raw.get("fertilized_until_day", -1)) < state.day + 2:
                tasks.append(
                    WorkTask(
                        f"fertilize:{tile.position}", "FERTILIZE", ("FERTILIZE",), 10,
                        end_of_day, tile.position, required_item="FERTILIZER",
                        dependency="fertilizer-in-worker-inventory",
                    )
                )

        feed_need = 0
        for tile in state.animal_tiles():
            raw = tile.raw
            if int(raw.get("yield_units", 0) or 0) > 0:
                tasks.append(WorkTask(f"animal-harvest:{tile.position}", "HARVEST", ("HARVEST",), 6, end_of_day, tile.position))
            if not bool(raw.get("fed_today", False)):
                feed_need += 1
                tasks.append(
                    WorkTask(
                        f"feed:{tile.position}", "FEED", ("FEED",), 8, end_of_day,
                        tile.position, required_item="WHEAT", dependency="wheat-in-worker-inventory",
                    )
                )
            if not bool(raw.get("cared_today", False)):
                tasks.append(WorkTask(f"care:{tile.position}", "CARE", ("CARE",), 15, end_of_day, tile.position))
            if bool(raw.get("fertilizer_available", False)):
                tasks.append(WorkTask(f"collect:{tile.position}", "COLLECT_FERTILIZER", ("COLLECT_FERTILIZER",), 20, end_of_day, tile.position))

        carried_wheat = state.carried_total("WHEAT")
        if feed_need > carried_wheat and state.shed.get("WHEAT", 0) > 0:
            quantity = min(feed_need - carried_wheat, state.shed.get("WHEAT", 0), 4)
            tasks.append(WorkTask("pickup:wheat", "PICKUP", ("PICKUP", "WHEAT", quantity), 5, end_of_day, at_shed=True))
        carried_fertilizer = state.carried_total("FERTILIZER")
        if len(plan.fertilize_targets) > carried_fertilizer and state.shed.get("FERTILIZER", 0) > 0:
            quantity = min(len(plan.fertilize_targets) - carried_fertilizer, state.shed.get("FERTILIZER", 0))
            tasks.append(WorkTask("pickup:fertilizer", "PICKUP", ("PICKUP", "FERTILIZER", quantity), 9, end_of_day, at_shed=True))

        for animal in rules.ANIMALS:
            if state.owned_total(animal) <= 0:
                continue
            structure = rules.ANIMALS[animal].structure
            empties = state.empty_structures(structure)
            carried = [worker for worker in state.workers if worker.inventory.get(animal, 0) > 0]
            for index, worker in enumerate(carried[: len(empties)]):
                target = empties[index].position
                tasks.append(
                    WorkTask(
                        f"place:{animal}:{target}", "PLACE", ("PLACE", animal), 4,
                        end_of_day, target, required_item=animal, eligible_worker=worker.index,
                        dependency=f"structure:{target}",
                    )
                )
            if empties and state.shed.get(animal, 0) > 0:
                tasks.append(
                    WorkTask(
                        f"pickup:{animal}", "PICKUP", ("PICKUP", animal, 1), 3,
                        end_of_day, at_shed=True, dependency=f"structure:{empties[0].position}",
                    )
                )

        build_requests: dict[Position, str] = {}
        for project in (*plan.obligations, *plan.selected):
            if project.kind not in ("ANIMAL", "ANIMAL_PLACEMENT") or project.target is None:
                continue
            if state.tile_at(project.target).is_empty:
                build_requests[project.target] = str(project.metadata["structure"])
        for target, structure in sorted(build_requests.items()):
            operation = "BUILD_COOP" if structure == "COOP" else "BUILD_PASTURE"
            tasks.append(WorkTask(f"build:{target}", "BUILD", (operation,), 25, end_of_day, target))

        for target, crop in sorted(plan.crop_targets.items()):
            if state.tile_at(target).is_empty and state.seeds.get(crop, 0) > 0:
                tasks.append(
                    WorkTask(
                        f"plant:{crop}:{target}", "PLANT", ("PLANT", crop), 24,
                        end_of_day, target, shared_item=f"{crop}_SEED",
                    )
                )
        for tile in state.tiles_of_kind("WEED"):
            tasks.append(WorkTask(f"dig:{tile.position}", "DIG", ("DIG",), 70, rules.TERMINAL_ACTION_STEP, tile.position))

    return tuple(sorted(tasks, key=lambda task: (task.priority, task.deadline_step, task.identifier)))


def _worker_can_do(worker: WorkerState, task: WorkTask) -> bool:
    return not (
        (task.eligible_worker is not None and task.eligible_worker != worker.index)
        or (task.required_item is not None and worker.inventory.get(task.required_item, 0) <= 0)
    )


def schedule(state: OwnedState, tasks: tuple[WorkTask, ...]) -> tuple[tuple[list[object], ...], dict[int, str]]:
    """Greedy deadline scheduler with deterministic travel and resource safety."""
    remaining = list(tasks)
    actions: list[list[object]] = [["PASS"] for _ in state.workers]
    assignments: dict[int, str] = {}
    emitted_resources: dict[str, int] = {}
    unassigned = set(range(len(state.workers)))
    while unassigned and remaining:
        choices: list[tuple[tuple[object, ...], int, int, Position]] = []
        for worker_index in sorted(unassigned):
            worker = state.workers[worker_index]
            for task_index, task in enumerate(remaining):
                if not _worker_can_do(worker, task):
                    continue
                if task.shared_item:
                    crop = task.shared_item.removesuffix("_SEED")
                    if emitted_resources.get(task.shared_item, 0) >= state.seeds.get(crop, 0):
                        continue
                target = _task_target(task, worker, state.board_size)
                distance = rules.manhattan(worker.position, target)
                if (
                    state.step >= 707
                    and distance + 1 + task.followup_actions > state.turns_left
                ):
                    continue
                continuity = (-2 if worker.position == target else 0) + (-1 if task.required_item else 0)
                # Deadline is ordered before route length.  Folding distance into
                # slack would perversely favor farther work with the same deadline.
                score = (task.priority, task.deadline_step, distance + continuity, task.identifier, worker.index)
                choices.append((score, worker_index, task_index, target))
        if not choices:
            break
        _, worker_index, task_index, target = min(choices)
        worker = state.workers[worker_index]
        task = remaining.pop(task_index)
        if worker.position == target:
            action = list(task.action)
            if task.shared_item:
                emitted_resources[task.shared_item] = emitted_resources.get(task.shared_item, 0) + 1
        else:
            action = rules.move_toward(worker.position, target)
        actions[worker_index] = action
        assignments[worker_index] = task.identifier
        unassigned.remove(worker_index)
    return tuple(actions), assignments


def build_market_orders(state: OwnedState, plan: Plan, worker_actions: tuple[list[object], ...]) -> tuple[list[object], ...]:
    """Construct ordered sales and purchases using post-unit-action inventory."""
    from .realization import Realization, bind_plan
    if not isinstance(plan, Realization):
        plan = bind_plan(state, plan)
    projected_shed = dict(state.shed)
    used_carried: dict[str, int] = {}
    for worker, action in zip(state.workers, worker_actions):
        op = action[0] if action else "PASS"
        if op == "DROP":
            for item, quantity in worker.inventory.items():
                projected_shed[item] = projected_shed.get(item, 0) + quantity
        elif op == "PLACE" and len(action) > 1:
            item = str(action[1])
            used_carried[item] = used_carried.get(item, 0) + 1
        elif op == "FEED":
            used_carried["WHEAT"] = used_carried.get("WHEAT", 0) + 1
        elif op == "FERTILIZE":
            used_carried["FERTILIZER"] = used_carried.get("FERTILIZER", 0) + 1
    terminal = state.step >= 707
    pending_fertilizer = (
        sum(
            state.tile_at(target).kind == "PLANT"
            and int(state.tile_at(target).raw.get("fertilized_until_day", -1)) < state.day + 2
            for target in plan.fertilize_targets
        )
        if plan.fertilize_targets
        else plan.fertilizer_reserve
    )
    reserves = {
        "WHEAT": 0 if terminal else plan.feed_reserve,
        "FERTILIZER": 0 if terminal else pending_fertilizer,
    }
    sales: list[list[object]] = []
    for item in sorted(rules.SELLABLE_PRODUCTS, key=lambda name: (-state.market_prices.get(name, 0), name)):
        quantity = max(0, projected_shed.get(item, 0) - reserves.get(item, 0))
        if quantity:
            sales.append(["SELL", item, quantity])
    if terminal:
        return tuple(sales[: rules.MAX_MARKET_ORDERS])

    commitment_orders: list[list[object]] = []
    for crop in rules.CROPS:
        remaining_targets = sum(
            planned_crop == crop and state.tile_at(target).is_empty
            for target, planned_crop in plan.crop_targets.items()
        )
        shortfall = max(0, remaining_targets - state.seeds.get(crop, 0))
        if shortfall:
            commitment_orders.append(["BUY_SEED", crop, shortfall])
    animal_counts = Counter(plan.animal_purchases)
    for animal in rules.ANIMALS:
        acquired_today = max(
            0,
            state.owned_animals(animal) - plan.starting_animals.get(animal, 0),
        )
        shortfall = max(0, animal_counts[animal] - acquired_today)
        if shortfall:
            commitment_orders.append(
                ["BUY_ANIMAL", animal, shortfall]
            )
    fertilizer_after_actions = max(0, state.owned_total("FERTILIZER") - used_carried.get("FERTILIZER", 0))
    fertilizer_shortfall = max(0, pending_fertilizer - fertilizer_after_actions)
    if fertilizer_shortfall:
        commitment_orders.append(
            ["BUY_PRODUCT", "FERTILIZER", fertilizer_shortfall]
        )
    wheat_after_actions = max(0, state.owned_total("WHEAT") - used_carried.get("WHEAT", 0))
    wheat_shortfall = max(0, plan.feed_reserve - wheat_after_actions)
    if wheat_shortfall:
        commitment_orders.append(["BUY_PRODUCT", "WHEAT", wheat_shortfall])
    land_target = next(
        (
            str(commitment.metadata.get("quadrant"))
            for commitment in plan.support
            if commitment.kind == "LAND"
        ),
        None,
    )
    if plan.buy_land and land_target not in state.unlocked_quadrants:
        commitment_orders.append(["BUY_LAND"])

    # Reserve the finite entry budget for commitment inputs.  Sales still execute
    # first (and can finance later entries), but only occupy genuinely spare slots.
    if len(commitment_orders) > rules.MAX_MARKET_ORDERS:
        raise ValueError("planner commitments exceed the market-entry budget")
    hire_slots = max(
        0, rules.MAX_MARKET_ORDERS - len(commitment_orders)
    )
    outstanding_hires = max(0, plan.hire_count - state.hires_today)
    hire_orders = [["HIRE"] for _ in range(min(outstanding_hires, hire_slots))]
    required_orders = [*hire_orders, *commitment_orders]
    sale_slots = rules.MAX_MARKET_ORDERS - len(required_orders)
    return tuple([*sales[:sale_slots], *required_orders])


def execute(state: OwnedState, plan: Plan) -> Execution:
    from .realization import Realization, bind_plan
    if not isinstance(plan, Realization):
        plan = bind_plan(state, plan)
    tasks = generate_tasks(state, plan)
    worker_actions, assignments = schedule(state, tasks)
    market_orders = build_market_orders(state, plan, worker_actions)
    return Execution(worker_actions, tasks, assignments, market_orders)
