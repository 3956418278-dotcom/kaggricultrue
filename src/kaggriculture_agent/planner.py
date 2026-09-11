"""Deterministic day-level rolling-horizon economic planner.

The planner filters projects through cash, horizon, land, storage, and dated labor
constraints.  Among feasible projects it uses transparent derived dominance keys;
there is no weighted utility that erases the six source dimensions. ``make_plan``
forms one operating plan from a day's opening state; the planning session owns
when that plan may be replaced.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping

from . import rules
from .economics import (
    ActionDimension,
    CashDimension,
    EconomicCommitment,
    LandDimension,
    OccupancyInterval,
    PhysicalDimension,
    RevenueDimension,
    TimeDimension,
    TimedAmount,
    TimedCash,
    WorkAmount,
)
from .state import OwnedState, Position, TileState
from .programme import (
    BindingConstraint,
    PrebuildCaps,
    ThirdLandFacts,
    animal_addition_cap,
    prebuild_count,
    programme_pace,
    programme_priority,
    projected_next_day_cash,
    third_land_allowed,
    worker_limit,
)


@dataclass(frozen=True)
class PlannerConfig:
    cash_reserve: int = 150
    max_daily_hands: int = 5
    latest_plant_hour: int = 18
    latest_hire_hour: int = 2
    working_shed_limit: int = 82
    terminal_liquidation_step: int = 710
    max_intraday_replans: int = 1


@dataclass(frozen=True)
class EconomicWindow:
    """A rare Plan-owned economic timing requirement within the current day.

    Turns are day-relative (0..23).  The listed orders must execute together on
    one legal turn in the inclusive window, after worker actions for that turn.
    Shed and cash minima are checked immediately before those orders.  Extra
    hands merely widen the Plan-authorized staffing cap; they are not required.
    """

    start_turn: int
    end_turn: int
    market_orders: tuple[tuple[object, ...], ...] = ()
    required_shed: Mapping[str, int] = field(default_factory=dict)
    minimum_cash: int = 0
    extra_hands_allowed: int = 0


@dataclass(frozen=True)
class Plan:
    obligations: tuple[EconomicCommitment, ...]
    selected: tuple[EconomicCommitment, ...]
    support: tuple[EconomicCommitment, ...]
    rejected: Mapping[str, str]
    fertilize_targets: frozenset[Position]
    animal_purchases: tuple[str, ...]
    buy_land: bool
    feed_reserve: int
    fertilizer_reserve: int
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    starting_animals: Mapping[str, int] = field(default_factory=dict)
    day: int = -1
    formed_step: int = -1
    revision: int = 0
    replan_reason: str | None = None
    max_hands: int | None = 5  # None: constrained only by cash and remaining time
    economic_windows: tuple[EconomicWindow, ...] = ()


@dataclass(frozen=True)
class FertilizerOpportunity:
    position: Position
    outputs: tuple[TimedAmount, ...]
    projected_gross: int
    input_cost: int
    from_stock: bool


def _sale_step(state: OwnedState, output_step: int) -> int:
    return min(rules.TERMINAL_ACTION_STEP, output_step + 4)


def _revenue_for_outputs(
    state: OwnedState,
    outputs: Iterable[TimedAmount],
    prior_sales: Mapping[str, int] | None = None,
) -> int:
    prior = dict(prior_sales or {})
    gross = 0
    for output in sorted(outputs, key=lambda flow: (flow.step, flow.item)):
        quantity = max(0, output.quantity)
        gross += rules.projected_sale_revenue(
            output.item,
            quantity,
            state.market_inventory.get(output.item, rules.MARKET_I0),
            _sale_step(state, output.step),
            state.step,
            state.unlocked_shops,
            prior.get(output.item, 0),
        )
        prior[output.item] = prior.get(output.item, 0) + quantity
    return gross


def _crop_output_schedule(state: OwnedState, crop: str) -> tuple[TimedAmount, ...]:
    rule = rules.CROPS[crop]
    if rule.ongoing:
        return tuple(
            TimedAmount(
                step=min(
                    rules.TERMINAL_ACTION_STEP,
                    (state.day + rule.first_yield_day + index * rule.interval)
                    * rules.TURNS_PER_DAY,
                ),
                item=crop,
                quantity=1,
            )
            for index in range(rule.max_yield)
            if (state.day + rule.first_yield_day + index * rule.interval)
            * rules.TURNS_PER_DAY
            <= rules.TERMINAL_ACTION_STEP
        )
    expected_yield = {"WHEAT": 4, "CARROT": 3, "MELON": 6}[crop]
    harvest_day = state.day + rule.max_yield_day
    output_step = harvest_day * rules.TURNS_PER_DAY
    if output_step > rules.TERMINAL_ACTION_STEP:
        return ()
    return (TimedAmount(output_step, crop, expected_yield),)


def _crop_commitment(state: OwnedState, crop: str, tile: TileState) -> EconomicCommitment:
    rule = rules.CROPS[crop]
    outputs = _crop_output_schedule(state, crop)
    completion = max((output.step for output in outputs), default=rules.TERMINAL_ACTION_STEP + 1)
    active_days = max(0, completion // rules.TURNS_PER_DAY - state.day + 1)
    harvest_actions = len(outputs) if rule.ongoing else (1 if outputs else 0)
    distance = rules.distance_to_shed(tile.position, state.board_size)
    # Simple travel approximation: one shed round trip plus one local reposition
    # per four service days.  The dated service actions remain separately visible.
    travel = 2 * distance + max(0, active_days - 1) // 4
    work = [
        WorkAmount(
            state.day,
            "PLANT",
            1,
            travel_actions=travel,
            position=tile.position,
            deadline_step=(state.day + 1) * rules.TURNS_PER_DAY - 1,
        )
    ]
    for offset in range(active_days):
        work.append(
            WorkAmount(
                state.day + offset,
                "WATER",
                1,
                position=tile.position,
                deadline_step=(state.day + offset + 1) * rules.TURNS_PER_DAY - 1,
            )
        )
    for output in outputs:
        work.append(
            WorkAmount(
                output.step // rules.TURNS_PER_DAY,
                "HARVEST",
                1,
                position=tile.position,
                deadline_step=min(rules.TERMINAL_ACTION_STEP, output.step + 23),
            )
        )
    gross = _revenue_for_outputs(state, outputs)
    return EconomicCommitment(
        identifier=f"crop:{crop}:{tile.position[0]}:{tile.position[1]}",
        kind="CROP",
        target=tile.position,
        existing=False,
        cash=CashDimension(upfront=rule.seed_cost, reserve_required=0),
        time=TimeDimension(
            start_step=state.step,
            completion_step=completion,
            last_value_step=min(rules.TERMINAL_ACTION_STEP, completion + 23),
            deadlines=tuple(
                amount.deadline_step
                for amount in work
                if amount.deadline_step is not None
            ),
        ),
        land=LandDimension(
            (OccupancyInterval(tile.position, state.step, completion),)
        ),
        actions=ActionDimension(tuple(work)),
        physical=PhysicalDimension(
            inputs=(TimedAmount(state.step, f"{crop}_SEED", 1),),
            outputs=outputs,
            peak_shed_units=max((output.quantity for output in outputs), default=0),
        ),
        revenue=RevenueDimension(
            projected_sales=outputs,
            projected_gross=gross,
            terminal_unsold_units=0,
        ),
        metadata={"crop": crop},
    )


def _project_animal_outputs(
    state: OwnedState,
    animal: str,
    placed_day: int,
    pending_care_bonus: int = 0,
) -> tuple[TimedAmount, ...]:
    """Project exact production under the planner's daily feed/care promise.

    The engine produces one base unit.  On a fed production refresh it consumes
    every care bonus accumulated since the prior production, then today's care
    starts the next bonus accumulation.  Harvests are assumed prompt enough that
    the per-tile held-yield cap applies to each projected event independently.
    """
    rule = rules.ANIMALS[animal]
    outputs: list[TimedAmount] = []
    pending = max(0, pending_care_bonus)
    for service_day in range(state.day, 29):
        production_day = service_day + 1
        days_since_first = production_day - placed_day - rule.first_yield_day
        if days_since_first >= 0 and days_since_first % rule.interval == 0:
            quantity = min(rule.max_held, 1 + pending)
            outputs.append(
                TimedAmount(production_day * rules.TURNS_PER_DAY, rule.product, quantity)
            )
            pending = 0
        pending += 1
    return tuple(outputs)


def _animal_commitment(
    state: OwnedState,
    animal: str,
    target: Position,
    *,
    existing: bool,
    placed_day: int | None = None,
    current_yield: int = 0,
    fertilizer_available: bool = False,
    needs_structure: bool = False,
    pending_care_bonus: int = 0,
) -> EconomicCommitment:
    rule = rules.ANIMALS[animal]
    placed_day = state.day if placed_day is None else placed_day
    future_outputs = _project_animal_outputs(
        state,
        animal,
        placed_day,
        pending_care_bonus=pending_care_bonus,
    )
    existing_outputs = (
        (TimedAmount(state.step, rule.product, current_yield),)
        if existing and current_yield > 0
        else ()
    )
    outputs = (*existing_outputs, *future_outputs)
    # Servicing day 29 has no refresh before the step-718/719 terminal boundary.
    end_day = 28
    service_days = max(0, end_day - state.day + 1)
    feed_price = max(1, state.market_prices.get("WHEAT", rules.MARKET["WHEAT"].base))
    feed_schedule = tuple(
        TimedCash(day * rules.TURNS_PER_DAY, "one wheat feed", feed_price)
        for day in range(state.day, end_day + 1)
    )
    work: list[WorkAmount] = []
    if not existing:
        travel = 2 * rules.distance_to_shed(target, state.board_size)
        if needs_structure:
            work.append(WorkAmount(state.day, "BUILD", 1, travel, target))
        work.append(WorkAmount(state.day, "PICKUP_PLACE", 2, travel, target))
    for day in range(state.day, end_day + 1):
        work.extend(
            (
                WorkAmount(day, "FEED", 1, position=target, deadline_step=(day + 1) * 24 - 1),
                WorkAmount(day, "CARE", 1, position=target, deadline_step=(day + 1) * 24 - 1),
                WorkAmount(day, "COLLECT_FERTILIZER", 1, position=target),
            )
        )
    for output in outputs:
        work.append(WorkAmount(output.step // 24, "HARVEST", 1, position=target))
    fertilizer_outputs = (
        ((TimedAmount(state.step, "FERTILIZER", 1),) if existing and fertilizer_available else ())
        + tuple(
            TimedAmount(day * 24, "FERTILIZER", 1)
            for day in range(state.day + 1, 30)
            if day * 24 <= rules.TERMINAL_ACTION_STEP
        )
    )
    all_outputs = (*outputs, *fertilizer_outputs)
    gross = _revenue_for_outputs(state, all_outputs)
    purchase = 0 if existing else rule.cost
    return EconomicCommitment(
        identifier=("maintain" if existing else "animal")
        + f":{animal}:{target[0]}:{target[1]}",
        kind="ANIMAL_MAINTENANCE" if existing else "ANIMAL",
        target=target,
        existing=existing,
        cash=CashDimension(
            upfront=purchase,
            scheduled=feed_schedule,
            reserve_required=feed_price,
            sunk_cost=rule.cost if existing else 0,
        ),
        time=TimeDimension(
            state.step,
            max((item.step for item in outputs), default=state.step),
            rules.TERMINAL_ACTION_STEP,
        ),
        land=LandDimension(
            (OccupancyInterval(target, state.step, rules.TERMINAL_ACTION_STEP),)
        ),
        actions=ActionDimension(tuple(work)),
        physical=PhysicalDimension(
            inputs=(TimedAmount(state.step, "WHEAT", service_days),),
            outputs=all_outputs,
            peak_shed_units=min(rule.max_held, 2) + 1,
        ),
        revenue=RevenueDimension(projected_sales=all_outputs, projected_gross=gross),
        metadata={"animal": animal, "structure": rule.structure},
    )


def _existing_crop_obligation(state: OwnedState, tile: TileState) -> EconomicCommitment:
    raw = tile.raw
    crop = str(raw["crop"])
    rule = rules.CROPS[crop]
    age = state.day - int(raw.get("planted_day", state.day))
    current_yield = max(0, int(raw.get("yield_units", 0) or 0))
    if rule.ongoing:
        remaining_events = max(0, rule.max_yield - max(0, (age - rule.first_yield_day) // max(1, rule.interval) + 1))
        output_qty = current_yield + remaining_events
        completion_day = min(29, state.day + remaining_events * max(1, rule.interval))
    else:
        planted_day = int(raw.get("planted_day", state.day))
        completion_day = min(29, planted_day + rule.max_yield_day)
        window_start = planted_day + (rule.max_yield_day + 1) // 2
        possible_bonus = 0
        for day in range(max(state.day, window_start), completion_day + 1):
            if day == state.day and bool(raw.get("watered_today", False)):
                continue
            possible_bonus += 2 if int(raw.get("fertilized_until_day", -1)) >= day else 1
        output_qty = min(rule.max_yield, max(current_yield, 1) + possible_bonus)
    output_step = min(rules.TERMINAL_ACTION_STEP, completion_day * 24)
    outputs = (TimedAmount(output_step, crop, output_qty),) if output_qty else ()
    remaining_days = max(1, completion_day - state.day + 1)
    work = tuple(
        WorkAmount(day, "WATER", 1, position=tile.position, deadline_step=(day + 1) * 24 - 1)
        for day in range(state.day, completion_day + 1)
    ) + (WorkAmount(completion_day, "HARVEST", 1, position=tile.position),)
    return EconomicCommitment(
        identifier=f"existing-crop:{crop}:{tile.position[0]}:{tile.position[1]}",
        kind="CROP_MAINTENANCE",
        target=tile.position,
        existing=True,
        cash=CashDimension(sunk_cost=rule.seed_cost),
        time=TimeDimension(state.step, output_step, min(rules.TERMINAL_ACTION_STEP, output_step + 23)),
        land=LandDimension((OccupancyInterval(tile.position, state.step, output_step),)),
        actions=ActionDimension(work),
        physical=PhysicalDimension(outputs=outputs, peak_shed_units=output_qty),
        revenue=RevenueDimension(projected_sales=outputs, projected_gross=_revenue_for_outputs(state, outputs)),
        metadata={"crop": crop, "remaining_days": remaining_days},
    )


def existing_obligations(state: OwnedState) -> tuple[EconomicCommitment, ...]:
    obligations: list[EconomicCommitment] = []
    for tile in state.tiles:
        if tile.kind == "PLANT":
            obligations.append(_existing_crop_obligation(state, tile))
        elif tile.animal:
            obligations.append(
                _animal_commitment(
                    state,
                    tile.animal,
                    tile.position,
                    existing=True,
                    placed_day=int(tile.raw.get("placed_day", state.day)),
                    current_yield=int(tile.raw.get("yield_units", 0) or 0),
                    fertilizer_available=bool(tile.raw.get("fertilizer_available", False)),
                    pending_care_bonus=int(tile.raw.get("pending_care_bonus", 0) or 0),
                )
            )
        elif tile.kind == "WEED":
            obligations.append(
                EconomicCommitment(
                    identifier=f"weed:{tile.position[0]}:{tile.position[1]}",
                    kind="RECOVERY",
                    target=tile.position,
                    existing=True,
                    cash=CashDimension(),
                    time=TimeDimension(state.step, state.step, rules.TERMINAL_ACTION_STEP),
                    land=LandDimension(),
                    actions=ActionDimension((WorkAmount(state.day, "DIG", 1, position=tile.position),)),
                    physical=PhysicalDimension(),
                    revenue=RevenueDimension(),
                )
            )
    # Purchased animals are already-paid physical commitments.  Their purchase
    # price is sunk; only placement, service, and later proceeds remain relevant.
    pending_animals = {
        animal: state.owned_total(animal)
        for animal in rules.ANIMALS
        if state.owned_total(animal) > 0
    }
    targets = list(_candidate_targets(state))
    for animal, quantity in pending_animals.items():
        structure = rules.ANIMALS[animal].structure
        structure_targets = list(state.empty_structures(structure))
        for index in range(quantity):
            target_tile = (
                structure_targets[index]
                if index < len(structure_targets)
                else targets[min(index, len(targets) - 1)] if targets else None
            )
            if target_tile is None:
                continue
            commitment = _animal_commitment(
                state, animal, target_tile.position, existing=True, placed_day=state.day
            )
            travel = 2 * rules.distance_to_shed(target_tile.position, state.board_size)
            setup = []
            if target_tile.is_empty:
                setup.append(WorkAmount(state.day, "BUILD", 1, travel, target_tile.position))
            setup.append(WorkAmount(state.day, "PICKUP_PLACE", 2, travel, target_tile.position))
            obligations.append(
                replace(
                    commitment,
                    identifier=f"place-existing:{animal}:{index}:{target_tile.position[0]}:{target_tile.position[1]}",
                    kind="ANIMAL_PLACEMENT",
                    actions=ActionDimension((*setup, *commitment.actions.work)),
                    metadata={**commitment.metadata, "pending": True},
                )
            )
    return tuple(obligations)


def _candidate_targets(state: OwnedState) -> tuple[TileState, ...]:
    return tuple(
        sorted(
            state.empty_tiles(),
            key=lambda tile: (
                rules.distance_to_shed(tile.position, state.board_size),
                tile.position[1],
                tile.position[0],
            ),
        )
    )


def enumerate_projects(state: OwnedState, config: PlannerConfig) -> tuple[EconomicCommitment, ...]:
    if state.step >= config.terminal_liquidation_step:
        return ()
    targets = _candidate_targets(state)
    candidates: list[EconomicCommitment] = []
    if state.hour <= config.latest_plant_hour:
        # A bounded frontier is enough: farther identical tiles are dominated by
        # these targets under the explicit Manhattan travel approximation.
        for tile in targets:
            for crop in rules.CROPS:
                project = _crop_commitment(state, crop, tile)
                if project.physical.outputs:
                    candidates.append(project)
    animal_targets: dict[str, Position] = {}
    for animal, animal_rule in rules.ANIMALS.items():
        empty_structures = state.empty_structures(animal_rule.structure)
        if empty_structures:
            animal_targets[animal] = empty_structures[0].position
        elif targets:
            animal_targets[animal] = targets[0].position
    # Finish placing anything already bought before committing cash to another
    # animal.  This prevents repeated market purchases while a worker is in transit.
    if not any(state.owned_total(animal) for animal in rules.ANIMALS):
        for animal, target in animal_targets.items():
            project = _animal_commitment(
                state,
                animal,
                target,
                existing=False,
                needs_structure=state.tile_at(target).is_empty,
            )
            if project.physical.outputs:
                candidates.append(project)
    return tuple(candidates)


def _reprice(state: OwnedState, project: EconomicCommitment, own_sales: Mapping[str, int]) -> EconomicCommitment:
    gross = _revenue_for_outputs(state, project.physical.outputs, own_sales)
    return replace(project, revenue=replace(project.revenue, projected_gross=gross))


def _daily_load(commitments: Iterable[EconomicCommitment]) -> dict[int, int]:
    result: dict[int, int] = {}
    for commitment in commitments:
        for work in commitment.actions.work:
            result[work.day] = result.get(work.day, 0) + work.actions + work.travel_actions
    return result


def _capacity(state: OwnedState, config: PlannerConfig, day: int) -> int:
    if day == state.day:
        affordable_hands = config.max_daily_hands if state.money >= 20 else 0
        workers = max(len(state.workers), 1 + affordable_hands)
        return workers * state.turns_left_today
    return rules.TURNS_PER_DAY * (1 + config.max_daily_hands)


def _selection_reason(
    state: OwnedState,
    config: PlannerConfig,
    project: EconomicCommitment,
    selected: list[EconomicCommitment],
    cash_spent: int,
    occupied: set[Position],
) -> str | None:
    if project.time.completion_step > rules.TERMINAL_ACTION_STEP:
        return "after terminal boundary"
    if project.terminal_profit <= 0:
        return "non-positive marginal terminal profit"
    reserve = config.cash_reserve + state.projected_feed_need_today() * max(
        1, state.market_prices.get("WHEAT", 25)
    )
    if cash_spent + project.cash.upfront > max(0, state.money - reserve):
        return "insufficient unreserved cash"
    if project.target is not None and project.target in occupied:
        return "land interval conflicts with selected project"
    peak = state.shed_used + sum(item.physical.peak_shed_units for item in selected)
    if peak + project.physical.peak_shed_units > config.working_shed_limit:
        return "projected working storage exceeds reserve limit"
    loads = _daily_load([*selected, project])
    for day, load in loads.items():
        if load > _capacity(state, config, day):
            return f"dated labor demand exceeds day-{day} capacity"
    return None


def _hire_target(
    state: OwnedState,
    commitments: tuple[EconomicCommitment, ...],
    config: PlannerConfig,
    additional_today_work: int = 0,
) -> int:
    """Return the target number of hands for this day, not an order count."""
    if state.hour > config.latest_hire_hour or state.step >= config.terminal_liquidation_step:
        return state.hires_today
    today_work = _daily_load(commitments).get(state.day, 0) + additional_today_work
    carried = sum(worker.carried for worker in state.workers)
    today_work += min(8, carried)
    # Staffing is a day-opening decision. Do not create extra demand for hands
    # merely because the same workload is observed one hour later during a
    # bounded repair; elapsed turns are handled by feasibility/local execution.
    capacity_per_worker = rules.TURNS_PER_DAY
    workers_needed = max(1, (today_work + capacity_per_worker - 1) // capacity_per_worker)
    current_hands = max(0, len(state.workers) - 1)
    available_slots = max(0, config.max_daily_hands - current_hands)
    additional = max(0, min(available_slots, workers_needed - len(state.workers)))
    affordable = 0
    cash = max(0, state.money - config.cash_reserve)
    for index in range(state.hires_today, state.hires_today + additional):
        cost = rules.fibonacci_hire_cost(index)
        if cost > cash:
            break
        cash -= cost
        affordable += 1
    return state.hires_today + affordable


def _fertilizer_marginal_outputs(
    state: OwnedState, tile: TileState
) -> tuple[TimedAmount, ...]:
    """Return only output units caused by fertilizing this crop now."""
    raw = tile.raw
    crop = str(raw["crop"])
    rule = rules.CROPS[crop]
    planted_day = int(raw.get("planted_day", state.day))
    existing_until = int(raw.get("fertilized_until_day", -1))
    treated_until = max(existing_until, state.day + 2)
    if rule.ongoing:
        outputs: list[TimedAmount] = []
        for service_day in range(state.day, min(28, state.day + 2) + 1):
            production_day = service_day + 1
            days_since_first = production_day - planted_day - rule.first_yield_day
            if days_since_first < 0 or days_since_first % rule.interval != 0:
                continue
            production_count = days_since_first // rule.interval + 1
            if production_count > rule.max_yield:
                continue
            if existing_until < service_day <= treated_until:
                outputs.append(
                    TimedAmount(production_day * rules.TURNS_PER_DAY, crop, 1)
                )
        return tuple(outputs)

    completion_day = min(29, planted_day + rule.max_yield_day)
    if completion_day < state.day:
        return ()
    baseline_yield = max(0, int(raw.get("yield_units", 0) or 0))
    treated_yield = baseline_yield
    for day in range(state.day, completion_day + 1):
        already_watered = day == state.day and bool(raw.get("watered_today", False))
        baseline_yield += rules.one_time_water_gain(
            crop,
            planted_day=planted_day,
            day=day,
            yield_units=baseline_yield,
            fertilized_until_day=existing_until,
            watered_today=already_watered,
        )
        treated_yield += rules.one_time_water_gain(
            crop,
            planted_day=planted_day,
            day=day,
            yield_units=treated_yield,
            fertilized_until_day=treated_until,
            watered_today=already_watered,
        )
    marginal = max(0, treated_yield - baseline_yield)
    if not marginal:
        return ()
    return (
        TimedAmount(
            min(rules.TERMINAL_ACTION_STEP, completion_day * rules.TURNS_PER_DAY),
            crop,
            marginal,
        ),
    )


def _fertilizer_opportunities(
    state: OwnedState,
) -> tuple[FertilizerOpportunity, ...]:
    candidates: list[tuple[int, int, int, Position, tuple[TimedAmount, ...]]] = []
    for tile in state.crop_tiles():
        outputs = _fertilizer_marginal_outputs(state, tile)
        if not outputs:
            continue
        gross = _revenue_for_outputs(state, outputs)
        candidates.append(
            (
                -gross,
                rules.distance_to_shed(tile.position),
                tile.position[1],
                tile.position,
                outputs,
            )
        )
    candidates.sort()
    owned = state.owned_total("FERTILIZER")
    selected: list[FertilizerOpportunity] = []
    purchased = 0
    for _, _, _, position, outputs in candidates:
        from_stock = len(selected) < owned
        carried_routes = [
            rules.manhattan(worker.position, position) + 1
            for worker in state.workers
            if worker.inventory.get("FERTILIZER", 0) > 0
        ]
        if from_stock and carried_routes:
            actions_to_apply = min(carried_routes)
        else:
            approach_shed = min(
                rules.distance_to_shed(worker.position) for worker in state.workers
            )
            if not from_stock:
                # A market purchase arrives after this turn's worker actions.
                approach_shed = max(1, approach_shed)
            actions_to_apply = (
                approach_shed + 1 + rules.distance_to_shed(position) + 1
            )
        raw = state.tile_at(position).raw
        if not bool(raw.get("watered_today", False)):
            # Execution deliberately applies fertilizer before the day's water so
            # the marginal schedule is physically realizable for one-time crops.
            actions_to_apply += 1
        if actions_to_apply > state.turns_left_today:
            continue
        if from_stock:
            input_cost = rules.projected_sale_revenue(
                "FERTILIZER",
                1,
                state.market_inventory.get("FERTILIZER", rules.MARKET_I0),
                state.step,
                state.step,
                state.unlocked_shops,
            )
        else:
            inventory = state.market_inventory.get("FERTILIZER", rules.MARKET_I0)
            input_cost = rules.market_price("FERTILIZER", inventory - purchased - 1)
        gross = _revenue_for_outputs(state, outputs)
        if gross <= input_cost:
            continue
        if not from_stock and state.money < input_cost:
            continue
        selected.append(
            FertilizerOpportunity(position, outputs, gross, input_cost, from_stock)
        )
        purchased += not from_stock
        if len(selected) == 2:
            break
    return tuple(selected)


def hiring_commitments(state: OwnedState, hire_target: int) -> tuple[EconomicCommitment, ...]:
    """Dated labor capacity: a market-hired hand first acts NEXT turn."""
    capacity = max(0, min(state.turns_left_today, state.turns_left) - 1)
    return tuple(EconomicCommitment(
        identifier=f"hire:{state.day}:{index}", kind="HIRE", target=None, existing=False,
        cash=CashDimension(upfront=rules.fibonacci_hire_cost(index)),
        time=TimeDimension(state.step, min(718, (state.day + 1) * 24 - 1), min(718, (state.day + 1) * 24 - 1)),
        land=LandDimension(), actions=ActionDimension(capacity_supplied={state.day: capacity}),
        physical=PhysicalDimension(outputs=(TimedAmount(state.step + 1, "WORKER_ACTION_CAPACITY", capacity),)),
        revenue=RevenueDimension(), metadata={"hire_index": index})
        for index in range(state.hires_today, hire_target))


def _support_commitments(
    state: OwnedState,
    fertilizer_opportunities: tuple[FertilizerOpportunity, ...],
    buy_land: bool,
) -> tuple[EconomicCommitment, ...]:
    support: list[EconomicCommitment] = []
    for opportunity in fertilizer_opportunities:
        position = opportunity.position
        crop = str(state.tile_at(position).raw["crop"])
        support.append(
            EconomicCommitment(
                identifier=f"fertilize:{position[0]}:{position[1]}",
                kind="FERTILIZER",
                target=position,
                existing=opportunity.from_stock,
                cash=CashDimension(
                    upfront=0 if opportunity.from_stock else opportunity.input_cost,
                    scheduled=(
                        TimedCash(
                            state.step,
                            "foregone fertilizer sale",
                            opportunity.input_cost,
                        ),
                    )
                    if opportunity.from_stock
                    else (),
                    sunk_cost=opportunity.input_cost if opportunity.from_stock else 0,
                ),
                time=TimeDimension(
                    state.step,
                    max(output.step for output in opportunity.outputs),
                    min(718, max(output.step for output in opportunity.outputs) + 23),
                ),
                land=LandDimension(
                    (OccupancyInterval(position, state.step, min(718, state.step + 71)),)
                ),
                actions=ActionDimension(
                    (WorkAmount(state.day, "FERTILIZE", 1, position=position),)
                ),
                physical=PhysicalDimension(
                    inputs=(TimedAmount(state.step, "FERTILIZER", 1),),
                    outputs=opportunity.outputs,
                ),
                revenue=RevenueDimension(
                    projected_sales=opportunity.outputs,
                    projected_gross=opportunity.projected_gross,
                ),
                metadata={
                    "crop": crop,
                    "from_stock": opportunity.from_stock,
                    "marginal_units": sum(
                        output.quantity for output in opportunity.outputs
                    ),
                    "input_cost": opportunity.input_cost,
                },
            )
        )
    if buy_land:
        quadrant = rules.LAND_ORDER[len(state.unlocked_quadrants) - 1]
        price = rules.LAND_PRICES[len(state.unlocked_quadrants) - 1]
        support.append(
            EconomicCommitment(
                identifier=f"land:{quadrant}",
                kind="LAND",
                target=None,
                existing=False,
                cash=CashDimension(upfront=price),
                time=TimeDimension(state.step, state.step, rules.TERMINAL_ACTION_STEP),
                land=LandDimension(capacity_created=25),
                actions=ActionDimension(),
                physical=PhysicalDimension(outputs=(TimedAmount(state.step, "LAND_TILE_CAPACITY", 25),)),
                revenue=RevenueDimension(),
                metadata={"quadrant": quadrant},
            )
        )
    inventory_outputs = tuple(
        TimedAmount(state.step, item, state.shed.get(item, 0) + state.carried_total(item))
        for item in rules.SELLABLE_PRODUCTS
        if state.shed.get(item, 0) + state.carried_total(item) > 0
    )
    tile_outputs: list[TimedAmount] = []
    tile_work: list[WorkAmount] = []
    for tile in state.tiles:
        raw = tile.raw
        quantity = int(raw.get("yield_units", 0) or 0) if isinstance(raw, Mapping) else 0
        if quantity <= 0:
            continue
        if tile.kind == "PLANT":
            crop = str(raw["crop"])
            rule = rules.CROPS[crop]
            age = state.day - int(raw.get("planted_day", state.day))
            if age < rule.first_yield_day:
                continue
            gain = rules.one_time_water_gain(
                crop,
                planted_day=int(raw.get("planted_day", state.day)),
                day=state.day,
                yield_units=quantity,
                fertilized_until_day=int(raw.get("fertilized_until_day", -1)),
                watered_today=bool(raw.get("watered_today", False)),
            )
            item = crop
            setup_actions = 1 if gain else 0
            quantity += gain
        elif tile.animal:
            item = rules.ANIMALS[tile.animal].product
            setup_actions = 0
        else:
            continue
        route_actions = min(
            rules.manhattan(worker.position, tile.position)
            for worker in state.workers
        ) + setup_actions + 1 + rules.distance_to_shed(tile.position) + 1
        if route_actions > state.turns_left:
            continue
        realization_step = min(
            rules.TERMINAL_ACTION_STEP, state.step + route_actions - 1
        )
        tile_outputs.append(TimedAmount(realization_step, item, quantity))
        tile_work.append(
            WorkAmount(
                state.day,
                "WATER_HARVEST_TRANSPORT" if setup_actions else "HARVEST_TRANSPORT",
                2 + setup_actions,
                travel_actions=route_actions - 2 - setup_actions,
                position=tile.position,
                deadline_step=realization_step,
            )
        )
    if tile_outputs and state.day == 29:
        for index, (output, work) in enumerate(zip(tile_outputs, tile_work)):
            support.append(EconomicCommitment(
                identifier=f"recover:{state.step}:{index}",
                kind="RECOVERY",
                target=work.position,
                existing=True,
                cash=CashDimension(),
                time=TimeDimension(state.step, rules.TERMINAL_ACTION_STEP, rules.TERMINAL_ACTION_STEP),
                land=LandDimension(),
                actions=ActionDimension((work,)),
                physical=PhysicalDimension(outputs=(output,)),
                revenue=RevenueDimension(
                    projected_sales=(output,),
                    projected_gross=_revenue_for_outputs(state, (output,)),
                ),
                metadata={
                    "inventory_is_sunk": True,
                    "recoverable_tile_units": output.quantity,
                    "item": output.item,
                },
            ))
    return tuple(support)


def _canonical_daily_commitment(
    state: OwnedState, project: EconomicCommitment
) -> EconomicCommitment:
    """Attach today's economic outcome without prescribing primitive actions."""

    if project.kind in ("LAND", "HIRE"):
        return project
    if project.target is None:
        raise ValueError(
            f"daily commitment {project.identifier!r} has no fixed placement"
        )
    raw = state.tile_at(project.target).raw
    required: dict[str, object] = {}
    outputs: dict[str, int] = {}

    if project.kind == "CROP":
        required = {
            "kind": "PLANT",
            "crop": str(project.metadata["crop"]),
            "watered_today": True,
        }
    elif project.kind == "CROP_MAINTENANCE":
        crop = str(project.metadata["crop"])
        harvest_today = project.time.completion_step // rules.TURNS_PER_DAY <= state.day
        if harvest_today and not rules.CROPS[crop].ongoing:
            required = {"$tile": None}
            if isinstance(raw, Mapping):
                opening_yield = max(0, int(raw.get("yield_units", 0) or 0))
                water_gain = rules.one_time_water_gain(
                    crop,
                    planted_day=int(raw.get("planted_day", state.day)),
                    day=state.day,
                    yield_units=opening_yield,
                    fertilized_until_day=int(raw.get("fertilized_until_day", -1)),
                    watered_today=bool(raw.get("watered_today", False)),
                )
                outputs[crop] = opening_yield + water_gain
        else:
            required = {"watered_today": True}
            if harvest_today:
                required["yield_units"] = 0
                if isinstance(raw, Mapping):
                    outputs[crop] = max(0, int(raw.get("yield_units", 0) or 0))
    elif project.kind in ("ANIMAL", "ANIMAL_PLACEMENT"):
        required = {
            "kind": str(project.metadata["structure"]),
            "animal": str(project.metadata["animal"]),
            "fed_today": True,
            "cared_today": True,
        }
    elif project.kind == "ANIMAL_MAINTENANCE":
        required = {"fed_today": True, "cared_today": True}
        if isinstance(raw, Mapping) and int(raw.get("yield_units", 0) or 0) > 0:
            required["yield_units"] = 0
            item = rules.ANIMALS[str(project.metadata["animal"])].product
            outputs[item] = int(raw["yield_units"])
        if isinstance(raw, Mapping) and raw.get("fertilizer_available", False):
            required["fertilizer_available"] = False
            outputs["FERTILIZER"] = 1
    elif project.kind == "FERTILIZER":
        required = {"fertilized_until_day": {"at_least": state.day + 2}}
    elif project.kind == "RECOVERY":
        if isinstance(raw, Mapping) and raw.get("kind") == "PLANT" and rules.CROPS[str(raw["crop"])].ongoing:
            required = {"yield_units": 0}
        elif isinstance(raw, Mapping) and "animal" in raw:
            required = {"yield_units": 0}
        else:
            required = {"$tile": None}
        item = project.metadata.get("item")
        quantity = int(project.metadata.get("recoverable_tile_units", 0) or 0)
        if item and quantity:
            outputs[str(item)] = quantity
    else:
        raise ValueError(
            f"daily commitment {project.identifier!r} has no outcome semantics"
        )
    return replace(
        project,
        required_state=required,
        required_outputs=outputs,
    )


def _programme_deadline(
    state: OwnedState, commitment: EconomicCommitment, turn: int = 20
) -> EconomicCommitment:
    deadline = min(
        rules.TERMINAL_ACTION_STEP,
        state.day * rules.TURNS_PER_DAY + turn,
    )
    return replace(
        commitment,
        time=replace(commitment.time, deadlines=(deadline,)),
    )


def _programme_maintenance(
    state: OwnedState,
) -> tuple[tuple[EconomicCommitment, ...], dict[str, int], dict[str, int], int]:
    """Build only D5-D10 mandatory outcomes and guaranteed turn-7 sales."""

    obligations: list[EconomicCommitment] = []
    guaranteed: dict[str, int] = {}
    replacement_seeds: dict[str, int] = {}
    service_actions = 0
    for tile in state.crop_tiles():
        raw = tile.raw
        crop = str(raw["crop"])
        rule = rules.CROPS[crop]
        age = state.day - int(raw.get("planted_day", state.day))
        current_yield = max(0, int(raw.get("yield_units", 0) or 0))
        base = _existing_crop_obligation(state, tile)
        if crop == "WHEAT" and age >= rule.max_yield_day:
            commitment = _programme_crop_project(state, "WHEAT", tile)
            quantity = current_yield + rules.one_time_water_gain(
                crop,
                planted_day=int(raw.get("planted_day", state.day)),
                day=state.day,
                yield_units=current_yield,
                fertilized_until_day=int(raw.get("fertilized_until_day", -1)),
                watered_today=bool(raw.get("watered_today", False)),
            )
            commitment = replace(
                commitment,
                required_outputs={"WHEAT": quantity},
            )
            guaranteed["WHEAT"] = guaranteed.get("WHEAT", 0) + quantity
            replacement_seeds["WHEAT"] = replacement_seeds.get("WHEAT", 0) + 1
            service_actions += 4 if quantity > current_yield else 3
        elif crop == "MELON" or age < rule.max_yield_day:
            commitment = replace(
                base,
                required_state={"watered_today": True},
                required_outputs={},
            )
            service_actions += 1
        elif rule.ongoing:
            required = {"watered_today": True}
            outputs = {}
            if current_yield:
                required["yield_units"] = 0
                outputs[crop] = current_yield
                guaranteed[crop] = guaranteed.get(crop, 0) + current_yield
                service_actions += 1
            commitment = replace(
                base, required_state=required, required_outputs=outputs
            )
            service_actions += 1
        else:
            gain = rules.one_time_water_gain(
                crop,
                planted_day=int(raw.get("planted_day", state.day)),
                day=state.day,
                yield_units=current_yield,
                fertilized_until_day=int(raw.get("fertilized_until_day", -1)),
                watered_today=bool(raw.get("watered_today", False)),
            )
            quantity = current_yield + gain
            commitment = replace(
                base,
                required_state={"$tile": None},
                required_outputs={crop: quantity},
            )
            guaranteed[crop] = guaranteed.get(crop, 0) + quantity
            service_actions += 2
        obligations.append(_programme_deadline(state, commitment))

    for tile in state.animal_tiles():
        raw = tile.raw
        animal = str(tile.animal)
        commitment = _canonical_daily_commitment(
            state,
            _animal_commitment(
                state,
                animal,
                tile.position,
                existing=True,
                placed_day=int(raw.get("placed_day", state.day)),
                current_yield=int(raw.get("yield_units", 0) or 0),
                fertilizer_available=bool(raw.get("fertilizer_available", False)),
                pending_care_bonus=int(raw.get("pending_care_bonus", 0) or 0),
            ),
        )
        obligations.append(_programme_deadline(state, commitment))
        for item, quantity in commitment.required_outputs.items():
            guaranteed[item] = guaranteed.get(item, 0) + int(quantity)
        service_actions += (
            2
            + int(int(raw.get("yield_units", 0) or 0) > 0)
            + int(bool(raw.get("fertilizer_available", False)))
        )
    return tuple(obligations), guaranteed, replacement_seeds, service_actions


def _programme_orders(
    opening_sales: Mapping[str, int],
    main_sales: Mapping[str, int],
    emergency_wheat: int,
    animal_buys: Mapping[str, int],
    seed_buys: Mapping[str, int],
    buy_land: bool,
) -> tuple[tuple[tuple[object, ...], ...], tuple[tuple[object, ...], ...]]:
    opening: list[tuple[object, ...]] = []
    for item in ("FERTILIZER", "EGG", "MILK", "WOOL", "CARROT", "WHEAT"):
        quantity = max(0, int(opening_sales.get(item, 0)))
        if quantity:
            opening.append(("SELL", item, quantity))
    if emergency_wheat:
        opening.append(("BUY_PRODUCT", "WHEAT", emergency_wheat))

    main: list[tuple[object, ...]] = []
    for item in ("FERTILIZER", "EGG", "MILK", "WOOL", "CARROT", "WHEAT"):
        quantity = max(0, int(main_sales.get(item, 0)))
        if quantity:
            main.append(("SELL", item, quantity))
    if buy_land:
        main.append(("BUY_LAND",))
    for animal in ("COW", "SHEEP", "GOOSE"):
        quantity = max(0, int(animal_buys.get(animal, 0)))
        if quantity:
            main.append(("BUY_ANIMAL", animal, quantity))
    for crop in ("STRAWBERRY", "CARROT", "WHEAT", "TOMATO", "MELON"):
        quantity = max(0, int(seed_buys.get(crop, 0)))
        if quantity:
            main.append(("BUY_SEED", crop, quantity))
    return tuple(opening), tuple(main)


def _programme_cash_ledger(
    state: OwnedState,
    opening_sales: Mapping[str, int],
    main_sales: Mapping[str, int],
    emergency_wheat: int,
    animal_buys: Mapping[str, int],
    seed_buys: Mapping[str, int],
    total_worker_cap: int,
    buy_land: bool = False,
) -> tuple[int, int]:
    """Return exact conservative cash after t0 and t7 order sequences.

    The worker cap is reserved in the ledger, but it is not an instruction to
    hire that many workers.  Intraday still stops at the first feasible W.
    """

    market = dict(state.market_inventory)
    money = state.money

    def sell(group: Mapping[str, int]) -> None:
        nonlocal money
        for item in ("FERTILIZER", "EGG", "MILK", "WOOL", "CARROT", "WHEAT"):
            for _ in range(max(0, int(group.get(item, 0)))):
                inventory = market.get(item, rules.MARKET_I0)
                price = rules.market_price(item, inventory)
                money += price
                if price > rules.PRICE_FLOOR:
                    market[item] = inventory + 1

    sell(opening_sales)
    extra_hands = max(0, total_worker_cap - len(state.workers))
    money -= rules.hire_expenditure(state.hires_today, extra_hands)
    if emergency_wheat:
        for _ in range(emergency_wheat):
            inventory = market.get("WHEAT", rules.MARKET_I0)
            money -= rules.market_price("WHEAT", inventory - 1)
            market["WHEAT"] = inventory - 1
    opening_cash = money

    sell(main_sales)
    if buy_land and len(state.unlocked_quadrants) < 4:
        money -= rules.LAND_PRICES[len(state.unlocked_quadrants) - 1]
    money -= sum(rules.ANIMALS[item].cost * int(quantity)
                 for item, quantity in animal_buys.items())
    money -= sum(rules.CROPS[item].seed_cost * int(quantity)
                 for item, quantity in seed_buys.items())
    return opening_cash, money


def _programme_crop_project(
    state: OwnedState, crop: str, tile: TileState
) -> EconomicCommitment:
    project = _canonical_daily_commitment(
        state, _crop_commitment(state, crop, tile)
    )
    if tile.kind == "PLANT" and tile.raw.get("crop") == "WHEAT":
        opening_yield = max(0, int(tile.raw.get("yield_units", 0) or 0))
        project = replace(
            project,
            required_outputs={
                "WHEAT": opening_yield + rules.one_time_water_gain(
                    "WHEAT",
                    planted_day=int(tile.raw.get("planted_day", state.day)),
                    day=state.day,
                    yield_units=opening_yield,
                    fertilized_until_day=int(
                        tile.raw.get("fertilized_until_day", -1)
                    ),
                    watered_today=bool(tile.raw.get("watered_today", False)),
                )
            },
        )
    return _programme_deadline(state, project)


def _programme_animal_project(
    state: OwnedState, animal: str, tile: TileState
) -> EconomicCommitment:
    project = _canonical_daily_commitment(
        state,
        _animal_commitment(
            state,
            animal,
            tile.position,
            existing=False,
            needs_structure=tile.kind not in ("COOP", "PASTURE"),
        ),
    )
    if tile.kind == "PLANT" and tile.raw.get("crop") == "WHEAT":
        opening_yield = max(0, int(tile.raw.get("yield_units", 0) or 0))
        project = replace(
            project,
            required_outputs={
                "WHEAT": opening_yield + rules.one_time_water_gain(
                    "WHEAT",
                    planted_day=int(tile.raw.get("planted_day", state.day)),
                    day=state.day,
                    yield_units=opening_yield,
                    fertilized_until_day=int(
                        tile.raw.get("fertilized_until_day", -1)
                    ),
                    watered_today=bool(tile.raw.get("watered_today", False)),
                )
            },
        )
    return _programme_deadline(state, project)


def _programme_land_project(state: OwnedState) -> EconomicCommitment:
    quadrant = rules.LAND_ORDER[len(state.unlocked_quadrants) - 1]
    price = rules.LAND_PRICES[len(state.unlocked_quadrants) - 1]
    return EconomicCommitment(
        identifier=f"land:{quadrant}",
        kind="LAND",
        target=None,
        existing=False,
        cash=CashDimension(upfront=price),
        time=TimeDimension(state.step, state.step + 7, state.step + 7),
        land=LandDimension(capacity_created=25),
        actions=ActionDimension(),
        physical=PhysicalDimension(
            outputs=(TimedAmount(state.step + 7, "LAND_TILE_CAPACITY", 25),)
        ),
        revenue=RevenueDimension(),
        metadata={"quadrant": quadrant},
    )


def _programme_structure_project(
    state: OwnedState, animal: str, tile: TileState
) -> EconomicCommitment:
    structure = rules.ANIMALS[animal].structure
    outputs: dict[str, int] = {}
    if tile.kind == "PLANT" and tile.raw.get("crop") == "WHEAT":
        opening_yield = max(0, int(tile.raw.get("yield_units", 0) or 0))
        outputs["WHEAT"] = opening_yield + rules.one_time_water_gain(
            "WHEAT",
            planted_day=int(tile.raw.get("planted_day", state.day)),
            day=state.day,
            yield_units=opening_yield,
            fertilized_until_day=int(tile.raw.get("fertilized_until_day", -1)),
            watered_today=bool(tile.raw.get("watered_today", False)),
        )
    deadline = min(
        rules.TERMINAL_ACTION_STEP,
        state.day * rules.TURNS_PER_DAY + 20,
    )
    return EconomicCommitment(
        identifier=f"prebuild:{animal}:{tile.position[0]}:{tile.position[1]}",
        kind="PREBUILD",
        target=tile.position,
        existing=False,
        cash=CashDimension(),
        time=TimeDimension(
            state.step,
            deadline,
            rules.TERMINAL_ACTION_STEP,
            deadlines=(deadline,),
        ),
        land=LandDimension((OccupancyInterval(
            tile.position, state.step, rules.TERMINAL_ACTION_STEP
        ),)),
        actions=ActionDimension(),
        physical=PhysicalDimension(),
        revenue=RevenueDimension(),
        metadata={
            "animal": animal,
            "structure": structure,
            "programme_role": "NEXT_DAY_STRUCTURE",
        },
        required_state={"kind": structure},
        required_outputs=outputs,
    )


def _make_programme_plan(
    state: OwnedState,
    *,
    revision: int = 0,
    replan_reason: str | None = None,
    _allow_third_land: bool = True,
) -> Plan:
    """Create the deterministic D5-D10 Plan with fixed t0/t7 financing."""

    game_day = state.day + 1
    pace = programme_pace(game_day, state.unlocked_shops)
    priority = programme_priority(state.unlocked_shops)
    total_worker_cap = worker_limit(game_day)
    if total_worker_cap is None:
        raise ValueError("D5-D10 programme has no worker cap")

    obligations, guaranteed, replacement_seeds, base_services = (
        _programme_maintenance(state)
    )
    placed = {animal: state.owned_animals(animal) for animal in rules.ANIMALS}
    crops = {
        crop: sum(tile.kind == "PLANT" and tile.raw.get("crop") == crop
                  for tile in state.tiles)
        for crop in rules.CROPS
    }
    opening_sales: dict[str, int] = {}
    for item in ("FERTILIZER", "EGG", "MILK", "WOOL"):
        opening_sales[item] = state.shed.get(item, 0)
    carrot_keep = pace.carrot if pace.carrot else 0
    opening_sales["CARROT"] = max(
        0, state.shed.get("CARROT", 0) - carrot_keep
    )
    # Operating Wheat is not ordinary opening finance.  Keeping it here also
    # avoids the economically pointless sell-at-t0 / buy-back-at-t0 pattern.
    opening_sales["WHEAT"] = 0
    main_sales = {
        item: int(guaranteed.get(item, 0))
        for item in ("FERTILIZER", "EGG", "MILK", "WOOL", "CARROT", "WHEAT")
        if guaranteed.get(item, 0)
    }
    def wheat_finance(final_animals: int) -> tuple[int, int]:
        # One unit for today's FEED plus a two-day operating reserve.  Only the
        # remainder is a financing asset; emergency Wheat buys cover a genuine
        # shortfall instead of being paired with a same-day sale.
        required = 3 * max(0, final_animals)
        available = state.owned_total("WHEAT") + guaranteed.get("WHEAT", 0)
        emergency = max(0, required - available)
        sale = max(0, available + emergency - required)
        return sale, emergency

    main_sales["WHEAT"], _ = wheat_finance(sum(placed.values()))

    empty = [tile for tile in state.tiles
             if tile.raw is None
             and rules.quadrant(tile.position, state.board_size)
             in state.unlocked_quadrants]
    empty.sort(key=lambda tile: (
        rules.distance_to_shed(tile.position, state.board_size),
        tile.position[1], tile.position[0],
    ))
    operating_positions = set(
        tile.position for tile in sorted(
            (tile for tile in state.crop_tiles()
             if tile.raw.get("crop") == "WHEAT"),
            key=lambda tile: (
                rules.distance_to_shed(tile.position, state.board_size),
                tile.position[1], tile.position[0],
            ),
        )[:pace.operating_wheat]
    )
    convertible = [tile for tile in state.crop_tiles()
                   if tile.raw.get("crop") == "WHEAT"
                   and tile.position not in operating_positions]
    convertible.sort(key=lambda tile: (
        -rules.distance_to_shed(tile.position, state.board_size),
        tile.position[1], tile.position[0],
    ))
    targets = [*empty, *convertible]

    animal_shortfall = {
        "COW": max(0, pace.cow - placed["COW"]),
        "SHEEP": max(0, pace.sheep - placed["SHEEP"]),
        "GOOSE": max(0, pace.goose - placed["GOOSE"]),
    }
    crop_shortfall = (
        max(0, pace.strawberry - crops["STRAWBERRY"])
        + max(0, pace.carrot - crops["CARROT"])
    )
    permanent_tiles_needed = sum(animal_shortfall.values()) + crop_shortfall
    new_land_positions: set[Position] = set()
    buy_land = False
    if (
        _allow_third_land
        and 7 <= game_day <= 10
        and len(state.unlocked_quadrants) == 2
    ):
        price = rules.LAND_PRICES[len(state.unlocked_quadrants) - 1]
        no_hire_cash = _programme_cash_ledger(
            state,
            opening_sales,
            main_sales,
            0,
            {},
            {},
            len(state.workers),
        )[1]
        day_worker_cost = rules.hire_expenditure(
            state.hires_today,
            max(0, total_worker_cap - len(state.workers)),
        )
        animal_slots = animal_addition_cap(game_day)
        planned_animal_cost = 0
        for item in priority:
            if item not in rules.ANIMALS:
                continue
            take = min(animal_slots, animal_shortfall[item])
            planned_animal_cost += take * rules.ANIMALS[item].cost
            animal_slots -= take
        mandatory_seed_cost = (
            sum(replacement_seeds.values()) + 25
        ) * rules.CROPS["WHEAT"].seed_cost
        facts = ThirdLandFacts(
            game_day=game_day,
            two_land_space_insufficient=permanent_tiles_needed > len(targets),
            permanent_tiles_needed=permanent_tiles_needed,
            cash_after_purchase=no_hire_cash - price,
            day_worker_cost=day_worker_cost,
            next_day_feed_cost=0,
            planned_animal_cost=planned_animal_cost,
            mandatory_seed_cost=mandatory_seed_cost,
            has_same_day_layout=(
                base_services + 2 * 25 <= total_worker_cap * 20
            ),
        )
        buy_land = third_land_allowed(facts)
        if buy_land:
            quadrant = rules.LAND_ORDER[len(state.unlocked_quadrants) - 1]
            locked = [
                tile for tile in state.tiles
                if tile.is_locked
                and rules.quadrant(tile.position, state.board_size) == quadrant
            ]
            locked.sort(key=lambda tile: (
                rules.distance_to_shed(tile.position, state.board_size),
                tile.position[1], tile.position[0],
            ))
            new_land_positions = {tile.position for tile in locked}
            targets.extend(locked)

    chosen: list[tuple[str, str, TileState]] = []
    animal_buys: dict[str, int] = {}
    seed_buys: dict[str, int] = dict(replacement_seeds)
    rejected: dict[str, str] = {}
    daily_animals = 0
    estimated_services = base_services

    def target_counts():
        animals = dict(placed)
        plants = dict(crops)
        for kind, item, tile in chosen:
            if tile.kind == "PLANT":
                prior_crop = str(tile.raw.get("crop"))
                plants[prior_crop] = max(0, plants.get(prior_crop, 0) - 1)
            if kind == "ANIMAL":
                animals[item] += 1
            else:
                plants[item] += 1
        return animals, plants

    def try_group(kind: str, item: str, quantity: int) -> bool:
        nonlocal estimated_services, daily_animals
        if quantity <= 0:
            return False
        used_positions = {used.position for _, _, used in chosen}
        matching_structures = []
        if kind == "ANIMAL":
            structure = rules.ANIMALS[item].structure
            matching_structures = [
                tile for tile in state.empty_structures(structure)
                if tile.position not in used_positions
            ]
            matching_structures.sort(key=lambda tile: (
                rules.distance_to_shed(tile.position, state.board_size),
                tile.position[1], tile.position[0],
            ))
        available = [
            *matching_structures,
            *(tile for tile in targets if tile.position not in used_positions),
        ]
        if len(available) < quantity:
            rejected[f"{kind}:{item}"] = BindingConstraint.LAND.value
            return False
        if kind == "ANIMAL" and daily_animals + quantity > animal_addition_cap(game_day):
            rejected[f"{kind}:{item}"] = BindingConstraint.ROUTE.value
            return False
        extra_services = quantity * (7 if kind == "ANIMAL" else 3)
        if estimated_services + extra_services + len(chosen) + quantity \
                > total_worker_cap * 20:
            reason = (BindingConstraint.WATER_DEADLINE if kind == "CROP"
                      else BindingConstraint.ROUTE)
            rejected[f"{kind}:{item}"] = reason.value
            return False
        proposed_animals = dict(animal_buys)
        proposed_seeds = dict(seed_buys)
        proposed_main_sales = dict(main_sales)
        selected = available[:quantity]
        if kind == "ANIMAL":
            proposed_animals[item] = proposed_animals.get(item, 0) + quantity
        else:
            proposed_seeds[item] = proposed_seeds.get(item, 0) + quantity
        # A released reservation-W tile already contributed one replacement
        # Wheat seed to the mandatory baseline.  Conversion replaces that seed
        # with the selected asset's own seed (or no seed for an animal).
        for tile in selected:
            if tile.kind == "PLANT" and tile.raw.get("crop") == "WHEAT":
                proposed_seeds["WHEAT"] = max(
                    0, proposed_seeds.get("WHEAT", 0) - 1
                )
        future_animals = sum(placed.values()) + sum(proposed_animals.values())
        proposed_main_sales["WHEAT"], emergency = wheat_finance(
            future_animals
        )
        opening_cash, ending_cash = _programme_cash_ledger(
            state, opening_sales, proposed_main_sales, emergency,
            proposed_animals, proposed_seeds, total_worker_cap, buy_land,
        )
        if opening_cash < 0 or ending_cash < 0:
            rejected[f"{kind}:{item}"] = BindingConstraint.CASH.value
            return False
        chosen.extend((kind, item, tile) for tile in selected)
        animal_buys.clear(); animal_buys.update(proposed_animals)
        seed_buys.clear(); seed_buys.update(proposed_seeds)
        main_sales.clear(); main_sales.update(proposed_main_sales)
        estimated_services += extra_services
        if kind == "ANIMAL":
            daily_animals += quantity
        return True

    animals_now, crops_now = target_counts()
    for item in priority:
        if item in rules.ANIMALS:
            floor = {
                "COW": pace.cow,
                "SHEEP": pace.sheep,
                "GOOSE": pace.goose,
            }[item]
            while daily_animals < animal_addition_cap(game_day):
                animals_now, _ = target_counts()
                if not try_group("ANIMAL", item, 1):
                    break
                if animals_now[item] >= floor and floor > 0:
                    # Pace is not a cap: one more candidate is still attempted
                    # on the next iteration until a real constraint binds.
                    continue
        elif item in ("STRAWBERRY", "SUPPORTED_CROP"):
            crop = ("STRAWBERRY" if pace.strawberry
                    else "CARROT" if pace.carrot else None)
            floor = pace.strawberry if crop == "STRAWBERRY" else pace.carrot
            if crop:
                while True:
                    _, crops_now = target_counts()
                    if not try_group("CROP", crop, 4):
                        break
                    if crops_now[crop] >= floor and floor > 0:
                        continue
        elif item == "OPERATING_WHEAT":
            while True:
                _, crops_now = target_counts()
                if crops_now["WHEAT"] >= pace.operating_wheat:
                    break
                if not try_group("CROP", "WHEAT", 1):
                    break
        elif item == "RESERVATION_WHEAT":
            while try_group("CROP", "WHEAT", 4):
                pass

    if buy_land:
        while True:
            used = {tile.position for _, _, tile in chosen}
            if new_land_positions <= used:
                break
            if not try_group("CROP", "WHEAT", 1):
                # A land purchase without same-day use of every remaining tile
                # violates the programme's no-bare-new-land rule.  Re-form the
                # same day with the gate disabled; this is not route repair.
                return _make_programme_plan(
                    state,
                    revision=revision,
                    replan_reason=replan_reason,
                    _allow_third_land=False,
                )

    prebuilds: list[tuple[str, TileState]] = []
    if game_day < 10:
        next_day = game_day + 1
        next_priority = programme_priority(state.unlocked_shops)
        next_animal = next(
            (item for item in next_priority if item in rules.ANIMALS), None
        )
        if next_animal is not None:
            final_animals_by_kind, _ = target_counts()
            used = {tile.position for _, _, tile in chosen}
            current_structures = sum(
                tile.position not in used
                for tile in state.empty_structures(
                    rules.ANIMALS[next_animal].structure
                )
            )
            main_sales["WHEAT"], current_emergency = wheat_finance(
                sum(final_animals_by_kind.values())
            )
            _, cash_end = _programme_cash_ledger(
                state,
                opening_sales,
                main_sales,
                current_emergency,
                animal_buys,
                seed_buys,
                total_worker_cap,
                buy_land,
            )
            next_cap = worker_limit(next_day) or total_worker_cap
            next_hires = rules.hire_expenditure(0, max(0, next_cap - 1))
            fertilizer_revenue = sum(final_animals_by_kind.values()) * rules.market_price(
                "FERTILIZER",
                state.market_inventory.get("FERTILIZER", rules.MARKET_I0),
            )
            next_cash = projected_next_day_cash(
                cash_end,
                fertilizer_revenue,
                next_hires,
                0,
                0,
                0,
            )
            unit_cost = (
                rules.ANIMALS[next_animal].cost
                + 3 * rules.market_price(
                    "WHEAT",
                    state.market_inventory.get("WHEAT", rules.MARKET_I0) - 1,
                )
            )
            available = [tile for tile in targets if tile.position not in used]
            route_capacity = max(
                0, (total_worker_cap * 20 - estimated_services) // 3
            )
            count = prebuild_count(PrebuildCaps(
                finance=max(0, next_cash // max(1, unit_cost)),
                feed=max(0, next_cash // max(1, unit_cost)),
                route=min(animal_addition_cap(next_day), route_capacity),
                land=len(available),
            ))
            count = max(0, count - current_structures)
            for tile in available[:count]:
                prebuilds.append((next_animal, tile))
                estimated_services += 3 if tile.kind == "PLANT" else 1
                if tile.kind == "PLANT" and tile.raw.get("crop") == "WHEAT":
                    seed_buys["WHEAT"] = max(
                        0, seed_buys.get("WHEAT", 0) - 1
                    )

    total_animals = sum(placed.values()) + sum(animal_buys.values())
    main_sales["WHEAT"], emergency_wheat = wheat_finance(total_animals)
    feasible_animals, feasible_crops = target_counts()
    animal_floors = {
        "COW": pace.cow,
        "SHEEP": pace.sheep,
        "GOOSE": pace.goose,
    }
    for animal, floor in animal_floors.items():
        if feasible_animals[animal] >= floor:
            continue
        rejected.setdefault(
            f"ANIMAL:{animal}",
            (BindingConstraint.ROUTE.value
             if daily_animals >= animal_addition_cap(game_day)
             else BindingConstraint.CASH.value),
        )
    crop_floors = {
        "STRAWBERRY": pace.strawberry,
        "CARROT": pace.carrot,
        "WHEAT": pace.operating_wheat,
    }
    for crop, floor in crop_floors.items():
        if feasible_crops[crop] < floor:
            rejected.setdefault(f"CROP:{crop}", BindingConstraint.LAND.value)
    opening_orders, main_orders = _programme_orders(
        opening_sales, main_sales, emergency_wheat,
        animal_buys, seed_buys, buy_land,
    )
    if len(main_orders) > rules.MAX_MARKET_ORDERS:
        raise ValueError("programme MAIN_FINANCE exceeds ten market entries")

    selected: list[EconomicCommitment] = []
    for kind, item, tile in chosen:
        project = (_programme_animal_project(state, item, tile)
                   if kind == "ANIMAL"
                   else _programme_crop_project(state, item, tile))
        selected.append(project)

    selected.extend(
        _programme_structure_project(state, animal, tile)
        for animal, tile in prebuilds
    )

    converted_positions = {
        tile.position for _, _, tile in chosen if tile.kind == "PLANT"
    }
    converted_positions.update(
        tile.position for _, tile in prebuilds if tile.kind == "PLANT"
    )
    obligations = tuple(
        commitment for commitment in obligations
        if commitment.target not in converted_positions
    )

    windows = []
    if opening_orders:
        windows.append(EconomicWindow(
            0, 0,
            market_orders=opening_orders,
            required_shed={str(order[1]): int(order[2])
                           for order in opening_orders
                           if order[0] == "SELL"},
        ))
    if main_orders:
        windows.append(EconomicWindow(
            7, 7,
            market_orders=main_orders,
            required_shed={str(order[1]): int(order[2])
                           for order in main_orders
                           if order[0] == "SELL"},
        ))
    animals_final, crops_final = target_counts()
    opening_cash, ending_cash = _programme_cash_ledger(
        state,
        opening_sales,
        main_sales,
        emergency_wheat,
        animal_buys,
        seed_buys,
        total_worker_cap,
        buy_land,
    )
    return Plan(
        obligations=obligations,
        selected=tuple(selected),
        support=((_programme_land_project(state),) if buy_land else ()),
        rejected=rejected,
        fertilize_targets=frozenset(),
        animal_purchases=tuple(
            animal for animal in ("COW", "SHEEP", "GOOSE")
            for _ in range(animal_buys.get(animal, 0))
        ),
        buy_land=buy_land,
        feed_reserve=2 * total_animals,
        fertilizer_reserve=0,
        diagnostics={
            "programme": pace.programme.value,
            "pace_floor": {
                "COW": pace.cow, "SHEEP": pace.sheep,
                "GOOSE": pace.goose, "STRAWBERRY": pace.strawberry,
                "CARROT": pace.carrot, "OPERATING_WHEAT": pace.operating_wheat,
            },
            "planned_target": {**animals_final, **crops_final},
            "requested_target": {
                **animal_floors,
                **crop_floors,
            },
            "max_feasible_target": {
                **feasible_animals,
                **feasible_crops,
            },
            "opening_finance_turn": 0,
            "main_finance_turn": 7,
            "cash_after_opening_finance": opening_cash,
            "cash_after_conservative_ledger": ending_cash,
            "next_day_prebuilds": tuple(
                (animal, tile.position) for animal, tile in prebuilds
            ),
            "third_land": buy_land,
            "binding_constraints": dict(rejected),
        },
        starting_animals=placed,
        day=state.day,
        formed_step=state.step,
        revision=revision,
        replan_reason=replan_reason,
        max_hands=total_worker_cap - 1,
        economic_windows=tuple(windows),
    )


def make_plan(
    state: OwnedState,
    config: PlannerConfig | None = None,
    *,
    revision: int = 0,
    replan_reason: str | None = None,
) -> Plan:
    """Form the economic and production commitments for ``state.day``."""
    if 4 <= state.day <= 9:
        return _make_programme_plan(
            state, revision=revision, replan_reason=replan_reason
        )
    config = config or PlannerConfig()
    obligations = existing_obligations(state)
    candidates = list(enumerate_projects(state, config))
    selected: list[EconomicCommitment] = []
    rejected: dict[str, str] = {}
    occupied: set[Position] = {
        project.target for project in obligations if project.target is not None
    }
    own_sales: dict[str, int] = {}
    cash_spent = 0

    while candidates:
        repriced = []
        for candidate in candidates:
            candidate = _reprice(state, candidate, own_sales)
            if candidate.kind == "CROP":
                crop = str(candidate.metadata["crop"])
                used = sum(
                    item.kind == "CROP" and item.metadata.get("crop") == crop
                    for item in selected
                )
                from_stock = state.seeds.get(crop, 0) > used
                if from_stock:
                    candidate = replace(
                        candidate,
                        cash=replace(candidate.cash, upfront=0, sunk_cost=rules.CROPS[crop].seed_cost),
                        metadata={**candidate.metadata, "seed_purchase_required": False},
                    )
                else:
                    candidate = replace(
                        candidate,
                        metadata={**candidate.metadata, "seed_purchase_required": True},
                    )
            repriced.append(candidate)
        # Lexicographic dominance on derived quantities, not a weighted collapse.
        repriced.sort(
            key=lambda item: (
                -item.profit_per_action,
                -item.terminal_profit,
                -item.profit_per_tile_day,
                item.kind,
                item.identifier,
            )
        )
        chosen = None
        for project in repriced:
            reason = _selection_reason(
                state, config, project, [*obligations, *selected], cash_spent, occupied
            )
            if reason is None:
                chosen = project
                break
            rejected.setdefault(project.identifier, reason)
        if chosen is None:
            break
        selected.append(chosen)
        cash_spent += chosen.cash.upfront
        if chosen.target is not None:
            occupied.add(chosen.target)
        for output in chosen.physical.outputs:
            own_sales[output.item] = own_sales.get(output.item, 0) + output.quantity
        candidates = [item for item in candidates if item.identifier != chosen.identifier]

    animal_purchases = tuple(
        str(project.metadata["animal"])
        for project in selected
        if project.kind == "ANIMAL"
    )
    today_feed = state.projected_feed_need_today()
    feed_reserve = max(today_feed, len(state.animal_tiles()) * 2) + len(animal_purchases) + sum(
        state.owned_total(animal) for animal in rules.ANIMALS)
    fertilizer_opportunities = _fertilizer_opportunities(state)
    fertilize = frozenset(
        opportunity.position for opportunity in fertilizer_opportunities
    )
    fertilizer_reserve = len(fertilizer_opportunities)

    buy_land = False
    if (
        not _candidate_targets(state)
        and len(state.unlocked_quadrants) < 4
        and state.days_left >= 5
    ):
        price = rules.LAND_PRICES[len(state.unlocked_quadrants) - 1]
        # Capacity is purchased only when one conservative wheat project per new
        # tile could repay the land; the land project is not valued in isolation.
        unit_margin = max(0, 4 * state.market_prices.get("WHEAT", 25) - rules.CROPS["WHEAT"].seed_cost)
        buy_land = state.money - cash_spent - config.cash_reserve >= price and 25 * unit_margin > price

    support = _support_commitments(state, fertilizer_opportunities, buy_land)
    obligations = tuple(_canonical_daily_commitment(state, item) for item in obligations)
    selected = tuple(_canonical_daily_commitment(state, item) for item in selected)
    support = tuple(_canonical_daily_commitment(state, item) for item in support)
    return Plan(
        obligations=obligations,
        selected=selected,
        support=support,
        rejected=rejected,
        fertilize_targets=fertilize,
        animal_purchases=animal_purchases,
        buy_land=buy_land,
        feed_reserve=feed_reserve,
        fertilizer_reserve=fertilizer_reserve,
        diagnostics={
            "selected_terminal_profit": sum(item.terminal_profit for item in selected),
            "selected_actions": sum(item.actions.total_actions for item in selected),
            "existing_obligation_value": sum(max(0, item.terminal_profit) for item in obligations),
        },
        starting_animals={
            animal: state.owned_animals(animal) for animal in rules.ANIMALS
        },
        day=state.day,
        formed_step=state.step,
        revision=revision,
        replan_reason=replan_reason,
        max_hands=config.max_daily_hands,
    )
