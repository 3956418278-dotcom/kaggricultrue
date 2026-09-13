"""Event-driven, next-cycle handling of assets already on the farm."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from . import rules
from .market import buy_cost, known_demand_events, sell_revenue
from .midgame_config import DEFAULT_MIDGAME_PARAMETERS, MidgameParameters
from .programme import AssetProgramme, CurrentAssetState, ProgrammeEvent
from .state import AssetState, Position, State

MAINTAIN = "MAINTAIN"
EXIT = "EXIT"


@dataclass(frozen=True)
class AnimalCycleValue:
    next_production_step: int | None
    forecast_inventory: int
    product_value: int
    wheat_units: int
    wheat_cost: int
    hire_cost: int

    @property
    def base_gain(self) -> int:
        return self.product_value - self.wheat_cost - self.hire_cost


@dataclass(frozen=True)
class AnimalDecisionInputs:
    """Facts proven by other planner owners, without route search here.

    Replacement values and incremental hire costs are cash amounts, not scores,
    worker counts, or locally invented shadow prices.
    """

    replacement_advantages: Mapping[Position, int] | None = None
    base_incremental_hire_costs: Mapping[str, int] | None = None
    care_incremental_hire_costs: Mapping[str, int] | None = None
    liquidation_incremental_hire_costs: Mapping[str, int] | None = None

    def replacement_at(self, tile: Position) -> int:
        return max(0, int((self.replacement_advantages or {}).get(tile, 0)))

    def hire_cost_for(self, mapping: Mapping[str, int] | None,
                      asset_id: str) -> int:
        return max(0, int((mapping or {}).get(asset_id, 0)))


def _event(step: int, priority: int, event_id: str, kind: str, *,
           tile: Position, asset_id: str, item: str | None = None,
           quantity: int = 0, action=("PASS",), mandatory=True,
           deadline: int | None = None, source: str | None = None) -> ProgrammeEvent:
    return ProgrammeEvent(step, priority, event_id, kind, tile, asset_id,
                          item, quantity, tuple(action), mandatory, deadline,
                          None, source)


def _asset_id(asset: AssetState) -> str:
    x, y = asset.position
    return f"current:{asset.asset_type}:{x}:{y}"


def _day_end(state: State) -> int:
    return min((state.day + 1) * rules.TURNS_PER_DAY - 1,
               rules.TERMINAL_ACTION_STEP)


def next_animal_production_day(raw: Mapping[str, object], day: int) -> int | None:
    """Return the next closing production day in O(1)."""
    rule = rules.ANIMALS[str(raw["animal"])]
    first = int(raw["placed_day"]) + rule.first_yield_day - 1
    candidate = max(day, first)
    remainder = (candidate - first) % rule.interval
    if remainder:
        candidate += rule.interval - remainder
    last_refresh_day = rules.TERMINAL_ACTION_STEP // rules.TURNS_PER_DAY - 1
    return candidate if candidate <= last_refresh_day else None


def _production_step(production_day: int | None) -> int | None:
    return None if production_day is None else (
        production_day + 1) * rules.TURNS_PER_DAY


def _production_is_cashable(state: State, tile: Position,
                            output_step: int | None) -> bool:
    if output_step is None:
        return False
    distance = rules.distance_to_shed(tile, state.board_size)
    earliest_drop = output_step + 2 * distance + 2
    return earliest_drop <= rules.TERMINAL_ACTION_STEP


def _minimal_feed_days(raw: Mapping[str, object], start_day: int,
                       production_day: int) -> tuple[int, ...]:
    consecutive = int(raw.get("consecutive_unfed", 0))
    result = []
    for day in range(start_day, production_day + 1):
        if day == start_day and bool(raw.get("fed_today", False)):
            consecutive = 0
        elif consecutive >= 1:
            result.append(day)
            consecutive = 0
        else:
            consecutive += 1
    return tuple(result)


def _animal_output_days(raw: Mapping[str, object], start_day: int,
                        last_output_step: int) -> Iterable[int]:
    day = next_animal_production_day(raw, start_day)
    interval = rules.ANIMALS[str(raw["animal"])].interval
    while day is not None and (day + 1) * 24 < last_output_step:
        yield day
        day += interval


def _crop_output_days(raw: Mapping[str, object], start_day: int,
                      last_output_step: int) -> Iterable[int]:
    crop = str(raw["crop"])
    rule = rules.CROPS[crop]
    if not rule.ongoing:
        return
    first = int(raw["planted_day"]) + rule.first_yield_day - 1
    day = max(start_day, first)
    while (day + 1) * 24 < last_output_step:
        age = day + 1 - int(raw["planted_day"]) - rule.first_yield_day
        if age >= 0 and age % rule.interval == 0:
            count = age // rule.interval + 1
            if count <= rule.max_yield:
                yield day
        day += 1


def _base_supply_calendar(state: State, last_step: int
                          ) -> Mapping[str, Mapping[int, int]]:
    calendar = defaultdict(lambda: defaultdict(int))
    for animal in (*state.own.animals, *state.opp.visible_animals):
        product = rules.ANIMALS[animal.asset_type].product
        for day in _animal_output_days(animal.official, state.day, last_step):
            calendar[product][(day + 1) * 24] += 1
    for crop in (*state.own.crops, *state.opp.visible_crops):
        for day in _crop_output_days(crop.official, state.day, last_step):
            calendar[crop.asset_type][(day + 1) * 24] += 1
    return {product: dict(events) for product, events in calendar.items()}


def _forecast_inventory(state: State, product: str, output_step: int,
                        supply, demand) -> int:
    inventory = int(state.market.inventory.get(product, rules.MARKET_I0))
    inventory += sum(quantity for step, quantity in
                     supply.get(product, {}).items() if step < output_step)
    inventory -= sum(quantity for step, quantity in demand.get(product, ())
                     if step <= output_step)
    return inventory


def _cycle_value(state: State, asset: AssetState, output_step: int | None,
                 supply, demand, incremental_hire_cost: int
                 ) -> AnimalCycleValue:
    raw = asset.official
    product = rules.ANIMALS[asset.asset_type].product
    opening = int(state.market.inventory.get(product, rules.MARKET_I0))
    if output_step is None or not _production_is_cashable(
            state, asset.position, output_step):
        return AnimalCycleValue(output_step, opening, 0, 0, 0, 0)
    production_day = output_step // 24 - 1
    feed_days = _minimal_feed_days(raw, state.day, production_day)
    wheat_units = len(feed_days)
    wheat_cost = buy_cost("WHEAT", wheat_units,
                          int(state.market.inventory.get(
                              "WHEAT", rules.MARKET_I0)))
    forecast = _forecast_inventory(state, product, output_step, supply, demand)
    value = sell_revenue(product, 1, forecast)
    hire_cost = max(0, incremental_hire_cost)
    return AnimalCycleValue(output_step, forecast, value, wheat_units,
                            wheat_cost, hire_cost)


def _care_decision(state: State, asset: AssetState, cycle: AnimalCycleValue,
                   incremental_hire_cost: int
                   ) -> tuple[bool, int, int | None]:
    raw = asset.official
    rule = rules.ANIMALS[asset.asset_type]
    if (cycle.next_production_step is None or cycle.product_value <= 0 or
            int(raw.get("pending_care_bonus", 0)) > 0 or
            bool(raw.get("cared_today", False))):
        return False, 0, None
    production_day = cycle.next_production_step // 24 - 1
    if production_day <= state.day:
        return False, 0, None
    held = int(raw.get("yield_units", 0))
    if held + 2 > rule.max_held:
        return False, 0, None

    base_feed = set(_minimal_feed_days(raw, state.day, production_day))
    reusable = sorted(day for day in base_feed if day < production_day)
    care_day = reusable[0] if reusable else state.day
    care_feed = {care_day, production_day}
    extra_wheat_units = len(care_feed - base_feed)
    extra_wheat_cost = buy_cost(
        "WHEAT", extra_wheat_units,
        int(state.market.inventory.get("WHEAT", rules.MARKET_I0)))
    after_base_inventory = cycle.forecast_inventory
    if rules.market_price(rule.product, after_base_inventory) > rules.PRICE_FLOOR:
        after_base_inventory += 1
    bonus_value = sell_revenue(rule.product, 1, after_base_inventory)
    hire_cost = max(0, incremental_hire_cost)
    gain = bonus_value - extra_wheat_cost - hire_cost
    return gain > 0, gain, care_day * rules.TURNS_PER_DAY


def animal_locality_bonus(asset_type: str, tile: Position, day: int,
                          board_size: int = rules.BOARD_SIZE,
                          params: MidgameParameters = DEFAULT_MIDGAME_PARAMETERS) -> float:
    if asset_type not in rules.ANIMALS:
        return 0.0
    if day <= params.locality_full_through_day:
        day_factor = 1.0
    elif day >= params.locality_zero_day:
        day_factor = 0.0
    else:
        span = params.locality_zero_day - params.locality_full_through_day
        day_factor = (params.locality_zero_day - day) / span
    distance = rules.distance_to_shed(tile, board_size)
    distance_factor = max(0.0, 1.0 - distance /
                          max(1, params.locality_distance_span))
    return params.locality_peak_bonus * day_factor * distance_factor


def _decision_event(state: State, raw: Mapping[str, object],
                    prior: CurrentAssetState | None, shops_changed: bool,
                    replacement_advantage: int,
                    next_production_step: int | None,
                    next_production_cashable: bool) -> str | None:
    if prior is None:
        return "ATTACH"
    if int(raw.get("placed_day", -1)) != int(
            prior.official.get("placed_day", -1)):
        return "ATTACH"
    if shops_changed:
        return "SHOP_REVEAL"
    if (prior.next_production_step is not None and
            state.step >= prior.next_production_step):
        return "PRODUCTION"
    if int(raw.get("yield_units", 0)) != prior.held_quantity:
        return "HELD_CAPACITY_CHANGE"
    if replacement_advantage > 0 and (
            prior.replacement_advantage <= 0 or
            replacement_advantage != prior.replacement_advantage):
        return "REPLACEMENT"
    if ((next_production_step is None and
         prior.next_production_step is not None) or
            (prior.next_production_cashable and
             not next_production_cashable)):
        return "TERMINAL"
    return None


def _animal_current_state(
    state: State, asset: AssetState, prior: CurrentAssetState | None,
    shops_changed: bool, inputs: AnimalDecisionInputs, supply, demand, *,
    force_exit: bool = False,
) -> CurrentAssetState:
    raw = dict(asset.official)
    identifier = _asset_id(asset)
    rule = rules.ANIMALS[asset.asset_type]
    held = int(raw.get("yield_units", 0))
    production_day = next_animal_production_day(raw, state.day)
    next_step = _production_step(production_day)
    next_cashable = _production_is_cashable(
        state, asset.position, next_step)
    replacement = inputs.replacement_at(asset.position)
    decision_reason = "FORCED_REPLACEMENT" if force_exit else _decision_event(
        state, raw, prior, shops_changed, replacement, next_step,
        next_cashable)

    if decision_reason is None and prior is not None:
        mode = prior.animal_mode or MAINTAIN
        base_gain = prior.base_gain if prior.base_gain is not None else 0
        replacement = prior.replacement_advantage
        care_approved = prior.care_approved
        care_gain = prior.care_gain
        care_step = prior.care_step
        next_step = prior.next_production_step
        next_cashable = prior.next_production_cashable
    else:
        cycle = _cycle_value(
            state, asset, next_step, supply, demand,
            inputs.hire_cost_for(
                inputs.base_incremental_hire_costs, identifier))
        base_gain = cycle.base_gain
        mode = EXIT if force_exit or base_gain <= replacement else MAINTAIN
        if mode == MAINTAIN:
            care_approved, care_gain, care_step = _care_decision(
                state, asset, cycle,
                inputs.hire_cost_for(
                    inputs.care_incremental_hire_costs, identifier))
        else:
            care_approved, care_gain, care_step = False, 0, None

    consecutive = int(raw.get("consecutive_unfed", 0))
    fed = bool(raw.get("fed_today", False))
    fertilizer = bool(raw.get("fertilizer_available", False))
    today: list[ProgrammeEvent] = []
    next_events: list[ProgrammeEvent] = []
    end = _day_end(state)

    if mode == MAINTAIN:
        production_today = (next_step is not None and
                            next_step == (state.day + 1) * 24)
        care_today = (care_approved and care_step is not None and
                      care_step // 24 == state.day and
                      not raw.get("cared_today", False) and
                      int(raw.get("pending_care_bonus", 0)) == 0)
        realize_pending = (production_today and
                           int(raw.get("pending_care_bonus", 0)) > 0)
        must_feed = not fed and consecutive >= 1
        if not fed and (must_feed or care_today or realize_pending):
            today.append(_event(
                state.step, 20, identifier + ":feed", "FEED",
                tile=asset.position, asset_id=identifier, item="WHEAT",
                quantity=1, action=("FEED",), mandatory=True,
                deadline=end, source=("MINIMUM_MAINTENANCE" if must_feed
                                      else "CARE_CYCLE")))
        if care_today:
            today.append(_event(
                state.step, 21, identifier + ":care", "CARE",
                tile=asset.position, asset_id=identifier, action=("CARE",),
                mandatory=True, deadline=end, source="CARE_CYCLE"))

    projected_raw = dict(raw)
    if any(event.kind == "FEED" for event in today):
        projected_raw["fed_today"] = True
    production_tonight = next_step == (state.day + 1) * 24
    incoming = (rules.animal_production_on_refresh(projected_raw, state.day)
                if production_tonight else 0)
    overflow = incoming > 0 and held + incoming > rule.max_held
    terminal_liquidation = state.day >= 29 and held > 0
    exit_stage = mode == EXIT and consecutive >= 1 and not fed
    planned_visit = bool(today)

    if mode == EXIT and held:
        liquidation = sell_revenue(
            rule.product, held,
            int(state.market.inventory.get(rule.product, rules.MARKET_I0)))
        hire_cost = inputs.hire_cost_for(
            inputs.liquidation_incremental_hire_costs, identifier)
        if planned_visit or exit_stage or liquidation > hire_cost:
            today.append(_event(
                state.step, 30, identifier + ":liquidate", "HARVEST",
                tile=asset.position, asset_id=identifier, item=rule.product,
                quantity=held, action=("HARVEST",), deadline=end,
                source="EXIT_LIQUIDATION"))
            planned_visit = True
    elif held and (overflow or terminal_liquidation):
        harvest_reason = ("HELD_OVERFLOW" if overflow else
                          "TERMINAL_LIQUIDATION")
        today.append(_event(
            state.step, 30, identifier + ":harvest:" + harvest_reason,
            "HARVEST", tile=asset.position, asset_id=identifier,
            item=rule.product, quantity=held, action=("HARVEST",),
            deadline=end, source=harvest_reason))
        planned_visit = True

    if fertilizer and planned_visit:
        today.append(_event(
            state.step, 35, identifier + ":collect-f", "COLLECT_F",
            tile=asset.position, asset_id=identifier, item="FERTILIZER",
            quantity=1, action=("COLLECT_FERTILIZER",), mandatory=False,
            deadline=end, source="VISIT_ONLY"))

    survives_refresh = mode != EXIT or consecutive < 1 or fed
    if production_tonight and survives_refresh:
        held_before_output = 0 if overflow else held
        quantity = min(rule.max_held - held_before_output, incoming)
        if quantity > 0:
            next_events.append(_event(
                next_step, 0, identifier + ":next-output", "OUTPUT",
                tile=asset.position, asset_id=identifier, item=rule.product,
                quantity=quantity, mandatory=False,
                source="OFFICIAL_REFRESH"))
    if survives_refresh:
        next_events.append(_event(
            (state.day + 1) * 24, 1, identifier + ":next-fertilizer",
            "F_AVAILABLE", tile=asset.position, asset_id=identifier,
            item="FERTILIZER", quantity=1, mandatory=False,
            source="OFFICIAL_REFRESH"))

    release = None
    if mode == EXIT:
        nights = 3 if fed else 1 if consecutive >= 1 else 2
        release = min((state.day + nights) * 24,
                      rules.TERMINAL_ACTION_STEP + 1)
    elif production_day is not None:
        feed_days = _minimal_feed_days(raw, state.day, production_day)
        if feed_days:
            next_feed_day = feed_days[0]
            next_events.append(_event(
                max(state.step, next_feed_day * 24), 20,
                identifier + ":next-maintenance", "FEED_DUE",
                tile=asset.position, asset_id=identifier, item="WHEAT",
                quantity=1, mandatory=True,
                deadline=min(next_feed_day * 24 + 23,
                             rules.TERMINAL_ACTION_STEP),
                source="MAINTAIN"))

    economic_exit_step = None
    if mode == EXIT:
        economic_exit_step = (
            prior.economic_exit_step
            if prior is not None and prior.animal_mode == EXIT
            else state.step)

    return CurrentAssetState(
        asset_id=identifier, asset_type=asset.asset_type, tile=asset.position,
        official=raw, held_product=rule.product, held_quantity=held,
        animal_mode=mode, base_gain=base_gain,
        replacement_advantage=replacement, care_approved=care_approved,
        care_gain=care_gain, care_step=care_step,
        decision_event=decision_reason, next_production_step=next_step,
        next_production_cashable=next_cashable,
        today_events=tuple(sorted(today)), next_events=tuple(sorted(next_events)),
        economic_exit_step=economic_exit_step,
        physical_release_step=release,
    )


def exit_current_asset(
    state: State, asset: AssetState,
    inputs: AnimalDecisionInputs | None = None,
) -> CurrentAssetState:
    next_step = _production_step(next_animal_production_day(
        asset.official, state.day))
    demand = known_demand_events(state, end_step=next_step or state.step)
    supply = _base_supply_calendar(state, next_step or state.step)
    return _animal_current_state(
        state, asset, None, False, inputs or AnimalDecisionInputs(),
        supply, demand, force_exit=True)


def _next_crop_production_day(raw: Mapping[str, object], day: int) -> int | None:
    rule = rules.CROPS[str(raw["crop"])]
    if not rule.ongoing:
        mature = int(raw["planted_day"]) + rule.first_yield_day
        return max(day, mature) if mature <= 29 else None
    first = int(raw["planted_day"]) + rule.first_yield_day - 1
    candidate = max(day, first)
    remainder = (candidate - first) % rule.interval
    if remainder:
        candidate += rule.interval - remainder
    return candidate if candidate <= 28 else None


def _crop_current_state(state: State, asset: AssetState) -> CurrentAssetState:
    raw = dict(asset.official)
    identifier = _asset_id(asset)
    crop = asset.asset_type
    rule = rules.CROPS[crop]
    held = int(raw.get("yield_units", 0))
    end = _day_end(state)
    today: list[ProgrammeEvent] = []
    next_events: list[ProgrammeEvent] = []
    if (not raw.get("watered_today", False) and
            int(raw.get("consecutive_unwatered", 0)) >= 1):
        today.append(_event(
            state.step, 20, identifier + ":water", "WATER",
            tile=asset.position, asset_id=identifier, action=("WATER",),
            deadline=end, source="SURVIVAL"))
    next_day = _next_crop_production_day(raw, state.day)
    if next_day is not None:
        next_events.append(_event(
            max(state.step, next_day * 24), 0, identifier + ":next-output",
            "OUTPUT_DUE", tile=asset.position, asset_id=identifier,
            item=crop, quantity=1, mandatory=False,
            source="OFFICIAL_REFRESH"))
    mature = state.day - int(raw.get("planted_day", state.day)) >= rule.first_yield_day
    terminal = state.day >= 29
    expiring = int(raw.get("max_lifespan_step", -1)) in range(
        state.step, end + 2)
    release = None
    if held and mature and (terminal or expiring):
        today.append(_event(
            state.step, 30, identifier + ":harvest", "HARVEST",
            tile=asset.position, asset_id=identifier, item=crop,
            quantity=held, action=("HARVEST",), deadline=end,
            source="TERMINAL" if terminal else "EXPIRY"))
        if not rule.ongoing:
            release = min(state.step + 1, rules.TERMINAL_ACTION_STEP + 1)
    return CurrentAssetState(
        asset_id=identifier, asset_type=crop, tile=asset.position,
        official=raw, held_product=crop, held_quantity=held,
        today_events=tuple(sorted(today)),
        next_events=tuple(sorted(next_events)),
        physical_release_step=release,
    )


def read_current_assets(
    state: State, *,
    prior_assets: Iterable[CurrentAssetState] = (),
    prior_shops: Iterable[str] = (),
    inputs: AnimalDecisionInputs | None = None,
) -> tuple[CurrentAssetState, ...]:
    inputs = inputs or AnimalDecisionInputs()
    prior = {asset.asset_id: asset for asset in prior_assets}
    shops_changed = bool(prior) and tuple(prior_shops) != tuple(state.shops)
    next_steps = {
        _asset_id(asset): _production_step(next_animal_production_day(
            asset.official, state.day))
        for asset in state.own.animals
    }
    needs_rejudge = any(
        _decision_event(
            state, asset.official, prior.get(identifier), shops_changed,
            inputs.replacement_at(asset.position), next_steps[identifier],
            _production_is_cashable(
                state, asset.position, next_steps[identifier])) is not None
        for asset in state.own.animals
        for identifier in (_asset_id(asset),)
    )
    if needs_rejudge:
        horizon = max((step for step in next_steps.values()
                       if step is not None), default=state.step)
        supply = _base_supply_calendar(state, horizon)
        demand = known_demand_events(state, end_step=horizon)
    else:
        # A frozen cycle does not rebuild its supply or demand forecast.
        supply = {}
        demand = {}
    assets = []
    for animal in state.own.animals:
        identifier = _asset_id(animal)
        assets.append(_animal_current_state(
            state, animal, prior.get(identifier), shops_changed, inputs,
            supply, demand))
    assets.extend(_crop_current_state(state, asset)
                  for asset in state.own.crops)
    return tuple(sorted(assets, key=lambda asset: (
        asset.tile[1], asset.tile[0], asset.asset_id)))


def current_asset_programmes(
    current_assets: Iterable[CurrentAssetState],
) -> tuple[AssetProgramme, ...]:
    result = []
    for current in current_assets:
        service = tuple(event for event in current.today_events
                        if event.kind != "HARVEST")
        harvest = tuple(event for event in current.today_events
                        if event.kind == "HARVEST")
        result.append(AssetProgramme(
            current.asset_id, current.asset_type, current.tile,
            "EXIT" if current.animal_mode == EXIT else "KEEP", True,
            "CURRENT", current.physical_release_step, service, (), harvest,
        ))
    return tuple(result)
