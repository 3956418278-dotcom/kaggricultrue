"""Lightweight observation-backed decisions for assets already on the farm."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from . import rules
from .market import buy_cost, forecast_inventory, next_reveal, sell_revenue
from .midgame_config import DEFAULT_MIDGAME_PARAMETERS, MidgameParameters
from .programme import AssetProgramme, CurrentAssetState, ProgrammeEvent
from .state import AssetState, Position, State

PRODUCE = "PRODUCE"
MAINTAIN = "MAINTAIN"
EXIT = "EXIT"
GROW = "GROW"
HARVEST = "HARVEST"


@dataclass(frozen=True)
class DailyValue:
    normal: float
    fertilized: float
    selected: float
    production_days: int
    normal_output: int
    fertilized_output: int
    fertilizer_units: int


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


def _price(state: State, item: str) -> int:
    return int(state.market.price.get(
        item, rules.market_price(item, state.market.inventory[item])))


def conservative_f_price(
    state: State,
    params: MidgameParameters = DEFAULT_MIDGAME_PARAMETERS,
) -> int:
    current = _price(state, "FERTILIZER")
    return max(1, int(current * params.fertilizer_value_discount)) \
        if current > 0 else 0


def _fertilizer_use_cost(state: State, quantity: int) -> int:
    owned = min(max(0, quantity), state.owned_total("FERTILIZER"))
    deficit = max(0, quantity - owned)
    return (
        sell_revenue(
            "FERTILIZER", owned,
            int(state.market.inventory["FERTILIZER"]))
        + buy_cost(
            "FERTILIZER", deficit,
            int(state.market.inventory["FERTILIZER"])))


def _animal_cycle_days(animal: str) -> int:
    rule = rules.ANIMALS[animal]
    return rule.first_yield_day + (rule.max_held - 1) * rule.interval


def _minimum_survival_feed_units(days: int) -> int:
    consecutive = 0
    units = 0
    for _ in range(max(0, days)):
        if consecutive:
            units += 1
            consecutive = 0
        else:
            consecutive = 1
    return units


def animal_daily_value(
    state: State,
    animal: str,
    params: MidgameParameters = DEFAULT_MIDGAME_PARAMETERS,
    *, product_price: int | None = None,
) -> float:
    """Current snapshot value of one maximum effective animal cycle."""
    rule = rules.ANIMALS[animal]
    days = _animal_cycle_days(animal)
    wheat_units = _minimum_survival_feed_units(days)
    wheat_cost = buy_cost(
        "WHEAT", wheat_units,
        int(state.market.inventory.get("WHEAT", rules.MARKET_I0)))
    # Fertilizer cannot stack.  Count only the units available on already
    # necessary maintenance/harvest visits, not every theoretical future night.
    realizable_f = max(1, wheat_units)
    numerator = (
        rule.max_held * (_price(state, rule.product) if product_price is None else product_price)
        + realizable_f * conservative_f_price(state, params)
        - rule.cost
        - wheat_cost
    )
    return numerator / max(1, days)


def _fertilizer_applications(crop: str) -> int:
    rule = rules.CROPS[crop]
    window_start = (rule.max_yield_day + 1) // 2
    if rule.ongoing:
        output_days = [
            rule.first_yield_day + offset * rule.interval
            for offset in range(rule.max_yield)
        ]
        applications = 0
        covered_through = -1
        for day in output_days:
            if day > covered_through:
                applications += 1
                covered_through = day + 2
        return applications
    units = 1
    applications = 0
    covered_through = -1
    for day in range(window_start, rule.max_yield_day + 1):
        if units >= rule.max_yield:
            break
        if day > covered_through:
            applications += 1
            covered_through = day + 2
        units += min(2, rule.max_yield - units)
    return applications


def crop_daily_value(state: State, crop: str) -> DailyValue:
    """Current maximum-production snapshot for a newly planted crop."""
    rule = rules.CROPS[crop]
    price = _price(state, crop)
    days = max(1, (
        rule.first_yield_day + (rule.max_yield - 1) * rule.interval
        if rule.ongoing else rule.max_yield_day
    ))
    if rule.ongoing:
        normal_output = rule.max_yield
        # Maximum effective operation harvests before the official held cap
        # would discard a fertilized production unit.
        fertilized_output = 2 * rule.max_yield
    else:
        productive_days = max(
            0, rule.max_yield_day - (rule.max_yield_day + 1) // 2 + 1)
        normal_output = min(rule.max_yield, 1 + productive_days)
        fertilized_output = min(rule.max_yield, 1 + 2 * productive_days)
    fertilizer_units = _fertilizer_applications(crop)
    normal = (normal_output * price - rule.seed_cost) / days
    fertilized = (
        fertilized_output * price - rule.seed_cost
        - _fertilizer_use_cost(state, fertilizer_units)
    ) / days
    return DailyValue(
        normal, fertilized, max(normal, fertilized), days,
        normal_output, fertilized_output, fertilizer_units)


def next_animal_production_day(
    raw: Mapping[str, object], day: int, *, strictly_after: bool = False
) -> int | None:
    rule = rules.ANIMALS[str(raw["animal"])]
    first = int(raw["placed_day"]) + rule.first_yield_day - 1
    candidate = max(day + int(strictly_after), first)
    remainder = (candidate - first) % rule.interval
    if remainder:
        candidate += rule.interval - remainder
    last_refresh_day = rules.TERMINAL_ACTION_STEP // rules.TURNS_PER_DAY - 1
    return candidate if candidate <= last_refresh_day else None


def _production_step(day: int | None) -> int | None:
    return None if day is None else (day + 1) * rules.TURNS_PER_DAY


def completed_production_count(raw: Mapping[str, object], day: int) -> int:
    """Count productions already completed in the observation on `day`."""
    crop = str(raw["crop"])
    rule = rules.CROPS[crop]
    if not rule.ongoing:
        return 0
    # A refresh on closing day D becomes visible on observation day D+1.
    first_production_day = int(raw["planted_day"]) + rule.first_yield_day
    if day < first_production_day:
        return 0
    count = (day - first_production_day) // rule.interval + 1
    return min(count, rule.max_yield)


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
    if candidate > 28:
        return None
    # Check if this production would exceed max_yield.
    production_number = (candidate - first) // rule.interval + 1
    if production_number > rule.max_yield:
        return None
    return candidate


def _care_gain(state: State, asset: AssetState) -> tuple[int, bool]:
    raw = asset.official
    rule = rules.ANIMALS[asset.asset_type]
    if bool(raw.get("cared_today", False)):
        return 0, False
    if int(raw.get("pending_care_bonus", 0)) > 0:
        return _price(state, rule.product), True
    production_day = next_animal_production_day(
        raw, state.day, strictly_after=True)
    if production_day is None:
        return 0, False
    held = int(raw.get("yield_units", 0))
    # The next base unit and one CARE unit must both fit.
    if held + 2 > rule.max_held:
        return 0, False
    extra_wheat = 0 if (
        bool(raw.get("fed_today", False))
        or int(raw.get("consecutive_unfed", 0)) >= 1
    ) else 1
    wheat_cost = buy_cost(
        "WHEAT", extra_wheat,
        int(state.market.inventory.get("WHEAT", rules.MARKET_I0)))
    gain = _price(state, rule.product) - wheat_cost
    return gain, gain > 0


def animal_locality_bonus(
    asset_type: str,
    tile: Position,
    day: int,
    board_size: int = rules.BOARD_SIZE,
    params: MidgameParameters = DEFAULT_MIDGAME_PARAMETERS,
) -> float:
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
    distance_factor = max(
        0.0, 1.0 - distance / max(1, params.locality_distance_span))
    return params.locality_peak_bonus * day_factor * distance_factor


def _animal_current_state(
    state: State,
    asset: AssetState,
    prior: CurrentAssetState | None,
    params: MidgameParameters,
) -> CurrentAssetState:
    raw = dict(asset.official)
    identifier = _asset_id(asset)
    rule = rules.ANIMALS[asset.asset_type]
    held = int(raw.get("yield_units", 0))
    care_gain, care_possible = _care_gain(state, asset)
    production_day = next_animal_production_day(raw, state.day)
    first_output_complete = state.day >= int(raw["placed_day"]) + rule.first_yield_day
    forecast_price = (rules.market_price(rule.product, forecast_inventory(
        state, rule.product, _production_step(production_day), params=params))
        if production_day is not None else 0)
    daily = animal_daily_value(
        state, asset.asset_type, params, product_price=forecast_price)

    # EXIT is sticky until the official escape transition releases the tile.
    if not first_output_complete:
        mode = MAINTAIN
    elif prior is not None and prior.mode == EXIT:
        mode = EXIT
    elif daily <= 0 or production_day is None:
        mode = EXIT
    elif care_possible:
        mode = PRODUCE
    else:
        mode = MAINTAIN

    end = _day_end(state)
    today: list[ProgrammeEvent] = []
    next_events: list[ProgrammeEvent] = []
    consecutive = int(raw.get("consecutive_unfed", 0))
    fed = bool(raw.get("fed_today", False))
    production_step = _production_step(production_day)
    production_tonight = production_step == (state.day + 1) * 24

    if mode != EXIT:
        realize_pending = (
            production_tonight
            and int(raw.get("pending_care_bonus", 0)) > 0)
        minimum_feed = not fed and consecutive >= 1
        care_target_day = next_animal_production_day(
            raw, state.day, strictly_after=True)
        start_care = (
            mode == PRODUCE and care_possible
            and int(raw.get("pending_care_bonus", 0)) == 0
            and care_target_day is not None)
        if not fed and (minimum_feed or realize_pending or start_care):
            today.append(_event(
                state.step, 20, identifier + ":feed", "FEED",
                tile=asset.position, asset_id=identifier, item="WHEAT",
                quantity=1, action=("FEED",), deadline=end,
                source=("MINIMUM_MAINTENANCE" if minimum_feed
                        else "PRODUCTION_CYCLE")))
        if start_care:
            today.append(_event(
                state.step, 21, identifier + ":care", "CARE",
                tile=asset.position, asset_id=identifier,
                action=("CARE",), deadline=end, source="PRODUCTION_CYCLE"))

    projected = dict(raw)
    if any(event.kind == "FEED" for event in today):
        projected["fed_today"] = True
    incoming = (
        rules.animal_production_on_refresh(projected, state.day)
        if production_tonight else 0)
    overflow = incoming > 0 and held + incoming > rule.max_held

    if mode == EXIT and held > 0:
        today.append(_event(
            state.step, 30, identifier + ":liquidate", "HARVEST",
            tile=asset.position, asset_id=identifier, item=rule.product,
            quantity=held, action=("HARVEST",), deadline=end,
            source="EXIT_LIQUIDATION"))
    elif held > 0 and (overflow or state.day >= 29):
        source = "HELD_OVERFLOW" if overflow else "TERMINAL_LIQUIDATION"
        today.append(_event(
            state.step, 30, identifier + ":harvest", "HARVEST",
            tile=asset.position, asset_id=identifier, item=rule.product,
            quantity=held, action=("HARVEST",), deadline=end, source=source))

    fertilizer_available = bool(raw.get("fertilizer_available", False))
    if fertilizer_available and (
        mode == EXIT or bool(today)
    ) and conservative_f_price(state, params) > 0:
        today.append(_event(
            state.step, 35, identifier + ":collect-f", "COLLECT_F",
            tile=asset.position, asset_id=identifier, item="FERTILIZER",
            quantity=1, action=("COLLECT_FERTILIZER",), deadline=end,
            source=("EXIT_LIQUIDATION" if mode == EXIT else "VISIT_ONLY")))

    survives = mode != EXIT or consecutive < 1 or fed
    if production_tonight and survives and incoming > 0:
        held_after_harvest = 0 if any(
            event.kind == "HARVEST" for event in today) else held
        quantity = min(rule.max_held - held_after_harvest, incoming)
        if quantity > 0:
            next_events.append(_event(
                production_step, 0, identifier + ":next-output", "OUTPUT",
                tile=asset.position, asset_id=identifier, item=rule.product,
                quantity=quantity, mandatory=False,
                source="OFFICIAL_REFRESH"))
    if survives:
        next_events.append(_event(
            (state.day + 1) * 24, 1, identifier + ":next-f", "F_AVAILABLE",
            tile=asset.position, asset_id=identifier, item="FERTILIZER",
            quantity=1, mandatory=False, source="OFFICIAL_REFRESH"))

    release = None
    liquidation = None
    if mode == EXIT:
        liquidation = state.step
        nights = 1 if consecutive >= 1 and not fed else 2
        release = min(
            (state.day + nights) * 24,
            rules.TERMINAL_ACTION_STEP + 1)

    return CurrentAssetState(
        asset_id=identifier,
        asset_type=asset.asset_type,
        tile=asset.position,
        official=raw,
        held_product=rule.product,
        held_quantity=held,
        mode=mode,
        daily_value=daily,
        care_approved=(mode == PRODUCE and care_possible),
        input_gain=care_gain,
        next_production_step=production_step,
        today_events=tuple(sorted(today)),
        next_events=tuple(sorted(next_events)),
        liquidation_step=liquidation,
        physical_release_step=release,
    )


def _one_time_plan(
    state: State, asset: AssetState
) -> tuple[int | None, bool, bool, int]:
    """Return harvest day, WATER today, FERTILIZE today, incremental gain."""
    raw = asset.official
    crop = asset.asset_type
    rule = rules.CROPS[crop]
    planted = int(raw["planted_day"])
    age = state.day - planted
    held = int(raw.get("yield_units", 0))
    first_day = planted + rule.first_yield_day
    last_day = min(planted + rule.max_yield_day, 29)
    if crop in {"WHEAT", "CARROT"}:
        reveal = next_reveal(state.day)
        last_day = min(last_day, reveal)
    candidates = range(max(state.day, first_day), last_day + 1)
    price = _price(state, crop)
    land_daily = max(
        0.0,
        *(animal_daily_value(state, animal)
          for animal in rules.ANIMALS),
        *(crop_daily_value(state, candidate).selected
          for candidate in ("TOMATO", "STRAWBERRY", "MELON")),
    )
    if state.day >= first_day and state.day > last_day and held > 0:
        return state.day, False, False, held * price
    best: tuple[int, int, int, bool, bool] | None = None
    for harvest_day in candidates:
        for use_f in (False, True):
            quantity = held
            applications = 0
            covered = int(raw.get("fertilized_until_day", -1))
            for day in range(state.day, harvest_day + 1):
                day_age = day - planted
                window_start = (rule.max_yield_day + 1) // 2
                if not window_start <= day_age <= rule.max_yield_day:
                    continue
                if quantity >= rule.max_yield:
                    continue
                if use_f and day > covered:
                    applications += 1
                    covered = day + 2
                quantity += min(
                    2 if covered >= day else 1,
                    rule.max_yield - quantity)
            value = (
                quantity * price
                - _fertilizer_use_cost(state, applications)
                - (harvest_day - state.day) * land_daily)
            # Prefer earlier harvest, then no F, on exact cash ties.
            candidate = (
                value, -harvest_day, int(not use_f),
                use_f, harvest_day == state.day)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    if best is None:
        return None, False, False, 0
    _, neg_day, _, use_f, harvest_now = best
    harvest_day = -neg_day
    productive_water = rules.one_time_water_gain(
        crop,
        planted_day=planted,
        day=state.day,
        yield_units=held,
        fertilized_until_day=(
            state.day + 2 if use_f else
            int(raw.get("fertilized_until_day", -1))),
        watered_today=bool(raw.get("watered_today", False)),
    ) > 0
    water = (
        not harvest_now
        and not bool(raw.get("watered_today", False))
        and (
            productive_water
            or int(raw.get("consecutive_unwatered", 0)) >= 1
        )
    )
    fertilize = (
        water and productive_water and use_f
        and int(raw.get("fertilized_until_day", -1)) < state.day)
    baseline = held * price if age >= rule.first_yield_day else 0
    return harvest_day, water, fertilize, best[0] - baseline


def _crop_current_state(
    state: State, asset: AssetState
) -> CurrentAssetState:
    raw = dict(asset.official)
    identifier = _asset_id(asset)
    crop = asset.asset_type
    rule = rules.CROPS[crop]
    held = int(raw.get("yield_units", 0))
    end = _day_end(state)
    today: list[ProgrammeEvent] = []
    next_events: list[ProgrammeEvent] = []
    value = crop_daily_value(state, crop)
    input_gain = 0
    next_harvest = None
    mode = GROW
    lifecycle_done = False

    if rule.ongoing:
        production_day = _next_crop_production_day(raw, state.day)
        production_step = _production_step(production_day)
        production_tonight = production_step == (state.day + 1) * 24
        # Lifecycle completion: all productions exhausted.
        lifecycle_done = (
            production_day is None
            and completed_production_count(raw, state.day) >= rule.max_yield
        )
        incoming = (
            rules.crop_production_on_refresh(raw, state.day)
            if production_tonight else 0)
        terminal = state.day >= 29 or lifecycle_done
        survival_water = (
            not terminal
            and
            not bool(raw.get("watered_today", False))
            and int(raw.get("consecutive_unwatered", 0)) >= 1)
        fertilized = int(raw.get("fertilized_until_day", -1)) >= state.day
        fertilize = False
        bonus_water = False
        if (not terminal and production_tonight
                and not bool(raw.get("watered_today", False))):
            if fertilized:
                input_gain = _price(state, crop)
                bonus_water = input_gain > 0
            else:
                input_gain = (
                    _price(state, crop)
                    - _fertilizer_use_cost(state, 1))
                fertilize = input_gain > 0
                bonus_water = fertilize
        if fertilize:
            today.append(_event(
                state.step, 18, identifier + ":fertilize", "FERTILIZE",
                tile=asset.position, asset_id=identifier,
                item="FERTILIZER", quantity=1, action=("FERTILIZE",),
                deadline=end, source="NEXT_PRODUCTION"))
        if survival_water or bonus_water:
            today.append(_event(
                state.step, 20, identifier + ":water", "WATER",
                tile=asset.position, asset_id=identifier,
                action=("WATER",), deadline=end,
                source=("SURVIVAL" if survival_water else
                        "NEXT_PRODUCTION")))
        projected = dict(raw)
        if survival_water or bonus_water:
            projected["watered_today"] = True
        if fertilize:
            projected["fertilized_until_day"] = state.day + 2
        incoming = (
            rules.crop_production_on_refresh(projected, state.day)
            if production_tonight else 0)
        overflow = held > 0 and incoming > 0 and held + incoming > rule.max_yield
        lifespan = int(raw.get("max_lifespan_step", -1))
        expiring = 0 <= lifespan <= end
        if held > 0 and (overflow or expiring or state.day >= 29 or lifecycle_done):
            source = (
                "HELD_OVERFLOW" if overflow else
                "TERMINAL_LIQUIDATION" if state.day >= 29 else
                "LIFECYCLE_DONE" if lifecycle_done else "EXPIRY")
            today.append(_event(
                state.step, 30, identifier + ":harvest", "HARVEST",
                tile=asset.position, asset_id=identifier, item=crop,
                quantity=held, action=("HARVEST",), deadline=end,
                source=source))
            mode = HARVEST
            next_harvest = state.step
        if lifecycle_done:
            # Ongoing HARVEST never removes the plant. Keep cleanup in the
            # same tile bundle, strictly after the final harvest if needed.
            today.append(_event(
                state.step + int(held > 0), 31, identifier + ":dig", "DIG",
                tile=asset.position, asset_id=identifier,
                action=("DIG",), deadline=end,
                source="LIFECYCLE_DONE"))
            mode = HARVEST
            next_harvest = state.step
        elif production_step is not None:
            next_events.append(_event(
                production_step, 0, identifier + ":next-output",
                "OUTPUT_DUE", tile=asset.position, asset_id=identifier,
                item=crop, quantity=1, mandatory=False,
                source="OFFICIAL_REFRESH"))
    else:
        harvest_day, water, fertilize, input_gain = _one_time_plan(
            state, asset)
        next_harvest = (
            None if harvest_day is None else
            max(state.step, harvest_day * rules.TURNS_PER_DAY))
        if fertilize:
            today.append(_event(
                state.step, 18, identifier + ":fertilize", "FERTILIZE",
                tile=asset.position, asset_id=identifier,
                item="FERTILIZER", quantity=1, action=("FERTILIZE",),
                deadline=end, source="MAX_DAILY_VALUE"))
        if water:
            today.append(_event(
                state.step, 20, identifier + ":water", "WATER",
                tile=asset.position, asset_id=identifier,
                action=("WATER",), deadline=end,
                source=("SURVIVAL" if int(
                    raw.get("consecutive_unwatered", 0)) >= 1
                    else "MAX_DAILY_VALUE")))
        if harvest_day == state.day and held > 0:
            source = (
                "TERMINAL_LIQUIDATION" if state.day >= 29
                else "CURRENT_HARVEST_VALUE")
            today.append(_event(
                state.step, 30, identifier + ":harvest", "HARVEST",
                tile=asset.position, asset_id=identifier, item=crop,
                quantity=held, action=("HARVEST",), deadline=end,
                source=source))
            mode = HARVEST
        elif harvest_day is not None:
            next_events.append(_event(
                next_harvest, 0, identifier + ":next-harvest",
                "HARVEST_DUE", tile=asset.position, asset_id=identifier,
                item=crop, quantity=held, mandatory=False,
                source="CURRENT_HARVEST_VALUE"))

    release = (
        state.step + 1 + int(rule.ongoing and held > 0)
        if mode == HARVEST and (not rule.ongoing or lifecycle_done) else None)
    return CurrentAssetState(
        asset_id=identifier,
        asset_type=crop,
        tile=asset.position,
        official=raw,
        held_product=crop,
        held_quantity=held,
        mode=mode,
        daily_value=value.selected,
        input_gain=input_gain,
        next_production_step=(
            _production_step(_next_crop_production_day(raw, state.day))
            if rule.ongoing else None),
        next_harvest_step=next_harvest,
        today_events=tuple(sorted(today)),
        next_events=tuple(sorted(next_events)),
        physical_release_step=release,
    )


def read_current_assets(
    state: State,
    *,
    prior_assets: Iterable[CurrentAssetState] = (),
    prior_shops: Iterable[str] = (),
    params: MidgameParameters = DEFAULT_MIDGAME_PARAMETERS,
) -> tuple[CurrentAssetState, ...]:
    # Prior state is used only to keep an in-progress EXIT sticky.  Every other
    # decision is derived from the same real observation path used at runtime.
    del prior_shops
    prior = {asset.asset_id: asset for asset in prior_assets}
    assets = [
        _animal_current_state(
            state, animal, prior.get(_asset_id(animal)), params)
        for animal in state.own.animals
    ]
    assets.extend(_crop_current_state(state, crop)
                  for crop in state.own.crops)
    return tuple(sorted(
        assets,
        key=lambda asset: (asset.tile[1], asset.tile[0], asset.asset_id)))


def exit_current_asset(
    state: State,
    asset: AssetState,
    params: MidgameParameters = DEFAULT_MIDGAME_PARAMETERS,
) -> CurrentAssetState:
    prior = CurrentAssetState(
        _asset_id(asset), asset.asset_type, asset.position,
        dict(asset.official), mode=EXIT)
    return _animal_current_state(state, asset, prior, params)


def current_asset_programmes(
    current_assets: Iterable[CurrentAssetState],
) -> tuple[AssetProgramme, ...]:
    result = []
    for current in current_assets:
        service = tuple(
            event for event in current.today_events
            if event.kind != "HARVEST")
        harvest = tuple(
            event for event in current.today_events
            if event.kind == "HARVEST")
        result.append(AssetProgramme(
            current.asset_id,
            current.asset_type,
            current.tile,
            "EXIT" if current.mode == EXIT else "KEEP",
            True,
            "CURRENT",
            current.physical_release_step,
            service,
            (),
            harvest,
        ))
    return tuple(result)
