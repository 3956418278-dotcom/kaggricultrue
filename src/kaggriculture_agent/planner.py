"""Observation-driven midgame planner; it may attach on any game day."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Iterable

from . import rules
from .current_assets import (
    _minimum_survival_feed_units,
    animal_daily_value,
    animal_locality_bonus,
    conservative_f_price,
    crop_daily_value,
    current_asset_programmes,
    read_current_assets,
)
from .intraday import PlanningFailure, clear_route_cache, solve_intraday
from .market import buy_cost, forecast_inventory, next_reveal, optimize_short_sales
from .midgame_config import (
    MidgameParameters,
    coerce_midgame_parameters,
)
from .programme import (
    AssetProgramme,
    CurrentAssetState,
    LandProgramme,
    Programme,
    ProgrammeEvent,
)
from .return_requirements import annotate_programme_returns
from .simulation import drop_arrivals
from .state import Position, State

LONG_ASSETS = (
    "COW", "SHEEP", "GOOSE", "STRAWBERRY", "TOMATO", "MELON")


def _pe(step, priority, identifier, kind, *, tile=None, asset=None,
        item=None, quantity=0, action=("PASS",), mandatory=True,
        deadline=None, source=None):
    return ProgrammeEvent(
        step, priority, identifier, kind, tile, asset, item, quantity,
        tuple(action), mandatory, deadline, None, source)


def _day_end(state: State) -> int:
    return min((state.day + 1) * rules.TURNS_PER_DAY - 1,
               rules.TERMINAL_ACTION_STEP)


def _tile_raw_for_plan(
    state: State, tile: Position, unlocked: set[str]
):
    observed = state.tile_at(tile)
    if observed.is_locked and rules.quadrant(
            tile, state.board_size) in unlocked:
        return None
    return observed.raw


def _animal_programme(
    state: State,
    asset_type: str,
    tile: Position,
    *,
    existing: bool,
    raw=None,
    start_step=None,
    purpose="LONG",
    unlocked_quadrants: Iterable[str] | None = None,
) -> AssetProgramme:
    """Compile only today's purchase/build/place commitment for a new animal."""
    if existing:
        raise ValueError(
            "existing animals must be read through CurrentAssetState")
    start = max(state.step + 1, state.step if start_step is None else start_step)
    end = _day_end(state)
    identifier = f"new:{asset_type}:{tile[0]}:{tile[1]}:{start}"
    structure = rules.ANIMALS[asset_type].structure
    unlocked = set(unlocked_quadrants or state.own.owned_land)
    opening = _tile_raw_for_plan(state, tile, unlocked)
    kind = opening.get("kind") if isinstance(opening, dict) else None
    service: list[ProgrammeEvent] = []
    cursor = start
    if kind == "WEED" or (
            kind in {"COOP", "PASTURE"} and kind != structure):
        service.append(_pe(
            cursor, 9, identifier + ":dig", "DIG", tile=tile,
            asset=identifier, action=("DIG",), deadline=end))
        cursor += 1
        kind = None
    if kind != structure:
        service.append(_pe(
            cursor, 10, identifier + ":build", "BUILD", tile=tile,
            asset=identifier, action=("BUILD_" + structure,),
            deadline=end))
        cursor += 1
    service.append(_pe(
        cursor, 11, identifier + ":place", "PLACE", tile=tile,
        asset=identifier, item=asset_type, quantity=1,
        action=("PLACE", asset_type), deadline=end))
    return AssetProgramme(
        identifier, asset_type, tile, "NEW", False, purpose, None,
        tuple(service), (), ())


def _crop_programme(
    state: State,
    crop: str,
    tile: Position,
    *,
    existing: bool,
    raw=None,
    start_step=None,
    purpose="LONG",
    release_step=None,
    unlocked_quadrants: Iterable[str] | None = None,
) -> AssetProgramme:
    """Compile only today's plant and opening WATER for a new crop."""
    if existing:
        raise ValueError(
            "existing crops must be read through CurrentAssetState")
    start = max(state.step + 1, state.step if start_step is None else start_step)
    end = _day_end(state)
    identifier = f"new:{crop}:{tile[0]}:{tile[1]}:{start}"
    unlocked = set(unlocked_quadrants or state.own.owned_land)
    opening = _tile_raw_for_plan(state, tile, unlocked)
    kind = opening.get("kind") if isinstance(opening, dict) else None
    service: list[ProgrammeEvent] = []
    cursor = start
    if kind in {"WEED", "COOP", "PASTURE"}:
        service.append(_pe(
            cursor, 9, identifier + ":dig", "DIG", tile=tile,
            asset=identifier, action=("DIG",), deadline=end))
        cursor += 1
    service.append(_pe(
        cursor, 10, identifier + ":plant", "PLANT", tile=tile,
        asset=identifier, item=crop, quantity=1,
        action=("PLANT", crop), deadline=end, source=purpose))
    if cursor + 1 <= end:
        service.append(_pe(
            cursor + 1, 11, identifier + ":water", "WATER", tile=tile,
            asset=identifier, action=("WATER",), deadline=end,
            source=purpose))
    return AssetProgramme(
        identifier, crop, tile, "NEW", False, purpose, release_step,
        tuple(service), (), ())


def _programme_from_assets(
    state: State,
    assets: Iterable[AssetProgramme],
    extra_events=(),
    land=(),
    current_assets: Iterable[CurrentAssetState] = (),
):
    """Build commitments with one global animal/seed/input ledger."""
    assets = tuple(assets)
    events = list(extra_events)
    for asset in assets:
        events.extend(asset.service_schedule)
        events.extend(asset.harvest_schedule)

    animal_need = Counter()
    seed_need = Counter()
    for asset in assets:
        if asset.existing:
            continue
        if asset.asset_type in rules.ANIMALS:
            animal_need[asset.asset_type] += 1
        elif any(
                event.kind == "PLANT"
                for event in asset.service_schedule):
            seed_need[asset.asset_type] += 1

    for animal, quantity in sorted(animal_need.items()):
        deficit = max(0, quantity - state.owned_total(animal))
        if deficit:
            events.append(_pe(
                state.step, -30, f"buy:animal:{animal}", "BUY_ANIMAL",
                item=animal, quantity=deficit, source="NEW_ASSET"))
    for crop, quantity in sorted(seed_need.items()):
        deficit = max(0, quantity - int(state.seeds.get(crop, 0)))
        if deficit:
            events.append(_pe(
                state.step, -29, f"buy:seed:{crop}", "BUY_SEED",
                item=crop, quantity=deficit, source="SEED_LEDGER"))

    unique = {event.event_id: event for event in events}
    return Programme(
        state.step,
        state.day,
        state.shops,
        assets,
        tuple(sorted(unique.values())),
        land=tuple(land),
        current_assets=tuple(current_assets),
    )


def _arrivals(state: State, programme: Programme):
    """Only route-proven DROP/EOD arrivals are sellable."""
    return drop_arrivals(state, programme) if programme.routes else {}


def _shed_consumptions(programme: Programme):
    result: dict[int, Counter] = {}
    for route in programme.routes:
        for step, action in route.actions.items():
            if action and action[0] == "PICKUP":
                amount = int(action[2]) if len(action) > 2 else 1
                result.setdefault(step, Counter())[str(action[1])] += amount
    return {step: dict(amounts) for step, amounts in result.items()}


def _planned_quadrants(
    state: State, land: Iterable[LandProgramme]
) -> set[str]:
    return set(state.own.owned_land) | {entry.quadrant for entry in land}


def _unused_tiles(state: State, programme: Programme):
    occupied = {
        asset.tile for asset in programme.assets
        if asset.release_turn is None
        or asset.release_turn > _day_end(state)
    }
    unlocked = _planned_quadrants(state, programme.land)
    return tuple(
        tile.position
        for tile in state.own.tiles
        if tile.position not in occupied
        and (
            not tile.is_locked
            or rules.quadrant(tile.position, state.board_size) in unlocked
        )
        and tile.animal is None
        and (tile.crop is None or any(
            a.tile == tile.position and a.release_turn is not None
            and a.release_turn <= _day_end(state)
            and (not rules.CROPS[a.asset_type].ongoing or any(
                e.kind == "DIG" for e in a.service_schedule))
            for a in programme.assets if a.asset_type in rules.CROPS))
    )


def _mature_wheat_assets(
    state: State,
    current_assets: Iterable[CurrentAssetState],
):
    result = []
    for current in current_assets:
        if current.asset_type != "WHEAT" or current.held_quantity <= 0:
            continue
        planted = int(current.official.get("planted_day", state.day))
        if state.day - planted >= rules.CROPS["WHEAT"].first_yield_day:
            result.append(current)
    return sorted(
        result,
        key=lambda asset: (
            rules.distance_to_shed(asset.tile, state.board_size),
            asset.tile[1], asset.tile[0]))


def _replace_asset_events(
    assets: tuple[AssetProgramme, ...],
    additions: Iterable[ProgrammeEvent],
):
    by_asset: dict[str, list[ProgrammeEvent]] = {}
    for event in additions:
        by_asset.setdefault(event.asset_id or "", []).append(event)
    result = []
    for asset in assets:
        extra = by_asset.get(asset.asset_id, ())
        if extra:
            result.append(replace(
                asset,
                harvest_schedule=tuple(sorted(
                    (*asset.harvest_schedule, *extra)))))
        else:
            result.append(asset)
    return tuple(result)


def _resolve_inputs(
    state: State,
    programme: Programme,
) -> Programme:
    """Use real stock, then mature Wheat, then BUY; never plant for FEED."""
    events = [e for e in programme.events if not (
        e.kind == "BUY_PRODUCT" and e.item in {"WHEAT", "FERTILIZER"})]
    assets = programme.assets
    feed_events = [
        event for event in events if event.kind == "FEED"]
    current_wheat = state.shed.get("WHEAT", 0)
    needed_from_harvest = max(0, len(feed_events) - current_wheat)
    promoted: set[str] = set()
    for event in sorted(events):
        if (needed_from_harvest > 0 and event.kind == "HARVEST"
                and event.item == "WHEAT"
                and event.asset_id is not None
                and event.asset_id.startswith("current:")):
            promoted.add(event.event_id)
            needed_from_harvest -= event.quantity
    if promoted:
        events = [
            replace(event, source="FEED_SUPPLY")
            if event.event_id in promoted else event
            for event in events
        ]
        assets = tuple(replace(
            asset,
            harvest_schedule=tuple(
                replace(event, source="FEED_SUPPLY")
                if event.event_id in promoted else event
                for event in asset.harvest_schedule),
        ) for asset in assets)

    wheat_available = current_wheat + sum(
        event.quantity for event in events
        if event.kind == "HARVEST" and event.item == "WHEAT"
        and event.asset_id is not None
        and event.asset_id.startswith("current:"))
    deficit = max(0, len(feed_events) - wheat_available)

    forced_harvests: list[ProgrammeEvent] = []
    if deficit:
        existing = {
            current.asset_id: current
            for current in programme.current_assets
        }
        already_harvested = {
            event.asset_id for event in events
            if event.kind == "HARVEST" and event.item == "WHEAT"
        }
        for current in _mature_wheat_assets(
                state, programme.current_assets):
            if current.asset_id in already_harvested:
                continue
            quantity = min(deficit, current.held_quantity)
            event = _pe(
                state.step, 29,
                current.asset_id + ":feed-supply-harvest",
                "HARVEST",
                tile=current.tile,
                asset=current.asset_id,
                item="WHEAT",
                quantity=current.held_quantity,
                action=("HARVEST",),
                deadline=min(
                    (event_.deadline for event_ in feed_events),
                    default=_day_end(state)),
                source="FEED_SUPPLY",
            )
            forced_harvests.append(event)
            events.append(event)
            deficit -= quantity
            if deficit <= 0:
                break
        del existing

    if deficit:
        events.append(_pe(
            state.step, -28, "buy:product:WHEAT", "BUY_PRODUCT",
            item="WHEAT", quantity=deficit, source="FEED_DEFICIT"))
        events = [
            replace(event, step=max(event.step, state.step + 1))
            if event.kind == "FEED" else event
            for event in events
        ]

    fertilizer_need = sum(
        max(1, event.quantity)
        for event in events if event.kind == "FERTILIZE")
    collect_events = [
        event for event in events if event.kind == "COLLECT_F"]
    collected_for_use = min(
        max(0, fertilizer_need - state.shed.get("FERTILIZER", 0)),
        len(collect_events),
    )
    relay_ids = {
        event.event_id for event in collect_events[:collected_for_use]}
    if relay_ids:
        events = [
            replace(event, source="F_RELAY")
            if event.event_id in relay_ids else event
            for event in events
        ]
        assets = tuple(replace(
            asset,
            service_schedule=tuple(
                replace(event, source="F_RELAY")
                if event.event_id in relay_ids else event
                for event in asset.service_schedule),
        ) for asset in assets)
    fertilizer_deficit = max(
        0,
        fertilizer_need
        - state.shed.get("FERTILIZER", 0)
        - collected_for_use,
    )
    if fertilizer_deficit:
        events.append(_pe(
            state.step, -27, "buy:product:FERTILIZER", "BUY_PRODUCT",
            item="FERTILIZER", quantity=fertilizer_deficit,
            source="APPROVED_FERTILIZE"))
        events = [
            replace(event, step=max(event.step, state.step + 1))
            if event.kind == "FERTILIZE" else event
            for event in events
        ]

    if forced_harvests:
        assets = _replace_asset_events(assets, forced_harvests)
    unique = {event.event_id: event for event in events}
    return replace(
        programme,
        assets=assets,
        events=tuple(sorted(unique.values())))


def _asset_daily_value(
    state: State, kind: str, params: MidgameParameters, programme=None
) -> float:
    if kind in rules.ANIMALS:
        return _forecast_animal_daily_value(state, kind, params, programme)
    return _forecast_crop_daily_value(state, kind, params, programme)


def _forecast_animal_daily_value(
    state: State, animal: str, params: MidgameParameters, programme=None
) -> float:
    """Forecast-aware animal daily value for new placement decisions."""
    rule = rules.ANIMALS[animal]
    # Count productions achievable before terminal.
    first_prod_day = state.day + rule.first_yield_day - 1
    if first_prod_day > 28:
        return 0.0
    productions = 0
    for k in range(rule.max_held):
        prod_day = first_prod_day + k * rule.interval
        if prod_day > 28:
            break
        productions += 1
    if productions == 0:
        return 0.0
    forecast_inv = forecast_inventory(
        state, rule.product, (first_prod_day + 1) * 24,
        planned_assets=programme.additions if programme else (),
        commitments=programme.events if programme else (), params=params,
    )
    forecast_price = rules.market_price(rule.product, forecast_inv)
    days = rule.first_yield_day + (productions - 1) * rule.interval
    wheat_units = _minimum_survival_feed_units(days)
    wheat_cost = buy_cost(
        "WHEAT", wheat_units,
        int(state.market.inventory.get("WHEAT", rules.MARKET_I0)))
    realizable_f = max(1, wheat_units)
    numerator = (
        productions * forecast_price
        + realizable_f * conservative_f_price(state, params)
        - rule.cost
        - wheat_cost
    )
    return numerator / max(1, days)


def _forecast_crop_daily_value(state: State, crop: str, params=None, programme=None) -> float:
    """Forecast-aware crop daily value for new placement decisions."""
    rule = rules.CROPS[crop]
    params = coerce_midgame_parameters(params)
    target_step = (state.day + rule.first_yield_day) * 24
    if target_step > rules.TERMINAL_ACTION_STEP:
        return 0.0
    forecast_price = rules.market_price(crop, forecast_inventory(
        state, crop, target_step,
        planned_assets=programme.additions if programme else (),
        commitments=programme.events if programme else (), params=params))
    if not rule.ongoing:
        # Preserve the existing Melon normal/fertilized day convention, but
        # cap both yields to the water window still reachable before terminal.
        days = min(rule.max_yield_day, 29 - state.day)
        productive = max(0, days - (rule.max_yield_day + 1) // 2 + 1)
        normal_output = min(rule.max_yield, 1 + productive)
        fertilized_output = min(rule.max_yield, 1 + 2 * productive)
        snapshot = crop_daily_value(state, crop)
        # Use the same explicit F input ledger as the existing crop snapshot.
        from .current_assets import _fertilizer_use_cost
        f_cost = _fertilizer_use_cost(state, snapshot.fertilizer_units)
        return max(normal_output * forecast_price - rule.seed_cost,
                   fertilized_output * forecast_price - rule.seed_cost - f_cost) / max(1, days)
    # For ongoing crops, count achievable productions before terminal.
    first_prod_day = state.day + rule.first_yield_day - 1
    if first_prod_day > 28:
        return 0.0
    productions = 0
    for k in range(rule.max_yield):
        prod_day = first_prod_day + k * rule.interval
        if prod_day > 28:
            break
        productions += 1
    if productions == 0:
        return 0.0
    days = max(1, rule.first_yield_day + (productions - 1) * rule.interval)
    numerator = productions * forecast_price - rule.seed_cost
    return numerator / days


def _production_days(kind: str) -> int:
    if kind in rules.ANIMALS:
        rule = rules.ANIMALS[kind]
        return rule.first_yield_day + (
            rule.max_held - 1) * rule.interval
    value = rules.CROPS[kind]
    return (
        value.first_yield_day + (value.max_yield - 1) * value.interval
        if value.ongoing else value.max_yield_day)


def _remaining_effective_days(
    state: State, kind: str, start_day: int | None = None
) -> int:
    start_day = state.day if start_day is None else start_day
    first = (
        rules.ANIMALS[kind].first_yield_day
        if kind in rules.ANIMALS
        else rules.CROPS[kind].first_yield_day)
    if start_day + first > 29:
        return 0
    return max(0, 30 - start_day)


def _unit_cash_cost(state: State, kind: str) -> int:
    if kind in rules.ANIMALS:
        return rules.ANIMALS[kind].cost
    return rules.CROPS[kind].seed_cost


def _committed_purchase_cost(state: State, programme: Programme) -> int:
    cost = 0
    product_inventory = Counter()
    hires = state.hires_today
    for event in sorted(programme.events):
        if event.kind == "BUY_ANIMAL":
            cost += rules.ANIMALS[event.item].cost * event.quantity
        elif event.kind == "BUY_SEED":
            cost += rules.CROPS[event.item].seed_cost * event.quantity
        elif event.kind == "BUY_PRODUCT":
            inventory = int(state.market.inventory[event.item]) \
                - product_inventory[event.item]
            cost += sum(
                rules.market_price(event.item, inventory - offset - 1)
                for offset in range(event.quantity))
            product_inventory[event.item] += event.quantity
        elif event.kind == "BUY_LAND":
            cost += event.quantity
        elif event.kind == "HIRE":
            for _ in range(event.quantity):
                cost += rules.fibonacci_hire_cost(hires)
                hires += 1
    return cost


def _candidate_tile(
    state: State,
    programme: Programme,
    kind: str,
) -> Position | None:
    tiles = _unused_tiles(state, programme)
    if not tiles:
        return None
    expected = (
        rules.ANIMALS[kind].structure
        if kind in rules.ANIMALS else None)

    def terrain_rank(tile: Position):
        raw = _tile_raw_for_plan(
            state, tile, _planned_quadrants(state, programme.land))
        if any(a.tile == tile and a.release_turn is not None
               and a.release_turn <= _day_end(state) for a in programme.assets):
            raw = None
        tile_kind = raw.get("kind") if isinstance(raw, dict) else None
        if kind in rules.ANIMALS:
            rank = 0 if tile_kind == expected else 1 if raw is None else 2
        else:
            rank = 0 if raw is None else 1
        return (
            rank,
            rules.distance_to_shed(tile, state.board_size),
            tile[1],
            tile[0],
        )
    return min(tiles, key=terrain_rank)


def _wait_reveal_allows_start(
    state: State,
    kind: str,
    daily_value: float,
) -> bool:
    reveal = next_reveal(state.day)
    if reveal >= 31:
        return True
    now_days = _remaining_effective_days(state, kind, state.day)
    wait_days = _remaining_effective_days(state, kind, reveal)
    # Same current snapshot value is used on both sides.  No unknown shop is
    # assigned a value.
    now_value = daily_value * now_days
    buffer_daily = max(
        0.0,
        crop_daily_value(state, "WHEAT").selected,
        crop_daily_value(state, "CARROT").selected,
    )
    buffer_days = min(
        reveal - state.day,
        max(
            crop_daily_value(state, "WHEAT").production_days,
            crop_daily_value(state, "CARROT").production_days,
        ),
    )
    wait_value = buffer_daily * buffer_days + daily_value * wait_days
    return now_value >= wait_value


def _new_asset(
    state: State,
    kind: str,
    tile: Position,
    programme: Programme,
) -> AssetProgramme:
    unlocked = _planned_quadrants(state, programme.land)
    release = max((a.release_turn for a in programme.assets
                   if a.tile == tile and a.release_turn is not None),
                  default=state.step + 1)
    if kind in rules.ANIMALS:
        return _animal_programme(
            state, kind, tile, existing=False,
            unlocked_quadrants=unlocked, start_step=release)
    return _crop_programme(
        state, kind, tile, existing=False,
        unlocked_quadrants=unlocked, start_step=release)


def _ranked_kind(
    state: State,
    programme: Programme,
    params: MidgameParameters,
) -> tuple[str, Position, float] | None:
    if state.turn > rules.TURNS_PER_DAY - 4:
        return None
    candidates = []
    for kind in LONG_ASSETS:
        daily = _asset_daily_value(state, kind, params, programme)
        if daily <= 0 or _remaining_effective_days(state, kind) <= 0:
            continue
        if not _wait_reveal_allows_start(state, kind, daily):
            continue
        tile = _candidate_tile(state, programme, kind)
        if tile is None:
            continue
        score = daily
        if kind in rules.ANIMALS:
            score *= 1.0 + animal_locality_bonus(
                kind, tile, state.day, state.board_size, params)
        candidates.append((score, kind, tile, daily))
    if not candidates:
        return None
    _, kind, tile, daily = max(
        candidates,
        key=lambda entry: (
            entry[0], -entry[2][1], -entry[2][0], entry[1]))
    return kind, tile, daily


def _land_expansion(
    state: State,
    programme: Programme,
    params: MidgameParameters,
) -> Programme:
    owned = len(_planned_quadrants(state, programme.land))
    if owned >= 3 or state.turn >= rules.TURNS_PER_DAY - 3:
        return programme
    quadrant = rules.LAND_ORDER[owned - 1]
    price = rules.LAND_PRICES[owned - 1]

    # Compute free cash after all existing commitments.
    free_cash = state.money - _committed_purchase_cost(state, programme)
    # BUY_LAND immediately converts the new quadrant's LOCKED tiles to EMPTY.
    free_tiles = sum(
        rules.quadrant(tile.position, state.board_size) == quadrant
        and (tile.is_locked or tile.is_empty) for tile in state.own.tiles)
    wheat_seed_fallback = free_tiles * rules.CROPS["WHEAT"].seed_cost
    threshold = price + wheat_seed_fallback + params.fixed_cash_reserve
    if free_cash <= threshold:
        return programme

    tiles = sorted(
        (
            tile.position for tile in state.own.tiles
            if rules.quadrant(tile.position, state.board_size) == quadrant
        ),
        key=lambda tile: (
            rules.distance_to_shed(tile, state.board_size),
            tile[1], tile[0]))
    exact = tiles[0]
    event = _pe(
        state.step, -40, f"land:{quadrant}", "BUY_LAND",
        item=quadrant, quantity=price, source="CASH_FUNDED_EXPANSION")
    land = LandProgramme(
        quadrant, state.step, price, exact,
        "CONTINUE_PLACEMENT")
    return replace(
        programme,
        events=tuple(sorted((*programme.events, event))),
        land=(*programme.land, land))


def _long_candidate_loop(
    state: State,
    programme: Programme,
    params: MidgameParameters,
) -> Programme:
    """Greedily deploy positive current-value assets on exact chosen tiles."""
    while True:
        ranked = _ranked_kind(state, programme, params)
        if ranked is None:
            return programme
        kind, tile, _ = ranked
        candidate = _new_asset(state, kind, tile, programme)
        trial_assets = (*programme.assets, candidate)
        trial = _programme_from_assets(
            state,
            trial_assets,
            extra_events=tuple(
                event for event in programme.events
                if event.kind == "BUY_LAND"),
            land=programme.land,
            current_assets=programme.current_assets,
        )
        trial = _resolve_inputs(state, trial)
        if _committed_purchase_cost(state, trial) > state.money:
            # This kind is unaffordable; try no lower-valued forced substitute.
            return programme
        programme = trial


def _buffer_daily_value(state: State, crop: str) -> float:
    reveal = min(next_reveal(state.day), 29)
    available_days = reveal - state.day
    rule = rules.CROPS[crop]
    if available_days < rule.first_yield_day:
        return 0.0
    if available_days >= rule.max_yield_day:
        return crop_daily_value(state, crop).selected
    price = int(state.market.price.get(
        crop, rules.market_price(crop, state.market.inventory[crop])))
    return (price - rule.seed_cost) / rule.first_yield_day


def _buffer_loop(state: State, programme: Programme) -> Programme:
    choices = sorted(
        ((
            _buffer_daily_value(state, crop),
            1 if crop == "WHEAT" else 0,
            crop,
        ) for crop in ("WHEAT", "CARROT")),
        reverse=True,
    )
    if not choices or choices[0][0] <= 0:
        return programme
    _, _, crop = choices[0]
    while True:
        tile = _candidate_tile(state, programme, crop)
        if tile is None:
            return programme
        candidate = replace(_new_asset(state, crop, tile, programme),
                            purpose="W_BUFFER" if crop == "WHEAT" else "C_BUFFER")
        trial = _programme_from_assets(
            state,
            (*programme.assets, candidate),
            tuple(event for event in programme.events
                  if event.kind == "BUY_LAND"),
            programme.land,
            programme.current_assets,
        )
        if _committed_purchase_cost(state, trial) > state.money:
            return programme
        programme = trial


def _annotate_buffers(programme: Programme) -> Programme:
    wheat = {}
    carrot = {}
    for asset in programme.additions:
        starts = tuple(sorted(
            event.step for event in asset.service_schedule
            if event.kind == "PLANT"))
        if asset.purpose == "W_BUFFER":
            wheat[asset.tile] = starts
        elif asset.purpose == "C_BUFFER":
            carrot[asset.tile] = starts
    return replace(
        programme, wheat_buffer=wheat, carrot_buffer=carrot)


def _market_event_count(programme: Programme, step: int) -> int:
    count = 0
    for event in programme.events_at(step):
        if event.kind == "HIRE":
            count += event.quantity
        elif event.kind in {
                "BUY_LAND", "BUY_ANIMAL", "BUY_SEED", "BUY_PRODUCT"}:
            count += 1
    return count


def _validate_market_capacity(programme: Programme) -> Programme:
    for step in sorted({event.step for event in programme.events}):
        if _market_event_count(programme, step) > rules.MAX_MARKET_ORDERS:
            return replace(
                programme,
                feasible=False,
                diagnostics={
                    **programme.diagnostics,
                    "failure": f"market order capacity exceeded at {step}",
                })
    return programme


def _route_input_deficits(state: State, programme: Programme):
    """Replay private carry and shared shed in official worker action order.

    Missing pickup units are counted as required JIT purchases. They are never
    borrowed from a different worker's inventory. No route search occurs here.
    """
    items = {"WHEAT", "FERTILIZER"}
    shed = Counter({p: state.shed.get(p, 0) for p in items})
    inventory = {w: Counter(state.workers[w].inventory if w < len(state.workers) else {})
                 for w in range(programme.worker_count)}
    required = Counter()
    positions = {w: state.workers[w].position if w < len(state.workers)
                 else rules.shed_access(state.board_size)[w % 4]
                 for w in inventory}
    harvests = {a.tile: {e.item: e.quantity for e in a.harvest_schedule}
                for a in programme.assets if a.harvest_schedule}
    moves = {"NORTH": (0, -1), "SOUTH": (0, 1), "WEST": (-1, 0), "EAST": (1, 0)}
    actions = sorted((step, route.worker, action) for route in programme.routes
                     for step, action in route.actions.items() if step >= state.step)
    for step, worker, action in actions:
        op = action[0]
        inv = inventory[worker]
        if op in moves:
            dx, dy = moves[op]
            x, y = positions[worker]
            positions[worker] = (x + dx, y + dy)
        elif op == "PICKUP" and action[1] in items:
            item, qty = action[1], int(action[2])
            missing = max(0, qty - shed[item])
            required[item] += missing
            shed[item] += missing - qty
            inv[item] += qty
        elif op == "DROP":
            for item in items:
                shed[item] += inv[item]
                inv[item] = 0
        elif op == "HARVEST":
            for item, qty in harvests.get(positions[worker], {}).items():
                if item in items:
                    inv[item] += qty
        elif op == "COLLECT_FERTILIZER":
            inv["FERTILIZER"] += 1
        elif op in {"FEED", "FERTILIZE"}:
            item = "WHEAT" if op == "FEED" else "FERTILIZER"
            if inv[item] <= 0:
                raise PlanningFailure("route consumes an input not carried by its worker")
            inv[item] -= 1
    return required


def _reconcile_carried_inputs(state: State, solved: Programme) -> Programme:
    required = _route_input_deficits(state, solved)
    purchased = Counter()
    for event in solved.events:
        if event.kind == "BUY_PRODUCT":
            purchased[event.item] += event.quantity
    if any(required[p] > purchased[p] for p in ("WHEAT", "FERTILIZER")):
        raise PlanningFailure("route's actual pickup precedes its available input supply")
    events = tuple(replace(e, quantity=min(e.quantity, required[e.item]))
                   if e.kind == "BUY_PRODUCT" and e.item in {"WHEAT", "FERTILIZER"}
                   else e for e in solved.events)
    events = tuple(e for e in events if e.kind != "BUY_PRODUCT" or e.quantity > 0)
    if events == solved.events:
        return solved
    # At most one recompilation. If the reduced purchase changes assignment in
    # a way that loses the proven resource path, retain the conservative plan.
    try:
        adjusted = solve_intraday(state, replace(solved, events=events))
        needs = _route_input_deficits(state, adjusted)
        buys = Counter()
        for e in adjusted.events:
            if e.kind == "BUY_PRODUCT":
                buys[e.item] += e.quantity
        if any(needs[p] > buys[p] for p in ("WHEAT", "FERTILIZER")):
            return solved
        return adjusted
    except PlanningFailure:
        return solved


def _actual_cleanup_releases(state: State, solved: Programme) -> Programme:
    releases = {}
    ids = {a.asset_id for a in solved.current_assets
           if any(e.kind == "DIG" and e.source == "LIFECYCLE_DONE"
                  for e in a.today_events)}
    cleanup_tiles = {a.tile for a in solved.current_assets if a.asset_id in ids}
    retimed = {}
    for route in solved.routes:
        position = (state.workers[route.worker].position
                    if route.worker < len(state.workers) else
                    rules.shed_access(state.board_size)[route.worker % 4])
        moves = {"NORTH": (0, -1), "SOUTH": (0, 1), "WEST": (-1, 0), "EAST": (1, 0)}
        for step, action in sorted(route.actions.items()):
            if action[0] in moves:
                dx, dy = moves[action[0]]
                position = (position[0] + dx, position[1] + dy)
            elif action[0] == "DIG":
                releases[position] = step + 1
            if position in cleanup_tiles and action[0] not in moves and action[0] != "PASS":
                event = next((e for e in solved.events if e.tile == position
                              and e.action == action and e.event_id not in retimed), None)
                if event is not None:
                    retimed[event.event_id] = replace(event, step=step)
    def events(values):
        return tuple(sorted(retimed.get(e.event_id, e) for e in values))
    return replace(solved,
        events=events(solved.events),
        assets=tuple(replace(a,
                     release_turn=releases.get(a.tile) if a.asset_id in ids else a.release_turn,
                     service_schedule=events(a.service_schedule),
                     harvest_schedule=events(a.harvest_schedule)) for a in solved.assets),
        current_assets=tuple(replace(a, physical_release_step=releases.get(a.tile),
                             today_events=events(a.today_events))
                             if a.asset_id in ids else a for a in solved.current_assets))


def _solve_daily(
    state: State,
    programme: Programme,
    params: MidgameParameters,
) -> Programme:
    programme = annotate_programme_returns(state, programme, params=params)
    try:
        solved = solve_intraday(state, programme)
        solved = _reconcile_carried_inputs(state, solved)
        solved = _actual_cleanup_releases(state, solved)
    except PlanningFailure as exc:
        return replace(
            programme,
            feasible=False,
            diagnostics={
                **programme.diagnostics,
                "failure": str(exc),
            })
    solved = _validate_market_capacity(solved)
    if not solved.feasible and solved.diagnostics.get("failure"):
        return solved
    arrivals = _arrivals(state, solved)
    sales = optimize_short_sales(
        state,
        arrivals,
        consumptions=_shed_consumptions(solved),
        commitments=solved.events,
    )
    if not sales.feasible:
        return replace(
            solved,
            feasible=False,
            diagnostics={**solved.diagnostics, "failure": sales.failure},
        )
    return replace(
        solved,
        planned_sale=sales.planned_sale,
        market_inventory=sales.market_inventory,
        terminal_cash=state.money + sales.revenue
                      - _committed_purchase_cost(state, solved),
        feasible=True,
    )


def make_plan(
    state: State,
    config=None,
    *,
    prior_programme: Programme | None = None,
) -> Programme:
    """Build one observation-backed daily plan; no terminal economic rollout."""
    params = coerce_midgame_parameters(config)
    clear_route_cache()
    prior_assets = (
        prior_programme.current_assets
        if prior_programme is not None else ())
    current = read_current_assets(
        state,
        prior_assets=prior_assets,
        prior_shops=(
            prior_programme.shops if prior_programme is not None else ()),
        params=params,
    )
    assets = current_asset_programmes(current)
    programme = _programme_from_assets(
        state, assets, current_assets=current)
    programme = _resolve_inputs(state, programme)

    programme = _long_candidate_loop(state, programme, params)
    programme = _resolve_inputs(state, programme)
    programme = _buffer_loop(state, programme)
    programme = _resolve_inputs(state, programme)

    # Worker hire expenditure is known only after the real route is compiled.
    # If it breaks cash feasibility, remove the latest new commitment and
    # rebuild the same daily line; no asset is accepted on paper and funded by
    # a nonexistent balance.
    without_land = None
    while True:
        solved = _solve_daily(state, programme, params)
        if solved.feasible:
            if without_land is None:
                # Existing land is completely funded, including real hires,
                # before the strict cash gate considers one new quadrant.
                without_land = solved
                expanded = _land_expansion(state, solved, params)
                if expanded.land != solved.land:
                    programme = _long_candidate_loop(state, expanded, params)
                    programme = _buffer_loop(state, programme)
                    programme = _resolve_inputs(state, programme)
                    continue
            return _annotate_buffers(solved)
        if without_land is not None:
            return _annotate_buffers(without_land)
        additions = [
            asset for asset in programme.assets if not asset.existing]
        if additions:
            remove_id = additions[-1].asset_id
            remaining = tuple(
                asset for asset in programme.assets
                if asset.asset_id != remove_id)
            land_events = tuple(
                event for event in programme.events
                if event.kind == "BUY_LAND")
            programme = _programme_from_assets(
                state, remaining, land_events, programme.land,
                programme.current_assets)
            programme = _resolve_inputs(state, programme)
            continue
        if programme.land:
            programme = _programme_from_assets(
                state, programme.assets, (), (), programme.current_assets)
            programme = _resolve_inputs(state, programme)
            continue
        return solved
