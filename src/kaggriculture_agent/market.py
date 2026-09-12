"""Event-accurate known demand, visible pressure, and sale optimization."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping

from . import rules
from .state import AssetState, State

REVEAL_DAYS = (4, 7, 10, 13, 16, 19, 22, 25)


def next_reveal(day: int) -> int:
    return next((candidate for candidate in REVEAL_DAYS if candidate > day), 31)


@lru_cache(maxsize=None)
def sell_revenue(product: str, quantity: int, inventory: int) -> int:
    revenue = 0
    for _ in range(max(0, quantity)):
        price = rules.market_price(product, inventory)
        revenue += price
        if price > rules.PRICE_FLOOR:       # pinned engine's floor exception
            inventory += 1
    return revenue


def buy_cost(product: str, quantity: int, inventory: int) -> int:
    cost = 0
    for _ in range(max(0, quantity)):
        inventory -= 1
        cost += rules.market_price(product, inventory)
    return cost


def known_demand_events(state: State) -> dict[str, tuple[tuple[int, int], ...]]:
    events: dict[str, list[tuple[int, int]]] = {item: [] for item in rules.PRODUCTS}
    for step in range(state.step, rules.TERMINAL_ACTION_STEP + 1):
        amounts: dict[str, int] = defaultdict(int)
        if step % 4 == 0:
            for shop in state.shops:
                basket = rules.SHOPS.get(shop, ())
                for item in basket:
                    amounts[item] += 2 if len(basket) == 1 else 1
        if step % rules.TURNS_PER_DAY == 0:
            for item in rules.PRODUCTS:
                if item != "FERTILIZER":
                    amounts[item] += 1
        for item, quantity in amounts.items():
            events[item].append((step, quantity))
    return {item: tuple(values) for item, values in events.items()}


def _minimal_feed_days(asset: AssetState, start_day: int, last_day: int) -> set[int]:
    consecutive = int(asset.official.get("consecutive_unfed", 0))
    fed_today = bool(asset.official.get("fed_today", False))
    days: set[int] = set()
    for day in range(start_day, last_day + 1):
        if day == start_day and fed_today:
            consecutive = 0
            continue
        if consecutive >= 1:
            days.add(day)
            consecutive = 0
        else:
            consecutive += 1
    return days


def _animal_pressure(asset: AssetState, state: State) -> list[tuple[int, int]]:
    rule = rules.ANIMALS[asset.asset_type]
    raw = asset.official
    result: list[tuple[int, int]] = []
    held = int(raw.get("yield_units", 0))
    if held:
        result.append((state.step, held))
        held = 0
    last_day = rules.TERMINAL_ACTION_STEP // rules.TURNS_PER_DAY
    feed = _minimal_feed_days(asset, state.day, last_day)
    pending = int(raw.get("pending_care_bonus", 0))
    for day in range(state.day, last_day):
        next_day = day + 1
        since = next_day - int(raw["placed_day"]) - rule.first_yield_day
        if since >= 0 and since % rule.interval == 0:
            quantity = 1 + (pending if day in feed else 0)
            pending = 0
            step = next_day * rules.TURNS_PER_DAY
            if step <= rules.TERMINAL_ACTION_STEP:
                result.append((step, quantity))
    return result


def _crop_pressure(asset: AssetState, state: State) -> list[tuple[int, int]]:
    rule = rules.CROPS[asset.asset_type]
    raw = asset.official
    result: list[tuple[int, int]] = []
    held = int(raw.get("yield_units", 0))
    mature = state.day - int(raw["planted_day"]) >= rule.first_yield_day
    if held and mature:
        result.append((state.step, held))
    if not rule.ongoing:
        return result
    last_day = rules.TERMINAL_ACTION_STEP // rules.TURNS_PER_DAY
    for day in range(state.day, last_day):
        next_day = day + 1
        since = next_day - int(raw["planted_day"]) - rule.first_yield_day
        if since >= 0 and since % rule.interval == 0:
            count = since // rule.interval + 1
            step = next_day * rules.TURNS_PER_DAY
            if count <= rule.max_yield and step <= rules.TERMINAL_ACTION_STEP:
                result.append((step, 1))
    return result


def opponent_pressure(state: State) -> dict[str, tuple[tuple[int, int], ...]]:
    pressure: dict[str, list[tuple[int, int]]] = {item: [] for item in rules.PRODUCTS}
    for asset in state.opp.visible_animals:
        pressure[rules.ANIMALS[asset.asset_type].product].extend(_animal_pressure(asset, state))
    for asset in state.opp.visible_crops:
        pressure[asset.asset_type].extend(_crop_pressure(asset, state))
    merged: dict[str, tuple[tuple[int, int], ...]] = {}
    for item, events in pressure.items():
        by_step: dict[int, int] = defaultdict(int)
        for step, quantity in events:
            by_step[step] += quantity
        merged[item] = tuple(sorted(by_step.items()))
    return merged


@dataclass(frozen=True)
class SalePlan:
    revenue: int
    planned_sale: Mapping[int, Mapping[str, int]]
    market_inventory: Mapping[int, Mapping[str, int]]


_PRODUCT_DP_CACHE = {}


def clear_sale_cache():
    _PRODUCT_DP_CACHE.clear()


def _product_dp(product: str, opening_stock: int, opening_inventory: int,
                arrivals: Mapping[int, int], demand: Mapping[int, int],
                pressure: Mapping[int, int], start_step: int,
                forced_minimum: Mapping[int, int] | None = None):
    forced = dict(forced_minimum or {})
    cache_key=(product,opening_stock,opening_inventory,tuple(sorted(arrivals.items())),
               tuple(sorted(demand.items())),tuple(sorted(pressure.items())),start_step,
               tuple(sorted(forced.items())))
    if cache_key in _PRODUCT_DP_CACHE:
        return _PRODUCT_DP_CACHE[cache_key]
    if not any(pressure.values()):
        # With demand only, later inventory is never higher absent our sales;
        # postponing every non-forced unit weakly dominates selling it earlier.
        event_steps=sorted({start_step,rules.TERMINAL_ACTION_STEP,*arrivals,*forced,*demand})
        stock=opening_stock; inventory=opening_inventory; schedule=[]; revenue=0
        for step in event_steps:
            stock+=arrivals.get(step,0)
            inventory-=demand.get(step,0)
            quantity=stock if step==rules.TERMINAL_ACTION_STEP else min(stock,forced.get(step,0))
            schedule.append((step,quantity)); stock-=quantity
            revenue+=sell_revenue(product,quantity,inventory)
            for _ in range(quantity):
                if rules.market_price(product,inventory)>rules.PRICE_FLOOR:inventory+=1
        result=(revenue,tuple(schedule))
        _PRODUCT_DP_CACHE[cache_key]=result
        return result
    event_steps = sorted({start_step, rules.TERMINAL_ACTION_STEP, *arrivals, *demand, *pressure})

    @lru_cache(maxsize=None)
    def value(index: int, stock: int, inventory: int):
        step = event_steps[index]
        stock += arrivals.get(step, 0)
        inventory += pressure.get(step, 0)
        inventory -= demand.get(step, 0)
        minimum = forced.get(step, 0)
        best = None
        min_q = min(minimum, stock)
        for quantity in range(min_q, stock + 1):
            revenue = sell_revenue(product, quantity, inventory)
            next_inventory = inventory
            for _ in range(quantity):
                if rules.market_price(product, next_inventory) > rules.PRICE_FLOOR:
                    next_inventory += 1
            if index + 1 == len(event_steps):
                candidate = (revenue, -quantity, quantity, ())
            else:
                future, schedule = value(index + 1, stock - quantity, next_inventory)
                candidate = (revenue + future, -quantity, quantity, schedule)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        assert best is not None
        return best[0], ((step, best[2]), *best[3])

    result=value(0, opening_stock, opening_inventory)
    _PRODUCT_DP_CACHE[cache_key]=result
    return result


def optimize_sales(state: State, arrivals: Mapping[int, Mapping[str, int]],
                   demand_events: Mapping[str, Iterable[tuple[int, int]]],
                   pressure_events: Mapping[str, Iterable[tuple[int, int]]],
                   commitments: Iterable[object] = ()) -> SalePlan:
    """Solve each product DP, then enforce shared shed capacity and cash deadlines jointly.

    At an overflow event or cash deficit event, forced sales are selected across all
    products by their exact re-solved continuation loss, maintaining exact market
    inventory trajectories.
    """
    demand = {p: dict(events) for p, events in demand_events.items()}
    pressure = {p: dict(events) for p, events in pressure_events.items()}
    by_product_arrivals = {p: {} for p in rules.PRODUCTS}
    for step, amounts in arrivals.items():
        for product, quantity in amounts.items():
            by_product_arrivals[product][step] = by_product_arrivals[product].get(step, 0) + quantity

    commit_list = [e for e in commitments if getattr(e, "kind", None) in {"BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "BUY_LAND", "HIRE"}]

    forced: dict[str, dict[int, int]] = {p: {} for p in rules.PRODUCTS}

    def solve_all():
        total = 0
        schedules = {}
        for product in rules.PRODUCTS:
            result = _product_dp(product, state.shed.get(product, 0), state.market.inventory[product],
                by_product_arrivals[product], demand.get(product, {}), pressure.get(product, {}),
                state.step, forced[product])
            total += result[0]
            schedules[product] = dict(result[1])
        return total, schedules

    revenue, schedules = solve_all()
    event_steps = sorted({state.step, *arrivals, *(s for d in demand.values() for s in d),
                          *(s for d in pressure.values() for s in d),
                          *(e.step for e in commit_list if hasattr(e, "step")),
                          rules.TERMINAL_ACTION_STEP})

    while True:
        stock = {p: state.shed.get(p, 0) for p in rules.PRODUCTS}
        market_inv = {p: state.market.inventory[p] for p in rules.PRODUCTS}
        money = state.money
        hires = state.hires_today
        land_unlocked_count = len(state.own.owned_land)

        commit_by_step = defaultdict(list)
        for e in commit_list:
            commit_by_step[e.step].append(e)

        overflow = None
        cash_deficit = None

        for step in event_steps:
            for product, quantity in arrivals.get(step, {}).items():
                stock[product] += quantity
            for product, quantity in pressure.get(step, {}).items():
                market_inv[product] += quantity
            for product, quantity in demand.get(step, {}).items():
                market_inv[product] -= quantity

            for product in rules.PRODUCTS:
                quantity = schedules[product].get(step, 0)
                if quantity:
                    stock[product] -= quantity
                    for _ in range(quantity):
                        price = rules.market_price(product, market_inv[product])
                        money += price
                        if price > rules.PRICE_FLOOR:
                            market_inv[product] += 1

            for e in commit_by_step.get(step, ()):
                if e.kind == "HIRE":
                    for _ in range(getattr(e, "quantity", 1)):
                        cost = rules.fibonacci_hire_cost(hires)
                        money -= cost
                        hires += 1
                elif e.kind == "BUY_LAND":
                    if land_unlocked_count < 4:
                        cost = rules.LAND_PRICES[land_unlocked_count - 1]
                        money -= cost
                        land_unlocked_count += 1
                elif e.kind == "BUY_SEED":
                    money -= rules.CROPS[e.item].seed_cost * getattr(e, "quantity", 1)
                elif e.kind == "BUY_ANIMAL":
                    money -= rules.ANIMALS[e.item].cost * getattr(e, "quantity", 1)
                elif e.kind == "BUY_PRODUCT":
                    for _ in range(getattr(e, "quantity", 1)):
                        cost = rules.market_price(e.item, market_inv[e.item] - 1)
                        money -= cost
                        market_inv[e.item] -= 1

            excess = sum(stock.values()) - rules.SHED_CAPACITY
            if excess > 0:
                overflow = (step, excess)
                break

            if money < 0:
                cash_deficit = (step, -money)
                break

        if overflow is None and cash_deficit is None:
            break

        if overflow is not None:
            step, excess = overflow
            neutral = []
            for product in rules.PRODUCTS:
                already = forced[product].get(step, 0)
                available = stock.get(product, 0)
                avail_at_step = state.shed.get(product, 0) + sum(by_product_arrivals[product].get(s, 0) for s in by_product_arrivals[product] if s <= step)
                sold_by_step = sum(schedules[product].get(s, 0) for s in schedules[product] if s <= step)
                future_sales = [s for s, q in schedules[product].items() if s > step and q > 0]
                if available <= 0 or avail_at_step <= sold_by_step or not future_sales:
                    continue
                future = min(future_sales)
                if not any(step < s <= future for s in demand.get(product, {}) if demand[product][s]) and not any(
                        step < s <= future for s in pressure.get(product, {}) if pressure[product][s]):
                    neutral.append((product, available))
            if neutral:
                for product, available in neutral:
                    take = min(excess, available)
                    forced[product][step] = forced[product].get(step, 0) + take
                    excess -= take
                    if excess == 0:
                        break
                revenue, schedules = solve_all()
                continue

            for _ in range(excess):
                choices = []
                for product in rules.PRODUCTS:
                    already = forced[product].get(step, 0)
                    available = stock.get(product, 0)
                    avail_at_step = state.shed.get(product, 0) + sum(by_product_arrivals[product].get(s, 0) for s in by_product_arrivals[product] if s <= step)
                    sold_by_step = sum(schedules[product].get(s, 0) for s in schedules[product] if s <= step)
                    if available <= 0 or avail_at_step <= sold_by_step:
                        continue
                    trial = {p: dict(v) for p, v in forced.items()}
                    trial[product][step] = already + 1
                    result = _product_dp(product, state.shed.get(product, 0), state.market.inventory[product],
                        by_product_arrivals[product], demand.get(product, {}), pressure.get(product, {}),
                        state.step, trial[product])
                    old = _product_dp(product, state.shed.get(product, 0), state.market.inventory[product],
                        by_product_arrivals[product], demand.get(product, {}), pressure.get(product, {}),
                        state.step, forced[product])
                    choices.append((old[0] - result[0], product))
                if not choices:
                    break
                _, chosen = min(choices)
                forced[chosen][step] = forced[chosen].get(step, 0) + 1
            revenue, schedules = solve_all()
            continue

        if cash_deficit is not None:
            step, deficit = cash_deficit
            choices = []
            for product in rules.PRODUCTS:
                already = forced[product].get(step, 0)
                avail_at_step = state.shed.get(product, 0) + sum(by_product_arrivals[product].get(s, 0) for s in by_product_arrivals[product] if s <= step)
                sold_by_step = sum(schedules[product].get(s, 0) for s in schedules[product] if s <= step)
                if avail_at_step <= sold_by_step:
                    continue
                trial = {p: dict(v) for p, v in forced.items()}
                trial[product][step] = already + 1
                result = _product_dp(product, state.shed.get(product, 0), state.market.inventory[product],
                    by_product_arrivals[product], demand.get(product, {}), pressure.get(product, {}),
                    state.step, trial[product])
                old = _product_dp(product, state.shed.get(product, 0), state.market.inventory[product],
                    by_product_arrivals[product], demand.get(product, {}), pressure.get(product, {}),
                    state.step, forced[product])
                loss = old[0] - result[0]
                price = rules.market_price(product, market_inv[product] + sold_by_step)
                if price > 0:
                    choices.append((loss / price, loss, price, product, avail_at_step - sold_by_step))
            if not choices:
                break
            _, _, chosen_price, chosen, avail = min(choices)
            increment = max(1, min(avail, deficit // chosen_price)) if chosen_price > 0 else 1
            forced[chosen][step] = forced[chosen].get(step, 0) + increment
            revenue, schedules = solve_all()
            continue

    planned: dict[int, dict[str, int]] = defaultdict(dict)
    trajectory: dict[int, dict[str, int]] = defaultdict(dict)
    for product in rules.PRODUCTS:
        inventory = state.market.inventory[product]
        for step in event_steps:
            inventory += pressure.get(product, {}).get(step, 0)
            inventory -= demand.get(product, {}).get(step, 0)
            quantity = schedules[product].get(step, 0)
            if quantity:
                planned[step][product] = quantity
            for _ in range(quantity):
                if rules.market_price(product, inventory) > rules.PRICE_FLOOR:
                    inventory += 1
            trajectory[step][product] = inventory
    return SalePlan(revenue, dict(planned), dict(trajectory))

