"""Pinned Kaggriculture 1.32.7 economics and timing rules.

This module is the policy's single executable owner for constants copied from the
verified official environment.  Keeping them here prevents the planner and executor
from drifting into subtly different games.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


TERMINAL_ACTION_STEP = 718
TURNS_PER_DAY = 24
BOARD_SIZE = 10
SHED_CAPACITY = 100
MAX_MARKET_ORDERS = 10
MARKET_I0 = 10_000
PRICE_FLOOR = 1
LAND_ORDER = ("NE", "SW", "SE")
LAND_PRICES = (1_000, 2_000, 4_000)
PRODUCTS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)
SELLABLE_PRODUCTS = PRODUCTS


@dataclass(frozen=True)
class CropRule:
    seed_cost: int
    first_yield_day: int
    max_yield_day: int
    interval: int
    max_yield: int
    ongoing: bool


@dataclass(frozen=True)
class AnimalRule:
    cost: int
    structure: str
    first_yield_day: int
    interval: int
    max_held: int
    product: str


@dataclass(frozen=True)
class MarketRule:
    base: int
    throughput: int
    below_func: str
    below_target: float
    above_func: str
    above_target: float


CROPS: Mapping[str, CropRule] = {
    "WHEAT": CropRule(10, 2, 4, 0, 6, False),
    "CARROT": CropRule(20, 2, 3, 0, 4, False),
    "TOMATO": CropRule(50, 8, 8, 1, 4, True),
    "STRAWBERRY": CropRule(100, 10, 10, 2, 4, True),
    "MELON": CropRule(80, 10, 12, 0, 6, False),
}

ANIMALS: Mapping[str, AnimalRule] = {
    "GOOSE": AnimalRule(300, "COOP", 4, 1, 4, "EGG"),
    "COW": AnimalRule(400, "PASTURE", 8, 2, 6, "MILK"),
    "SHEEP": AnimalRule(500, "PASTURE", 6, 3, 6, "WOOL"),
}

MARKET: Mapping[str, MarketRule] = {
    "WHEAT": MarketRule(25, 400, "sqrt", 0.80, "log", 0.20),
    "CARROT": MarketRule(35, 450, "hinge", 1.00, "sqrt", 0.70),
    "TOMATO": MarketRule(60, 200, "hinge", 0.40, "sqrt", 0.60),
    "STRAWBERRY": MarketRule(120, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON": MarketRule(250, 300, "log", 0.20, "sq", 3.60),
    "EGG": MarketRule(50, 332, "hinge", 0.40, "log", 0.20),
    "MILK": MarketRule(160, 122, "sqrt", 0.60, "linear", 1.60),
    "WOOL": MarketRule(200, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": MarketRule(100, 200, "linear", 0.40, "linear", 0.40),
}

SHOPS: Mapping[str, tuple[str, ...]] = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}


def _shape(name: str, displacement: float, throughput: float) -> float:
    x = max(0.0, displacement)
    if name == "linear":
        return x
    if name == "sq":
        return x * x
    if name == "sqrt":
        return math.sqrt(x)
    if name == "log":
        return math.log1p(x)
    if name == "hinge":
        u = x / throughput
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    raise ValueError(f"unknown market curve: {name}")


def market_price(item: str, inventory: int) -> int:
    """Return the exact default-environment price at an inventory level."""
    rule = MARKET[item]
    if inventory == MARKET_I0:
        return rule.base
    if inventory < MARKET_I0:
        fn, target = rule.below_func, rule.below_target
        normalizer = _shape(fn, rule.throughput, rule.throughput)
        amplitude = target * rule.base / normalizer
        price = rule.base + amplitude * _shape(
            fn, MARKET_I0 - inventory, rule.throughput
        )
    else:
        fn, target = rule.above_func, rule.above_target
        normalizer = _shape(fn, rule.throughput, rule.throughput)
        amplitude = target * rule.base / normalizer
        price = rule.base - amplitude * _shape(
            fn, inventory - MARKET_I0, rule.throughput
        )
    return max(PRICE_FLOOR, int(round(price)))


def town_demand_per_day(item: str, unlocked_shops: tuple[str, ...]) -> float:
    """Known demand only: town center plus already unlocked shop instances."""
    demand = 0.0 if item == "FERTILIZER" else 1.0
    for shop in unlocked_shops:
        products = SHOPS.get(shop, ())
        if item in products:
            per_tick = 2 if len(products) == 1 else 1
            demand += per_tick * (TURNS_PER_DAY / 4)
    return demand


def projected_sale_revenue(
    item: str,
    quantity: int,
    current_inventory: int,
    sale_step: int,
    current_step: int,
    unlocked_shops: tuple[str, ...],
    prior_own_sales: int = 0,
) -> int:
    """Estimate proceeds from sequential own sales after known town demand.

    Opponent trades and future random shops are deliberately absent in this first
    baseline.  A floor-price sale does not add market supply, matching the engine.
    """
    days = max(0.0, (sale_step - current_step) / TURNS_PER_DAY)
    known_demand = int(town_demand_per_day(item, unlocked_shops) * days)
    inventory = current_inventory - known_demand + prior_own_sales
    revenue = 0
    for _ in range(max(0, quantity)):
        price = market_price(item, inventory)
        revenue += price
        if price > PRICE_FLOOR:
            inventory += 1
    return revenue


def fibonacci_hire_cost(index: int) -> int:
    a, b = 1, 1
    for _ in range(max(0, index)):
        a, b = b, a + b
    return a


def one_time_water_gain(
    crop: str,
    *,
    planted_day: int,
    day: int,
    yield_units: int,
    fertilized_until_day: int,
    watered_today: bool = False,
) -> int:
    """Immediate yield gained by a legal WATER on a one-time crop."""
    rule = CROPS[crop]
    if rule.ongoing or watered_today or yield_units >= rule.max_yield:
        return 0
    age = day - planted_day
    window_start = (rule.max_yield_day + 1) // 2
    if not window_start <= age <= rule.max_yield_day:
        return 0
    bonus = 2 if fertilized_until_day >= day else 1
    return min(bonus, rule.max_yield - yield_units)


def quadrant(position: tuple[int, int], board_size: int = BOARD_SIZE) -> str:
    x, y = position
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def shed_access(board_size: int = BOARD_SIZE) -> tuple[tuple[int, int], ...]:
    half = board_size // 2
    return (
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    )


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def distance_to_shed(position: tuple[int, int], board_size: int = BOARD_SIZE) -> int:
    return min(manhattan(position, access) for access in shed_access(board_size))


def move_toward(start: tuple[int, int], target: tuple[int, int]) -> list[str]:
    """Deterministic one-step Manhattan movement; y grows downward."""
    x, y = start
    tx, ty = target
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["PASS"]


def advance_owned(state, actions, orders=()):
    """One default-contract turn for our farm, with no opponent transactions.

    Branch copies are isolated. Units resolve before orders, town, decay, then
    deterministic refresh. Random weeds/shop unlocks are intentionally omitted
    at the boundary; search stops there and observes the real next day.
    """
    from collections import Counter
    from dataclasses import replace
    from .state import TileState, WorkerState

    tiles = [dict(t.raw) if isinstance(t.raw, Mapping) else t.raw for t in state.tiles]
    positions = [w.position for w in state.workers]
    inventories = [dict(w.inventory) for w in state.workers]
    shed, seeds, market = dict(state.shed), dict(state.seeds), dict(state.market_inventory)
    money, hires = state.money, state.hires_today
    unlocked = list(state.unlocked_quadrants)
    access = shed_access(state.board_size)
    requests = Counter(a[1] for a in actions if a and a[0] == "PLANT")

    def take(inv, item, n=1):
        if inv.get(item, 0) < n:
            return False
        inv[item] -= n
        if not inv[item]:
            del inv[item]
        return True

    def drop(inv):
        for item, n in list(inv.items()):
            shed[item] = shed.get(item, 0) + min(n, max(0, SHED_CAPACITY - sum(shed.values())))
        inv.clear()

    moves = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
    for i, action in enumerate(actions[:len(positions)]):
        if not action:
            continue
        op, pos, inv = action[0], positions[i], inventories[i]
        index = pos[1] * state.board_size + pos[0]
        tile = tiles[index]
        if op in moves:
            dx, dy = moves[op]
            new = (pos[0] + dx, pos[1] + dy)
            if all(0 <= v < state.board_size for v in new):
                positions[i] = new
        elif op == "DROP" and pos in access:
            drop(inv)
        elif op == "PICKUP" and pos in access:
            item = action[1]
            n = min(max(0, int(action[2]) if len(action) > 2 else 1), shed.get(item, 0))
            if n:
                shed[item] -= n
                inv[item] = inv.get(item, 0) + n
        elif op == "PLACE":
            item = action[1]
            if item in ANIMALS and isinstance(tile, dict) and tile.get("kind") == ANIMALS[item].structure and "animal" not in tile:
                if take(inv, item):
                    tiles[index] = dict(kind=ANIMALS[item].structure, animal=item,
                        placed_day=state.day, yield_units=0, consecutive_unfed=0,
                        fed_today=False, cared_today=False, fertilizer_available=False,
                        pending_care_bonus=0)
            elif pos in access:
                n = min(inv.get(item, 0), max(0, int(action[2]) if len(action) > 2 else 1),
                        max(0, SHED_CAPACITY - sum(shed.values())))
                if n and take(inv, item, n):
                    shed[item] = shed.get(item, 0) + n
        elif tile == "LOCKED":
            continue
        elif op == "PLANT" and tile is None:
            crop = action[1]
            if crop in CROPS and requests[crop] <= state.seeds.get(crop, 0):
                # Atomic check uses pre-turn seeds, not the decremented stock.
                pass
            else:
                continue
            seeds[crop] -= 1
            rule = CROPS[crop]
            tiles[index] = dict(kind="PLANT", crop=crop, planted_day=state.day,
                watered_today=False, consecutive_unwatered=1, yield_units=0 if rule.ongoing else 1,
                max_lifespan_step=-1 if rule.ongoing else (state.day + rule.max_yield_day + 1) * TURNS_PER_DAY,
                fertilized_until_day=-1)
        elif op in ("BUILD_COOP", "BUILD_PASTURE") and tile is None:
            tiles[index] = {"kind": "COOP" if op == "BUILD_COOP" else "PASTURE"}
        elif isinstance(tile, dict):
            if op == "WATER" and tile.get("kind") == "PLANT" and not tile.get("watered_today", False):
                tile["yield_units"] = tile.get("yield_units", 0) + one_time_water_gain(
                    tile["crop"], planted_day=tile["planted_day"], day=state.day,
                    yield_units=tile.get("yield_units", 0), fertilized_until_day=tile.get("fertilized_until_day", -1))
                tile["watered_today"] = True
            elif op == "FERTILIZE" and tile.get("kind") == "PLANT" and take(inv, "FERTILIZER"):
                tile["fertilized_until_day"] = max(tile.get("fertilized_until_day", -1), state.day + 2)
            elif op == "HARVEST" and tile.get("yield_units", 0) > 0:
                item = None
                if tile.get("kind") == "PLANT" and state.day - tile["planted_day"] >= CROPS[tile["crop"]].first_yield_day:
                    item = tile["crop"]
                    if not CROPS[item].ongoing:
                        tiles[index] = None
                elif "animal" in tile:
                    item = ANIMALS[tile["animal"]].product
                if item:
                    inv[item] = inv.get(item, 0) + tile["yield_units"]
                    tile["yield_units"] = 0
            elif op == "DIG" and "animal" not in tile:
                tiles[index] = None
            elif "animal" in tile:
                if op == "FEED" and not tile.get("fed_today", False) and take(inv, "WHEAT"):
                    tile["fed_today"] = True
                elif op == "CARE":
                    tile["cared_today"] = True
                elif op == "COLLECT_FERTILIZER" and tile.get("fertilizer_available", False):
                    tile["fertilizer_available"] = False
                    inv["FERTILIZER"] = inv.get("FERTILIZER", 0) + 1

    for order in orders[:MAX_MARKET_ORDERS]:
        op = order[0]
        if op == "HIRE":
            cost = fibonacci_hire_cost(hires)
            if money >= cost:
                money -= cost
                hires += 1
                positions.append(min(access, key=lambda p: (positions.count(p), access.index(p))))
                inventories.append({})
        elif op == "BUY_LAND":
            if len(unlocked) < 4 and money >= LAND_PRICES[len(unlocked) - 1]:
                money -= LAND_PRICES[len(unlocked) - 1]
                new = LAND_ORDER[len(unlocked) - 1]
                unlocked.append(new)
                for i, tile in enumerate(tiles):
                    if tile == "LOCKED" and quadrant((i % state.board_size, i // state.board_size), state.board_size) == new:
                        tiles[i] = None
        else:
            item = order[1]
            for _ in range(max(0, int(order[2]) if len(order) > 2 else 1)):
                if op == "SELL" and shed.get(item, 0):
                    price = market_price(item, market[item])
                    shed[item] -= 1
                    money += price
                    market[item] += price > 1
                elif op in ("BUY_PRODUCT", "BUY_ANIMAL", "BUY_SEED"):
                    price = (market_price(item, market[item] - 1) if op == "BUY_PRODUCT"
                             else ANIMALS[item].cost if op == "BUY_ANIMAL" else CROPS[item].seed_cost)
                    if money < price or (op != "BUY_SEED" and sum(shed.values()) >= SHED_CAPACITY):
                        break
                    money -= price
                    dest = seeds if op == "BUY_SEED" else shed
                    dest[item] = dest.get(item, 0) + 1
                    if op == "BUY_PRODUCT":
                        market[item] -= 1
    if state.step % 4 == 0:
        for shop in state.unlocked_shops:
            basket = SHOPS[shop]
            for item in basket:
                market[item] -= 2 if len(basket) == 1 else 1
    if state.step % TURNS_PER_DAY == 0:
        for item in PRODUCTS:
            if item != "FERTILIZER":
                market[item] -= 1
    for i, tile in enumerate(tiles):
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            lifespan = tile.get("max_lifespan_step", -1)
            if lifespan >= 0 and state.step >= lifespan and (state.step - lifespan) % 2 == 0:
                tile["yield_units"] -= 1
                if tile["yield_units"] <= 0:
                    tiles[i] = {"kind": "WEED"}
    if state.hour == TURNS_PER_DAY - 1:
        for i, tile in enumerate(tiles):
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                watered = tile.get("watered_today", False)
                tile["consecutive_unwatered"] = 0 if watered else tile.get("consecutive_unwatered", 0) + 1
                tile["watered_today"] = False
                if tile["consecutive_unwatered"] >= 2:
                    tiles[i] = {"kind": "WEED"}
                    continue
                rule = CROPS[tile["crop"]]
                age = state.day + 1 - tile["planted_day"] - rule.first_yield_day
                if rule.ongoing and age >= 0 and age % rule.interval == 0:
                    count = age // rule.interval + 1
                    if count <= rule.max_yield:
                        tile["yield_units"] = min(rule.max_yield, tile["yield_units"] + (2 if watered and tile.get("fertilized_until_day", -1) >= state.day else 1))
                        if count == rule.max_yield:
                            tile["max_lifespan_step"] = (state.day + 2) * TURNS_PER_DAY
            elif "animal" in tile:
                rule = ANIMALS[tile["animal"]]
                fed = tile.get("fed_today", False)
                tile["consecutive_unfed"] = 0 if fed else tile.get("consecutive_unfed", 0) + 1
                if tile["consecutive_unfed"] >= 2:
                    tiles[i] = {"kind": rule.structure}
                    continue
                age = state.day + 1 - tile["placed_day"] - rule.first_yield_day
                if age >= 0 and age % rule.interval == 0:
                    bonus = tile.get("pending_care_bonus", 0) if fed else 0
                    tile["yield_units"] = min(rule.max_held, tile["yield_units"] + 1 + bonus)
                    tile["pending_care_bonus"] = 0
                if fed and tile.get("cared_today", False):
                    tile["pending_care_bonus"] = tile.get("pending_care_bonus", 0) + 1
                tile.update(fertilizer_available=True, fed_today=False, cared_today=False)
        for inv in inventories:
            drop(inv)
        positions, inventories, hires = [access[0]], [{}], 0
    step = state.step + 1
    return replace(state, step=step, day=step // TURNS_PER_DAY, hour=step % TURNS_PER_DAY,
        money=money, hires_today=hires, unlocked_quadrants=tuple(unlocked), shed=shed, seeds=seeds,
        market_inventory=market, market_prices={i: market_price(i, n) for i, n in market.items()},
        tiles=tuple(TileState(t.position, raw) for t, raw in zip(state.tiles, tiles)),
        workers=tuple(WorkerState(i, p, inv) for i, (p, inv) in enumerate(zip(positions, inventories))))
