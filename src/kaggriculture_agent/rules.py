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
