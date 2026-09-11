"""Pure policy rules for the D5+ Kaggriculture programme.

This module does not build a :class:`Plan`, schedule workers, or emit actions.
It owns only the deterministic economic pace rules supplied by the programme
specification.  The daily planner may consume these results once its solving
path is ready; intraday must never import this module.

All ``game_day`` arguments are one-based: D5 is ``game_day=5``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

from . import rules


class Programme(str, Enum):
    DAIRY_GROWTH = "DAIRY_GROWTH"
    WOOL_GROWTH = "WOOL_GROWTH"
    EGG_ST_GROWTH = "EGG_ST_GROWTH"
    GOOSE_BRIDGE_MIXED = "GOOSE_BRIDGE_MIXED"


class BindingConstraint(str, Enum):
    """The complete allowed vocabulary for a rejected expansion candidate."""

    CASH = "CASH"
    FEED = "FEED"
    LAND = "LAND"
    ROUTE = "ROUTE"
    WATER_DEADLINE = "WATER_DEADLINE"
    HORIZON = "HORIZON"


class TileRole(str, Enum):
    PERMANENT = "PERMANENT"
    NEXT_DAY_STRUCTURE = "NEXT_DAY_STRUCTURE"
    OPERATING_WHEAT = "OPERATING_WHEAT"
    RESERVATION_WHEAT = "RESERVATION_WHEAT"


@dataclass(frozen=True)
class ShopCounters:
    milk: int = 0
    yarn: int = 0
    egg: int = 0
    strawberry: int = 0
    wheat: int = 0
    carrot_units: int = 0


@dataclass(frozen=True)
class ProgrammePace:
    """Pace floors for one D5+ day; every quantity is a floor, never a cap."""

    programme: Programme
    cow: int = 0
    sheep: int = 0
    goose: int = 0
    strawberry: int = 0
    carrot: int = 0
    operating_wheat: int = 8
    combined_cow_sheep: int = 0


@dataclass(frozen=True)
class PrebuildCaps:
    finance: int
    feed: int
    route: int
    land: int


@dataclass(frozen=True)
class ThirdLandFacts:
    game_day: int
    two_land_space_insufficient: bool
    permanent_tiles_needed: int
    cash_after_purchase: int
    day_worker_cost: int
    next_day_feed_cost: int
    planned_animal_cost: int
    mandatory_seed_cost: int
    has_same_day_layout: bool


@dataclass(frozen=True)
class ExpansionDiagnostic:
    game_day: int
    programme: Programme
    requested_target: int
    max_feasible_target: int
    binding_constraint: BindingConstraint


@dataclass(frozen=True)
class ReservationWheat:
    position: tuple[int, int]
    candidate_use: str
    preferred_release_day: int


MAX_TOTAL_WORKERS: Mapping[int, int] = {
    5: 10,
    6: 10,
    7: 11,
    8: 11,
    9: 12,
    10: 12,
    11: 10,
}
EXACT_TOTAL_WORKERS: Mapping[int, int] = {11: 10}

ORDINARY_DEADLINE = 20
NEW_ANIMAL_PLACE_DEADLINE = 19
NEW_CROP_DEADLINE = 20
D11_MELON_CRITICAL_DEADLINE = 19
D11_ORDER = (
    "ANIMAL_FEED_AND_FERTILIZER",
    "MANDATORY_CROP_WATER",
    "MELON_LIQUIDATION",
    "PRESELECTED_EXPANSION_BLOCK",
)

DAILY_ANIMAL_ADDITION_CAP: Mapping[int, int] = {
    4: 4,
    5: 3,
    6: 2,
    7: 2,
    8: 2,
    9: 3,
    10: 3,
}

_STRAWBERRY_ONE = {4: 6, 5: 12, 6: 16, 7: 20, 8: 22, 9: 24, 10: 24}
_STRAWBERRY_TWO = {5: 12, 6: 18, 7: 24, 8: 28, 9: 30, 10: 32}
_STRAWBERRY_THREE = {5: 12, 6: 18, 7: 24, 8: 28, 9: 34, 10: 36}

_TYPICAL = {
    "SMOOTHIE_SHOP": {
        "cow": {5: 6, 6: 7, 7: 8, 8: 9, 9: 10},
        "strawberry": {5: 12, 6: 16, 7: 20, 8: 22, 9: 24},
        "operating_wheat": {5: 9, 6: 10, 7: 11, 8: 12, 9: 13},
    },
    "ICE_CREAM_SHOP": {
        "cow": {5: 6, 6: 7, 7: 8, 8: 9, 9: 10},
        "strawberry": {5: 10, 6: 14, 7: 18, 8: 22, 9: 24},
        "operating_wheat": {5: 12, 6: 13, 7: 14, 8: 15, 9: 16},
    },
    "PIZZA_SHOP": {
        "cow": {5: 7, 6: 8, 7: 9, 8: 10},
        "operating_wheat": {5: 15, 6: 16, 7: 17, 8: 18},
    },
    "YARN_STORE": {
        "sheep": {5: 6, 6: 7, 7: 8, 8: 8},
        "operating_wheat": {5: 12, 6: 13, 7: 14, 8: 14},
    },
    "BRUNCH_SPOT": {
        "goose": {5: 4, 6: 4, 7: 4},
        "strawberry": {5: 14, 6: 18, 7: 22},
        "operating_wheat": {5: 10, 6: 11, 7: 12},
    },
    "BAKERY": {
        "goose": {5: 4, 6: 4},
        "operating_wheat": {5: 20, 6: 20},
    },
}


def _require_programme_day(game_day: int) -> None:
    if game_day < 5:
        raise ValueError("programme starts on D5")


def shop_counters(unlocked_shops: Iterable[str]) -> ShopCounters:
    counts = {shop: 0 for shop in rules.SHOPS}
    for shop in unlocked_shops:
        name = str(shop)
        if name in counts:
            counts[name] += 1
    return ShopCounters(
        milk=(counts["PIZZA_SHOP"] + counts["ICE_CREAM_SHOP"]
              + counts["SMOOTHIE_SHOP"]),
        yarn=counts["YARN_STORE"],
        egg=counts["BAKERY"] + counts["BRUNCH_SPOT"],
        strawberry=(counts["ICE_CREAM_SHOP"] + counts["SMOOTHIE_SHOP"]
                    + counts["BRUNCH_SPOT"] + counts["FARMERS_MARKET"]),
        wheat=(counts["BAKERY"] + counts["PIZZA_SHOP"]
               + counts["BRUNCH_SPOT"] + counts["ICE_CREAM_SHOP"]
               + counts["FARMERS_MARKET"]),
        carrot_units=2 * counts["PET_CAFE"] + counts["FARMERS_MARKET"],
    )


def programme_for(counters: ShopCounters) -> Programme:
    if counters.milk:
        return Programme.DAIRY_GROWTH
    if counters.yarn:
        return Programme.WOOL_GROWTH
    if counters.egg:
        return Programme.EGG_ST_GROWTH
    return Programme.GOOSE_BRIDGE_MIXED


def worker_limit(game_day: int) -> int | None:
    """Return the total-worker cap, including the farmer, when specified."""

    _require_programme_day(game_day)
    return MAX_TOTAL_WORKERS.get(game_day)


def exact_worker_requirement(game_day: int) -> int | None:
    _require_programme_day(game_day)
    return EXACT_TOTAL_WORKERS.get(game_day)


def animal_addition_cap(game_day: int) -> int:
    return DAILY_ANIMAL_ADDITION_CAP.get(game_day, 0)


def goose_bridge_target(counters: ShopCounters) -> int:
    """Return zero as soon as a Milk or Wool signal ends bridge expansion."""

    if counters.milk or counters.yarn:
        return 0
    if counters.egg == 0:
        return 3
    if counters.egg == 1:
        return 4
    return 6


def d10_animal_floor(counters: ShopCounters) -> tuple[int, int]:
    """Return ``(Cow, Sheep)`` D10 floors from the signal-count table."""

    milk, yarn = counters.milk, counters.yarn
    if milk and yarn:
        cows = 9 if milk >= 2 else 6
        sheep = 8 if yarn >= 2 else 5
        return cows, sheep
    if milk:
        return (7 if milk == 1 else 10 if milk == 2 else 12), 0
    if yarn:
        return 0, (8 if yarn == 1 else 13 if yarn == 2 else 15)
    return 0, 0


def strawberry_floor(game_day: int, counters: ShopCounters) -> int:
    _require_programme_day(game_day)
    if not counters.strawberry:
        return 0
    table = (_STRAWBERRY_ONE if counters.strawberry == 1
             else _STRAWBERRY_TWO if counters.strawberry == 2
             else _STRAWBERRY_THREE)
    if game_day not in table:
        return table[max(table)] if game_day > max(table) else 0
    return table[game_day]


def carrot_floor(unlocked_shops: Iterable[str]) -> int:
    shops = tuple(str(shop) for shop in unlocked_shops)
    pets = shops.count("PET_CAFE")
    farmers = shops.count("FARMERS_MARKET")
    if pets >= 2:
        return 24
    if pets and farmers:
        return 18
    if pets:
        return 14
    if farmers:
        return 6
    return 0


def operating_wheat_floor(total_livestock: int, counters: ShopCounters) -> int:
    return max(8, max(0, total_livestock) + 2 + 4 * counters.wheat)


def _typical_floor(game_day: int, shops: tuple[str, ...], field: str) -> int:
    # A typical table describes one named line.  With several different shop
    # signals the general counter rules remain authoritative.
    if len(shops) != 1 or shops[0] not in _TYPICAL:
        return 0
    shop = shops[0]
    table = _TYPICAL[shop].get(field, {})
    if game_day in table:
        return int(table[game_day])
    earlier = [day for day in table if day <= game_day]
    return int(table[max(earlier)]) if earlier else 0


def programme_pace(game_day: int, unlocked_shops: Iterable[str]) -> ProgrammePace:
    """Resolve the stated floor tables without applying feasibility truncation."""

    _require_programme_day(game_day)
    shops = tuple(str(shop) for shop in unlocked_shops)
    counters = shop_counters(shops)
    cow = _typical_floor(game_day, shops, "cow")
    sheep = _typical_floor(game_day, shops, "sheep")
    goose = _typical_floor(game_day, shops, "goose")
    strawberry = strawberry_floor(game_day, counters)
    if counters.strawberry == 1 and shops.count("ICE_CREAM_SHOP") == 1:
        strawberry = max(0, strawberry - 2)
    if counters.strawberry == 1 and shops.count("FARMERS_MARKET") == 1:
        strawberry = max(0, strawberry - 4)
    strawberry = max(strawberry, _typical_floor(game_day, shops, "strawberry"))
    if shops.count("YARN_STORE") >= 2:
        sheep = max(sheep, {7: 9, 8: 11, 9: 13}.get(game_day, 0))
    if game_day >= 10:
        terminal_cow, terminal_sheep = d10_animal_floor(counters)
        cow, sheep = max(cow, terminal_cow), max(sheep, terminal_sheep)
    goose = max(goose, goose_bridge_target(counters))
    livestock_floor = cow + sheep + goose
    operating_wheat = max(
        operating_wheat_floor(livestock_floor, counters),
        _typical_floor(game_day, shops, "operating_wheat"),
    )
    return ProgrammePace(
        programme=programme_for(counters),
        cow=cow,
        sheep=sheep,
        goose=goose,
        strawberry=strawberry,
        carrot=carrot_floor(shops),
        operating_wheat=operating_wheat,
        combined_cow_sheep=6 if game_day >= 5 and (counters.milk or counters.yarn) else 0,
    )


def prebuild_count(caps: PrebuildCaps) -> int:
    return max(0, min(caps.finance, caps.feed, caps.route, caps.land))


def block_addition_to_floor(current: int, floor: int, block: int = 4) -> int:
    """Return the minimum whole expansion blocks needed to reach a pace floor."""

    if block <= 0:
        raise ValueError("expansion block must be positive")
    shortfall = max(0, floor - current)
    return ((shortfall + block - 1) // block) * block


def reserve_wheat(
    position: tuple[int, int],
    candidate_use: str,
    game_day: int,
    release_in_days: int,
) -> ReservationWheat:
    if release_in_days not in (1, 2, 3):
        raise ValueError("reservation Wheat release must be 1-3 days ahead")
    if not candidate_use:
        raise ValueError("reservation Wheat requires a candidate use")
    return ReservationWheat(
        position=position,
        candidate_use=candidate_use,
        preferred_release_day=game_day + release_in_days,
    )


def projected_next_day_cash(
    cash_now: int,
    guaranteed_sales: int,
    next_day_hires: int,
    mandatory_feed: int,
    mandatory_seeds: int,
    other_mandatory_costs: int,
) -> int:
    return (
        cash_now
        + guaranteed_sales
        - next_day_hires
        - mandatory_feed
        - mandatory_seeds
        - other_mandatory_costs
    )


def programme_priority(unlocked_shops: Iterable[str]) -> tuple[str, ...]:
    shops = tuple(str(shop) for shop in unlocked_shops)
    kind = programme_for(shop_counters(shops))
    if kind is Programme.DAIRY_GROWTH:
        return ("COW", "STRAWBERRY", "OPERATING_WHEAT", "RESERVATION_WHEAT")
    if kind is Programme.WOOL_GROWTH:
        return ("SHEEP", "OPERATING_WHEAT", "SUPPORTED_CROP", "RESERVATION_WHEAT")
    if kind is Programme.EGG_ST_GROWTH:
        if "BRUNCH_SPOT" in shops:
            return ("STRAWBERRY", "GOOSE", "OPERATING_WHEAT", "RESERVATION_WHEAT")
        return ("GOOSE", "OPERATING_WHEAT", "RESERVATION_WHEAT")
    return ("SUPPORTED_CROP", "GOOSE", "OPERATING_WHEAT", "RESERVATION_WHEAT")


def optional_reduction_order() -> tuple[str, ...]:
    """What may be removed after the day's worker cap is actually exhausted."""

    return ("CARE", "MARGINAL_HARVEST", "RESERVATION_WHEAT", "LAST_MARGINAL_ASSET")


def operating_wheat_releasable(
    shed_wheat: int,
    guaranteed_harvest: int,
    next_two_days_feed: int,
) -> bool:
    return shed_wheat + guaranteed_harvest >= next_two_days_feed


def third_land_allowed(facts: ThirdLandFacts) -> bool:
    """Apply the four conjunctive D7+ third-land rules exactly."""

    if facts.game_day < 7:
        return False
    mandatory_after_purchase = (
        facts.day_worker_cost
        + facts.next_day_feed_cost
        + facts.planned_animal_cost
        + facts.mandatory_seed_cost
    )
    return (
        facts.two_land_space_insufficient
        and facts.permanent_tiles_needed >= 8
        and facts.cash_after_purchase >= mandatory_after_purchase
        and facts.has_same_day_layout
    )


def strawberry_investment_allowed(game_day: int, remaining_harvest_events: int) -> bool:
    """D22+ Strawberry requires two remaining harvest events; D24+ exits."""

    if game_day >= 24:
        return False
    return game_day < 22 or remaining_harvest_events >= 2


def animal_investment_allowed(game_day: int) -> bool:
    """Cow/Sheep expansion stops by default from D22 onward."""

    return game_day < 22


def fertilizer_should_sell(
    crop: str | None,
    marginal_crop_value: int,
    sale_value: int,
) -> bool:
    """Melon never consumes F; otherwise sell unless the proven margin is higher."""

    return crop == "MELON" or marginal_crop_value <= sale_value
