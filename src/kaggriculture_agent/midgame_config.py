"""Replay-tunable parameters for the observation-driven midgame controller."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class MidgameParameters:
    # Value only the fertilizer collectable on already useful visits.
    fertilizer_value_discount: float = 0.80

    # Ranking-only animal locality preference.
    locality_peak_bonus: float = 0.30
    locality_full_through_day: int = 11
    locality_zero_day: int = 24
    locality_distance_span: int = 8

    # New land purchase: cash reserve kept after land + fallback seed.
    fixed_cash_reserve: int = 200

    # D1-D10 force harvested bundles this close to the shed to return.  Later
    # days use only the real EOD-capacity split.
    early_return_last_day: int = 9
    early_return_distance: int = 3
    # Units per future reveal, initially one average basket tick across the
    # eight official shop types. This is a tunable weight, not a shop forecast.
    expected_shop_demand_per_reveal: Mapping[str, float] = field(
        default_factory=lambda: {
            "WHEAT": 0.625, "CARROT": 0.375, "TOMATO": 0.25,
            "STRAWBERRY": 0.5, "MELON": 0.0, "EGG": 0.25,
            "MILK": 0.375, "WOOL": 0.25,
        })


DEFAULT_MIDGAME_PARAMETERS = MidgameParameters()


def coerce_midgame_parameters(value) -> MidgameParameters:
    if value is None:
        return DEFAULT_MIDGAME_PARAMETERS
    if isinstance(value, MidgameParameters):
        return value
    if isinstance(value, dict):
        known = MidgameParameters.__dataclass_fields__
        return MidgameParameters(**{key: item for key, item in value.items()
                                    if key in known})
    raise TypeError("config must be MidgameParameters, a parameter mapping, or None")
