"""Replay-tunable parameters for the observation-driven midgame controller."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MidgameParameters:
    # Value only the fertilizer collectable on already useful visits.
    fertilizer_value_discount: float = 0.80

    # Ranking-only animal locality preference.
    locality_peak_bonus: float = 0.30
    locality_full_through_day: int = 11
    locality_zero_day: int = 24
    locality_distance_span: int = 8

    # A 25-tile quadrant is bought only for a genuine scale deployment.
    land_min_deployable_count: int = 8
    land_value_cover_ratio: float = 1.25


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
