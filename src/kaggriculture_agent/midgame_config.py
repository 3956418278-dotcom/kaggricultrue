"""Replay-tunable parameters for the observation-driven midgame controller."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MidgameParameters:
    # Ranking-only animal locality preference.
    locality_peak_bonus: float = 0.30
    locality_full_through_day: int = 11
    locality_zero_day: int = 24
    locality_distance_span: int = 8


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
