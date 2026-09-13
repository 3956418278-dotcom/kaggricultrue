# Observation-driven midgame parameters

This file records replay-tunable policy parameters. They are centralized in
`src/kaggriculture_agent/midgame_config.py`; game-rule constants remain in
`rules.py` and are not tuning knobs. The controller can attach on any day. D11
is only the initial locality-calibration point, not an initialization requirement.

| Parameter | Initial value | Meaning |
|---|---:|---|
| `fertilizer_value_discount` | 0.80 | Conservative fraction of the current F price used in animal daily value. |
| `locality_peak_bonus` | 0.30 | Maximum ranking multiplier increment for a shed-adjacent animal. |
| `locality_full_through_day` | 11 | Last day at full locality strength; earlier attachment uses the same cap. |
| `locality_zero_day` | 24 | Day from which locality preference is zero. |
| `locality_distance_span` | 8 | Shed-distance at which locality preference reaches zero. |
| `land_min_deployable_count` | 8 | Minimum immediately deployable assets required for a 25-tile land purchase. |
| `land_value_cover_ratio` | 1.25 | Required expansion-value multiple over the official land price. |

Animal modes have no injected score maps or action shadow prices. EXIT is the
mechanical zero boundary of current daily value; among positive assets, a
positive next-cycle CARE increment selects PRODUCE and otherwise selects
MAINTAIN.  Daily value uses current product/Wheat prices and discounted current
F value. Shop bonuses, animal-count penalties, and terminal forecasts are
excluded.

Animal placement first rejects every candidate whose daily value is not
positive. Remaining animal candidates use
`ranking_score = daily_value * (1 + locality_bonus)`, where
`locality_bonus = 0.30 * day_factor * max(0, 1 - shed_distance / 8)` and
`day_factor` is 1 through D11, linear to 0 at D24, then remains 0.
