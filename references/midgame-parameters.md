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
| `fixed_cash_reserve` | 200 | Strict cash gate after funding the existing-land plan, next land price and new-quadrant Wheat seed fallback. Only quadrants 2 and 3 may be purchased. |
| `expected_shop_demand_per_reveal` | W .625; C .375; T .25; ST .5; M 0; Egg .25; Milk .375; Wool .25 | Macro inventory subtraction per future reveal, initially one average basket tick across the eight official shops. These uncalibrated weights do not select or simulate a future shop. |

Animal modes have no injected score maps or action shadow prices. EXIT is the
mechanical zero boundary of current daily value; among positive assets, a
positive next-cycle CARE increment selects PRODUCE and otherwise selects
MAINTAIN. Before the first production has completed, an animal always remains
MAINTAIN with minimum survival feeding; afterwards EXIT remains sticky.
Product prices for new long assets use the first output's supply estimate;
existing animals use their individual next output. Wheat/F input prices remain
current. CARE is still a separate next-cycle marginal decision.

The macro inventory estimate at post-refresh state T is real inventory plus
own existing production on (now,T], plus accepted new production on (now,T],
plus the sum of max(0, output-1) for each visible opponent asset, minus known
Town/shop consumption on [now,T), minus the number of upcoming reveals through
T times the configured product weight. No unapproved future CARE/F inputs are
invented. New candidates exclude themselves from supply until accepted. New
asset output counts stop at the pinned terminal boundary. This estimate never
changes real market inventory and never enters runtime trading.

Land requires strictly more cash than the official next price plus the actual
new quadrant's plantable tile count times Wheat seed cost plus 200, after all
existing-land commitments including actual hires. No scale-count/value-ratio
gate remains. The same sequential placement resumes after a purchase.

Input purchasing starts from public shed stock. One reconciliation against real
worker routes removes redundant W/F purchases justified by private carry or
DROP-before-PICKUP, with at most one recompilation. Trading reserves only future
route-proven shed F pickups and immediately sells the remaining shed F.

Animal placement first rejects every candidate whose daily value is not
positive. Remaining animal candidates use
`ranking_score = daily_value * (1 + locality_bonus)`, where
`locality_bonus = 0.30 * day_factor * max(0, 1 - shed_distance / 8)` and
`day_factor` is 1 through D11, linear to 0 at D24, then remains 0.
