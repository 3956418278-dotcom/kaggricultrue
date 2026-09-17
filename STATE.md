# Current State

## Project and runtime

The project targets full Kaggriculture under pinned
`kaggle-environments==1.32.7`. The maintained root `main.py` re-exports the
submission agent from `src/kaggriculture_agent/agent.py`. The submitted policy
runs the scripted fixed opening through D4, deliberately passes D5-D10, and
hands off to the observation-driven midgame controller from D11
(`observation.day >= 10`); malformed observations degrade to a legal PASS.

The runtime boundary is unchanged:

```text
observation -> canonical State -> macro Programme -> intraday routes -> action
```

Module ownership is documented in `src/kaggriculture_agent/README.md`.
Relative to the previous record, `intraday.py` now consumes the zonal
fixed-road catalogue (`zonal_templates.py`): layout selection and
minimum-workforce sizing run inside routing, combined with the farthest-first
EOD-capacity split and D1-D10 near-shed `must_return` labels from
`return_requirements.py`. The terminal sale DP remains an offline comparator
in `market.py`, not on the runtime path. Player-day extraction/reconstruction
and its pipeline contracts remain separate evaluation-side tooling under
`tests/test_player_days.py`, `tests/test_reference_pipeline.py`, and the
`scripts/` reference commands.

## Two promoted opening bases

`opening_bases.py` builds two local base agents from audited six-day
top-opening prefixes (`data/top_opening_bases.json`, schema
`top-opening-common-prefix-v1`, sourced from the official Kaggle top-15
snapshot 2026-09-13):

- `catalyst_base_agent` (rank 12);
- `subramanya_base_agent` (rank 14).

Each replays its recorded actions until its recorded `handoff_step` and then
continues through the same `DailyPlanningSession` midgame controller as the
submitted agent. `scripts/run_opening_base_arena.py` runs exploratory
side-swapped games between the two bases; `scripts/verify_baseline.py` is a
full-episode health check. Both are explicitly exploratory and are not
acceptance-quality evidence.

## Empirical animal scenario forecasting (this revision)

Empirical scenario forecast models (`src/kaggriculture_agent/scenario_forecast.py`)
replace legacy single-point heuristics (`forecast_inventory` and `expected_shop_demand_per_reveal`)
for animal products (WOOL, MILK, EGG):

- **Sources**: Cleaned model assets under `animal_calculation_value_model/{wool,milk,egg}/`.
- **Interface**: `forecast_product_distribution(state, product, context=...)` returns `ScenarioDistribution` containing
  scenario paths, exact sequential price paths via `rules.market_price`, normalized weights, ordered quantiles (Q10/Q25/Q50/Q75/Q90),
  OOD diagnostics, and confidence metrics.
- **Dynamic re-anchoring & context**:
  - `DailyPlanningSession` maintains a per-player `AnimalMarketContext` that captures observed D12 branches and real inventories at step 311.
  - Post-D12 residual update applies $\rho$ correction with the real anchor.
  - Late-attach fallback performs empirical scenario re-anchoring without $\rho$ correction when no anchor was recorded.
  - Missing public opponent money triggers an explicit insufficient-information fallback rather than defaulting to branch 2.
  - Single-pass turn reference caching eliminates repeated full-scenario generation across farm animals.
- **Planner & asset valuation**:
  - `is_valid_shops` validates 1-8 unique legal shops, allowing empirical forecasting across all reveal stages up to full 8-shop reveal.
  - `animal_batch_sale_events` separates production from market sales: held inventory accumulates to `max_held` before batch sale, terminal partial batches liquidate at the final production step, and existing animals accumulate from observed held quantity with pending CARE bonuses.
  - `scenario_batch_sale_value` evaluates multi-batch sales in strict chronological order, ensuring earlier sales impact later batches while avoiding intra-batch double counting.
  - Existing animals in `_animal_current_state` are valued across their full remaining horizon up to the final batch sale (`remaining_days`), calculating net value over total expected product revenue, remaining survival feed cost, and realizable fertilizer via `animal_daily_value`, eliminating 2~3x overvaluation from legacy single-cycle division.
  - Exiting animals (`excluded_own_tiles`) remove their full batch-sale counterfactual impact from market inventory.
  - `DailyPlanningSession.plan_for` resets per-player `AnimalMarketContext` when `step < last_step`, eliminating cross-episode anchor and cache leakage.
  - WOOL at D8 and earlier routes to legacy `forecast_inventory`; empirical gating begins from step 239 onwards. Crops retain legacy `forecast_inventory` behavior.

## Market prediction failure protection patch

A targeted protection mechanism in `operating.py` and `market.py` guards against market inventory prediction divergence:
- **Failure tracking**: `DailyPlanningSession._market_failure_steps` records `player -> {product: step}` whenever `state.market.inventory` differs from `programme.market_inventory` for any non-excluded product (`_DEVIATION_EXCLUDED = frozenset({"WHEAT", "CARROT"})`).
- **4-turn window**: Cooldown is active while `0 <= state.step - last_failure_step <= MARKET_FAILURE_SELL_WINDOW` (`MARKET_FAILURE_SELL_WINDOW = 4`). New failures within the window refresh `last_failure_step`.
- **New-arrival forced sale**: During cooldown, any new units arriving into the shed at the current step (`_arrivals(state, programme).get(state.step, {}).get(product, 0)`) are immediately scheduled for sale, while existing shed inventory remains subject to normal optimizer timing.
- **Optimizer integration**: Forced quantities are passed via `forced_sales` to `optimize_short_sales`, merging into `_short_product_plan`'s `forced` minimum constraints and prioritized as hard lines in `_defer_sale_lines`.
- **Per-product isolation & reset**: Unaffected products follow normal valuation; failure histories are cleared on `DailyPlanningSession.reset()` and whenever `state.step < last_step`.

## Validation

- Full suite run under runner: 136 passed tests (all midgame asset, market failure patch, scenario forecast, D11 controller,
  opening base, player day, reference pipeline, replay viewer, scripted opening, and zonal template tests).
- Dedicated market failure patch tests in `tests/test_market_failure_patch.py`:
  - Failure recording on non-excluded product mismatch;
  - Immediate sale of new arrivals at $t+1$;
  - Only new arrivals forced without liquidating existing shed stock;
  - Exact 4-turn boundary enforcement ($t..t+4$ active, $t+5$ normal);
  - Window refresh on repeated deviation;
  - Per-product isolation;
  - Zero-arrival safety;
  - Episode reset and session reset clearance.
- Dedicated scenario forecast tests in `tests/test_scenario_forecast.py`:
  - Full-horizon multi-batch daily value stability and monotonic scaling across 1, 2, and 3 batches for existing animals;
  - Remaining survival feed calculation (`_remaining_survival_feed_units`) considering `fed_today` and `consecutive_unfed`;
  - Golden comparison against supplied reference runtimes (`milk_runtime_v3.py`, `egg_runtime_v3.py`) across D9, D10, D11, D12, D12+1d, D12+3d, D12+5d;
  - Exact price path identity: `dist.price_paths[s, h] == rules.market_price(product, round(dist.inventory_paths[s, h]))`;
  - Exact batch schedule verification for new COW (15 productions -> [6, 6, 3]), existing COW (held=3, 10 productions -> [6, 6, 1]), and GOOSE (10 productions -> [4, 4, 2]);
  - Market isolation of intermediate production before batch sales;
  - Strict chronological batch ordering and sequential price impact;
  - Full batch-path counterfactual subtraction for EXIT animals;
  - Multi-shop (5, 6, 8 shops) empirical execution;
  - Cross-episode context reset verification;
  - Counterfactual candidate valuation incorporating accepted additions;
  - Re-valuation of remaining animals when exiting animals are excluded;
  - Sequential revenue exact comparison against manual hand calculation;
  - Inventory residual $\rho$ decay vs constant shift;
  - Late-attach and insufficient-information fallbacks.

## Limiting issue

No accepted baseline or competitive evidence exists. The two opening bases are
candidate anchors only; the bootstrap gates in `EVALUATION.md` (pinned
contract, determinism, terminal completion, packaging, replay reproduction,
user approval) have not been run or recorded, and `base`-runner results cannot
substitute for them. D5-D10 remain a deliberate PASS interval. Intraday
efficiency is bounded by the fixed-road catalogue: no return-mask enumeration,
pair repair, or dynamic partitioning. No competitive strength claim of any
kind is established.
