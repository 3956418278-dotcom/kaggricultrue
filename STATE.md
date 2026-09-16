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
  - `animal_incremental_market_path` computes cumulative prior market impact avoiding intra-transaction double counting.
  - `planner._forecast_animal_daily_value` evaluates new candidate incremental impact on top of already accepted programme additions.
  - `current_assets._animal_current_state` deducts future supply of exiting animals (`excluded_own_tiles`) and evaluates the full remaining achievable production timeline via `scenario_revenue_value`.
  - WOOL at D8 and earlier routes to legacy `forecast_inventory`; empirical gating begins from step 239 onwards. Crops retain legacy `forecast_inventory` behavior.

## Validation

- Full suite run under runner: 119 passed tests (all midgame asset, scenario forecast, D11 controller,
  opening base, player day, reference pipeline, replay viewer, scripted opening, and zonal template tests).
- Dedicated scenario forecast tests in `tests/test_scenario_forecast.py`:
  - Golden comparison against supplied reference runtimes (`milk_runtime_v3.py`, `egg_runtime_v3.py`) across D9, D10, D11, D12, D12+1d, D12+3d, D12+5d;
  - Exact price path identity: `dist.price_paths[s, h] == rules.market_price(product, round(dist.inventory_paths[s, h]))`;
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
