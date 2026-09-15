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

## Current-assets maintenance fixes (this revision)

`current_assets.py` was corrected without redesign; the observation-driven
daily re-read design is retained:

- assets at their tile held cap (`animal max_held`, ongoing-crop `max_yield`)
  now harvest immediately, including off-production days (`HELD_CAP`), while
  the earlier next-production-overflow harvest is retained;
- one-time crop task generation now implements the `_one_time_plan()` choice:
  a same-day harvest may still WATER (or FERTILIZE -> WATER when the bonus
  needs fresh fertilizer), the candidate simulation no longer counts a gain on
  an already-watered day, and the HARVEST event quantity equals the plan's
  post-water yield instead of the pre-action held count;
- survival WATER no longer depends on the harvest-candidate search: a one-time
  crop with no candidate still waters when `consecutive_unwatered >= 1`;
- existing ongoing crops price one new fertilizer by its full
  `[day, day+2]` coverage of remaining legal productions
  (`_fertilizer_covered_productions`), not by tonight's single product;
- existing animals are valued for KEEP/EXIT with
  `animal_daily_value(..., include_purchase_cost=False)`; new-asset valuation
  calls keep the previous behavior.

## Validation

- Full suite run under the WSL conda `base` runner: 126 tests plus 6 subtests,
  123 passed, 3 failed in 28s. Failures: two `test_environment_contract.py`
  identity checks, because `base` is Python 3.13.12 with a different
  distribution set than the pinned runtime identity (Python 3.12.3 plus
  `requirements.lock`); and `test_midgame_assets.py::
  test_existing_animals_use_individual_next_output`, which still asserts the
  pre-fix sunk-cost EXIT behavior and is obsolete pending removal.
- The two full pinned episodes (seeds 119-120) with the full D1-D30 controller
  coverage passed in the same run, as did the zonal-template, opening-base,
  scripted-opening, and replay-viewer suites.
- Runner identity note: tests execute in WSL conda `base`, which has
  `kaggle-environments==1.32.7` and pytest; the conda `kaggle` environment is
  the Kaggle CLI tooling environment without the engine. Results from `base`
  do not satisfy the pinned-environment identity required for competitive
  evidence under `EVALUATION.md`.

## Limiting issue

No accepted baseline or competitive evidence exists. The two opening bases are
candidate anchors only; the bootstrap gates in `EVALUATION.md` (pinned
contract, determinism, terminal completion, packaging, replay reproduction,
user approval) have not been run or recorded, and `base`-runner results cannot
substitute for them. D5-D10 remain a deliberate PASS interval. Intraday
efficiency is bounded by the fixed-road catalogue: no return-mask enumeration,
pair repair, or dynamic partitioning. No competitive strength claim of any
kind is established.
