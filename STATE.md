# Current State

## Project and baseline

The project targets full Kaggriculture under the pinned
`kaggle-environments==1.32.7` contract. `PROJECT.md` owns stable architecture and
environment semantics; `EVALUATION.md` owns evidence rules. The maintained root
`main.py` is a deterministic runnable candidate, not an accepted competitive
baseline. No competitive promotion claim has been accepted.

## Current planning boundary

The production direction is:

```text
OwnedState + Plan
    -> solve_intraday()
    -> Realization
    -> execute_realization()
    -> S_end
```

- `Plan` owns today's required economic/farm outcomes, exact placement of every
  affected project, required outputs and deadlines. It may contain a small
  `EconomicWindow` only when a sale, purchase, shed/cash condition or extra-hand
  allowance has economically important timing.
- `EconomicCommitment.required_state` and `required_outputs` are the canonical
  intraday requirements. `EconomicCommitment.target` is exact. There is no
  production `placement_domains` field.
- `(C,T,L,A,Q,R)` remains the economic-planning record. In particular,
  `ActionDimension.work` supports project scoring, labor estimation and capacity
  planning; intraday does not interpret it as primitive work.
- Intraday owns worker assignment, route/order, Manhattan travel and the sparse
  resource precedences/logistics needed to execute the fixed Plan. Normal
  purchase, hire and land orders are placed mechanically at the earliest legal
  turn. Market timing is otherwise absent.
- `Realization` remains the only public intraday result. A partial Plan is never
  returned. `execute_realization()` replays real rules and requires every Plan
  outcome/window. `end_value()` is diagnostic only and is not an intraday
  objective.

## Current intraday implementation

`src/kaggriculture_agent/intraday.py` contains one private event-based CP-SAT
solver. For each Plan outcome it mechanically derives one shortest legal local
action chain through `rules.advance_owned(..., unit_only=True)`. The CP model then
contains fixed-location service events, event-to-worker assignment, per-worker
route successor arcs, event times required for travel/deadlines, and sparse
source/pickup/drop/transfer choices.

The model has no project-placement variables, no local-chain-choice variables,
no normal market-time variables, and no worker-by-turn position/inventory or
storage-by-turn lattice. Fixed workforce values are tried from low to high; the
first complete exact solution wins. Within one workforce, transparent execution
preferences cover residual purchases, worker actions, movement, logistics and
makespan. No LNS, regret insertion, population or repair stack is present.

The compact solver's search strength at full development scale is not
established. The prior dense temporal model and the route population/support
pipeline are rejected designs and are not production fallbacks. Their historical
development outputs remain only in ignored run artifacts and git history.

## Reference data and adapters

The private player-day pilot remains exploratory development data, not
competitive evidence. It contains 2,880 player-days from 78 reproduced episodes
and 96 qualified sides; all source episodes passed official transition and
terminal-reward replay checks. Dataset and provenance details remain under
`.cache/reference-pilot-20260905/` and the private Kaggle Dataset recorded in
project references.

New extraction schema `player-day-v3` produces one outcome-oriented commitment
per achieved farm entity/day, fixes its demonstrated placement, aggregates its
required outputs, and excludes failed attempts and primitive action records from
Plan. The frozen dataset is older schema and remains immutable;
`kaggriculture_eval.plan_io.plan_from_sample()` migrates it only inside the
reference adapter before constructing the canonical production `Plan`.

The frozen nine-row development benchmark uses episodes 105527696 and 105448362
on days 8, 16 and 24. Held-out episodes remain untouched. A run of the new
responsibility boundary has not been authorized or completed; the benchmark is
paused pending user confirmation.

## Current validation

- Focused Plan/intraday/reference tests pass in the optional OR-Tools environment.
  They cover fixed placement, outcome-based exact validation, rejection of
  partial realizations, ActionDimension independence, minimal economic-window
  timing and reference extraction/audit.
- A production initial-state Plan expands mechanically to fixed jobs, but this is
  only a formulation check, not evidence of route-search strength or runtime.
- No new nine-row result, held-out result, competitive arena result or runtime
  acceptance claim exists for this boundary.

## Next confirmation point

Before any frozen nine-row run, report the canonical Plan fields, deleted
decision-variable families, focused test result, and one real development
player-day's fixed-job/precedence/assignment/route-variable profile. Only proceed
to benchmark after explicit user confirmation.
