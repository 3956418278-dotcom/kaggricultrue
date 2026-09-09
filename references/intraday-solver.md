# Production intraday solver

## Boundary

The day-level planner produces a fixed `Plan`: a contract for economic and farm
outcomes that must happen today. It owns WHAT, exact WHERE, and only explicitly
important economic WHEN through a small `EconomicWindow`. It does not prescribe
workers, routes, logistics or primitive operations. The intraday layer consumes
`OwnedState + Plan` and returns only `Realization(placements, turns)`. Exact
execution either verifies every outcome and window and returns `S_end`, or
rejects the realization.

## Private formulation

Each physical commitment has an exact `target`, `required_state`, optional
`required_outputs`, and deadlines. `ActionDimension.work` is retained solely for
economic labor estimates. Intraday mechanically derives a shortest local action
chain by applying the maintained unit transition; `_Event` remains solver-local.

For a fixed workforce the CP-SAT model decides mandatory event assignment and
route successor/order, derives feasible event times from Manhattan travel and
precedence, and represents only sparse resource source, pickup and transfer
choices. Project placement, local-chain choice, normal purchase time, hire time
and land time are not decision-variable families. There is no worker-by-turn
position or inventory lattice.

The outer loop considers workforce sizes from low to high and stops at the first
complete exact realization. Within a fixed workforce, transparent lexicographic
execution preferences minimize unplanned residual purchases, total worker
actions, movement, logistics, then makespan/slack. `end_value()` remains an
evaluation diagnostic and is not an intraday objective.

## Failure semantics

`solve_intraday()` never returns a partial trajectory. Exhausted search or a
proved contradiction raises `PlanningFailure` with diagnostics. The distinction
between solver `UNKNOWN` and `INFEASIBLE` remains visible in those diagnostics.
