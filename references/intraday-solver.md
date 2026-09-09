# Production intraday solver

## Boundary

The day-level planner produces a fixed `Plan`: a contract for economic and farm
state changes that must happen today. It does not prescribe workers, routes,
logistics or market timing. The intraday layer consumes `OwnedState + Plan` and
returns only a `Realization(placements, turns)`. Exact execution either verifies
every Plan requirement and returns `S_end`, or rejects the realization.

## Private formulation

For each Plan commitment, the solver mechanically derives the finite legal local
action chains by applying the maintained unit transition. These `_Event` choices
are private implementation data, not another Plan representation.

For a fixed workforce the CP-SAT model jointly decides event presence/order and
worker/time assignment, open placement, worker motion, hire availability, land
availability, opening/worker/shed inventories, production and consumption,
pickup/drop/placement, purchases, market-entry capacity and cash timing. Every
mandatory Plan event is constrained present exactly once. A standard capacitated
routing solve supplies only an incumbent/hint; it does not define or prune the
feasible set. A CP neighborhood may fix part of an incumbent while jointly
reoptimizing the released events and their relevant resource decisions.

The outer loop considers bounded workforce sizes independently. Only candidates
that replay exactly through `rules.advance_owned()` and satisfy all Plan
requirements survive. Full candidates are compared by exact `V(S_end)`; travel
and action count are search guidance, not alternative completion objectives.

## Failure semantics

`solve_intraday()` never returns a partial trajectory. Exhausted search or a
proved contradiction raises `PlanningFailure` with diagnostics. The distinction
between solver `UNKNOWN` and `INFEASIBLE` remains visible in those diagnostics.
