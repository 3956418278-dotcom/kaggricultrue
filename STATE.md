# Current State

## Project and runtime

The project targets full Kaggriculture under pinned
`kaggle-environments==1.32.7`. The maintained root `main.py` exposes the
submission agent. The fixed opening owns D1-D4 and D5-D10 deliberately pass
because the former programme/target-table strategy was removed. The submitted
handoff currently occurs at D11, while the midgame planner and current-asset
reader themselves can attach to any legal observation.

The runtime boundary is now:

```text
observation -> canonical State -> macro Programme -> intraday executor -> action
```

`State` retains private own state, public opponent state, complete official
animal/crop tile records, market inventory/prices, shops, land and worker state.
`Programme` freezes the current day's KEEP/EXIT/NEW assets, exact tiles,
service/harvest commitments, land use, staffing, routes, route-proven DROP
arrivals, and short-horizon sales. Existing assets remain observation-backed
short states rather than terminal programmes.

## Implemented D11+ mechanisms

- discrete known Town/shop demand events and the fixed reveal calendar;
- first/next-output macro supply estimates for animal/new long-asset product
  prices, with terminal-truncated new output and discounted realizable F;
- observation-backed animal PRODUCE/MAINTAIN/EXIT, minimum survival FEED,
  pre-first-output protection, one-cycle CARE, held-capacity harvest gates,
  a two-night EXIT tail, and nearest-first sequential EXIT revaluation;
- event-local decisions for existing Wheat/Carrot/Melon/Tomato/Strawberry,
  including actual HARVEST/DIG cleanup and same-day ongoing-crop replacement;
- feed sourcing from real stock or route-proven mature Wheat, followed only by
  direct Wheat purchase for any deficit; no feed-purpose crop is created;
- global seed accounting, positive current-value long-asset placement,
  low-cost NOW/WAIT comparison, decaying animal locality ranking, and
  cash-gated second/third land purchase after existing-land funding;
- private worker carry is reconciled against real routes before retaining W/F
  purchases; only future shed F pickups reserve F, and free shed F sells NOW;
- restored runnable daily intraday assignment with exact PICKUP/DROP routes and
  minimum feasible same-day staffing;
- explicit `must_return` production: D1-D10 shed-distance <=3 plus a
  farthest-first EOD-capacity split, with forced DROP no later than turn 22;
- a runtime trade machine that re-reads real inventory every four turns and
  evaluates only NOW/+4/+8, enforcing overflow, cash, terminal, and ten-order
  constraints before discretionary timing;
- the terminal sale DP remains available only for offline comparison and is not
  on the submitted runtime path.

The former `economics.py`, fixed programme/pace tables, generic D11+ heuristic
planner and valuation layer are deleted. Frozen player-day research records use
an evaluation-local schema and do not enter the submitted strategy.

## Validation

- The repository suite passes 122 tests, including the pinned environment,
  controller, seed ledger, asset modes, crop inputs, EXIT liquidation,
  short-horizon trade, order-limit, and runnable intraday regressions.
- Two full pinned episodes (seeds 119-120) completed through D30 against PASS
  in the regression suite, checking feasible daily plans and four-turn trade
  refreshes with uncaught controller calls (no exception fallback). These are
  integration smoke checks, not competitive evidence; prior-version planning
  time measurements do not describe this revision.

## Limiting issue

The production runtime is executable, but its intraday implementation is the
restored pre-zonal daily executor. The replacement two-/three-/four-land
fixed-road catalogue and its exact crop/connect/24-turn selector are still
independent and intentionally not connected by this change. The current
catalogue has 41 two-land and 55 three-land layouts through 12 workers,
including cross-quadrant spines and directional hotspot roads. Its 42
four-land entries are provisional and outside current integration/coverage
acceptance. Each normal layout has three fixed EOD/narrow/wide return versions;
selected high-load layouts add one authored overflow version. There is no
return-mask enumeration, pair repair, or dynamic partitioning. D5-D10 also
remain the previously declared PASS interval. No competitive strength or
accepted-baseline claim has been established.
