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
- current-price animal/crop daily-value formulas with discounted realizable F;
- observation-backed animal PRODUCE/MAINTAIN/EXIT, minimum survival FEED,
  one-cycle CARE, held-capacity harvest gates, and a two-night EXIT tail;
- event-local decisions for existing Wheat/Carrot/Melon/Tomato/Strawberry;
- feed sourcing from real stock or route-proven mature Wheat, followed only by
  direct Wheat purchase for any deficit; no feed-purpose crop is created;
- global seed accounting, positive current-value long-asset placement,
  low-cost NOW/WAIT comparison, decaying animal locality ranking, and
  scale-based land purchase;
- restored runnable daily intraday assignment with exact PICKUP/DROP routes and
  minimum feasible same-day staffing;
- a runtime trade machine that re-reads real inventory every four turns and
  evaluates only NOW/+4/+8, enforcing overflow, cash, terminal, and ten-order
  constraints before discretionary timing;
- the terminal sale DP remains available only for offline comparison and is not
  on the submitted runtime path.

The former `economics.py`, fixed programme/pace tables, generic D11+ heuristic
planner and valuation layer are deleted. Frozen player-day research records use
an evaluation-local schema and do not enter the submitted strategy.

## Validation

- The repository suite passes 93 tests, including the pinned environment,
  controller, seed ledger, asset modes, crop inputs, EXIT liquidation,
  short-horizon trade, order-limit, and runnable intraday regressions.
- Three uncaught-exception full pinned episodes (seeds 119-121) completed through
  D30 against PASS. Across each D11-D30 run, planner calls totaled 0.031-0.036s
  with observed maximum calls at or below 0.0033s. These are integration smoke checks,
  not competitive evidence.

## Limiting issue

The production runtime is executable, but its intraday implementation is the
restored pre-zonal daily executor; the hand-authored zonal template library is
still independent and is intentionally not connected by this change. D5-D10
also remain the previously declared PASS interval. No competitive strength or
accepted-baseline claim has been established.
