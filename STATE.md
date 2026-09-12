# Current State

## Project and runtime

The project targets full Kaggriculture under pinned
`kaggle-environments==1.32.7`. The maintained root `main.py` exposes the
submission agent. The fixed opening owns D1-D4, D5-D10 deliberately pass because
the former programme/target-table strategy was removed, and the new controller
owns D11+.

The runtime boundary is now:

```text
observation -> canonical State -> macro Programme -> intraday executor -> action
```

`State` retains private own state, public opponent state, complete official
animal/crop tile records, market inventory/prices, shops, land and worker state.
`Programme` freezes KEEP/EXIT/NEW assets, exact tiles, service/output/harvest/
stock/sale schedules, feed and buffer Wheat, Carrot buffers, fertilizer flows,
land+use commitments, worker count, zones and routes.

## Implemented D11+ mechanisms

- discrete known Town/shop demand events and the fixed reveal calendar;
- visible existing-opponent base pressure without opponent route/new-strategy
  prediction;
- sequential official-price BUY/SELL accounting and event sale DP;
- separated farm, field, worker, shed, sale and shared-market trajectories;
- iterative existing-animal EXIT and optional-service comparisons;
- dated mandatory survival/base-production service;
- W_FEED sourcing, W/C/EMPTY buffers, exact-tile long candidates, WAIT_REVEAL,
  and combined BUY_LAND+exact-use comparisons;
- minimum feasible staffing search, exact pickups, bundled fixed-tile tasks,
  INNER/OUTER derivation, nearest-neighbor outer sweep and strict 2-opt;
- complete final pinned-rule replay and intraday programme freezing, with only
  remaining sales re-solved at a real sale checkpoint.

The former `economics.py`, fixed programme/pace tables, generic D11+ heuristic
planner and valuation layer are deleted. Frozen player-day research records use
an evaluation-local schema and do not enter the submitted strategy.

## Validation

- All 47 maintained unit/contract/reference tests pass in the pinned `.venv`.
- Six focused D11+ checks cover canonical state, reveal/demand events,
  sequential pricing, sale-stock separation, macro placement freezing and full
  simulator separation.
- A synthetic empty D11 state produced and fully replayed a feasible programme
  with 3 Cow, 1 Sheep and 4 W_FEED tiles; planned 18 Milk and 5 Wool sales were
  physically present, sold, and matched terminal cash.
- The same construction took roughly 4.6 seconds after equivalence caching.

## Limiting issue

Runtime acceptance is not established. Replanning the next heavier day in the
same synthetic trajectory did not finish inside a 30-second diagnostic bound.
The required accept-one/recompute-all optional and exact-tile candidate loops are
present, but their current implementation is not yet safe against the official
per-action/overage budget over a full D11-D30 episode. No claim of a complete
runtime-ready D11+ controller or competitive strength is accepted until this is
resolved and a full pinned episode finishes without timeout or fallback passes.
