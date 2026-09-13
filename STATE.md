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
`Programme` freezes KEEP/EXIT/NEW assets, exact tiles, service/output/harvest/
stock/sale schedules, feed and buffer Wheat, Carrot buffers, fertilizer flows,
land+use commitments, worker count, zones and routes.

## Implemented D11+ mechanisms

- discrete known Town/shop demand events and the fixed reveal calendar;
- visible existing-opponent base pressure without opponent route/new-strategy
  prediction;
- sequential official-price BUY/SELL accounting and event sale DP;
- separated farm, field, worker, shed, sale and shared-market trajectories;
- event-driven existing-animal MAINTAIN/EXIT decisions using only the next
  production cycle, with frozen state between explicit key events;
- separate one-cycle CARE checks, held-capacity harvest gates, bounded EXIT
  liquidation tails, and minimum official survival FEED;
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

- 16 focused current-asset regressions pass. They cover attach-anytime behavior,
  frozen no-event days,
  shop/production/harvest/replacement triggers, long-cycle Cow valuation,
  CARE capacity, EXIT liquidation, held-product harvest gates, unified land
  placement, and locality decay.
- The repository suite currently passes 67 of 69 tests. The two errors are the
  existing D11 controller checks that call the intentionally disconnected
  `intraday.solve_intraday` zonal integration stub; this current-asset change
  does not modify that subsystem.
- A prior synthetic empty D11 baseline produced and fully replayed a feasible
  programme with 3 Cow, 1 Sheep and 4 W_FEED tiles before the intraday routing
  replacement was disconnected.

## Limiting issue

Runtime acceptance is not established. The zonal replacement currently leaves
`intraday.solve_intraday` disconnected, so full programme replay and episode
timing cannot be accepted. No claim of a complete runtime-ready controller or
competitive strength is accepted until that separate subsystem is connected
and a full pinned episode finishes without timeout or fallback passes.
