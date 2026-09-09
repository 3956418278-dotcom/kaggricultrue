# Planning implementation guide

`PROJECT.md` owns architecture; `STATE.md` owns current capability and evidence.

| Concern | Owner |
| --- | --- |
| Observation/action contract | `state.py`, `contract.py`, root `main.py` |
| Exact owned-state transition | `rules.py` |
| Economic dimensions and Daily Plan | `economics.py`, `planner.py` |
| Intraday optimization | `intraday.py` |
| Public solved trajectory | `realization.py` |
| Exact execution/completion validation | `execution.py` |
| Reached-state value | `valuation.py` |
| Plan lifecycle and runtime composition | `operating.py`, `agent.py` |

The production boundary is exactly:

```text
OwnedState + Plan
        -> solve_intraday()
        -> Realization
        -> execute_realization()
        -> S_end
```

`Plan` describes the day's required economic/farm state changes and deliberately
does not prescribe the intraday implementation. `Realization` contains only open
placement commitments and actual per-turn worker/market actions. Assignment,
order, travel, resource flow, transfers, purchases, hiring and cash timing are
private solver decisions. Failure to find a fully executable Plan raises
`PlanningFailure`; no partial realization crosses the API.

The solver privately derives legal local event chains from Plan and the exact
rules, then jointly constrains event assignment/timing, placement, movement,
inventory, market acquisition, hiring and land. Its constructive route is only
a CP-SAT incumbent. Exact execution through `rules.advance_owned()` is the final
completion authority, and only full realizations are compared by reached-state
economic value.

Run focused checks with the OR-Tools environment:

```bash
.cache/planner-runtime/bin/python -m unittest tests.test_intraday tests.test_planner_model -v
```

Run the frozen development-only benchmark with:

```bash
.cache/planner-runtime/bin/python scripts/benchmark_intraday.py \
  .cache/reference-pilot-20260905/dataset-v1 runs/<unique-id>
```
