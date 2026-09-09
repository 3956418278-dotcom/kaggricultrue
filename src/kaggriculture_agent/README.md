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

`Plan` describes the day's required economic/farm outcomes, their exact project
placements, required outputs/deadlines, and any exceptional `EconomicWindow`.
`ActionDimension` is only a macro labor estimate. `Realization` contains the
fixed placement commitments and actual per-turn worker/market actions. Worker
assignment, order, travel and sparse resource logistics are private solver
decisions. Failure to find a fully executable Plan raises `PlanningFailure`; no
partial realization crosses the API.

The solver privately derives one shortest legal local chain for each outcome by
calling the exact unit transition. It then constrains fixed-position event
assignment/order, Manhattan travel and sparse source/pickup/transfer relations.
Normal acquisition, hiring and land orders are placed at the earliest required
turn; market timing exists only for an explicit economic window. Workforce is
tried from low to high and the first complete exact realization is selected by
transparent execution effort, never by `V(S_end)`.

Run focused checks with the OR-Tools environment:

```bash
.cache/planner-runtime/bin/python -m unittest tests.test_intraday tests.test_planner_model -v
```

Run the frozen development-only benchmark with:

```bash
.cache/planner-runtime/bin/python scripts/benchmark_intraday.py \
  .cache/reference-pilot-20260905/dataset-v1 runs/<unique-id>
```
