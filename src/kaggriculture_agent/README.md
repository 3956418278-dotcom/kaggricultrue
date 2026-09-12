# D11+ programme controller

The maintained runtime boundary is:

```text
observation -> canonical State -> macro Programme -> intraday routes -> action
```

- `state.py` retains complete official animal/crop records for both visible farms.
- `rules.py` owns pinned `kaggle-environments==1.32.7` transitions and prices.
- `market.py` owns reveal timing, known demand events, visible opponent pressure,
  and sequential sale DP with shared shed-capacity repair.
- `programme.py` is the frozen macro/executor contract.
- `planner.py` owns KEEP/EXIT, mandatory/optional service, W_FEED, W/C buffers,
  exact-tile long candidates, land+use comparisons, and terminal-cash selection.
- `intraday.py` may assign workers and construct routes, but cannot alter macro
  asset, placement, service, fertilizer, CARE, or lifecycle decisions.
- `simulation.py` performs the final complete pinned-rule replay and maintains
  separate output, field, worker, shed, sale, and market trajectories.
- `operating.py` freezes the programme within a day and only re-solves remaining
  sales at a real SELL checkpoint.

The prior programme/target tables and fixed-count policy are not present. D5-D10
pass after the existing fixed four-day opening; this controller starts at D11.
