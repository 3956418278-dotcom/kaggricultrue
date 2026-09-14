# Observation-driven programme controller

The maintained runtime boundary is:

```text
observation -> canonical State -> macro Programme -> intraday routes -> action
```

- `state.py` retains complete official animal/crop records for both visible farms.
- `rules.py` owns pinned `kaggle-environments==1.32.7` transitions and prices.
- `market.py` owns reveal timing, known demand events, the runtime NOW/+4/+8
  trade machine, and the retained offline terminal sale comparator.
- `programme.py` is the frozen macro/executor contract; `CurrentAssetState`
  carries existing assets without daily terminal expansion.
- `current_assets.py` owns short PRODUCE/MAINTAIN/EXIT and crop checkpoint
  decisions, including nearest-first sequential EXIT revaluation and the
  separate one-cycle CARE/WATER/FERTILIZE checks.
- `return_requirements.py` owns the farthest-first EOD-capacity split and the
  D1-D10 near-shed `must_return` labels consumed by intraday routing.
- `planner.py` owns current-asset compilation, mandatory inputs, seed ledger,
  current-value long candidates, and scale-based land purchase.
- `intraday.py` may assign workers and construct routes, but cannot alter macro
  asset, placement, service, fertilizer, CARE, or lifecycle decisions.
- `simulation.py` retains pinned-rule replay for validation and route-proven
  DROP accounting; it is not the runtime economic valuation path.
- `operating.py` freezes the programme within a day and re-solves short-horizon
  sales from real state every four turns.

The prior programme/target tables and fixed-count policy are not present. The
planner/current-asset APIs can attach to any legal day. The maintained submission
currently chooses its handoff after the fixed opening; that activation policy is
separate from the current-asset mechanism.
