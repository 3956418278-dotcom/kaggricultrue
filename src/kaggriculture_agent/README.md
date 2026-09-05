# Planning implementation guide

`PROJECT.md` owns the architecture and contract; `STATE.md` owns current evidence
and limitations. This guide explains where to change the implementation.

| Concern | Owner |
| --- | --- |
| Observation/action contract | `state.py`, `contract.py`, thin root `main.py` |
| Constants, production arithmetic, one-turn owned-farm transition | `rules.py` |
| Structured `(C,T,L,A,Q,R)` records | `economics.py` |
| Daily economic intent and future marginal asset estimates | `planner.py` |
| Spatial binding and realized hiring commitments | `realization.py` |
| Full remaining-day trajectory search and retention | `intraday.py` |
| Concrete tasks, market orders, retained greedy benchmark | `execution.py` |
| Daily plan lifecycle and composition | `operating.py`, `agent.py` |

## Search boundary

`Plan.selected` contains production goals, not predetermined new-asset sites.
`Realization` binds the same commitments, including their land intervals and
dated action schedules. Its `crop_targets` are execution state, not a second
economic plan. Existing fixed assets and fertilizer applications retain their
locations. Hiring uses the same commitment dimensions and supplies only the
turns after the hiring market phase.

`search_day` seeds complete trajectories for each admissible staffing count and
distinct spatial binding. Seeds include the unchanged greedy controller and
route-continuous completion policies. A bounded beam then branches at dispersed
times along retained trajectories, substitutes alternative joint assignments,
and simulates the entire tail before comparing outcomes. Further substitutions
can combine improvements. Prefixes that spend time on pickup and travel are
therefore not rejected solely for failing to produce immediate revenue.

Joint proposals reserve tasks, seeds, and pickup stock across workers. Workers
retain routes while those tasks remain available. Candidate orders and actions
are evaluated in official order by `advance_owned`; no worker movement or
logistics cost is subtracted as invented currency. End-of-day automatic drop,
ordered shed overflow, hand removal and farmer reset are actual transitions.

The comparison is lexicographic: estimated cash plus marginal remaining-asset
value, maintenance debt, then today's movement/logistics plus approximate future
service travel, then today's movement/logistics alone. At the episode boundary
only banked cash counts. Thus additional transport is worthwhile when it realizes
more money, while unnecessary logistics loses among equivalent outcomes.
Future service travel counts scheduled visit days and distance to the shed; it
is a coarse tie-breaker, not a multi-day route proof.

Search effort is deterministic (`SearchConfig`), not a wall-clock cutoff: three
retained trajectories, up to three placement constructions, all allowed early
staffing counts, and up to 64 tail-rollout substitutions. The number of edge
simulations and wall time are reported. This is bounded neighborhood search,
not exhaustive optimization or a proof that every feasible schedule is found.
Task proposals and completion policies restrict the reachable search space;
changing them does not require replacing the economic representation or rules.

`IntradaySession` stores predicted owned states and actions. Ordinary calls use
the retained tail. Routing/inventory divergence or a trade change that alters
the next attainable owned state triggers repair. Mere price changes do not
recompute economic commitments. Inputs remain reserved, while sales can be held
or reprioritized using current prices. Unexpected next-day randomness is handled
by the fresh daily observation, never by forecasting unseen shop unlocks.

## Checking changes

```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python scripts/verify_intraday.py --output runs/<unique-id>/scenarios.json
.venv/bin/python scripts/verify_baseline.py --output runs/<unique-id>/episodes.json --replay-dir runs/<unique-id>/replays
```

The scenario runner compares identical daily intent and initial conditions,
replays both controllers through the official interpreter against PASS, and
checks each simulated transition. It reports actual work, movement, logistics,
maintenance, inventory and cash separately from the model's future-value proxy.
The full-episode runner checks both seats, repeated-observation determinism,
terminal completion and inventory, timing, repair frequency, and source hashes.
Outputs use exclusive creation; choose a fresh run directory. These are
development/correctness checks, not acceptance-quality competitive evidence.
