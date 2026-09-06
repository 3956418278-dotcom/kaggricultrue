# Planning implementation guide

`PROJECT.md` owns architecture; `STATE.md` owns current capability and evidence.

| Concern | Owner |
| --- | --- |
| Observation/action contract | `state.py`, `contract.py`, root `main.py` |
| Verified game arithmetic and owned-farm transition | `rules.py` |
| Structured `(C,T,L,A,Q,R)` records | `economics.py` |
| State-aware daily economic intent and placement domains | `planner.py` |
| Separate execution choices; provisional placement witnesses | `realization.py` |
| Remaining-day trajectory search and retention | `intraday.py` |
| Schedule-derived concrete tasks and greedy sanity benchmark | `execution.py` |
| Reactive selling and input-protected market orders | `market.py` |
| Fixed daily Plan lifecycle and composition | `operating.py`, `agent.py` |
| Replay-derived reference samples and cloud extraction | `src/kaggriculture_eval/player_days.py`, `reference_pipeline.py` |

## Boundary

The direction is `current real state -> fixed Daily Plan -> intraday realization`.
Plan retains existing-asset positions and real constraints on new placements.
Equivalent route-dependent placements stay open. An existing empty structure
and an empty tile requiring construction are different economic premises.

`ExecutionChoices` is not a Plan subclass. It holds placement witnesses and
staffing, never rewritten economic commitments. The next day forms a fresh
Plan; intraday divergence repairs the trajectory and leaves economic shortfalls
visible. Market selling is outside Plan. Expected revenue is not a sale schedule.

The current implementation deliberately retains the **legacy restricted search**
until the reference-data checkpoint. It uses three placement constructions,
a small beam, staffing variants and bounded tail substitutions. Its economic
value proxy and neighborhood coverage are not the requested maturity standard.
Do not present the interface repair as a completed planner redesign.

Worker/resource reservation, atomic seeds, transition parity and trajectory
retention remain useful infrastructure. The greedy controller is only a weak
sanity baseline. The reference extractor's generic `STATE_EFFECT` commitments
are not yet implemented by the legacy task generator.

## Checks

```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python scripts/verify_intraday.py --output runs/<unique-id>/scenarios.json
.venv/bin/python scripts/verify_baseline.py --output runs/<unique-id>/episodes.json --replay-dir runs/<unique-id>/replays
```

Full episodes check legality, determinism, terminal completion and runtime,
not reference-level execution strength or competitive promotion.

Inspect an extracted sample with the official game renderer:

```bash
.venv/bin/python scripts/inspect_player_day.py data/player-days/episode-<id>.jsonl.gz --sample <id>:<side>:<day> --output replays/player-day.html
```

The page contains day-start/day-end scenes, an economic-effect table and the
demonstrated trajectory. Opponent private inventory is unknown. Raw shards,
HTML viewers and cloud outputs stay in ignored data areas.
