# Current State

## Project definition and accepted baseline

The project targets full Kaggriculture, pinned to `kaggle-environments==1.32.7` and schema `0.1.0`. `PROJECT.md` owns stable semantics and architecture; `EVALUATION.md` owns evidence rules. The environment/reference bootstrap is complete.

The maintained root `main.py` is a deterministic runnable **candidate**, not an accepted baseline. No competitive evidence or baseline promotion has been accepted. Packaging and acceptance-quality arenas remain unfinished.

## Current responsibility boundary

- Runtime direction is `real current state -> fixed Daily Plan -> intraday planner -> trajectory`.
- Daily economics uses the full current farm, including land/layout and existing structures. Commitments retain `(C,T,L,A,Q,R)`; actual existing locations stay fixed and economically equivalent new placements have explicit state-derived domains.
- `ExecutionChoices` is separate from Plan. The former `Realization(Plan)`, `bind_plan` and `unbind_project` interfaces are removed. Choices hold placement witnesses and staffing, without replacing or modifying economic commitments.
- Plan no longer contains a hire count or HIRE support commitments. The retained weak benchmark estimates staffing in the execution layer; the provisional search compares staffing variants.
- Selling is owned by `market.py`, outside Plan. Expected revenue remains an economic estimate, not a sale schedule. Terminal recovery objectives concern recoverable farm output, not prescribed sales. Input-protected market entry ordering remains implemented.
- Task generation is restricted to today's commitment work schedules, plus supporting logistics. Ordinary intraday observations retain the same economic Plan. Physical divergence triggers execution repair, not silent economic replanning or reduction of targets.
- The current search is still the **legacy restricted neighborhood search**. Its variants, five-hand default and economic-state-first ranking are not the requested mature joint realization planner. Redesign is deliberately deferred until the dataset checkpoint.

## Maintained capabilities

- Thin submission entrypoint; contract normalization; immutable owned-state reconstruction; one policy rule/transition owner; structured economic commitments; daily lifecycle; execution tasks; reactive market module; retained trajectories and greedy sanity benchmark.
- Verified ordering and timing: atomic seed requests, ordered unit actions before market, market lockstep, town demand, crop/animal production, fertilizer, shed overflow, daily automatic inventory drop, hand removal/farmer reset, and the 718/719 terminal boundary.
- Earlier correctness fixes remain: valuable final watering before one-time harvest, feed/care-derived animal yield, marginal fertilizer output, recoverable terminal crop/animal yield, and protection of required purchases from market-entry truncation.
- Local official-scene replay viewer and deterministic environment, candidate, transition-parity and replay tests.
- Python 3.12.3 and the exact local dependency lock remain maintained in the project environment. The authenticated `kaggle` Conda environment is separate tooling, not the simulation runtime.

## Plan-to-realization dataset pilot

The user approved a private CPU-only Kaggle pilot, with checkpoints before planner redesign. No model training is in scope.

Maintained implementation:

- `src/kaggriculture_eval/player_days.py`: shared-field/clock normalization, official full-replay reexecution, semantic unit-effect extraction, day slicing, attempted-versus-achieved effects, placement domains and deterministic compressed episode shards.
- `src/kaggriculture_eval/reference_pipeline.py`: bounded cached metadata discovery, per-side qualification, deterministic candidate sampling, source/transition checks, quarantine reporting and hash-compatible episode checkpoints.
- `scripts/kaggle_reference_pilot.py`: explicit source allowlist and private CPU notebook preparation. Large extraction occurs in Kaggle, not locally. The notebook payload contains only candidate metadata, below Kaggle's verified 1 MB source limit. A prior Dataset can be attached for checkpoint resume.
- `scripts/inspect_player_day.py`: HTML inspection with official day-start/day-end scenes, a realization timeline, an economic-effect table and worker/market traces. Opponent public farm state is retained, but its private inventory is not reconstructed or represented as known.

Source and selection evidence:

- Official index `kaggle/kaggriculture-episodes-index`, inspected version 37: 37 daily datasets and 26,212 episodes. Index manifest SHA-256: `78c4d110c8e1f5c1b78654ed6d164dc1ac22f59b2098683d34622fc6356747fe`.
- Pilot source: `kaggle/kaggriculture-episodes-2026-09-04`, 668 candidates. Daily manifest SHA-256: `91b2037cdad455022cf90688d593b72cc93c84f767f7ad1be722d3eccd67c1a4`.
- Index/daily manifests have aggregate scores, not player-side ratings. Replays have names but no submission IDs. The public episode service actually returned submission ID, team ID, side index and pre-/post-game score; confidence was not present in the inspected responses. Metadata discovery is not a leaderboard-ranking method.
- Frozen leaderboard snapshot: 2026-09-05 13:44:52 UTC. Qualify each side only if its team is in that snapshot's top ten and its pre-game rating is at least the tenth-place score, **2828.9**. This is a snapshot-qualified high-rating cohort, not proven historical top-ten rank or optimal execution.
- A deterministic hash sample selects 100 episodes before extraction, without selecting winners. Cached metadata joins 94 candidates and qualifies 96 player-sides before replay validation. Six missing joins stay unqualified; the other side is not automatically included.
- The sampled official replay 105620288 reproduced all 719 joint transitions, 1,440 player observations and terminal rewards under the pinned engine. Local illustrative extraction produced 60 player-days; this is a pipeline check, not the final dataset.
- Frame `t+1` contains the action from state `t`; side 1 can omit shared `step`. Days 0–28 have 24 actionable turns; day 29 has 23.

Cloud status at handoff:

- Private notebook: https://www.kaggle.com/code/f7e6n5g4/kaggriculture-player-day-pilot
- Version 1 was successfully uploaded and Kaggle reported `RUNNING`. CPU requested, GPU disabled, one-hour run limit, 100-candidate cap.
- The user explicitly requested stopping monitoring once successful startup was confirmed. Monitoring has stopped. Completion, actual extracted counts, quarantines and cloud runtime are **not yet verified**.
- No final online player-day Dataset has been created or versioned yet. Retrieval, output audit and private Dataset publication remain pending.
- Preparation/cache: `.cache/reference-pilot-20260905/`; exploratory raw downloads: `.cache/reference-intake-20260905/`. These are ignored data, not maintained project authority.

## Current validation

All **71 maintained tests pass** under the pinned local environment. Eight new tests cover the fixed ownership boundary, real placement constraints, absence of opportunistic economic work outside Plan, official replay round-trip, shared clock normalization, final-day slicing, semantic removal of staffing/logistics/sales, atomic failed planting attempts, per-side qualification and deterministic shard round-trip.

Four full candidate-versus-starter episodes (seeds 17/29, both seats) completed 720 states with both players DONE and no terminal sellable inventory. Repeated-observation decisions were deterministic. Each episode used 30 daily searches without intraday repair. Maximum local decision times were 1.81–2.07 seconds; measured overage totals were 5.27–7.54 seconds, below the 60-second reserve locally. Reports, source hashes and complete replays are in `runs/boundary-20260905/`. These are integration results, not competitive evidence or proof of remote runtime equivalence.

The older handcrafted execution scenarios remain sanity checks only. Their historical score table is superseded as the maturity standard by the same-state/same-Plan reference benchmark, which has not yet been run.

## Remaining limitations and decisions

- The semantic player-day schema is a **pilot**, not an accepted benchmark. It captures direct tile effects, land unlocks and their physical input/output deltas; standalone stock-acquisition objectives, ambiguous no-ops, repeated/new entity lifecycles, and placement dependencies need review against extracted samples. Failed input-dependent actions are explicitly distinguished from achieved work.
- Reconstructed `STATE_EFFECT` commitments are not yet supported by the legacy search/task generator. There are no candidate-versus-reference realization results. Do not report the planner as reference-comparable or mature.
- Fixed-Plan completion evaluation, general coupled action-space search, better resource organization, broader placement coverage and elimination of arbitrary search templates belong to the next redesign, after checkpoint approval.
- Daily admission and multi-day asset value retain coarse labor/travel/storage assumptions and optimistic future maintenance. They are not proofs of executable terminal profit.
- Selling remains separate. Sale-financing effects have not been counterfactually measured; flag/control them only when they materially change achievable realization, not merely because hiring occurs.
- Cloud Python/image details and extraction diagnostics must be read from completed output. Local dependency identity does not establish byte-identical Kaggle infrastructure.
- Acceptance arena design, packaged submission validation, immutable baseline designation and competitive promotion remain separate unfinished work.

## Next meaningful work

At the next requested checkpoint, retrieve the completed notebook output without silently treating startup as success of extraction. Verify shard hashes, actual side/team/submission coverage, source identities, semantic goals/domains, exclusions and representative official-scene player-days. Publish the audited result as a private versioned Kaggle Dataset, preserving exact extractor source identity.

Then present the dataset evidence and proposed same-state/same-Plan development and held-out comparison before redesigning intraday search. Preserve the fixed Plan, separate market ownership and candidate status throughout.
