# Current State

## Project definition

The project targets Kaggle's full two-player `kaggriculture` environment. Stable competition semantics, submission contract, and conceptual architecture are owned by `PROJECT.md`. Competitive claims and baseline promotion are governed by `EVALUATION.md`.

The competition/reference/environment bootstrap is complete. A first deterministic rolling-horizon baseline candidate is implemented, but it has not been frozen, packaged as an immutable artifact, or accepted as the measurement anchor.

## Accepted baseline

No accepted baseline exists yet. The maintained root `main.py` now exposes a runnable project candidate; it remains a candidate until the bootstrap acceptance gates and user approval described in `EVALUATION.md` are satisfied. The official built-in `starter` agent remains only a smoke opponent.

The first accepted baseline will be a user-approved, immutable measurement anchor after contract, determinism, terminal-completion, packaging, and replay-reproduction checks. That bootstrap designation will not imply competitive strength. Later baseline promotions require acceptance-quality competitive evidence.

## Implemented capabilities

- Repository behavioral core and task-routing harness.
- Stable project, state, and competitive-evaluation authorities.
- General procedures for orientation, structural implementation, debugging, evidence validation, and independent review.
- Kaggriculture-specific procedure for strategy hypotheses, controlled arenas, replay analysis, opponent comparisons, and competitive claims.
- Maintained external-reference index with exact official distribution/source hashes, a dated compact competition snapshot, and versioned public-reference provenance.
- Reproducible Conda specification for Python `3.12.3`, pip `25.2`, and `kaggle-environments==1.32.7`, plus the exact verified transitive Python dependency lock. An ignored local `.venv/` currently realizes that specification.
- Contract verification command and 18 focused environment tests covering Python and exact dependency-lock conformance, distribution/source and schema identity, explicit seed privacy, private inventory visibility, unit-before-market ordering, lockstep purchases, step-zero town consumption, daily labor/shop refresh, market-entry truncation, atomic planting (including the nonexistent-hand edge case), malformed quantity failure, shed-overflow loss, animal placement, fertilizer lifetime, deterministic replay state, the step-718/719 terminal boundary, final-money rewards, and official root-entrypoint loading.
- Thin root `main.py` and an inspectable policy package under `src/kaggriculture_agent/`: contract normalization, immutable owned-state reconstruction, one executable Kaggriculture rule owner, the structured economic model, deterministic rolling-horizon planning, task generation/scheduling, ordered market construction, and a submission-safe wrapper.
- Six-dimensional `(C, T, L, A, Q, R)` commitments for crop projects, new and existing animals, fertilizer allocation, hiring, land expansion, and liquidation. Already-owned inputs and assets are recorded as sunk and are not charged again in marginal project value.
- Feasibility filtering for the terminal horizon, unreserved cash, land conflicts, working shed capacity, and dated labor capacity. Feasible projects are compared lexicographically through derived profit/action, terminal profit, and profit/tile-day views rather than a weighted score.
- Current-state execution tasks expose position, deadline, travel distance, carried-input eligibility, continuity, dependencies, and unit capacity. The scheduler caps simultaneous plant actions by owned seeds before the contract adapter performs a second atomic-plant guard.
- Seven candidate-focused tests cover exact market-price parity with the pinned implementation, six-dimensional project coverage, sunk seed treatment, support-project representation, shared-seed safety, one-time crop maturity, and deterministic JSON-safe output.
- `scripts/verify_baseline.py` runs an unguarded full-episode health sweep against the built-in `starter`, both seats for every explicit seed, verifies every repeated observation decision, requires zero terminal sellable inventory, and reports timing and operation traces. On 2026-09-05, seeds 17 and 29 completed all four 720-state games with `DONE` status in both seats; maximum observed local decision time was under 0.038 seconds. The traces exercised crops, animals, fertilizer, hiring, storage/transport, purchases/sales, and terminal liquidation. A focused synthetic state also exercises the land commitment and emitted `BUY_LAND` path. This is exploratory runtime evidence only.

All 25 maintained tests pass under the pinned Linux x86_64 environment. There is still no acceptance arena/result manifest, replay analyzer, submission builder, immutable candidate artifact, retained opponent, or accepted competitive strategy.

## Accepted competitive evidence

None. No local match result, Kaggle episode, leaderboard score, or baseline-promotion result has been accepted for this repository.

## Current decisions

- Use the full `kaggriculture` environment, not `kaggriculture_beginner`.
- The local executable authority is full `kaggriculture` schema `0.1.0` in `kaggle-environments==1.32.7`, matching official release source commit `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`. The key files also matched upstream commit `bbda347572cf5134e56f0eb49e8058e2560f9844` on 2026-09-04.
- Treat the executable pinned environment as the behavioral authority and re-audit contract changes before pooling evidence across versions.
- Establish deterministic local execution, exact contract tests, side-swapped evaluation, and replay retention before competitive strategy work.
- Prefer deterministic or planning-based baselines initially. Learning is justified only by a measured limitation and an evaluation design capable of testing the proposed improvement.
- Keep exploratory, acceptance-quality, and illustrative evidence distinct. Strategy changes are empirical claims, not accepted improvements by inspection.
- Keep agent mechanisms, evaluation infrastructure, durable state, and generated data in separate ownership areas as defined by `AGENTS.md` and `PROJECT.md`.
- Keep official/package identities and a small curated reference index in the project body; keep raw page responses, wheels, archives, full external repositories, generated replays, and downloaded opponents outside it unless deliberately promoted after provenance and license review.
- Use the structured `(C, T, L, A, Q, R)` commitment record as the common decision language. Historical acquisition cost is documentary for existing assets; only remaining marginal consequences affect current choice.
- Keep the first policy deterministic and observation-pure. It projects current prices, its own planned transactions, Town Center demand, and already-unlocked shops, while deliberately excluding opponent forecasts and future random shop unlocks.
- Treat the current implementation as `candidate`, not `baseline`, regardless of its smoke-match money results.

## Limiting uncertainties

- Kaggle does not publicly expose an immutable deployed evaluator image/package identity or stable numeric RAM, disk, and vCPU limits. Local byte identity with the remote runtime is therefore not established.
- Contract tests cover the implementation's central ordering, market-price, overflow, placement, fertilizer, and terminal dependencies but not every crop/animal timing path, invalid-action form, timeout/overage behavior, or multi-file `.tar.gz` import layout.
- The planner uses direct deterministic estimates: conservative crop yields, current-market liquidation values, known current town demand, own projected supply, coarse travel actions, and maximum configured daily hand capacity. It does not simulate the executor turn by turn, model price interactions among every support decision, forecast opponents, or predict future shop unlocks.
- Scheduling is a current-turn greedy deadline/travel assignment. Continuity is inferred from position and carried inputs rather than retained cross-turn assignments; build/place and pickup/use dependencies normally resolve over successive observations. No formal guarantee yet proves that every accepted long-horizon commitment will be serviced.
- Storage feasibility uses a working peak approximation. Cash feasibility reserves current feed and project upfront costs but does not construct a complete dated cash-flow proof for all future service purchases.
- The submission wrapper intentionally degrades unexpected internal exceptions to legal pass actions. The health command uses the unguarded decision core to expose such errors, but runtime telemetry and retained failure replays do not yet exist.
- The candidate has not been frozen or validated as a self-contained `.tar.gz`, and local remote-runtime equivalence remains unestablished.
- Acceptance-set opponents, their licensing/provenance policy, seed count, practical-improvement threshold, non-inferiority tolerances, interval method, and compute budget have not been chosen.
- The competition is active and the official environment may change. There is no update-monitoring or compatibility procedure implemented yet.
- Remote archive/package behavior has only been checked against the documented root-`main.py` contract and local single-file loader; actual Kaggle validation remains untested because no project submission exists.

## Next meaningful work

Characterize and freeze the candidate before any baseline designation:

1. add replay diagnostics that compare planned commitments, emitted tasks, realized service, asset escape/decay, discarded overflow, idle labor, and terminal inventory, then use them to test the planner's realizability assumptions;
2. implement the thin side-swapped arena, manifest/result schema, replay retention, source/package hashing, and deterministic audit path required by `EVALUATION.md`;
3. build and locally validate a self-contained `.tar.gz`, including its import layout and per-action timing, then freeze an immutable candidate identity;
4. select licensed, hashable development opponents and predeclare the bootstrap acceptance gates; obtain user approval before recording this or a revised candidate as the first accepted baseline;
5. defer competitive promotion claims and Kaggle submission interpretation until the corresponding evidence protocol is in place.
