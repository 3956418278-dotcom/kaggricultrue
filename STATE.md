# Current State

## Project definition

The project targets Kaggle's full two-player `kaggriculture` environment. Stable competition semantics, submission contract, and conceptual architecture are owned by `PROJECT.md`. Competitive claims and baseline promotion are governed by `EVALUATION.md`.

The competition/reference/environment bootstrap is complete. A first deterministic day-level rolling-horizon baseline candidate is implemented, but it has not been frozen, packaged as an immutable artifact, or accepted as the measurement anchor.

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
- Contract verification command and 19 focused environment tests covering Python and exact dependency-lock conformance, distribution/source and schema identity, explicit seed privacy, private inventory visibility, unit-before-market ordering, lockstep purchases, step-zero town consumption, daily labor/shop refresh (including ordered automatic inventory drop, shed overflow, hand removal, and farmer reset), market-entry truncation, atomic planting (including the nonexistent-hand edge case), malformed quantity failure, explicit shed-overflow loss, animal placement, fertilizer lifetime, deterministic replay state, the step-718/719 terminal boundary, final-money rewards, and official root-entrypoint loading.
- Thin root `main.py` and an inspectable policy package under `src/kaggriculture_agent/`: contract normalization, immutable owned-state reconstruction, one executable Kaggriculture rule owner, the structured economic model, deterministic day-level rolling-horizon planning, daily-plan lifecycle control, task generation/scheduling, ordered market construction, and a submission-safe wrapper.
- Six-dimensional `(C, T, L, A, Q, R)` commitments for crop projects, new and existing animals, fertilizer allocation, hiring, land expansion, and liquidation. Already-owned inputs and assets are recorded as sunk and are not charged again in marginal project value.
- Feasibility filtering for the terminal horizon, unreserved cash, land conflicts, working shed capacity, and dated labor capacity. Feasible projects are compared lexicographically through derived profit/action, terminal profit, and profit/tile-day views rather than a weighted score.
- Current-state execution tasks expose position, deadline, travel distance, carried-input eligibility, continuity, dependencies, and unit capacity. The scheduler caps simultaneous plant actions by owned seeds before the contract adapter performs a second atomic-plant guard.
- `DailyPlanningSession` forms one `Plan` at the first observation of each day and reuses the same `(C,T,L,A,Q,R)` commitments and staffing target during ordinary intraday observations. It permits at most one explicitly reasoned repair when a target premise is invalidated or mandatory same-day task count exceeds even the remaining zero-travel worker capacity. Intraday execution remains state-reactive, and product selling is reprioritized from current shared-market prices without changing the production plan.
- Daily staffing is a target count computed from the opening commitments' dated work and coarse travel load. Outstanding hands are ordered near day start and are not ordered again after the target is observed; fixed daily crop, animal, fertilizer, and land commitments likewise buy only their still-unfulfilled inputs. Late-day workers no longer return solely to preserve carried inventory because official refresh performs the capacity-limited drop before hands disappear and the farmer resets.
- Correctness repairs preserve the original planner and scheduler design while making five depended-on decisions rule-derived: one-time harvest waits for any valuable final same-day water; animal output follows base production, accumulated care bonuses, feeding, intervals, and held-yield caps; fertilizer projects contain the actual marginal output schedule and realizable revenue; terminal liquidation values and schedules recoverable yield on crop and animal tiles; and market construction reserves its ten-entry budget for required commitment inputs before admitting sales.
- Twenty candidate-focused tests cover exact market-price parity with the pinned implementation, six-dimensional project coverage, sunk seed treatment, support-project representation, shared-seed safety, deterministic JSON-safe output, the five earlier correctness repairs, stable intraday commitments and staffing, early non-repeated hiring, staffing invariance to elapsed hours alone, fulfillment-aware non-repeated commitment orders, adaptive routing and market selling under a fixed plan, bounded material replanning, next-day refresh, and reliance on the official end-of-day inventory drop. Animal and fertilizer projection tests directly challenge the pinned refresh functions.
- `scripts/verify_baseline.py` runs an isolated daily-planning session in an unguarded full-episode health sweep against the built-in `starter`, both seats for every explicit seed, verifies every repeated observation decision, requires zero terminal sellable inventory, and reports timing and operation traces. On 2026-09-05, the day-cadence candidate completed seeds 17 and 29 in both seats—all four 720-state games—with `DONE` status and zero terminal sellable inventory; maximum observed local decision time was under 0.038 seconds. The traces exercised crops, animals, fertilizer, hiring, storage/transport, purchases/sales, and terminal liquidation; focused synthetic states exercise land commitments and emitted land actions. This is exploratory runtime evidence only.
- An official-scene replay reader under `src/kaggriculture_eval/`, with thin command `scripts/view_replay.py`, loads validated `Environment.toJSON()` replays or runs a fresh current-candidate-versus-`starter` game, reports recorded environment identity and terminal results, and serves the installed Kaggriculture animated visualizer with timeline controls. It can also emit a single HTML viewer and retain the underlying replay at an explicitly requested ignored path. Seven focused tests cover round-trip loading, identity reporting, official-renderer use, timeline selection, wrapped renderer payloads, and malformed or wrong-environment rejection. A full 720-state seed-17 candidate-seat-0 game was loaded and rendered end to end under the pinned environment.

All 46 maintained tests pass under the pinned Linux x86_64 environment. There is still no acceptance arena/result manifest, semantic replay analyzer, submission builder, immutable candidate artifact, retained opponent, or accepted competitive strategy.

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
- Keep the first policy deterministic and episode-local: identical observations within the same planning session remain idempotent, while the only retained policy state is the current per-player daily plan and its bounded repair count. Daily plan formation projects opening prices, its own planned transactions, Town Center demand, and already-unlocked shops, while deliberately excluding opponent forecasts and future random shop unlocks; intraday selling may use current prices.
- Treat the current implementation as `candidate`, not `baseline`, regardless of its smoke-match money results.

## Limiting uncertainties

- Kaggle does not publicly expose an immutable deployed evaluator image/package identity or stable numeric RAM, disk, and vCPU limits. Local byte identity with the remote runtime is therefore not established.
- Contract tests cover the implementation's central ordering, market-price, overflow, placement, fertilizer, and terminal dependencies but not every crop/animal timing path, invalid-action form, timeout/overage behavior, or multi-file `.tar.gz` import layout.
- The planner uses direct deterministic estimates: conservative crop yields, current-market liquidation values, known current town demand, own projected supply, coarse travel actions, and maximum configured daily hand capacity. Animal projections now follow exact production arithmetic but assume daily feed/care, prompt harvesting, and the modeled placement date. Fertilizer projections now compute exact marginal physical yield under daily watering and prompt harvest and reject applications that cannot reach the crop before the relevant daily boundary, while their future sale value and input opportunity cost still use the baseline's simple market forecast. The planner does not simulate the executor turn by turn, model price interactions among every support decision, forecast opponents, or predict future shop unlocks.
- Daily staffing and feasibility still use aggregate commitment work, coarse travel, and a lower-bound remaining-task check rather than a joint route schedule. Required orders can delay some hires from hour 0 into the next few turns, and the single repair may detect infeasibility only after execution has already lost useful capacity. A process that first receives an episode in mid-day forms its plan from that current observation because the earlier opening state is unavailable.
- Scheduling is a current-turn greedy deadline/travel assignment. Continuity is inferred from position and carried inputs rather than retained cross-turn assignments; build/place and pickup/use dependencies normally resolve over successive observations. No formal guarantee yet proves that every accepted long-horizon commitment will be serviced.
- Storage feasibility uses a working peak approximation. Cash feasibility reserves current feed and project upfront costs but does not construct a complete dated cash-flow proof for all future service purchases. Automatic refresh drop avoids unnecessary return travel but may still discard carried overflow when the shed lacks capacity; the controller does not yet schedule a pre-refresh capacity clearance proof.
- Terminal recovery checks each on-tile yield against an individual worker-to-tile-to-shed route. It does not yet solve a joint multi-worker terminal routing problem, so several individually recoverable tiles may still compete for the same remaining worker capacity.
- Required non-labor market inputs are protected from sale-list truncation and same-animal purchases are aggregated. If an externally constructed plan exceeds ten required non-labor entries it is rejected; within normal planner bounds, daily hires use only the entry slots left after those inputs.
- The submission wrapper intentionally degrades unexpected internal exceptions to legal pass actions. The health command uses the unguarded decision core to expose such errors, but runtime telemetry and retained failure replays do not yet exist.
- The replay reader is a local visual inspection tool: it does not download Kaggle episodes, annotate frames with internal planner decisions, or derive service/overflow/idle-labor diagnostics. Replays produced by a different recorded package/schema are rendered with an explicit identity warning rather than treated as contract-equivalent evidence.
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
