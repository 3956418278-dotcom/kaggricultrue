# Current State

## Project definition

The project targets Kaggle's full two-player `kaggriculture` environment. Stable competition semantics, submission contract, and conceptual architecture are owned by `PROJECT.md`. Competitive claims and baseline promotion are governed by `EVALUATION.md`.

The competition/reference/environment bootstrap is complete. The deterministic candidate now combines daily economic intent with bounded full-remaining-day trajectory search. It has not been packaged as an immutable artifact or accepted as the measurement anchor.

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
- `Plan` owns daily economic intent, fixed existing-asset obligations, production goals, required inputs and constraints. New-asset locations are open in the same `(C,T,L,A,Q,R)` land/work schedules; `Realization` owns their spatial binding. The three-production-project quota and twelve-tile enumeration frontier are removed. Aggregate economic feasibility remains an admission estimate; the intraday transition search measures executable outcomes.
- `intraday.py` compares complete reachable trajectories through the remaining day, seeding an unchanged greedy incumbent and route-continuous completions for alternative placements and staffing counts. Its bounded beam substitutes joint task assignments and simulates whole tails, avoiding shallow-prefix rejection of pickup/travel/use chains. Task/resource reservation, retained routes, batched/split pickups, harvest/water dependencies, optional sales deferral, terminal fertilizer collection, and automatic refresh drop are executable choices. The original `execution.schedule`/`execute` remains the greedy benchmark, not the submitted controller.
- Staffing is chosen near day start by comparing realized outcomes for all counts up to the configured five-hand bound, not just by dividing aggregate work by shrinking remaining time. `Realization` records hiring in the same commitment model, with capacity beginning after the hiring market phase. Selected staffing and locations persist through ordinary execution; a repair preserves completed placements and the daily economic goals. Required feed now includes new and already-owned unplaced animals.
- `DailyPlanningSession` forms one economic plan per day and permits at most one repair of an invalidated existing physical premise. It no longer counts tasks from speculative fresh placements to trigger late economic replanning. `IntradaySession` retains predicted owned states and actions, repairs routing/inventory or input-affordability divergence, and constructs current-market orders without redoing daily economics. New-day observations start fresh plans after hand removal and farmer reset.
- The single rule owner now includes a branch-isolated default-contract owned-farm transition: ordered units, atomic planting, market orders, known town consumption, decay, feed/care production, fertilizer, daily refresh, overflow, land, hiring and terminal timing. Market inventory reconstruction preserves legal negative inventory. Opponent transactions and future random shop unlocks are deliberately excluded from prediction.
- Correctness repairs preserve the original planner and scheduler design while making five depended-on decisions rule-derived: one-time harvest waits for any valuable final same-day water; animal output follows base production, accumulated care bonuses, feeding, intervals, and held-yield caps; fertilizer projects contain the actual marginal output schedule and realizable revenue; terminal liquidation values and schedules recoverable yield on crop and animal tiles; and market construction reserves its ten-entry budget for required commitment inputs before admitting sales.
- Twenty candidate-focused tests cover exact market-price parity with the pinned implementation, six-dimensional project coverage, sunk seed treatment, support-project representation, shared-seed safety, deterministic JSON-safe output, the five earlier correctness repairs, stable intraday commitments and staffing, early non-repeated hiring, staffing invariance to elapsed hours alone, fulfillment-aware non-repeated commitment orders, adaptive routing and market selling under a fixed plan, bounded material replanning, next-day refresh, and reliance on the official end-of-day inventory drop. Animal and fertilizer projection tests directly challenge the pinned refresh functions.
- Seventeen additional tests cover full-trajectory quality, clustered/dispersed and competing routes, carry chains, terminal banked cash, marginal hiring, persistent placement, absence of the production-count quota, deterministic search, trajectory reuse and repair, cash invalidation, fresh next-day plans, and direct official-transition parity. Both controllers' complete scenario trajectories are replayed through the official interpreter, not just the local model.
- `scripts/verify_intraday.py` records nine maintained development scenarios, source hashes, timings, economic-state estimates and actual execution metrics. The 2026-09-05 final run is retained at `runs/intraday-20260905-final/scenarios.json`; search took 0.12–0.63 seconds per scenario. The one-turn-short clustered control deliberately cannot fulfill every fixed fertilizer/harvest obligation. Results below are development evidence, not a competitive promotion.
- `scripts/verify_baseline.py` uses an unguarded, isolated session, tests every repeated-observation decision, retains optional full replays and source-hashed diagnostics, and rejects source changes during a run. The final 2026-09-05 sweep completed seeds 17/29 in both seats: all four 720-state games were `DONE`, with zero terminal sellable inventory. Each episode used exactly 30 daily searches, zero intraday repairs, and zero predicted unfulfilled new-production goals. Maximum local decision time was 2.294 seconds; p95 was 0.0011–0.0013 seconds. Measured cumulative time above the one-second allowance was 7.80–10.54 seconds per episode, below the 60-second reserve locally. Reports and all four replays are under `runs/intraday-20260905-final/`. These timings do not prove remote resource equivalence.
- An official-scene replay reader under `src/kaggriculture_eval/`, with thin command `scripts/view_replay.py`, loads validated `Environment.toJSON()` replays or runs a fresh current-candidate-versus-`starter` game, reports recorded environment identity and terminal results, and serves the installed Kaggriculture animated visualizer with timeline controls. It can also emit a single HTML viewer and retain the underlying replay at an explicitly requested ignored path. Seven focused tests cover round-trip loading, identity reporting, official-renderer use, timeline selection, wrapped renderer payloads, and malformed or wrong-environment rejection. A full 720-state seed-17 candidate-seat-0 game was loaded and rendered end to end under the pinned environment.

All 63 maintained tests pass under the pinned Linux x86_64 environment. There is still no acceptance arena, general semantic replay analyzer, submission builder, immutable candidate artifact, retained external opponent, or accepted competitive strategy.

### Current execution-development evidence

Identical daily intent and initial states are compared against the retained greedy controller, with the official interpreter as transition oracle, PASS opposition, explicit seed 41, and weeds disabled to isolate execution. “Value” below is the model's cash-plus-marginal-assets estimate, not a competitive return; at the terminal boundary it is actual banked cash. Travel/logistics counts combine movement with pickup/drop/place actions.

| Scenario | Value: greedy → search | Travel/logistics: greedy → search |
| --- | ---: | ---: |
| Clustered, exactly feasible | 2,900 → 7,283 | 20 → 10 |
| Clustered, one turn short | 2,900 → 6,293 | 19 → 10 |
| Dispersed maintenance | 6,301 → 7,292 | 40 → 27 |
| Competing worker routes | 11,121 → 12,548 | 39 → 21 |
| Mixed crop/animal maintenance | 15,664 → 15,664 | 28 → 19 |
| Carry-dependent chains | 15,422 → 15,422 | 16 → 4 |
| Terminal sale | 5,000 → 5,247 | 10 → 12 |
| Hiring unlocks work | 5,305 → 9,256 | 18 → 34 |
| Persistent placement | 12,440 → 12,440 | 17 → 14 |

The feasible clustered case harvests all four crops; dispersed/competing routes eliminate the benchmark's missed daily maintenance. Hiring chooses two hands instead of the allowed three; an empty-work regression chooses none. Extra transport in terminal/hiring cases realizes more value, rather than being classified as waste merely because movement increased. These deliberately selected scenarios support improved execution here, not universal scheduling optimality or leaderboard strength.

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
- Keep the policy deterministic and episode-local: identical observations within a session are idempotent. Retain daily intent, bounded economic repair count, the remaining-day trajectory and compact planning diagnostics. Daily planning projects opening prices, own transactions and known town/shop demand; intraday transitions model own trades and known demand exactly while selling uses current prices. No opponent forecast or future random-shop prediction is introduced.
- Compare full reachable trajectories primarily by economic state and remaining maintenance debt, then by movement/logistics plus approximate future servicing travel. Search effort is deterministic, not a wall-clock-dependent policy choice. Keep the greedy controller available for repeatable execution comparisons; see `src/kaggriculture_agent/README.md` for ownership and search details.
- Treat the current implementation as `candidate`, not `baseline`, regardless of its smoke-match money results.

## Limiting uncertainties

- Kaggle does not publicly expose an immutable deployed evaluator image/package identity or stable numeric RAM, disk, and vCPU limits. Local byte identity with the remote runtime is therefore not established.
- Contract tests cover the implementation's central ordering, market-price, overflow, placement, fertilizer, and terminal dependencies but not every crop/animal timing path, invalid-action form, timeout/overage behavior, or multi-file `.tar.gz` import layout.
- Daily admission and beyond-day value remain approximate: coarse travel and future labor capacity, conservative crop forecasts, daily feed/care and prompt-harvest assumptions, and no complete multi-day cash-flow or routing proof. Future asset valuation does not jointly price every asset's supply interaction; unsown seeds/unplaced animals retain an acquisition-price option proxy before the terminal boundary. These are value assumptions, not exact realizable terminal cash. Future servicing uses scheduled visit days and distance rather than a solved multi-day route.
- Intraday search is bounded neighborhood search over complete trajectories: three retained trajectories, up to three placement constructions, the configured five-hand bound, and up to 64 tail substitutions. Joint proposal/completion heuristics restrict coverage. It genuinely compares resulting states but is neither exhaustive nor guaranteed to find every feasible schedule or globally optimal placement. Economic intent can itself be infeasible, as the one-turn-short control demonstrates; unfulfilled goals and maintenance debt remain explicit diagnostics.
- In the four full episodes, summed predicted day-end maintenance debt was 14–19 per episode despite fulfillment of every selected new-production goal. Replay inspection includes skipped pre-production care/feed where future yield caps can make today's input unproductive, and plants left within their one-day survival grace. This is not automatically a correctness failure, but optimality of these deferrals is not established. Counterfactual replay checks should distinguish missed valuable obligations from deliberate marginal-value decisions.
- Intraday storage/overflow and resource timing are simulated; future-day storage still uses the economic working-peak approximation. Known town demand and own market orders are simulated against a PASS opponent. Real opponent trades can change prices or purchase feasibility; the controller checks observations and repairs attainable-state divergence, but it does not forecast concurrent orders. Default board/configuration semantics are the tested transition contract, not arbitrary overrides or all malformed actions.
- Individual terminal return-feasibility filters bound task generation; joint trajectories then compete for the actual remaining worker capacity. This improves terminal routing in the maintained scenarios without proving all recoverable money is collected in arbitrary farms. Terminal inventory was zero in the four full episodes, but exhaustive terminal-routing coverage is absent.
- Required non-labor market inputs are protected from sale-list truncation and same-animal purchases are aggregated. If an externally constructed plan exceeds ten required non-labor entries it is rejected; within normal planner bounds, daily hires use only the entry slots left after those inputs.
- The submission wrapper intentionally degrades unexpected internal exceptions to legal pass actions. The health command uses the unguarded decision core and records runtime/trajectory diagnostics and optional replays; it is not yet a failure-resilient acceptance arena that always emits a complete failure manifest.
- The replay reader is a local visual inspection tool: it does not download Kaggle episodes, annotate frames with internal planner decisions, or derive service/overflow/idle-labor diagnostics. Replays produced by a different recorded package/schema are rendered with an explicit identity warning rather than treated as contract-equivalent evidence.
- The candidate has not been frozen or validated as a self-contained `.tar.gz`, and local remote-runtime equivalence remains unestablished.
- Acceptance-set opponents, their licensing/provenance policy, seed count, practical-improvement threshold, non-inferiority tolerances, interval method, and compute budget have not been chosen.
- The competition is active and the official environment may change. There is no update-monitoring or compatibility procedure implemented yet.
- Remote archive/package behavior has only been checked against the documented root-`main.py` contract and local single-file loader; actual Kaggle validation remains untested because no project submission exists.

## Next meaningful work

Characterize and freeze the candidate before any baseline designation:

1. extend held-out execution stress cases and replay counterfactuals for the remaining maintenance debt, dense-farm runtime, overflow logistics, placement neighborhoods and multi-day service feasibility; separate economically sensible deferral from search omissions;
2. implement the thin side-swapped arena, manifest/result schema, replay retention, source/package hashing, and deterministic audit path required by `EVALUATION.md`;
3. build and locally validate a self-contained `.tar.gz`, including its import layout and per-action timing, then freeze an immutable candidate identity;
4. select licensed, hashable development opponents and predeclare the bootstrap acceptance gates; obtain user approval before recording this or a revised candidate as the first accepted baseline;
5. defer competitive promotion claims and Kaggle submission interpretation until the corresponding evidence protocol is in place.
