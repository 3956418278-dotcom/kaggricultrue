# Current State

## Project definition

The project targets Kaggle's full two-player `kaggriculture` environment. Stable competition semantics, submission contract, and conceptual architecture are owned by `PROJECT.md`. Competitive claims and baseline promotion are governed by `EVALUATION.md`.

The competition/reference/environment bootstrap is complete. Competitive strategy remains intentionally unimplemented while the next phase establishes the first immutable baseline and executable evaluation infrastructure.

## Accepted baseline

No project agent or accepted baseline exists yet. The official built-in `starter` agent is a useful environment smoke opponent, not an accepted project baseline.

The first accepted baseline will be a user-approved, immutable measurement anchor after contract, determinism, terminal-completion, packaging, and replay-reproduction checks. That bootstrap designation will not imply competitive strength. Later baseline promotions require acceptance-quality competitive evidence.

## Implemented capabilities

- Repository behavioral core and task-routing harness.
- Stable project, state, and competitive-evaluation authorities.
- General procedures for orientation, structural implementation, debugging, evidence validation, and independent review.
- Kaggriculture-specific procedure for strategy hypotheses, controlled arenas, replay analysis, opponent comparisons, and competitive claims.
- Maintained external-reference index with exact official distribution/source hashes, a dated compact competition snapshot, and versioned public-reference provenance.
- Reproducible Conda specification for Python `3.12.3`, pip `25.2`, and `kaggle-environments==1.32.7`, plus the exact verified transitive Python dependency lock. An ignored local `.venv/` currently realizes that specification.
- Contract verification command and 16 focused tests covering Python and exact dependency-lock conformance, distribution/source and schema identity, explicit seed privacy, private inventory visibility, unit-before-market ordering, lockstep purchases, step-zero town consumption, daily labor/shop refresh, market-entry truncation, atomic planting (including the nonexistent-hand edge case), malformed quantity failure, deterministic replay state, the step-718/719 terminal boundary, final-money rewards, and official loading of a temporary root `main.py`.

The verification passed under Linux x86_64 on 2026-09-04, and `pip check` reported no broken requirements. There is still no maintained project `main.py`, policy implementation, local arena/result manifest, replay analyzer, submission builder, downloaded opponent, or competitive strategy.

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

## Limiting uncertainties

- Kaggle does not publicly expose an immutable deployed evaluator image/package identity or stable numeric RAM, disk, and vCPU limits. Local byte identity with the remote runtime is therefore not established.
- Contract tests cover the highest-risk framework and ordering boundaries but not every crop/animal timing rule, price-curve point, shed-overflow path, invalid-action form, timeout/overage behavior, or `.tar.gz` import layout. Those need targeted tests when their first maintained implementation depends on them.
- The first minimal deterministic project baseline and its immutable packaging form have not been defined.
- Acceptance-set opponents, their licensing/provenance policy, seed count, practical-improvement threshold, non-inferiority tolerances, interval method, and compute budget have not been chosen.
- The competition is active and the official environment may change. There is no update-monitoring or compatibility procedure implemented yet.
- Remote archive/package behavior has only been checked against the documented root-`main.py` contract and local single-file loader; actual Kaggle validation remains untested because no project submission exists.

## Next meaningful work

Complete the remaining evaluation foundation without making a competitive claim:

1. define the minimal deterministic bootstrap agent and its immutable package identity, then obtain user approval before recording it as the first accepted baseline;
2. implement the thin side-swapped match runner, manifest/result schema, replay retention, and deterministic audit path required by `EVALUATION.md`;
3. select or implement licensed, hashable smoke/development opponents and predeclare the first acceptance opponent set and statistical thresholds;
4. extend contract tests only where the baseline or runner relies on currently untested crop, animal, storage, price, archive, or timeout behavior;
5. perform local package validation before any authorized Kaggle submission or leaderboard interpretation.
