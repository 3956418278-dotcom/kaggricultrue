# Current State

## Project definition

The project targets Kaggle's full two-player `kaggriculture` environment. Stable competition semantics, submission contract, and conceptual architecture are owned by `PROJECT.md`. Competitive claims and baseline promotion are governed by `EVALUATION.md`.

The current phase is repository and evaluation-contract initialization. Competitive strategy implementation is out of scope until the official environment can be run and checked locally through a reproducible harness.

## Accepted baseline

No project agent or accepted baseline exists yet. The official built-in `starter` agent is a useful environment smoke opponent, not an accepted project baseline.

The first accepted baseline will be a user-approved, immutable measurement anchor after contract, determinism, terminal-completion, packaging, and replay-reproduction checks. That bootstrap designation will not imply competitive strength. Later baseline promotions require acceptance-quality competitive evidence.

## Implemented capabilities

- Repository behavioral core and task-routing harness.
- Stable project, state, and competitive-evaluation authorities.
- General procedures for orientation, structural implementation, debugging, evidence validation, and independent review.
- Kaggriculture-specific procedure for strategy hypotheses, controlled arenas, replay analysis, opponent comparisons, and competitive claims.

There is no agent entrypoint, local arena, replay analyzer, packaging path, test suite, environment pin, or strategy implementation yet.

## Accepted competitive evidence

None. No local match result, Kaggle episode, leaderboard score, or baseline-promotion result has been accepted for this repository.

## Current decisions

- Use the full `kaggriculture` environment, not `kaggriculture_beginner`.
- The initial external contract reference is schema version `0.1.0` at official upstream commit `bbda347572cf5134e56f0eb49e8058e2560f9844`, inspected 2026-09-04.
- Treat the executable pinned environment as the behavioral authority and re-audit contract changes before pooling evidence across versions.
- Establish deterministic local execution, exact contract tests, side-swapped evaluation, and replay retention before competitive strategy work.
- Prefer deterministic or planning-based baselines initially. Learning is justified only by a measured limitation and an evaluation design capable of testing the proposed improvement.
- Keep exploratory, acceptance-quality, and illustrative evidence distinct. Strategy changes are empirical claims, not accepted improvements by inspection.
- Keep agent mechanisms, evaluation infrastructure, durable state, and generated data in separate ownership areas as defined by `AGENTS.md` and `PROJECT.md`.

## Limiting uncertainties

- No executable `kaggle-environments` package/version is pinned locally, and equivalence between a future local package and Kaggle's deployed evaluator has not been established.
- The default 720-state terminal/action boundary observed in current source needs an executable contract test, along with same-turn inventory, market-order, end-of-day, privacy, timeout, and invalid-action behavior.
- The first minimal deterministic project baseline and its immutable packaging form have not been defined.
- Acceptance-set opponents, their licensing/provenance policy, seed count, practical-improvement threshold, non-inferiority tolerances, interval method, and compute budget have not been chosen.
- The exact Kaggle runtime dependency allowance, archive constraints beyond root `main.py`, and deployed environment identity need verification before packaging decisions.
- The competition is active and the official environment may change. There is no update-monitoring or compatibility procedure implemented yet.
- This directory is not currently initialized as a Git worktree, so revision identity and normal diff-based provenance are unavailable until repository initialization is decided.

## Next meaningful work

Create the minimal environment/evaluation contract layer without choosing a competitive strategy:

1. initialize version control and pin a reproducible Python plus `kaggle-environments` environment after confirming the desired dependency workflow;
2. build a thin local match runner that records the full identity and replay contract from `EVALUATION.md`;
3. add focused executable tests for observation privacy, actions, deterministic seeds, side swapping, terminal timing, market lockstep, and replay reproducibility;
4. verify a packaged no-op or official-example agent against `pass` and deterministic `starter` as contract smoke tests;
5. use the resulting timing and variability evidence to predeclare the first baseline and promotion thresholds before strategy implementation.

Dependency installation, Kaggle authentication, submission, and any other external write require explicit approval at the point of action.
