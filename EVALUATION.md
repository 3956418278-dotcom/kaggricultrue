# Kaggriculture Competitive Evaluation

## Purpose and authority

This document defines when match results support a competitive claim or the promotion of a new accepted baseline. It owns the evaluation protocol; `STATE.md` records which evidence and baseline have actually been accepted.

The executable local reference is full `kaggriculture` schema `0.1.0` from `kaggle-environments==1.32.7`, matching official source commit `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`. Its wheel and key-file hashes are in `references/official/kaggriculture-environment-1.32.7.json`. Evidence must use an executable environment pinned more precisely than the environment name or schema version alone.

## Evidence identity

Every retained evaluation must identify:

- installed `kaggle-environments` version plus an environment source commit or content hash;
- complete resolved environment configuration, including `episodeSteps`, market overrides, and the explicit environment seed;
- candidate and accepted-baseline source revision or content hash;
- packaged-entrypoint hash when the packaged form is under test;
- every opponent's stable name, exact content hash, provenance, and loading method;
- arena implementation revision, command/configuration, start and finish time, platform, and Python version;
- one row per game with seed, candidate seat, statuses, final money, outcome, replay path/hash, and any error, timeout, or invalid-result flag.

Changing any of these creates a new evidence set. Results from incompatible environment identities or configurations are not pooled silently.

The repository's contract verification proves behavior only for the pinned local distribution. The public competition does not expose an immutable deployed evaluator package or container identity, so remote equivalence remains bounded by the observed submission contract and replay behavior rather than asserted as byte identity.

## Seed and replication semantics

Set `configuration.seed` explicitly for every controlled game and verify that the resolved value recorded in `env.info["seed"]` and the replay matches the request. Environment-generated seeds are acceptable for casual smoke tests only.

For deterministic policies, rerunning the same agents, seats, seed, environment, and configuration must reproduce the result and replay-relevant state. Any policy randomness must have a separate explicit seed and be included in the design and manifest.

The environment uses its seed for weeds and town unlocks, but random draws depend on the evolving farm state. A common environment seed makes each trajectory reproducible; it does not guarantee the same realized exogenous events under different policies. Retain realized shops and replay state when diagnosing paired differences.

The analysis block is an opponent-and-environment-seed pairing with both candidate seats. The two side-swapped games in a block are paired observations, not two independent samples. Reusing a seed across opponents does not turn those opponents into interchangeable replicates.

The built-in `random` agent constructs its own unseeded random generator in the inspected official implementation. Matches against it are smoke or exploratory evidence unless that opponent is replaced by a version with controlled randomness.

## Side-swapped evaluation

For every controlled seed and opponent, run both:

```text
candidate as player 0 vs opponent as player 1
opponent as player 0 vs candidate as player 1
```

Use the same environment identity and configuration for the pair. Record candidate money and margin from the candidate's perspective in both games. Report seat-stratified results as well as the paired aggregate; do not average away a material seat interaction.

Any incomplete pair is a reliability failure for acceptance evidence. It may be retained for diagnosis but is excluded only by a rule declared before looking at competitive outcomes.

## Accepted baseline and opponent sets

The accepted baseline is an immutable, runnable agent identity recorded in `STATE.md`. A candidate is always compared directly with that exact baseline in a side-swapped, held-out arena. When the baseline is promoted, preserve its source or package hash and the evidence that justified promotion.

The first accepted baseline is a bootstrap anchor, not a claim of competitive strength. It may be designated after it passes the pinned-environment contract, determinism, terminal-completion, packaging, and replay-reproduction gates and after the user approves its role. Record its exact identity and limitations in `STATE.md`. Every later competitive promotion uses the comparison rules in this document.

Opponent sets have explicit roles:

- **Smoke opponents** check execution and obvious contract failures. `pass`, `starter`, and controlled toy agents may serve this role; performance against them is not broad competitive evidence.
- **Development opponents** expose weaknesses and support iteration. Once results influence a candidate, those matches are exploratory for that candidate.
- **Acceptance opponents** are versioned, hashed, representative policies whose seeds, weights, and aggregation are fixed before the acceptance run. They must include the accepted baseline directly; additional opponents measure breadth, not a substitute for the baseline comparison.
- **Kaggle opponents** are the unknown and changing population encountered by submitted agents. Their results are external evidence and are never assumed equivalent to the local acceptance set.

Report every opponent separately. A pooled summary is allowed only when the opponent list and weights were declared in advance; it must not conceal a material regression against an individual opponent class.

## Required statistics

For each opponent, seat, and prespecified aggregate, report:

- games, complete side-swapped blocks, wins, draws, and losses;
- win rate and match score, where a win is `1`, draw `0.5`, and loss `0`;
- candidate and opponent final-money mean, median, standard deviation, interquartile range, minimum, and maximum;
- final-money margin (`candidate - opponent`) mean, median, standard deviation, interquartile range, minimum, and maximum;
- a 95% uncertainty interval for the primary match-score and margin summaries, resampled or calculated at the side-swapped block level;
- player-0 and player-1 breakdowns and the paired average margin per block;
- non-`DONE` statuses, timeouts, exceptions, malformed outputs, and reproducibility failures.

Money distributions can be skewed and price-floor effects can create large tails, so means and win rates must not stand alone. Statistical units and intervals must respect paired seats and any opponent stratification. Do not describe overlapping intervals as proof of equality or a favorable point estimate as established superiority.

## Evidence classes

### Illustrative single-game inspection

A single game or selected replay can demonstrate an action sequence, reproduce a bug, explain a mechanism, or provide a concrete example. It cannot establish expected performance, robustness, a win-rate claim, or a strategy improvement.

### Exploratory evaluation

Exploration includes smoke matches, small or convenient seed sets, adaptive opponent selection, repeated tuning against observed results, unseeded opponents, partial seat coverage, and replay-driven strategy development. It may identify hypotheses, estimate variability, or reject a clearly broken candidate. Report it as observed behavior under named conditions, not as accepted competitive evidence.

### Acceptance-quality competitive evidence

Before running an acceptance arena, freeze and record:

- the candidate and accepted-baseline hashes;
- environment identity and full configuration;
- deterministic seed list and both-seat pairing;
- opponent identities and any aggregate weights;
- primary competitive claim, practical effect threshold, non-inferiority tolerances, uncertainty method, and promotion rule;
- reliability gates and any exclusion rule;
- replay and artifact retention location.

Acceptance requires all planned games and seat pairs to finish validly and reproduce on a declared audit subset. The primary baseline comparison must meet its prespecified superiority or practical-improvement rule, while prespecified opponent-suite checks must meet their non-inferiority rules. Any multiple opponent or metric claims must be handled as a declared family rather than selected after inspection.

The bootstrap baseline may be designated under the non-competitive gates above without claiming superiority. Numerical effect thresholds, non-inferiority tolerances, seed count, and power/precision targets govern the first candidate seeking promotion over that anchor; they remain undecided in `STATE.md`. Until they are fixed before that acceptance run, candidate results may be strong exploration but cannot justify a competitive baseline promotion.

Do not relabel a development arena as acceptance evidence after seeing its outcome. A candidate that changes in response to acceptance results requires a fresh candidate identity and fresh held-out evidence.

## Replay and artifact retention

Initially retain the complete replay for every acceptance-quality game, together with the immutable row-level result table and manifest. Retain logs for all abnormal games and enough stdout/stderr to diagnose loader or runtime failures. Store generated evidence under a unique ignored directory such as `runs/evaluation/<evaluation-id>/`; do not put it in `STATE.md` or treat it as source code.

An acceptance manifest must hash the candidate, baseline, opponents, configuration, result table, and replays. Completed evidence is immutable. Re-analysis writes a new derived artifact that names the source manifest. If storage pressure later requires selective replay retention, revise this contract explicitly before the affected run and preserve at least every failure, every draw, deterministic audit cases, and representative boundary outcomes.

Replay analysis is diagnostic. It may explain why a measured difference occurred or motivate a new hypothesis; replay selection alone does not increase the strength of the competitive estimate.

## Local evidence and Kaggle results

Local evidence establishes behavior only for the pinned local environment, declared seeds, and declared opponents. It does not by itself predict the Kaggle score, rank, or robustness to the leaderboard population.

Kaggle submission evidence must record the submission ID, uploaded artifact hash, remote status, environment information if exposed, episode IDs, seats, final rewards, timestamps, and downloaded replay/log hashes. Distinguish:

- local package validation;
- remote submission acceptance;
- completed Kaggle episodes;
- a timestamped public leaderboard snapshot;
- the post-deadline final tournament standing.

Leaderboard results are valuable external validation but are observational: opponent assignment, seeds, environment deployment, and population may be hidden or change over time. Use them to challenge local conclusions and opponent coverage. Do not tune repeatedly to a public score and then cite that same score as independent confirmation.

Under the official evaluation page inspected on 2026-09-04, an upload first runs a self-play validation episode. Valid submissions then play similarly rated agents; only the latest two submissions are tracked for scoring, and the higher-scoring one is displayed. The episode outcome is win/draw/loss from final coins after the 720-turn game, not the size of the money margin. After the deadline, submissions lock and a final Bradley-Terry tournament is computed from subsequent episodes. These remote ratings are population-dependent and are not interchangeable with the fixed local acceptance protocol.

## Intraday realization references

Player-day research is a separate development benchmark, not competitive baseline-promotion evidence. Its interface is `identical day-start state + identical fixed reconstructed Plan -> reference / candidate realizations`. The target is Plan fulfillment and the state reached, not imitation of actions or final-game strategy. Keep full episodes as legality/runtime/integration checks and greedy scheduling as a weak sanity baseline.

- Use the official episodes index only as a candidate pool. Qualify each player-side separately using metadata that was actually obtained. Freeze the leaderboard snapshot, pre-game score cutoff, candidate sampling rule and source versions before extracting or inspecting outcomes. Missing or ambiguous side joins remain unqualified.
- The approved initial cohort uses membership in a frozen top-10 team snapshot AND that particular side's pre-game rating at least the snapshot's tenth-place score. Call it a snapshot-qualified high-rating cohort, not historical top-10 rank or proof of optimal execution. Do not select winners or use post-game score to qualify a side.
- Normalize replay frame/shared fields explicitly. Frame `t+1` carries the action from state `t`; the final day has 23 actionable turns. Reexecute both players through the pinned official interpreter with the recorded seed and configuration; quarantine incompatible or divergent episodes.
- Reconstruct economic work from direct action/state effects. Preserve existing-location constraints and meaningful construction/land dependencies, while leaving equivalent new placements open. Keep worker identities, staffing, paths, pickup organization, order and selling in the demonstrated realization, not Plan. Distinguish successful effects, demonstrably failed input-dependent attempts and unresolved no-ops. Do not infer hidden intent from a route or pass.
- Compare Plan completion first; diagnose materials, coordination, timing, resulting assets and labor capacity. Movement, pickup, drop and placement are not waste by definition. Efficiency is relative to another feasible realization achieving the same or better outcome.
- Keep selling separate. Flag and control sale-financing only if it materially changes executable work or purchases; do not impose a financing correction merely because a replay includes hiring and sales.
- Treat the pilot as exploratory. Before planner tuning, freeze development/held-out episode groups (never split days of one episode across them), report team/submission overlap and duplicate behavior, and choose the completion/efficiency comparison from inspected samples. Keep held-out results untouched during tuning. No claim of planner maturity is established by a pilot extraction alone.
- Keep code, schemas, tests and inspection commands maintained locally. Extract large samples in private CPU Kaggle runs. Publish versioned private Kaggle Datasets with selection metadata, source hashes, per-episode checkpoints, exclusions and exact extractor identity. Resume only hash-compatible checkpoints; keep generated local copies ignored.

## Claim language

Claims must name their evidence boundary:

- “reproduced in one game” for illustrative inspection;
- “improved in the development arena against X on seeds Y” for exploratory comparisons;
- “met the prespecified local promotion rule against baseline B and acceptance set O” only for complete acceptance-quality evidence;
- “achieved Kaggle public score S at timestamp T” for a remote snapshot, without implying final rank or local causal attribution.

Use “not established” when coverage or precision is insufficient. Use “regressed” or “inconsistent with the claim” only when the design could detect the relevant failure.
