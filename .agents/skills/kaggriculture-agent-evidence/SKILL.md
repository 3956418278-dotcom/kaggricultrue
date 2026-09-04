---
name: kaggriculture-agent-evidence
description: Design, run, analyze, visualize, or review Kaggriculture strategy hypotheses, agent comparisons, controlled arenas, replays, opponent sets, and competitive-performance claims under the repository's evaluation contract. Use for competitive evidence, not ordinary repository engineering.
---

# Kaggriculture agent evidence

Build evidence for a competitive decision without weakening the repository's environment, comparison, or claim standards.

## Start from the competition contract

Read the relevant parts of `PROJECT.md`, `STATE.md`, and `EVALUATION.md` before planning or interpreting agent work.

Treat the official environment identity, observation/action contract, game semantics, visibility boundary, and architectural ownership in `PROJECT.md` as fixed unless the user approves a contract change. Treat the accepted baseline, capabilities, current decisions, and limiting uncertainties in `STATE.md` as current. Treat comparison, seed, side-swap, statistics, retention, and claim rules in `EVALUATION.md` as authoritative.

Code, historical branches, public notebooks, old replays, leaderboard anecdotes, and opponent names do not override these documents. Verify external-agent provenance and hashes before comparison.

## Frame the hypothesis and evidence class

State the strategy hypothesis, affected mechanism, expected observable effect, relevant opponents or game states, and the decision the result can support. Classify the work before matches are run:

- **Exploratory:** smoke testing, debugging, reconnaissance, variance estimation, opponent discovery, adaptive iteration, or evaluation-design selection. It can motivate or reject ideas but cannot promote a competitive claim.
- **Acceptance-quality:** a frozen candidate evaluated against the exact accepted baseline and prespecified opponent/seed/seat design under the promotion rule in `EVALUATION.md`.
- **Illustrative:** one selected game, replay segment, state calculation, or action trace used to explain a mechanism without estimating expected performance.

Do not relabel exploratory matches after seeing them. Do not present a selected win, striking money margin, or persuasive replay as evidence of average strength.

## Design controlled arenas

- Identify candidate, baseline, environment, arena, configuration, and every opponent by immutable revision or content hash.
- Use explicit deterministic environment seeds. If an agent is stochastic, control and record its policy seed separately.
- Run every seed/opponent block with the candidate in both seats. Preserve pairing in analysis; do not count the two seat swaps as independent replicates.
- Compare the candidate directly against the accepted baseline. Use additional opponents to assess breadth and failure modes, not to replace the baseline contest.
- Freeze acceptance opponents, weights, seed set, metrics, practical effect threshold, non-inferiority tolerances, exclusion rules, and uncertainty method before the acceptance run.
- Keep conditions compatible across comparisons unless the condition change is itself the hypothesis. Never pool incompatible environment versions or configurations silently.
- Recognize that equal environment seeds reproduce trajectories but do not force identical weeds or shop unlocks across behaviorally different agents, because random-stream consumption depends on farm state.
- Treat timeouts, exceptions, malformed actions, non-`DONE` statuses, and missing seat pairs as reliability results, not inconvenient samples to discard.

Choose the number of side-swapped blocks from the prespecified precision or decision requirement. More convenient matches do not repair opponent-set mismatch, post-hoc selection, uncontrolled randomness, or a weak claim definition.

## Analyze outcomes and replays

Report wins/draws/losses, match score, both players' final money, candidate money margin, distribution summaries, block-level uncertainty, and seat-stratified results as required by `EVALUATION.md`. Preserve opponent strata and paired-seat structure.

Inspect replays to diagnose mechanisms, legality failures, resource starvation, routing failures, terminal inventory, market interactions, or reactions to stochastic events. Prefer replay sets chosen by a declared rule: every failure, worst paired margins, deterministic audit cases, or stratified representative games.

Replay analysis explains measured outcomes and generates hypotheses. It does not turn one game into competitive evidence. If a replay reveals a defect and the candidate changes, assign a new candidate identity and rerun the appropriate comparison.

## Preserve reproducible evidence

For acceptance-quality work, retain the manifest, row-level results, complete replays, configuration, hashes, logs required for failures, analysis code identity, and derived report under the ignored evaluation-data area defined in `EVALUATION.md`. Completed evidence is immutable; re-analysis names the source manifest and writes a new artifact.

Keep raw match evidence, derived statistics, diagnostic replay notes, presentation, and maintained conclusions distinct. Promote only accepted outcomes, decisions, and material limitations into `STATE.md`; do not turn it into a match diary.

## Match claim wording to evidence

Use wording no stronger than the collected design:

- **Illustrative:** “In this replay, the agent sold before the opponent and received a higher unit price.”
- **Exploratory:** “Across this named development arena, the candidate had a higher observed match score; the result was used for iteration and is not promotion evidence.”
- **Acceptance-quality:** “The frozen candidate met the prespecified local promotion rule against baseline B and acceptance set O under environment E, seeds S, and both seats.”
- **Leaderboard:** “Submission artifact A had public score S at timestamp T,” without implying final rank, causal attribution, or equivalence to the local arena.

Distinguish:

- **established under the declared arena:** directly supported with the required coverage and uncertainty;
- **inconsistent with a claim:** challenged by evidence capable of detecting the claimed effect;
- **not established:** insufficient, incompatible, selected, or overly narrow evidence.

A favorable point estimate is not established superiority. Failure to demonstrate improvement is not proof of equivalence. Local strength against a fixed set is not general strength against the Kaggle population.

Before accepting a consequential baseline promotion or broad competitive conclusion, use `independent-review` when available to challenge environment identity, opponent provenance, arena implementation, raw rows, replays, statistics, exclusion handling, and claim wording. The main project context remains responsible for the final decision and for updating `STATE.md`.
