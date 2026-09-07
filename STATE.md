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
- The submitted candidate still uses the **legacy restricted neighborhood search**. A separate joint temporal/resource constraint model is now implemented for intraday research; it has not replaced the submitted policy and has not met the reference-quality or runtime standard. Dataset preparation is no longer the priority or a prerequisite to continuing model development.

## Maintained capabilities

- Thin submission entrypoint; contract normalization; immutable owned-state reconstruction; one policy rule/transition owner; structured economic commitments; daily lifecycle; execution tasks; reactive market module; retained trajectories and greedy sanity benchmark.
- Verified ordering and timing: atomic seed requests, ordered unit actions before market, market lockstep, town demand, crop/animal production, fertilizer, shed overflow, daily automatic inventory drop, hand removal/farmer reset, and the 718/719 terminal boundary.
- Earlier correctness fixes remain: valuable final watering before one-time harvest, feed/care-derived animal yield, marginal fertilizer output, recoverable terminal crop/animal yield, and protection of required purchases from market-entry truncation.
- Local official-scene replay viewer and deterministic environment, candidate, transition-parity and replay tests.
- Python 3.12.3 and the exact local dependency lock remain maintained in the project environment. The authenticated `kaggle` Conda environment is separate tooling, not the simulation runtime.

## Plan-to-realization reference collection

The user approved a private CPU-only Kaggle pilot, with checkpoints before planner redesign. No model training is in scope.

Maintained implementation:

- `src/kaggriculture_eval/player_days.py`: shared-field/clock normalization, official full-replay reexecution, semantic unit-effect extraction, day slicing, attempted-versus-achieved effects, placement domains and deterministic compressed episode shards.
- `src/kaggriculture_eval/reference_pipeline.py`: **one cloud run** collects metadata, qualifies sides, reads official Dataset-mounted replays, validates transitions, reconstructs Plans/realizations, checkpoints each episode and audits the result. Collection does not defer reconstruction to another job. Runtime-limited partial outputs are explicit; compatible checkpoints can resume without reexecuting valid shards.
- `src/kaggriculture_eval/reference_audit.py`: streaming checks for source/shard identity, actual qualification, complete day chains, clocks, goal/effect counts, attempts, placement membership and the Plan/execution boundary. These checks do not establish execution optimality.
- `scripts/kaggle_reference_pilot.py`: reusable private CPU notebook preparation, now with explicit source versions, candidate/runtime/query bounds, frozen selection reuse and optional metadata bootstrap. New metadata collection occurs in the cloud. The source allowlist is hashed, notebook size checked against Kaggle's 1 MB limit, and exact extractor/installed package identities retained in output. Usage: `references/player-day-pipeline.md`.
- `scripts/fetch_kaggle_outputs.py`, `scripts/audit_player_days.py` and `scripts/prepare_reference_dataset.py`: safe selected-output retrieval, audit and allowlisted private Dataset staging. Raw replays, credentials and disposable metadata caches are not included in the staged Dataset.
- `scripts/restore_reference_dataset.py` and `reference_storage.py`: lossless restoration of Kaggle-expanded JSONL into original hash-verified gzip checkpoints. This is storage handling, not a second Plan-reconstruction stage. Original expanded extractor files retain separate source-hash checks.
- `scripts/inspect_player_day.py`: HTML inspection with official day-start/day-end scenes, a realization timeline, an economic-effect table and worker/market traces. Opponent public farm state is retained, but its private inventory is not reconstructed or represented as known.

Source and selection evidence:

- Official index `kaggle/kaggriculture-episodes-index`, inspected version 37: 37 daily datasets and 26,212 episodes. Index manifest SHA-256: `78c4d110c8e1f5c1b78654ed6d164dc1ac22f59b2098683d34622fc6356747fe`.
- Source: `kaggle/kaggriculture-episodes-2026-09-04`, **verified version 1**, 668 candidates. Daily manifest SHA-256: `91b2037cdad455022cf90688d593b72cc93c84f767f7ad1be722d3eccd67c1a4`.
- Index/daily manifests have aggregate scores, not player-side ratings. Replays have names but no submission IDs. The public episode service actually returned submission ID, team ID, side index and pre-/post-game score; confidence was not present in the inspected responses. Metadata discovery is not a leaderboard-ranking method.
- Frozen leaderboard snapshot: 2026-09-05 13:44:52 UTC. Qualify each side only if its team is in that snapshot's top ten and its pre-game rating is at least the tenth-place score, **2828.9**. This is a snapshot-qualified high-rating cohort, not proven historical top-ten rank or optimal execution.
- A deterministic hash sample selects 100 episodes before extraction, without selecting winners. Cached metadata joins 94 candidates and qualifies 96 player-sides before replay validation. Six missing joins stay unqualified; the other side is not automatically included.
- The sampled official replay 105620288 reproduced all 719 joint transitions, 1,440 player observations and terminal rewards under the pinned engine. Local illustrative extraction produced 60 player-days; this is a pipeline check, not the final dataset.
- Frame `t+1` contains the action from state `t`; side 1 can omit shared `step`. Days 0–28 have 24 actionable turns; day 29 has 23.

Audited pilot:

- Private notebook: https://www.kaggle.com/code/f7e6n5g4/kaggriculture-player-day-pilot
- Notebook version 1 completed on 2026-09-06. Of 100 candidates, **78 extracted, 22 unqualified, zero quarantined**: **2,880 player-days, 96 qualified sides, six teams and ten submissions**. Six candidates lacked metadata; sixteen had no qualified side.
- All extracted episodes passed 719 official joint transitions, 1,440 observation comparisons and terminal-reward checks. Cloud Python was **3.12.13**, engine 1.32.7, official source SHA-256 `bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`. Summed recorded episode processing was **980.61 seconds**; this is not total notebook wall time.
- Local audit verified every shard hash and all 2,880 samples: **281,737 economic-effect goals**, including **218 unfulfilled attempt-only goals**, no empty Plans and no phantom retry entities in this pilot. Qualification is not a claim that demonstrations are flawless. Raw action/no-op counts are diagnostics, not waste scores.
- Extraction identity: `ae2878626e1b7d157cd9b26a47519daf17573d0aa513b018081125bc388a5323`. Immutable schema-v1 outputs and audit: `.cache/reference-pilot-20260905/cloud-output/player-days/` and `.cache/reference-pilot-20260905/audit.json`.
- Private Dataset https://www.kaggle.com/datasets/f7e6n5g4/kaggriculture-player-day-references: **ID 11917310, version 1, Ready**, with `isPrivate=true` verified through Kaggle's API. The staging set contains 87 allowlisted files (70.3 MB), exact original extractor source and separately identified inspection tools. Creation receipt: `.cache/reference-pilot-20260905/dataset-create-receipt.json`.
- Kaggle automatically expanded gzip shards to `.jsonl` and zipped sources into named directories, reporting **2.03 GB online**. Restore Dataset downloads/mounts with the maintained storage command before checkpoint audit/resume; notebook outputs still use original gzip paths. The pilot's archived inspection tools predate this adapter; use the maintained repository restoration command with v1.

Larger integrated run:

- Private CPU notebook https://www.kaggle.com/code/f7e6n5g4/kaggriculture-player-day-collection, **version 1**, successfully launched; Kaggle reports **RUNNING**. Startup confirmed 2026-09-06; further monitoring stopped as requested. Completion and output counts are not yet known.
- Expands to **all 668 candidates** in the same frozen source, without changing the leaderboard snapshot or threshold. Bootstrap metadata joins 631 episodes and qualifies 674 sides (up to 20,220 player-days before validation); cloud discovery may improve coverage but cannot relax qualification.
- Collection, validation, reconstruction and audit are in the same job, with 24 additional metadata-query maximum, 3-hour extraction budget and 4-hour notebook cap. No GPU or training. Preparation and launch receipt: `.cache/reference-scale-20260906/`.
- The launched notebook is frozen. A storage-restoration wrapper was added to future notebook preparation after observing Dataset archive expansion. Resuming this already-launched version requires restoring its Dataset checkpoint before passing it to the exact archived extractor; do not mix a freshly generated extractor identity with its checkpoints.
- Schema **player-day-v2** fixes a focused, reproducible edge case where repeated input-starved planting attempts became multiple new-asset goals. Retries now share a lifecycle-scoped identity; successful removal resets it. Unit effects also use the recorded shed capacity. The earlier pilot did not exhibit the retry defect and stays immutable; its v1 shards are not reused as v2 checkpoints.
- Generated caches, downloaded outputs and staging directories remain ignored data, not project authority. A larger completed Dataset version will preserve the cloud-produced references; publication is not a separate reconstruction run.

## Current validation

### Intraday model under development

- The first research candidate used **joint temporal constraint optimization**. `references/intraday-model-design.md` records that formulation. It is now rejected as the runtime representation: it remains useful for small correctness cases, but its dense turn/worker/position/resource lattice did not search effectively within the action budget.
- `intent.py` compiles native commitments and reconstructed `STATE_EFFECT` Plans into semantic service requirements and local legal orders using the shared unit transition. `temporal_model.py` represents dated worker positions, service assignment/order, open placements, hiring/spawn, carrying, shared inventories, acquisitions, market-entry capacity, decay and daily refresh. `temporal_session.py` retains trajectories and records acknowledged progress separately from Plan; repairs do not edit production intent. Selling remains in `market.py`.
- The optimizer includes full-space search and joint constraint neighborhoods. A constructive starting witness supplies a feasible incumbent, not restrictions on staffing, assignments, materials or routes. Its labeling is normalized only across genuinely equivalent new assets. Diagnostics explicitly distinguish a retained witness from a solution improved by joint search; merely returning the witness is not evidence of optimizer strength.
- The solver dependency is optional and isolated: OR-Tools 9.14.6206, protobuf 6.31.1 and immutabledict 4.3.1 in `.cache/planner-runtime`, layered over the pinned local game environment. `requirements-planning.txt` owns this overlay. The base `.venv`, main submission imports and accepted-baseline status are unchanged.
- Nineteen focused temporal-model tests pass in the optional solver environment. The canonical suite has 97 tests and passes under the pinned base environment, with the fourteen optional solver tests skipped there as intended. Coverage includes official trajectory parity, primitive-only projection, atomic seeds, shared pickup/feed, hiring availability, build/place ordering, fertilizer/water/harvest timing, decay, normal/terminal inventory handling, insertion-order overflow, land cost/availability, sale-entry truncation, retained execution and equivalent-asset witness labeling.
- `plan_io.py`, `realization_benchmark.py` and `scripts/benchmark_realizations.py` compare the exact same starting state and Plan using official reachable effects, maximum one-to-one semantic matching and resulting states. Candidate inputs exclude demonstrated actions. Both controlled trajectories use an opponent-PASS background with random weeds disabled; the recorded reference is scored separately, and a change in its fulfilled goals is explicitly flagged. No automatic financing adjustment is imposed.
- Episode membership was frozen before quality tuning: **59 development / 19 held-out episodes**, using the named SHA-256 split in `runs/temporal-split-20260906/manifest.json`. Every side and day of an episode stays in its partition. Held-out performance has not been inspected. Initial comparisons use a predetermined development episode; ten player-days from one episode are not ten independent games or broad strength evidence.
- The frozen development run contains fifteen player-days from two predetermined episodes. The temporal candidate matched reference completion on six, but averaged 12.33 fewer completed goals; representative midgame gaps were 62/82, 111/126, and 120/142. Median planning time was 24.93 seconds and maximum 32.41 seconds. Search improved its starting witness on only one of fifteen cases. Raising the deterministic budget to 20 units on two day-8 states took 35--39 seconds and left completion at 62/82 and 72/89. This is decisive rejection evidence for the dense representation, not evidence that constraint or route optimization in general cannot work.
- `references/intraday-formulation-study.md` compares four different representations against all 2,880 pilot player-days. A typical reference has 112 goals over 63 entities and about 12 workers, but compresses to about 70 same-position visit blocks. With demonstrated assignment/order fixed, Manhattan distances explain about 94% of movement on average. Event-route-skeleton search with deterministic primitive compilation is selected as the first engineering hypothesis; macro CP with validation cuts remains a credible peer, while event-driven forward search and route-column generation have recorded trigger conditions.
- Whole-entity jobs and static placement exclusion were rejected during independent design review. Strong replays commonly split one entity across workers, use same-turn ordered effects, and clear then reuse tiles. The selected representation must therefore route atomic service/logistics events under local precedence, permit explicit synchronized worker-order bundles, and model temporal tile leases. Hire-turn movement ties, ordered shed transfers, and capacity-sensitive deposit order also remain explicit when consequential.

### Maintained submission and collection checks

Before the model extension, all **78 maintained tests passed** under the pinned local environment. Coverage includes the fixed ownership boundary, official replay round-trip, final-day slicing, semantic removal of staffing/logistics/sales, atomic failed planting attempts, per-side qualification, deterministic shards, retry identity, audit corruption rejection, one-run collection/reconstruction, compatible resume, incompatible source rejection, partial runtime checkpoints and metadata-denial behavior. The earlier full-episode results below concern the maintained submission, not the new research optimizer.

Four full candidate-versus-starter episodes (seeds 17/29, both seats) completed 720 states with both players DONE and no terminal sellable inventory. Repeated-observation decisions were deterministic. Each episode used 30 daily searches without intraday repair. Maximum local decision times were 1.81–2.07 seconds; measured overage totals were 5.27–7.54 seconds, below the 60-second reserve locally. Reports, source hashes and complete replays are in `runs/boundary-20260905/`. These are integration results, not competitive evidence or proof of remote runtime equivalence.

The older handcrafted execution scenarios remain sanity checks only. Their historical score table is superseded as the maturity standard by the same-state/same-Plan reference benchmark. The initial development run is diagnostic rejection evidence for the temporal formulation, not acceptance evidence for another planner.

## Remaining limitations and decisions

- The semantic player-day schema is **exploratory**, not an accepted benchmark. It captures direct tile effects, land unlocks and their physical input/output deltas. Failed placement retries are now coalesced, but standalone stock-acquisition objectives, ambiguous no-ops, broader entity lifecycles, placement dependencies and alternative-order effect equivalence still need review before defining the planner evaluator. Failed input-dependent actions remain distinct from achieved work.
- Reconstructed `STATE_EFFECT` commitments remain unsupported by the legacy search/task generator but are supported by the new semantic compiler. Development comparisons now exist; they expose remaining deficiencies, not planner maturity.
- The rejected joint temporal encoding is expensive. Broad feasible domains and larger budgets did not yield useful improvement. It remains outside `main.py`; the maintained submission still uses the legacy scheduler.
- The selected event-route-skeleton formulation has not yet been implemented or benchmarked. Its main risks are search coverage, synchronized ordered-unit effects, newly produced material shared between routes, temporal tile reuse, and ensuring that deterministic compilation reports rather than hides structural infeasibility.
- State evaluation gives fulfillment precedence, then uses quoted inventory and surviving-asset option values with a simple future-service-distance estimate. Those are not exact long-horizon terminal-cash predictions. Native duplicate physical obligations share an execution acknowledgement; reference effect matching is exploratory and must not be mistaken for hidden-intent inference.
- Daily admission and multi-day asset value retain coarse labor/travel/storage assumptions and optimistic future maintenance. They are not proofs of executable terminal profit.
- Selling remains separate. Sale-financing effects have not been counterfactually measured; flag/control them only when they materially change achievable realization, not merely because hiring occurs.
- Pilot cloud Python and engine source identity are verified; the entire image was not pinned. The larger run records installed distribution versions as additional provenance. Local dependency identity does not establish byte-identical Kaggle infrastructure.
- Tooling environment Kaggle CLI 2.2.3 failed to read its upload-resume cache (`KaggleObject.from_dict` signature mismatch) following a transient SSL transfer failure, so the successful Dataset retry retransferred files. This is separate from the tested extraction checkpoint mechanism; large future uploads may need a tooling compatibility fix.
- Acceptance arena design, packaged submission validation, immutable baseline designation and competitive promotion remain separate unfinished work.

## Next meaningful work

Implement the complete event-route-skeleton plus deterministic trajectory-compiler model defined in `references/intraday-formulation-study.md`. First establish compiler expressiveness on development demonstrations with movement stripped; then search from `S_start + fixed Plan` only. Keep event assignment/order, synchronized bundles, batching, resource links, staffing, and temporal placement leases jointly mutable. Do not benchmark a deliberately incomplete version, and keep held-out episodes untouched until the model, budget, and scoring protocol are frozen.

The larger private collection was successfully started and is not being monitored, as requested. Its completion/publication can be handled at a separately requested checkpoint; it is not blocking planner development on the available audited pilot.
