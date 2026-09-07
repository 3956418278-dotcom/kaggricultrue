# Player-day collection and reconstruction

This is CPU reference-data preparation, not model training or a strategy-learning
dataset. `EVALUATION.md` owns qualification and claim rules; `STATE.md` owns actual
run status. Selling is outside the reconstructed Daily Plan.

## One cloud run

`scripts/kaggle_reference_pilot.py` prepares either a small pilot or a larger run.
Despite its historical filename, it does not impose a 100-episode cap. Preparation
bundles allowlisted code, frozen source/selection identities and optional cached
metadata. It does **not** collect new episode metadata or reconstruct replays locally.

The generated private CPU Kaggle notebook runs `run_collection`:

1. Collect additional public episode metadata, reusing the supplied cache. Discovery
   is bounded by the requested query count, caches responses, and stops on an access
   denial, rate limit or transport failure without bypassing it.
2. Join the attached official daily Dataset manifest to the frozen indexed source.
   Qualify player-sides, not entire episodes. Missing metadata stays unqualified.
3. Open each qualifying official replay directly from the Kaggle Dataset mount.
   Check source size/ID, side reward joins, configuration, seed and official engine
   identity, then reexecute all 719 joint transitions and compare 1,440 observations.
4. Reconstruct all 30 player-days for each qualifying side immediately, audit the
   effect-to-Plan correspondence and write a deterministic compressed episode shard.
5. Checkpoint after each episode. At the runtime budget, stop between episodes and
   save an explicitly incomplete but resumable collection.
6. Audit the collected shards and emit Dataset-ready references, provenance, exact
   extractor source, frozen manifests and installed Python package versions.

Raw replays are not copied into the final Dataset. Collection and reconstruction
are not separate jobs. Downloading/auditing completed output and uploading a Dataset
version are artifact management, not a later reconstruction step.

## Prepare and launch

Use the authenticated `kaggle` Conda environment. A future run should supply its own
verified manifest paths/versions and explicit candidate limit. Reuse the frozen
leaderboard only when that is the intended cohort; it is not historical rank.

```bash
conda run --no-capture-output -n kaggle python scripts/kaggle_reference_pilot.py \
  --output .cache/reference-scale-20260906 \
  --daily-manifest .cache/reference-intake-20260905/day-2026-09-04/manifest.csv \
  --index-manifest .cache/reference-intake-20260905/index/manifest.csv \
  --selection-source .cache/reference-pilot-20260905/selection.json \
  --metadata-cache .cache/reference-pilot-20260905/metadata/joined-metadata.json \
  --daily-version 1 --candidate-limit 668 --metadata-queries 24 \
  --kernel-slug kaggriculture-player-day-collection --extraction-seconds 10800
conda run -n kaggle kaggle kernels push -p .cache/reference-scale-20260906 -t 14400
```

Preparation performs no upload. Notebook metadata is private, GPU-disabled and
Internet-enabled for bounded metadata collection and the pinned engine installation.
The 4-hour notebook cap reserves time beyond the 3-hour extraction budget for audit
and output finalization. Confirm startup; do not continuously monitor unless asked.

To resume, attach the prior private checkpoint Dataset with `--resume-source
OWNER/DATASET`. Keep the exact extractor, selection, source hashes, bootstrap metadata
and run configuration. A changed identity is rejected rather than mixed. Valid shard
hashes skip reexecution; a damaged shard is recomputed. Schema-v1 pilot artifacts
are **not** compatible checkpoints for the schema-v2 collection.

## Retrieve, inspect, and version

**Kaggle Dataset ingestion expands archives automatically**, independently of
`--keep-tabular`: `.jsonl.gz` becomes `.jsonl`, and source `.zip` files become named
directories. The online pilot is therefore about 2.03 GB expanded versus 70.3 MB
staged. Notebook outputs retain their original compressed files.

When using a downloaded or mounted Dataset rather than notebook outputs, first run:

```bash
.venv/bin/python scripts/restore_reference_dataset.py .cache/downloaded-dataset \
  --output .cache/restored-player-days
```

This streams the expanded JSONL back into the original deterministic gzip encoding
and requires the original shard SHA-256 to match. It verifies expanded extractor
source files separately; it does not claim to recover original zip container bytes.
It does **not** reconstruct Plans, qualify players or rerun games. Use the restored
directory for the existing auditor/reader and when passing a checkpoint to
`run_collection`. Future generated resume notebooks perform this restoration before
loading a Dataset checkpoint. The already-launched collection notebook v1 predates
that storage adapter; resuming its frozen extractor requires the same restoration
step in the notebook wrapper, without changing its frozen selection/extractor identity.

`scripts/fetch_kaggle_outputs.py` retrieves selected outputs without printing signed
download URLs. It verifies known shard hashes and bounds transient transfer retries.
Generated files belong under ignored `.cache/`, `runs/` or `replays/` directories.

```bash
conda run --no-capture-output -n kaggle python scripts/fetch_kaggle_outputs.py \
  f7e6n5g4/kaggriculture-player-day-collection --output .cache/collection-output
.venv/bin/python scripts/audit_player_days.py .cache/collection-output/player-days
.venv/bin/python scripts/inspect_player_day.py \
  .cache/collection-output/player-days/episode-EPISODE.jsonl.gz \
  --sample EPISODE:SIDE:DAY --output replays/player-day.html
```

The HTML reader shows official game scenes for start, trajectory and end, alongside
the reconstructed Plan and demonstrated actions. Opponent private inventory remains
unknown; it must not be used as zero inventory in planner comparisons.

`scripts/prepare_reference_dataset.py` audits and stages an allowlisted Dataset
directory. It validates the original extractor hashes and preserves separately hashed
current inspection tools. For the initial pilot only, `--extractor` supplies downloaded
source because that older notebook did not yet archive it in the data directory.
Use a new empty staging directory, not a directory containing raw downloads or caches.
Upload privately with CSV conversion disabled. Subsequent versions must remain private;
never use the CLI's public flag. Dataset `Other` licensing does not relicense included
repository code; underlying official replays retain CC0-1.0 provenance.

```bash
.venv/bin/python scripts/prepare_reference_dataset.py .cache/collection-output/player-days \
  --output .cache/reference-dataset-v2 \
  --dataset f7e6n5g4/kaggriculture-player-day-references
conda run -n kaggle kaggle datasets version -p .cache/reference-dataset-v2 \
  --keep-tabular -m "Integrated collection and reconstruction; see extraction manifest"
```

The version command preserves earlier versions; do not pass `--delete-old-versions`.
For the first creation only, use `datasets create -p STAGING --keep-tabular` (private
by default). Check the existing Dataset's privacy before appending a later version.
Kaggle CLI 2.2.3 exhibited a resume-cache deserialization bug during the pilot upload;
a retry retransferred files. This does not invalidate extraction checkpoints, but
upload-time expectations must not assume that this CLI version resumes correctly.

## Evidence and schema limits

The audit checks hashes, side qualification, complete day chains, clocks, semantic
goal/effect agreement, observed counts, attempt flags and placement membership.
It is not a proof of optimal execution or a finished Plan-completion evaluator.
Action-type counts diagnose workload; they do not classify movement or logistics as
waste. Failed input-dependent attempts remain explicit goals with no demonstrated
completion; unresolved no-ops do not become guessed hidden intentions.

Schema `player-day-v2` coalesces repeated input-starved placement attempts into one
logical asset and uses recorded shed capacity in unit transitions. The pilot remains
immutable schema v1; its audited samples did not exhibit the retry-identity defect.
Standalone stock accumulation, alternative placement prerequisites and semantic
equivalence of differently ordered effects remain benchmark-evaluator review items.
Do not start planner redesign or describe extraction alone as planner maturity.
