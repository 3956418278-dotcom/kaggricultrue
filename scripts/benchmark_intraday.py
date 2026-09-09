#!/usr/bin/env python3
"""Run the fixed nine-row development benchmark through solve_intraday()."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kaggriculture_eval.player_days import read_samples
from src.kaggriculture_eval.realization_benchmark import (
    compare_intraday_sample,
    episode_partition,
    source_identity,
)


FROZEN_EPISODES = ("105527696", "105448362")
FROZEN_DAYS = (8, 16, 24)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    members = {}
    for episode in FROZEN_EPISODES:
        path = args.samples / f"episode-{episode}.jsonl.gz"
        if not path.exists():
            parser.error(f"missing frozen development shard: {path}")
        if episode_partition(episode) != "development":
            parser.error(f"frozen episode unexpectedly resolves outside development: {episode}")
        members[episode] = {
            "path": str(path),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "episodes": list(FROZEN_EPISODES),
        "days": list(FROZEN_DAYS),
        "partition": "development-only",
        "held_out_touched": False,
        "members": members,
        "solver_api": "OwnedState + Plan -> solve_intraday -> Realization -> execute",
        "sources": source_identity(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    for episode in FROZEN_EPISODES:
        for sample in read_samples(Path(members[episode]["path"])):
            if sample["day"] not in FROZEN_DAYS:
                continue
            row = compare_intraday_sample(sample)
            rows.append(row)
            target = args.output / f"{sample['sample_id'].replace(':', '-')}.json"
            target.write_text(json.dumps(row, separators=(",", ":")) + "\n")
            print(json.dumps(row), flush=True)
    if len(rows) != 9:
        raise RuntimeError(f"frozen benchmark expected 9 rows, found {len(rows)}")
    successes = [row for row in rows if row["candidate"]["status"] == "complete"]
    summary = {
        "rows": len(rows),
        "full_plan_success": len(successes),
        "planning_failure": len(rows) - len(successes),
        "same_goal_set_rows": sum(row["candidate"]["same_goal_set"] for row in rows),
        "candidate_end_value": [row["candidate"]["end_value"] for row in rows],
        "reference_end_value": [row["reference"]["end_value"] for row in rows],
        "candidate_workforce": [row["candidate"]["workforce"] for row in rows],
        "candidate_hire_expenditure": [row["candidate"]["hire_expenditure"] for row in rows],
        "runtime_seconds": [row["candidate"]["runtime_seconds"] for row in rows],
        "total_runtime_seconds": sum(row["candidate"]["runtime_seconds"] for row in rows),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
