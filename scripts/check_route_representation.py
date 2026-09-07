#!/usr/bin/env python3
"""Check event-route compiler expressiveness on development demonstrations."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kaggriculture_agent.route_compiler import compile_skeleton
from src.kaggriculture_eval.player_days import read_samples
from src.kaggriculture_eval.realization_benchmark import episode_partition
from src.kaggriculture_eval.route_reference import demonstrated_skeleton


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episode-limit", type=int, default=2)
    parser.add_argument("--days", type=int, nargs="+", default=[0, 8, 16, 24, 29])
    args = parser.parse_args()
    files = [path for path in sorted(args.samples.glob("episode-*.jsonl.gz"))
             if episode_partition(path.stem.split(".")[0].removeprefix("episode-")) == "development"]
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in files[:args.episode_limit]:
        for sample in read_samples(path):
            if sample["day"] not in args.days: continue
            try:
                problem, skeleton = demonstrated_skeleton(sample)
                result = compile_skeleton(problem, skeleton)
                row = {"sample_id": sample["sample_id"], "goal_count": len(problem.goals)+len(problem.intent.land),
                       "completed": len(result.completed), "unfulfilled": sorted(result.unfulfilled),
                       "failures": result.diagnostics["compile_failures"],
                       "movement": result.diagnostics["movement"], "logistics": result.diagnostics["logistics"]}
            except Exception as error:
                row = {"sample_id": sample["sample_id"], "error": f"{type(error).__name__}: {error}"}
            rows.append(row)
            (args.output/(sample["sample_id"].replace(":", "-")+".json")).write_text(json.dumps(row, indent=2)+"\n")
            print(json.dumps(row), flush=True)
    summary = {"samples": len(rows), "errors": sum("error" in row for row in rows),
               "full_completion": sum(row.get("completed") == row.get("goal_count") for row in rows),
               "missing_goals": sum(row.get("goal_count", 0)-row.get("completed", 0) for row in rows)}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
