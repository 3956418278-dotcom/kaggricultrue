#!/usr/bin/env python3
"""Benchmark the event-route planner on the frozen development player-days."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kaggriculture_agent.route_search import RouteSearchConfig, solve_routes
from src.kaggriculture_eval.player_days import read_samples
from src.kaggriculture_eval.realization_benchmark import (
    compare_sample, episode_partition, source_identity,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episode-limit", type=int, default=2)
    parser.add_argument("--days", type=int, nargs="+", default=[0, 8, 16, 24, 29])
    parser.add_argument("--iterations", type=int, default=1600)
    parser.add_argument("--exact-candidates", type=int, default=18)
    args = parser.parse_args()
    files = sorted(args.samples.glob("episode-*.jsonl.gz"))
    if not files:
        parser.error("no audited episode shards found")
    members = {path.stem.split(".")[0].removeprefix("episode-"): {
        "partition": episode_partition(path.stem.split(".")[0].removeprefix("episode-")),
        "path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
        for path in files}
    chosen = sorted((episode for episode, member in members.items()
                     if member["partition"] == "development"),
                    key=lambda episode: sha256(
                        f"temporal-development-order:{episode}".encode()).hexdigest())[:args.episode_limit]
    config = RouteSearchConfig(iterations=args.iterations,
                               exact_candidates=args.exact_candidates)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "split_rule": "sha256(temporal-reference-split-20260906:<episode>) mod 5; zero held-out",
        "chosen_episodes": chosen, "members": members, "days": args.days,
        "configuration": asdict(config), "evidence": "development-only",
        "candidate": "event-route structural search plus exact transition compilation",
        "sources": source_identity(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    rows = []
    for episode in chosen:
        for sample in read_samples(Path(members[episode]["path"])):
            if sample["day"] not in args.days:
                continue
            result = compare_sample(sample, config, solver=solve_routes)
            target = args.output / (sample["sample_id"].replace(":", "-")+".json")
            target.write_text(json.dumps(result, separators=(",", ":"))+"\n")
            row = {"sample": sample["sample_id"], **result["comparison"],
                   "reference_completion": result["reference"]["completed"],
                   "candidate_completion": result["candidate"]["completed"],
                   "diagnostics": {key: result["candidate"]["diagnostics"].get(key)
                                   for key in ("search_seconds", "search_generated",
                                               "search_exact_evaluations", "search_improved_start")}}
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = {
        "samples": len(rows),
        "matched_or_better": sum(row["completion_difference"] >= 0 for row in rows),
        "mean_completion_difference": (sum(row["completion_difference"] for row in rows)/len(rows)
                                       if rows else None),
        "max_search_seconds": max((row["diagnostics"]["search_seconds"] for row in rows), default=0),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
