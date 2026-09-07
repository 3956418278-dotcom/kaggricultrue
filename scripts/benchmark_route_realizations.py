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
    compare_route_sample, episode_partition, source_identity,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episode-limit", type=int, default=2)
    parser.add_argument("--episode-ids", nargs="+",
                        help="explicit development episode ids (overrides --episode-limit)")
    parser.add_argument("--days", type=int, nargs="+", default=[0, 8, 16, 24, 29])
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--exact-candidates", type=int, default=18)
    parser.add_argument("--refinement-candidates", type=int, default=4)
    parser.add_argument("--repair-rounds", type=int, default=6)
    parser.add_argument("--max-exact-evaluations", type=int, default=96)
    parser.add_argument("--compression-rounds", type=int, default=2)
    parser.add_argument("--compression-candidates", type=int, default=4)
    parser.add_argument("--ruin-probability", type=float, default=.35)
    args = parser.parse_args()
    files = sorted(args.samples.glob("episode-*.jsonl.gz"))
    if not files:
        parser.error("no audited episode shards found")
    members = {path.stem.split(".")[0].removeprefix("episode-"): {
        "partition": episode_partition(path.stem.split(".")[0].removeprefix("episode-")),
        "path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
        for path in files}
    if args.episode_ids:
        missing = sorted(set(args.episode_ids)-set(members))
        held_out = sorted(episode for episode in args.episode_ids
                          if episode in members and members[episode]["partition"] != "development")
        if missing or held_out:
            parser.error(f"invalid explicit episodes; missing={missing}, non-development={held_out}")
        chosen = list(args.episode_ids)
    else:
        chosen = sorted((episode for episode, member in members.items()
                         if member["partition"] == "development"),
                        key=lambda episode: sha256(
                            f"temporal-development-order:{episode}".encode()).hexdigest())[:args.episode_limit]
    config = RouteSearchConfig(iterations=args.iterations,
                               exact_candidates=args.exact_candidates,
                               refinement_candidates=args.refinement_candidates,
                               repair_rounds=args.repair_rounds,
                               max_exact_evaluations=args.max_exact_evaluations,
                               compression_rounds=args.compression_rounds,
                               compression_candidates=args.compression_candidates,
                               ruin_probability=args.ruin_probability)
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
            result = compare_route_sample(sample, config, solver=solve_routes)
            target = args.output / (sample["sample_id"].replace(":", "-")+".json")
            target.write_text(json.dumps(result, separators=(",", ":"))+"\n")
            row = {"sample": sample["sample_id"], **result["comparison"],
                   "reference_completion": result["reference"]["completed"],
                   "representation_completion": result["representation_witness"]["completed"],
                   "candidate_completion": result["candidate"]["completed"],
                   "structural_patterns": result["structural_patterns"],
                   "diagnostics": {key: result["candidate"]["diagnostics"].get(key)
                                   for key in ("search_seconds", "search_generated",
                                               "search_exact_evaluations", "search_improved_start",
                                               "search_economic_state_value")}}
            rows.append(row)
            print(json.dumps(row), flush=True)
    patterns = sorted({pattern for row in rows
                       for pattern in row["structural_patterns"]["present"]})
    pattern_summary = {}
    for pattern in patterns:
        members = [row for row in rows if pattern in row["structural_patterns"]["present"]]
        representation_members = [row for row in members if row["representation_missing_goals"]]
        witness_gap_members = [row for row in members
                               if row["demonstration_witness_missing_goals"]]
        pattern_summary[pattern] = {
            "observed_samples": len(members),
            "observed_frequency": len(members)/len(rows) if rows else None,
            # Co-occurrence is diagnostic, not causal attribution.
            "representation_gap_samples": len(representation_members),
            "representation_missing_goals_cooccurring": sum(
                row["representation_missing_goals"] for row in representation_members),
            "demonstration_witness_gap_samples": len(witness_gap_members),
            "demonstration_witness_missing_goals_cooccurring": sum(
                row["demonstration_witness_missing_goals"] for row in witness_gap_members),
        }
    summary = {
        "samples": len(rows),
        "matched_or_better": sum(row["completion_difference"] >= 0 for row in rows),
        "mean_completion_difference": (sum(row["completion_difference"] for row in rows)/len(rows)
                                       if rows else None),
        "gap_decomposition": {
            "representation_gap_samples": sum(bool(row["representation_missing_goals"])
                                              for row in rows),
            "representation_missing_goals": sum(row["representation_missing_goals"]
                                                for row in rows),
            "demonstration_witness_gap_samples": sum(
                bool(row["demonstration_witness_missing_goals"]) for row in rows),
            "demonstration_witness_missing_goals": sum(
                row["demonstration_witness_missing_goals"] for row in rows),
            "witness_missing_goals_recovered_by_blind_search": sum(
                row["witness_missing_goals_recovered_by_blind_search"] for row in rows),
            "unresolved_representation_or_search_goals": sum(
                row["unresolved_representation_or_search_goals"] for row in rows),
            "search_gap_samples": sum(bool(row["search_missing_goals"]) for row in rows),
            "search_missing_goals": sum(row["search_missing_goals"] for row in rows),
            "efficiency_comparable_samples": sum(
                row["efficiency_comparison_status"] != "not-comparable-different-goal-set"
                for row in rows),
        },
        "structural_pattern_frequency_and_representation_impact": pattern_summary,
        "max_search_seconds": max((row["diagnostics"]["search_seconds"] for row in rows), default=0),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
