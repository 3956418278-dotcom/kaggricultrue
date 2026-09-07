#!/usr/bin/env python3
"""Reanalyze immutable route-benchmark rows without rerunning planners."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    files = sorted(path for path in args.run.glob("*.json")
                   if path.name not in {"manifest.json", "summary.json"})
    if not files:
        parser.error("no benchmark result rows found")
    rows, patterns = [], Counter()
    pattern_witness_gap = Counter()
    for path in files:
        result = json.loads(path.read_text())
        reference = set(result["reference"]["completed_ids"])
        witness = set(result["representation_witness"]["completed_ids"])
        candidate = set(result["candidate"]["completed_ids"])
        witness_missing = reference-witness
        recovered = witness_missing & candidate
        unresolved = reference-(witness | candidate)
        search_missing = witness-candidate
        present = result["structural_patterns"]["present"]
        patterns.update(present)
        if witness_missing:
            pattern_witness_gap.update(present)
        rows.append({
            "sample_id": result["sample_id"],
            "reference_completed": len(reference),
            "demonstration_witness_completed": len(witness),
            "candidate_completed": len(candidate),
            "demonstration_witness_missing": len(witness_missing),
            "witness_missing_recovered_by_blind_search": len(recovered),
            "unresolved_representation_or_search": len(unresolved),
            "search_missing_from_compiler_witness": len(search_missing),
            "same_reference_goal_set": reference == candidate,
            "efficiency_comparison_status": result["efficiency"][
                "candidate_vs_controlled_reference"]["comparison_status"],
            "candidate_vs_reference_efficiency": result["efficiency"][
                "candidate_vs_controlled_reference"],
            "candidate_vs_reference_state": result["resulting_state"][
                "candidate_vs_controlled_reference"],
            "search_seconds": result["candidate"]["diagnostics"]["search_seconds"],
            "patterns": present,
        })
    report = {
        "source_run": str(args.run),
        "source_manifest_sha256": sha256((args.run/"manifest.json").read_bytes()).hexdigest(),
        "interpretation": {
            "reference_role": "strong feasible witness, not an optimum",
            "representation_rule": ("a demonstration-compiler miss is a translation/compiler-witness gap; "
                                    "it is not an established representation gap without a diagnosed "
                                    "unexpressible structural pattern"),
            "efficiency_rule": "interpret only rows with compatible achieved-goal sets",
        },
        "summary": {
            "samples": len(rows),
            "same_reference_goal_set": sum(row["same_reference_goal_set"] for row in rows),
            "demonstration_witness_gap_samples": sum(
                bool(row["demonstration_witness_missing"]) for row in rows),
            "demonstration_witness_missing_goals": sum(
                row["demonstration_witness_missing"] for row in rows),
            "witness_missing_goals_recovered_by_blind_search": sum(
                row["witness_missing_recovered_by_blind_search"] for row in rows),
            "unresolved_representation_or_search_goals": sum(
                row["unresolved_representation_or_search"] for row in rows),
            "established_unexpressible_patterns": [],
            "established_representation_missing_goals": 0,
            "search_gap_samples": sum(bool(row["search_missing_from_compiler_witness"])
                                      for row in rows),
            "search_missing_goals": sum(row["search_missing_from_compiler_witness"]
                                        for row in rows),
            "efficiency_comparable_samples": sum(
                row["efficiency_comparison_status"] != "not-comparable-different-goal-set"
                for row in rows),
            "mean_search_seconds": sum(row["search_seconds"] for row in rows)/len(rows),
            "max_search_seconds": max(row["search_seconds"] for row in rows),
        },
        "pattern_frequency_and_witness_gap_cooccurrence": {
            pattern: {"observed_samples": count, "observed_frequency": count/len(rows),
                      "demonstration_witness_gap_samples": pattern_witness_gap[pattern]}
            for pattern, count in sorted(patterns.items())
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
