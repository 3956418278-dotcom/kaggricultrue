#!/usr/bin/env python3
"""Run maintained execution challenges and print a reproducible JSON report."""
import hashlib
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kaggriculture_eval.intraday_benchmark import scenarios, run_scenario

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"scope": "development execution scenarios; not baseline acceptance",
        "environment": "kaggle-environments==1.32.7", "seed": 41,
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("src/kaggriculture_agent", "src/kaggriculture_eval")
            for p in sorted((ROOT / folder).glob("*.py"))},
        "scenarios": [run_scenario(*scenario) for scenario in scenarios()]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            json.dump(report, output, indent=2)
        print(f"Saved {len(report['scenarios'])} scenario comparisons to {args.output}")
    else:
        print(json.dumps(report, indent=2))
