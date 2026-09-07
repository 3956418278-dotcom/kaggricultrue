#!/usr/bin/env python3
"""Audit a collected player-day directory without modifying its evidence files."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kaggriculture_eval.reference_audit import audit_collection
from src.kaggriculture_eval.reference_pipeline import write_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit_collection(args.directory)
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, indent=2))
