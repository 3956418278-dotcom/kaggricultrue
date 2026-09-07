#!/usr/bin/env python3
"""Restore expanded Kaggle Dataset shards to hash-verified local checkpoints."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kaggriculture_eval.reference_storage import restore_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(restore_dataset(args.source, args.output)))
