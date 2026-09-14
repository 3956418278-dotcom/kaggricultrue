#!/usr/bin/env python3
"""Promote the two verified six-day common opening prefixes into runtime data."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs" / "opening-top15-20260913" / "strategies"
TARGET = ROOT / "src" / "kaggriculture_agent" / "data" / "top_opening_bases.json"
RANKS = (12, 14)
CUTOFF = 6 * 24


def normalized(action):
    return {
        "farmer": action.get("farmer") or ["PASS"],
        "hands": action.get("hands") or [],
        "market": action.get("market") or [],
    }


def main() -> None:
    records = []
    for rank in RANKS:
        path = next(path for path in SOURCE.glob("*.json.gz")
                    if int(path.name[:2]) == rank)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            strategy = json.load(handle)
        canonical = [normalized(action)
                     for action in strategy["canonical"]["actions"]]
        first_differences = []
        for variant in strategy["variants"]:
            actions = [normalized(action) for action in variant["actions"]]
            difference = next((step for step, pair in enumerate(
                zip(canonical, actions)) if pair[0] != pair[1]),
                min(len(canonical), len(actions)))
            first_differences.append(difference)
        if min(first_differences) != CUTOFF:
            raise ValueError(
                f"rank {rank}: expected first variant at {CUTOFF}, "
                f"got {min(first_differences)}")
        records.append({
            "rank": rank,
            "team_name": strategy["team_name"],
            "submission_id": strategy["submission_id"],
            "canonical_episode_id": strategy["canonical"]["episode_id"],
            "source_strategy_gzip_sha256": hashlib.sha256(
                path.read_bytes()).hexdigest(),
            "handoff_step": CUTOFF,
            "actions": canonical[:CUTOFF],
        })
    value = {
        "schema_version": "top-opening-common-prefix-v1",
        "source_census": "official Kaggle top-15 snapshot 2026-09-13",
        "strategies": records,
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")) + "\n", encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
