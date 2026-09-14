#!/usr/bin/env python3
"""Exploratory side-swapped arena for the two promoted opening bases."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

from kaggle_environments import make
import kaggle_environments

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kaggriculture_agent.opening_bases import make_opening_base  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_game(seed: int, rank12_seat: int, output: Path) -> dict[str, object]:
    policies = [None, None]
    policies[rank12_seat] = make_opening_base(12)
    policies[1 - rank12_seat] = make_opening_base(14)
    # kaggle-environments distinguishes plain functions from other callable
    # objects when preparing local agents.
    def player_zero(observation):
        return policies[0](observation)

    def player_one(observation):
        return policies[1](observation)

    agents = [player_zero, player_one]
    env = make("kaggriculture", configuration={"seed": seed})
    started = time.perf_counter()
    env.run(agents)
    elapsed = time.perf_counter() - started
    replay = env.toJSON()
    replay_path = output / "replays" / f"seed-{seed}-rank12-seat-{rank12_seat}.json"
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(json.dumps(
        replay, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    rewards = [float(value) if value is not None else float("nan")
               for value in replay["rewards"]]
    candidate = rewards[rank12_seat]
    opponent = rewards[1 - rank12_seat]
    return {
        "seed": seed,
        "rank12_seat": rank12_seat,
        "statuses": replay["statuses"],
        "rank12_money": candidate,
        "rank14_money": opponent,
        "rank12_margin": candidate - opponent,
        "outcome": "W" if candidate > opponent else "L" if candidate < opponent else "D",
        "elapsed_seconds": elapsed,
        "replay": str(replay_path.relative_to(ROOT)),
        "replay_sha256": sha256(replay_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[1701, 1702, 1703, 1704, 1705])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs" / "evaluation" /
                        "opening-bases-r12-vs-r14-exploratory")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for seed in args.seeds:
        for seat in (0, 1):
            row = run_game(seed, seat, output)
            rows.append(row)
            print(seed, seat, row["outcome"], row["rank12_money"],
                  row["rank14_money"], f"{row['elapsed_seconds']:.2f}s",
                  flush=True)
    outcomes = Counter(row["outcome"] for row in rows)
    summary = {
        "games": len(rows),
        "complete_blocks": len(args.seeds),
        "rank12_wins": outcomes["W"],
        "draws": outcomes["D"],
        "rank12_losses": outcomes["L"],
        "rank12_mean_margin": sum(row["rank12_margin"] for row in rows) / len(rows),
        "all_done": all(row["statuses"] == ["DONE", "DONE"] for row in rows),
        "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
    }
    source_paths = [
        ROOT / "src/kaggriculture_agent/opening_bases.py",
        ROOT / "src/kaggriculture_agent/data/top_opening_bases.json",
        ROOT / "src/kaggriculture_agent/planner.py",
        ROOT / "src/kaggriculture_agent/intraday.py",
        ROOT / "src/kaggriculture_agent/zonal_templates.py",
    ]
    result = {
        "evidence_class": "exploratory",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "environment": "kaggriculture",
        "kaggle_environments_version": kaggle_environments.__version__,
        "python": platform.python_version(),
        "seeds": args.seeds,
        "agents": {
            "rank12": "Catalyst D1-D6 prefix -> current zonal midgame",
            "rank14": "Subramanya N D1-D6 prefix -> current zonal midgame",
        },
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path)
                          for path in source_paths},
        "summary": summary,
        "games": rows,
    }
    result_path = output / "results.json"
    result_path.write_text(json.dumps(
        result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(result_path)


if __name__ == "__main__":
    main()
