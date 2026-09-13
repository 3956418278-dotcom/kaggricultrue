#!/usr/bin/env python3
"""Reexecute every collected D1-D8 trace with pinned official transitions."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

from kaggle_environments import make

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kaggriculture_eval.player_days import observation


LAST_OPENING_FRAME = 192


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def audit(root: Path) -> None:
    results = {}
    paths = sorted((root / "replays").glob("episode-*.json.gz"))
    for index, path in enumerate(paths, 1):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            replay = json.load(handle)
        episode_id = int(replay["info"]["EpisodeId"])
        config = deepcopy(replay["configuration"])
        config["seed"] = replay["info"]["seed"]
        environment = make("kaggriculture", configuration=config)
        environment.reset(2)
        checks = 0
        failure = None
        for frame in range(LAST_OPENING_FRAME + 1):
            if frame:
                environment.step([
                    deepcopy(side.get("action") or {}) for side in replay["steps"][frame]
                ])
            for side in range(2):
                expected = observation(replay, frame, side)
                actual = deepcopy(dict(environment.state[side].observation))
                actual.pop("remainingOverageTime", None)
                actual["step"] = frame
                if actual != expected:
                    failure = {"frame": frame, "side": side}
                    break
                checks += 1
            if failure:
                break
        results[str(episode_id)] = {
            "passed": failure is None,
            "failure": failure,
            "observation_checks": checks,
            "replay_gzip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "module_version": replay.get("module_version"),
            "seed": replay["info"].get("seed"),
        }
        _write_json(root / "opening-transition-audit.json", {
            "complete": False,
            "episodes": results,
        })
        print(f"audit {index:03d}/{len(paths)} episode={episode_id} passed={failure is None}", flush=True)
    _write_json(root / "opening-transition-audit.json", {
        "complete": len(results) == len(paths),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "scope": "official frames 0-192 inclusive for both sides; 386 observation checks per episode",
        "episodes": results,
        "passed": len(results) == len(paths) and all(row["passed"] for row in results.values()),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("collection", type=Path)
    args = parser.parse_args()
    audit(args.collection)


if __name__ == "__main__":
    main()
