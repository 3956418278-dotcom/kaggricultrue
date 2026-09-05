#!/usr/bin/env python3
"""Exploratory full-episode health check for the baseline candidate.

This is a correctness/runtime probe, not an EVALUATION.md competitive arena and
its money results must not be presented as acceptance-quality evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import platform
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_environments import make  # noqa: E402
from kaggle_environments.envs.kaggriculture import kaggriculture as official  # noqa: E402

from main import agent  # noqa: E402
from src.kaggriculture_agent.agent import decide  # noqa: E402
from src.kaggriculture_agent.operating import DailyPlanningSession  # noqa: E402


def run(seed: int, candidate_seat: int, replay_dir: Path | None = None) -> dict[str, object]:
    durations: list[float] = []
    planning_session = DailyPlanningSession()

    def measured_decide(observation):
        started = time.perf_counter()
        action = decide(observation, session=planning_session)
        durations.append(time.perf_counter() - started)
        if action != decide(copy.deepcopy(observation), session=planning_session):
            raise AssertionError(
                f"identical observation produced different actions at step {observation.step}"
            )
        return action

    # Use the unguarded core so an internal error fails the check instead of
    # being hidden by the submission wrapper's safe pass.
    agents = [measured_decide, "starter"] if candidate_seat == 0 else ["starter", measured_decide]
    environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
    initial = environment.reset(2)[candidate_seat].observation
    if agent(copy.deepcopy(initial)) != agent(copy.deepcopy(initial)):
        raise AssertionError("identical initial observations produced different actions")
    environment.run(agents)
    if len(environment.steps) != 720 or any(state.status != "DONE" for state in environment.state):
        raise AssertionError(f"episode did not complete cleanly: {[state.status for state in environment.state]}")
    final_private = environment.state[candidate_seat].observation.private
    terminal_inventory = sum(
        final_private["shed"].get(item, 0)
        + sum(inventory.get(item, 0) for inventory in final_private["inventories"])
        for item in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
    )
    if terminal_inventory:
        raise AssertionError(f"terminal sellable inventory was not liquidated: {terminal_inventory}")
    replay = None
    if replay_dir is not None:
        replay_dir.mkdir(parents=True, exist_ok=True)
        path = replay_dir / f"seed-{seed}-seat-{candidate_seat}.json"
        with path.open("x") as output:
            json.dump(environment.toJSON(), output)
        replay = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    unit_operations: Counter[str] = Counter()
    market_operations: Counter[str] = Counter()
    for step in environment.steps[:-1]:
        action = step[candidate_seat].action or {}
        unit_operations.update([action.get("farmer", ["MISSING"])[0]])
        unit_operations.update(item[0] for item in action.get("hands", []))
        market_operations.update(item[0] for item in action.get("market", []))
    return {
        "seed": seed,
        "candidate_seat": candidate_seat,
        "statuses": [state.status for state in environment.state],
        "configuration": {**dict(environment.configuration), "seed": environment.info["seed"]},
        "replay": replay,
        "rewards": [state.reward for state in environment.state],
        "decision_seconds_max": max(durations),
        "decision_seconds_p95": statistics.quantiles(durations, n=20)[18],
        "measured_overage_seconds": sum(max(0.0, duration - 1) for duration in durations),
        "planning": {
            "calls": len(planning_session.planning_diagnostics),
            "reasons": dict(Counter(row["reason"] for row in planning_session.planning_diagnostics)),
            "seconds_total": sum(row["seconds"] for row in planning_session.planning_diagnostics),
            "expanded_total": sum(row["expanded"] for row in planning_session.planning_diagnostics),
            "days": planning_session.planning_diagnostics,
        },
        "terminal_sellable_inventory": terminal_inventory,
        "unit_operations": dict(sorted(unit_operations.items())),
        "market_operations": dict(sorted(market_operations.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay-dir", type=Path)
    args = parser.parse_args()
    source_paths = [ROOT / "main.py", Path(__file__), *sorted((ROOT / "src/kaggriculture_agent").glob("*.py"))]
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    started = datetime.now(timezone.utc).isoformat()
    results = [run(seed, seat, args.replay_dir) for seed in args.seeds for seat in (0, 1)]
    if source_hashes != {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}:
        raise AssertionError("source changed during verification; discard this run and rerun")
    report = {"scope": "exploratory runtime health check", "environment": "kaggle-environments==1.32.7",
        "official_source_sha256": hashlib.sha256(Path(official.__file__).read_bytes()).hexdigest(),
        "opponent": {"name": "starter", "loader": "official registered builtin",
            "source_sha256": hashlib.sha256(Path(official.__file__).read_bytes()).hexdigest()},
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "command": sys.argv,
        "source_hashes": source_hashes,
        "episodes": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            json.dump(report, output, indent=2)
        print(f"Saved {len(results)} complete episode diagnostics to {args.output}")
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
