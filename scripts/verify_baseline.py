#!/usr/bin/env python3
"""Exploratory full-episode health check for the baseline candidate.

This is a correctness/runtime probe, not an EVALUATION.md competitive arena and
its money results must not be presented as acceptance-quality evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_environments import make  # noqa: E402

from main import agent  # noqa: E402
from src.kaggriculture_agent.agent import decide  # noqa: E402


def run(seed: int, candidate_seat: int) -> dict[str, object]:
    durations: list[float] = []

    def measured_decide(observation):
        started = time.perf_counter()
        action = decide(observation)
        durations.append(time.perf_counter() - started)
        if action != decide(copy.deepcopy(observation)):
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
        "rewards": [state.reward for state in environment.state],
        "decision_seconds_max": max(durations),
        "decision_seconds_p95": statistics.quantiles(durations, n=20)[18],
        "terminal_sellable_inventory": terminal_inventory,
        "unit_operations": dict(sorted(unit_operations.items())),
        "market_operations": dict(sorted(market_operations.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29])
    args = parser.parse_args()
    results = [run(seed, seat) for seed in args.seeds for seat in (0, 1)]
    print(json.dumps({"scope": "exploratory runtime health check", "episodes": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
