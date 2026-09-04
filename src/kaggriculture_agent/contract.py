"""Kaggle-facing action adapter and final defensive normalization."""

from __future__ import annotations

from typing import Any

from . import rules
from .execution import Execution
from .state import OwnedState


def pass_action(hand_count: int = 0) -> dict[str, list[Any]]:
    return {"farmer": ["PASS"], "hands": [["PASS"] for _ in range(max(0, hand_count))], "market": []}


def construct_action(state: OwnedState, execution: Execution) -> dict[str, list[Any]]:
    actions = list(execution.worker_actions)
    if not actions:
        return pass_action()
    requests: dict[str, int] = {}
    for action in actions:
        if len(action) >= 2 and action[0] == "PLANT":
            crop = str(action[1])
            requests[crop] = requests.get(crop, 0) + 1
    blocked = {crop for crop, amount in requests.items() if amount > state.seeds.get(crop, 0)}
    normalized = [
        ["PASS"] if len(action) >= 2 and action[0] == "PLANT" and str(action[1]) in blocked else action
        for action in actions
    ]
    return {
        "farmer": normalized[0],
        "hands": normalized[1 : len(state.workers)],
        "market": [list(order) for order in execution.market_orders[: rules.MAX_MARKET_ORDERS]],
    }
