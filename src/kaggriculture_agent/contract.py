"""Kaggle-facing serialization of an exact ``TurnDecision``."""

from __future__ import annotations

from typing import Any

from . import rules
from .realization import TurnDecision
from .state import OwnedState


def pass_action(hand_count: int = 0) -> dict[str, list[Any]]:
    return {"farmer": ["PASS"], "hands": [["PASS"] for _ in range(max(0, hand_count))], "market": []}


def construct_action(state: OwnedState, decision: TurnDecision) -> dict[str, list[Any]]:
    actions = [list(action) for action in decision.worker_actions]
    if not actions:
        return pass_action()
    if len(actions) != len(state.workers):
        raise ValueError("TurnDecision worker count does not match OwnedState")
    return {
        "farmer": actions[0],
        "hands": actions[1:],
        "market": [list(order) for order in decision.market_orders[: rules.MAX_MARKET_ORDERS]],
    }
