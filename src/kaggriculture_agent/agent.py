"""Pure deterministic baseline policy composition."""

from __future__ import annotations

from typing import Any

from .contract import construct_action, pass_action
from .execution import execute
from .planner import make_plan
from .state import get, reconstruct


def decide(observation: Any) -> dict[str, list[Any]]:
    state = reconstruct(observation)
    return construct_action(state, execute(state, make_plan(state)))


def agent(observation: Any) -> dict[str, list[Any]]:
    """Submission-safe wrapper: malformed observations degrade to a legal pass."""
    try:
        return decide(observation)
    except Exception:
        farms = get(observation, "farms", ()) or ()
        player = int(get(observation, "player", 0) or 0)
        hand_count = len(get(farms[player], "hands", ()) or ()) if 0 <= player < len(farms) else 0
        return pass_action(hand_count)
