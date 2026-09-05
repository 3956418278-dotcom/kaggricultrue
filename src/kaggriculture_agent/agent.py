"""Deterministic baseline composition with episode-local daily plans."""

from __future__ import annotations

from typing import Any

from .contract import construct_action, pass_action
from .operating import DailyPlanningSession
from .state import get, reconstruct


_DEFAULT_SESSION = DailyPlanningSession()


def decide(
    observation: Any,
    *,
    session: DailyPlanningSession | None = None,
) -> dict[str, list[Any]]:
    state = reconstruct(observation)
    owner = session or _DEFAULT_SESSION
    operating_plan = owner.plan_for(state)
    return construct_action(state, owner.execution_for(state, operating_plan))


def agent(observation: Any) -> dict[str, list[Any]]:
    """Submission-safe wrapper: malformed observations degrade to a legal pass."""
    try:
        return decide(observation)
    except Exception:
        farms = get(observation, "farms", ()) or ()
        player = int(get(observation, "player", 0) or 0)
        hand_count = len(get(farms[player], "hands", ()) or ()) if 0 <= player < len(farms) else 0
        return pass_action(hand_count)
