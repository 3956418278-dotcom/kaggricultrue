"""Two local base agents built from audited six-day top-opening prefixes."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .contract import construct_action
from .operating import DailyPlanningSession
from .state import reconstruct


_DATA_PATH = Path(__file__).with_name("data") / "top_opening_bases.json"


def _records() -> dict[int, dict[str, Any]]:
    raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    return {int(record["rank"]): record for record in raw["strategies"]}


OPENING_BASES = _records()


@dataclass
class OpeningBaseAgent:
    rank: int
    session: DailyPlanningSession = field(default_factory=DailyPlanningSession)
    _last_step: int = field(default=-1, init=False)

    @property
    def record(self) -> dict[str, Any]:
        return OPENING_BASES[self.rank]

    def __call__(self, observation: Any) -> dict[str, list[Any]]:
        state = reconstruct(observation)
        if state.step < self._last_step:
            self.session.reset()
        self._last_step = state.step
        if state.step < int(self.record["handoff_step"]):
            return deepcopy(self.record["actions"][state.step])
        programme = self.session.plan_for(state)
        return construct_action(
            state, self.session.execution_for(state, programme))


def make_opening_base(rank: int) -> OpeningBaseAgent:
    if rank not in OPENING_BASES:
        raise ValueError(f"no promoted opening base for rank {rank}")
    return OpeningBaseAgent(rank)


catalyst_base_agent = make_opening_base(12)
subramanya_base_agent = make_opening_base(14)

