"""Lifecycle owner for one fixed daily Plan and its exact Realization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .planner import Plan, PlannerConfig, make_plan
from .realization import Realization, TurnDecision
from .state import OwnedState


def _target_invalidation(state: OwnedState, plan: Plan) -> str | None:
    """Return a reason only when a commitment's physical premise was invalidated."""

    for commitment in plan.obligations:
        if commitment.target is None:
            continue
        tile = state.tile_at(commitment.target)
        if commitment.kind == "CROP_MAINTENANCE":
            crop = str(commitment.metadata["crop"])
            if tile.kind == "PLANT" and tile.raw.get("crop") == crop:
                continue
            if state.step >= commitment.time.completion_step:
                continue
            return f"existing {crop} commitment disappeared before completion"
        if commitment.kind == "ANIMAL_MAINTENANCE":
            animal = str(commitment.metadata["animal"])
            if tile.animal != animal:
                return f"existing {animal} commitment is no longer on its tile"
        if commitment.kind == "ANIMAL_PLACEMENT":
            animal = str(commitment.metadata["animal"])
            structure = str(commitment.metadata["structure"])
            if (
                state.owned_total(animal) > 0
                or any(candidate.animal == animal for candidate in state.animal_tiles())
                or tile.is_empty
                or (tile.kind == structure and tile.animal is None)
            ):
                continue
            return f"pending {animal} placement lost its feasible target"

    for commitment in plan.selected:
        if commitment.target is None:
            continue
        tile = state.tile_at(commitment.target)
        if commitment.kind == "CROP":
            crop = str(commitment.metadata["crop"])
            if tile.is_empty or (
                tile.kind == "PLANT" and tile.raw.get("crop") == crop
            ):
                continue
            return f"planned {crop} target {commitment.target} became unavailable"
        if commitment.kind == "ANIMAL":
            animal = str(commitment.metadata["animal"])
            structure = str(commitment.metadata["structure"])
            if (
                tile.is_empty
                or tile.animal == animal
                or (tile.kind == structure and tile.animal is None)
                or state.owned_total(animal) > 0
            ):
                continue
            return f"planned {animal} target {commitment.target} became unavailable"
    return None


def daily_plan_replan_reason(state: OwnedState, plan: Plan) -> str | None:
    """Invalidate owned physical premises, not a speculative spatial binding.

    Intraday search owns executable capacity and remaining work. Rebinding all
    new projects here would mistake completed plantings for more unfulfilled
    work and silently change the economic plan late in the day.
    """
    return _target_invalidation(state, plan)


@dataclass
class DailyPlanningSession:
    """Cache one ``Plan`` per player/day and permit a bounded invalidation repair."""

    config: PlannerConfig = field(default_factory=PlannerConfig)
    _plans: dict[int, Plan] = field(default_factory=dict, init=False)
    _last_steps: dict[int, int] = field(default_factory=dict, init=False)
    _realizations: dict[int, tuple[int, int, Realization]] = field(default_factory=dict, init=False)

    def execution_for(self, state: OwnedState, plan: Plan) -> TurnDecision:
        """Return the already solved action for this absolute turn.

        The lifecycle cache is not a second planner representation: its value is
        exactly the public ``Realization`` returned by ``solve_intraday``.
        """
        from .intraday import solve_intraday

        cached = self._realizations.get(state.player)
        if cached is None or cached[0] != plan.day or cached[1] > state.step:
            realization = solve_intraday(state, plan)
            cached = (plan.day, state.step, realization)
            self._realizations[state.player] = cached
        _, start_step, realization = cached
        offset = state.step - start_step
        if not 0 <= offset < len(realization.turns):
            return TurnDecision(tuple(("PASS",) for _ in state.workers))
        return realization.turns[offset]

    def reset(self) -> None:
        self._plans.clear()
        self._last_steps.clear()
        self._realizations.clear()

    def plan_for(self, state: OwnedState) -> Plan:
        prior = self._plans.get(state.player)
        last_step = self._last_steps.get(state.player, -1)
        new_episode = state.step < last_step
        new_day = prior is None or prior.day != state.day

        if new_episode or new_day:
            plan = make_plan(state, self.config)
        else:
            plan = prior
            # Even an impossible target stays fixed; realization reports it.

        self._plans[state.player] = plan
        self._last_steps[state.player] = state.step
        return plan

    @property
    def plans(self) -> Mapping[int, Plan]:
        return dict(self._plans)

    @property
    def planning_diagnostics(self) -> tuple[dict, ...]:
        return ()
