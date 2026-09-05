"""Lifecycle owner for fixed daily operating plans and bounded repair."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from . import rules
from .execution import generate_tasks
from .planner import Plan, PlannerConfig, make_plan
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
    """Detect only material invalidation or certain same-day infeasibility.

    Current market-price movement is deliberately absent: selling is reactive in
    execution and does not rewrite production commitments.  The task-count check
    is a lower bound, so it fires only if remaining mandatory operations cannot
    fit even with zero travel.
    """

    invalidation = _target_invalidation(state, plan)
    if invalidation:
        return invalidation

    end_of_day = min(rules.TERMINAL_ACTION_STEP, (state.day + 1) * rules.TURNS_PER_DAY - 1)
    mandatory_tasks = tuple(
        task
        for task in generate_tasks(state, plan)
        if task.deadline_step <= end_of_day and task.priority < 70
    )
    # Outstanding hires are part of the fixed plan and normally arrive after
    # this turn's unit actions. Do not declare the plan infeasible merely because
    # those already-planned workers are not visible in the current observation.
    planned_workers = max(len(state.workers), 1 + plan.hire_count)
    remaining_capacity = planned_workers * state.turns_left_today
    if len(mandatory_tasks) > remaining_capacity:
        return (
            f"{len(mandatory_tasks)} remaining same-day operations exceed "
            f"{remaining_capacity} worker-action slots"
        )
    return None


@dataclass
class DailyPlanningSession:
    """Cache one ``Plan`` per player/day and permit a bounded invalidation repair."""

    config: PlannerConfig = field(default_factory=PlannerConfig)
    _plans: dict[int, Plan] = field(default_factory=dict, init=False)
    _last_steps: dict[int, int] = field(default_factory=dict, init=False)

    def reset(self) -> None:
        self._plans.clear()
        self._last_steps.clear()

    def plan_for(self, state: OwnedState) -> Plan:
        prior = self._plans.get(state.player)
        last_step = self._last_steps.get(state.player, -1)
        new_episode = state.step < last_step
        new_day = prior is None or prior.day != state.day

        if new_episode or new_day:
            plan = make_plan(state, self.config)
        else:
            plan = prior
            if (
                state.step > plan.formed_step
                and plan.revision < self.config.max_intraday_replans
            ):
                reason = daily_plan_replan_reason(state, plan)
                if reason is not None:
                    plan = make_plan(
                        state,
                        self.config,
                        revision=plan.revision + 1,
                        replan_reason=reason,
                    )

        self._plans[state.player] = plan
        self._last_steps[state.player] = state.step
        return plan

    @property
    def plans(self) -> Mapping[int, Plan]:
        return dict(self._plans)
