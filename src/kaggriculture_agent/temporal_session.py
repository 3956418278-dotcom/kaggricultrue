"""Retained execution for the temporal model; fixed Plan, separate live sales.

This research controller is not selected by the submitted policy. Progress and
placement witnesses live here, never in a rewritten/subclassed economic Plan.
"""
from dataclasses import replace

from . import rules
from .intent import IntentProgress, compile_intent, matches, unit_event
from .intraday import farm_key
from .market import realization_orders
from .temporal_model import solve_temporal, TemporalConfig


class TemporalSession:
    def __init__(self, config=None):
        self.config = config or TemporalConfig()
        self.entries = {}
        self.diagnostics = []

    def execution_for(self, state, plan):
        entry = self.entries.get(state.player)
        if entry and entry["plan"] == plan and entry["observed"] == state:
            return entry["execution"]
        same_day = entry and entry["plan"] == plan and entry["day"] == state.day and state.step >= entry["observed"].step
        progress = entry["progress"] if same_day else IntentProgress()
        if same_day and state.step == entry["observed"].step + 1:
            # Units precede all external market effects. Reconstruct exactly
            # what the previously emitted units achieved from their real input.
            completed, placements = set(progress.completed), dict(progress.placements)
            before = entry["observed"]
            execution = entry["execution"]
            for w, action in enumerate(execution.worker_actions):
                before, old, new, delta = unit_event(before, w, action)
                identifier = execution.assignments.get(w)
                goal = entry["goals"].get(identifier)
                if goal and matches(goal, old, new, delta, action):
                    completed.add(identifier)
                    completed.update(goal.aliases)
                    if isinstance(new, dict):
                        placements[goal.entity] = before.workers[w].position
            progress = IntentProgress(frozenset(completed), placements)
        elif same_day:
            # A missing observation is not proof of intervening task completion.
            # Explicitly retain only acknowledged execution progress.
            pass
        trajectory = entry["trajectory"] if same_day else None
        offset = state.step - entry["origin"] if same_day else -1

        def live(result, index):
            e = result.executions[index]
            required = [o for o in e.market_orders if o[0] != "SELL"]
            orders = realization_orders(state, e.worker_actions,
                [o for o in required if o[0] not in ("HIRE", "BUY_LAND")],
                sum(o[0] == "HIRE" for o in required), sum(o[0] == "BUY_LAND" for o in required),
                result.remaining_inputs[index])
            return replace(e, market_orders=orders)

        valid = trajectory is not None and 0 <= offset < len(trajectory.executions)
        if valid:
            valid = farm_key(state) == farm_key(trajectory.expected[offset])
        if valid:
            execution = live(trajectory, offset)
            # Price/cash changes are repair-worthy only when they alter the
            # physical realization (e.g. a required purchase actually fails).
            actual = rules.advance_owned(state, execution.worker_actions, execution.market_orders)
            valid = farm_key(actual) == farm_key(trajectory.expected[offset+1])
        if not valid:
            trajectory = solve_temporal(state, plan, self.config, progress)
            offset, origin = 0, state.step
            goals = {g.identifier: g for g in compile_intent(state, plan, progress).goals}
            execution = live(trajectory, 0)
            self.diagnostics.append({"step": state.step, "reason": "repair" if same_day else "daily", **trajectory.diagnostics})
        else:
            origin, goals = entry["origin"], entry["goals"]
        self.entries[state.player] = dict(plan=plan, day=state.day, observed=state, execution=execution,
            trajectory=trajectory, origin=origin, progress=progress, goals=goals)
        return execution
