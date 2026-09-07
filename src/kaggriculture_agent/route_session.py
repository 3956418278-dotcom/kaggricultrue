"""Retain event-route trajectories while a fixed Daily Plan remains valid."""
from dataclasses import replace

from . import rules
from .intent import IntentProgress, compile_intent, matches, unit_event
from .intraday import farm_key
from .market import realization_orders
from .route_search import RouteSearchConfig, solve_routes


class RouteSession:
    """Execution memory is separate from the immutable economic Plan."""

    def __init__(self, config=None):
        self.config = config or RouteSearchConfig()
        self.entries = {}
        self.diagnostics = []

    def execution_for(self, state, plan):
        entry = self.entries.get(state.player)
        if entry and entry["plan"] == plan and entry["observed"] == state:
            return entry["execution"]
        same_day = bool(entry and entry["plan"] == plan and entry["day"] == state.day
                        and state.step >= entry["observed"].step)
        progress = entry["progress"] if same_day else IntentProgress()
        if same_day and state.step == entry["observed"].step+1:
            completed, placements = set(progress.completed), dict(progress.placements)
            micro = entry["observed"]
            execution = entry["execution"]
            for worker, action in enumerate(execution.worker_actions):
                if worker >= len(micro.workers): break
                following, before, after, delta = unit_event(micro, worker, action)
                identifier = execution.assignments.get(worker)
                goal = entry["goals"].get(identifier)
                if goal and matches(goal, before, after, delta, action):
                    completed.add(identifier); completed.update(goal.aliases)
                    placements[goal.entity] = micro.workers[worker].position
                micro = following
            progress = IntentProgress(frozenset(completed), placements)

        trajectory = entry["trajectory"] if same_day else None
        offset = state.step-entry["origin"] if same_day else -1

        def live(result, index):
            execution = result.executions[index]
            required = [order for order in execution.market_orders if order[0] != "SELL"]
            reserve = result.remaining_inputs[index] if index < len(result.remaining_inputs) else {}
            orders = realization_orders(state, execution.worker_actions,
                [order for order in required if order[0] not in ("HIRE", "BUY_LAND")],
                sum(order[0] == "HIRE" for order in required),
                sum(order[0] == "BUY_LAND" for order in required), reserve,
                reserve_is_shed=True, required_sequence=required)
            return replace(execution, market_orders=orders)

        valid = trajectory is not None and 0 <= offset < len(trajectory.executions)
        if valid:
            valid = farm_key(state) == farm_key(trajectory.expected[offset])
        if valid:
            execution = live(trajectory, offset)
            actual = rules.advance_owned(state, execution.worker_actions, execution.market_orders)
            valid = farm_key(actual) == farm_key(trajectory.expected[offset+1])
        if not valid:
            trajectory = solve_routes(state, plan, self.config, progress)
            offset, origin = 0, state.step
            goals = {goal.identifier: goal for goal in compile_intent(state, plan, progress).goals}
            if not trajectory.executions:
                from .execution import Execution
                execution = Execution(tuple(["PASS"] for _ in state.workers), (), {}, ())
            else:
                execution = live(trajectory, 0)
            self.diagnostics.append({"step": state.step,
                "reason": "repair" if same_day else "daily", **trajectory.diagnostics})
        else:
            origin, goals = entry["origin"], entry["goals"]
        self.entries[state.player] = dict(plan=plan, day=state.day, observed=state,
            execution=execution, trajectory=trajectory, origin=origin,
            progress=progress, goals=goals)
        return execution
