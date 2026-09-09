"""Public intraday contract and exact-completion invariants."""

from dataclasses import fields, replace
from pathlib import Path
import unittest

from kaggle_environments import make

from src.kaggriculture_agent.execution import InvalidRealization, execute_realization
from src.kaggriculture_agent.economics import ActionDimension, WorkAmount
from src.kaggriculture_agent.intraday import _expand_plan, solve_intraday
from src.kaggriculture_agent.planner import EconomicWindow
from src.kaggriculture_agent.realization import PlanningFailure, Realization, TurnDecision
from src.kaggriculture_agent.state import reconstruct
from src.kaggriculture_eval.plan_io import plan_from_dict
from src.kaggriculture_eval.player_days import reconstruct_day


class IntradayContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = make("kaggriculture", configuration={"seed": 123})

        def demonstration(observation):
            actions = {
                0: {"farmer": ["BUILD_PASTURE"], "market": [
                    ["BUY_ANIMAL", "COW", 1], ["BUY_PRODUCT", "WHEAT", 2]
                ]},
                1: {"farmer": ["PICKUP", "COW", 1]},
                2: {"farmer": ["PLACE", "COW"]},
                3: {"farmer": ["CARE"]},
                4: {"farmer": ["PICKUP", "WHEAT", 1]},
                5: {"farmer": ["FEED"]},
            }
            return actions.get(observation.step, {"farmer": ["PASS"]})

        env.run([demonstration, "pass"])
        replay = env.toJSON()
        replay["info"]["EpisodeId"] = 123
        sample = reconstruct_day(replay, 0, 0, {}, "test")
        cls.state = reconstruct(sample["day_start_state"])
        cls.plan = plan_from_dict(sample["plan"])
        cls.realization = solve_intraday(cls.state, cls.plan)

    def test_public_representation_is_only_turns_and_placements(self):
        self.assertEqual([field.name for field in fields(TurnDecision)],
                         ["worker_actions", "market_orders"])
        self.assertEqual([field.name for field in fields(Realization)],
                         ["placements", "turns"])

    def test_known_witness_plan_is_fully_realized_and_exactly_executable(self):
        ending = execute_realization(self.state, self.plan, self.realization)
        self.assertEqual(len(self.realization.turns), 24)
        self.assertTrue(any(action[0] == "BUILD_PASTURE"
                            for turn in self.realization.turns
                            for action in turn.worker_actions))
        self.assertTrue(any(tile.animal == "COW" for tile in ending.tiles))

    def test_action_dimension_is_not_an_intraday_prescription(self):
        altered = replace(self.plan,
            obligations=tuple(replace(project, actions=ActionDimension((
                WorkAmount(self.plan.day, "DIG", 99, position=(9, 9)),
            ))) for project in self.plan.obligations),
            selected=tuple(replace(project, actions=ActionDimension((
                WorkAmount(self.plan.day, "DIG", 99, position=(9, 9)),
            ))) for project in self.plan.selected))
        original = _expand_plan(self.state, self.plan)[0]
        changed = _expand_plan(self.state, altered)[0]
        signature = lambda projects: tuple(
            (project.position, tuple(event.action for event in project.events))
            for project in projects)
        self.assertEqual(signature(original), signature(changed))

    def test_explicit_economic_window_is_the_only_market_timing_input(self):
        empty = replace(self.plan, obligations=(), selected=(), support=(),
            economic_windows=(EconomicWindow(
                2, 3, market_orders=(("BUY_SEED", "WHEAT", 1),),
            ),))
        realization = solve_intraday(self.state, empty)
        execute_realization(self.state, empty, realization)
        timed = [(index, order) for index, turn in enumerate(realization.turns)
                 for order in turn.market_orders]
        self.assertEqual(timed, [(2, ("BUY_SEED", "WHEAT", 1))])

    def test_incomplete_realization_is_never_a_result(self):
        with self.assertRaises(InvalidRealization):
            execute_realization(self.state, self.plan, Realization({}, ()))

    def test_solver_failure_is_explicit(self):
        with self.assertRaises(PlanningFailure):
            solve_intraday(self.state, self.plan.__class__(
                **{**self.plan.__dict__, "day": self.plan.day + 1}
            ))

    def test_old_production_layers_are_deleted(self):
        root = Path("src/kaggriculture_agent")
        for name in ("intent.py", "route_search.py", "route_support.py",
                     "route_structure.py", "route_compiler.py",
                     "route_session.py", "temporal_model.py",
                     "temporal_session.py", "temporal_start.py"):
            self.assertFalse((root / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
