"""Public intraday contract and exact-completion invariants."""

from dataclasses import fields, replace
from pathlib import Path
import unittest

from kaggle_environments import make

from src.kaggriculture_agent.execution import InvalidRealization, execute_realization
from src.kaggriculture_agent.economics import ActionDimension, WorkAmount
from src.kaggriculture_agent.intraday import _expand_plan, solve_intraday
from src.kaggriculture_agent.planner import (
    EconomicWindow,
    Plan,
    _animal_commitment,
    _canonical_daily_commitment,
    _crop_commitment,
)
from src.kaggriculture_agent.realization import PlanningFailure, Realization, TurnDecision
from src.kaggriculture_agent.state import TileState, reconstruct
from src.kaggriculture_agent import rules
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


class ProgrammeFinanceRegressionTests(unittest.TestCase):
    def _state(self, *, money, shed, tile_updates):
        observation = make(
            "kaggriculture", configuration={"seed": 31}
        ).reset(2)[0].observation
        state = reconstruct(observation)
        replacements = {
            position: TileState(position, raw)
            for position, raw in tile_updates.items()
        }
        return replace(
            state,
            step=4 * rules.TURNS_PER_DAY,
            day=4,
            hour=0,
            money=money,
            shed={item: int(shed.get(item, 0)) for item in rules.PRODUCTS},
            tiles=tuple(replacements.get(tile.position, tile)
                        for tile in state.tiles),
        )

    def _plan(self, state, *, obligations=(), selected=(), window):
        return Plan(
            obligations=tuple(obligations),
            selected=tuple(selected),
            support=(),
            rejected={},
            fertilize_targets=frozenset(),
            animal_purchases=tuple(
                str(item.metadata["animal"])
                for item in selected if item.kind == "ANIMAL"
            ),
            buy_land=False,
            feed_reserve=0,
            fertilizer_reserve=0,
            day=state.day,
            formed_step=state.step,
            max_hands=0,
            economic_windows=(window,),
        )

    def _trace(self, state, realization):
        current = state
        trace = []
        money = [state.money]
        for turn, decision in enumerate(realization.turns):
            for worker, action in enumerate(decision.worker_actions):
                trace.append((turn, current.workers[worker].position, action))
            current = rules.advance_owned(
                current, decision.worker_actions, decision.market_orders
            )
            money.append(current.money)
        return current, trace, money

    def test_turn7_fertilizer_finances_cow_after_prebuild(self):
        goose_position = (4, 4)
        cow_position = (4, 3)
        state = self._state(
            money=300,
            shed={"WHEAT": 2},
            tile_updates={goose_position: {
                "kind": "COOP",
                "animal": "GOOSE",
                "placed_day": 0,
                "fed_today": False,
                "cared_today": False,
                "consecutive_unfed": 0,
                "yield_units": 0,
                "fertilizer_available": True,
                "pending_care_bonus": 0,
            }},
        )
        deadline = state.step + 20
        goose = _canonical_daily_commitment(
            state,
            _animal_commitment(
                state, "GOOSE", goose_position, existing=True,
                placed_day=0, fertilizer_available=True,
            ),
        )
        cow = _canonical_daily_commitment(
            state,
            _animal_commitment(
                state, "COW", cow_position, existing=False,
                needs_structure=True,
            ),
        )
        goose = replace(goose, time=replace(
            goose.time, deadlines=(deadline,)
        ))
        cow = replace(cow, time=replace(cow.time, deadlines=(deadline,)))
        window = EconomicWindow(
            7, 7,
            market_orders=(("SELL", "FERTILIZER", 1),
                           ("BUY_ANIMAL", "COW", 1)),
            required_shed={"FERTILIZER": 1},
        )
        plan = self._plan(
            state, obligations=(goose,), selected=(cow,), window=window
        )

        realization = solve_intraday(state, plan)
        ending = execute_realization(state, plan, realization)
        _, trace, money = self._trace(state, realization)
        self.assertTrue(all(value >= 0 for value in money))
        self.assertEqual(realization.turns[7].market_orders, (
            ("SELL", "FERTILIZER", 1),
            ("BUY_ANIMAL", "COW", 1),
        ))
        builds = [turn for turn, position, action in trace
                  if position == cow_position and action[0] == "BUILD_PASTURE"]
        places = [turn for turn, position, action in trace
                  if position == cow_position and action[0] == "PLACE"]
        mandatory = [turn for turn, _, action in trace
                     if action[0] in {"FEED", "WATER", "COLLECT_FERTILIZER",
                                      "BUILD_PASTURE", "PLACE", "CARE"}]
        self.assertLess(builds[0], 7)
        self.assertLessEqual(places[0], 19)
        self.assertLessEqual(max(mandatory), 20)
        self.assertEqual(ending.tile_at(cow_position).animal, "COW")

    def test_reservation_wheat_finances_strawberry_seed(self):
        position = (4, 3)
        wheat = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 0,
            "watered_today": False,
            "consecutive_unwatered": 0,
            "yield_units": 3,
            "max_lifespan_step": 5 * rules.TURNS_PER_DAY,
            "fertilized_until_day": -1,
        }
        state = self._state(money=25, shed={}, tile_updates={position: wheat})
        strawberry = _canonical_daily_commitment(
            state, _crop_commitment(state, "STRAWBERRY", state.tile_at(position))
        )
        strawberry = replace(
            strawberry,
            time=replace(strawberry.time, deadlines=(state.step + 20,)),
            required_outputs={"WHEAT": 3},
        )
        window = EconomicWindow(
            7, 7,
            market_orders=(("SELL", "WHEAT", 3),
                           ("BUY_SEED", "STRAWBERRY", 1)),
            required_shed={"WHEAT": 3},
        )
        plan = self._plan(
            state, selected=(strawberry,), window=window
        )

        realization = solve_intraday(state, plan)
        ending = execute_realization(state, plan, realization)
        _, trace, money = self._trace(state, realization)
        self.assertTrue(all(value >= 0 for value in money))
        self.assertEqual(realization.turns[7].market_orders, (
            ("SELL", "WHEAT", 3),
            ("BUY_SEED", "STRAWBERRY", 1),
        ))
        harvest = next(turn for turn, at, action in trace
                       if at == position and action[0] == "HARVEST")
        plant = next(turn for turn, at, action in trace
                     if at == position and action[0] == "PLANT")
        water = next(turn for turn, at, action in trace
                     if at == position and action[0] == "WATER")
        self.assertLess(harvest, 7)
        self.assertGreaterEqual(plant, 8)
        self.assertLessEqual(water, 20)
        self.assertEqual(ending.tile_at(position).raw["crop"], "STRAWBERRY")
        self.assertEqual(ending.tile_at(position).raw["consecutive_unwatered"], 0)


if __name__ == "__main__":
    unittest.main()
