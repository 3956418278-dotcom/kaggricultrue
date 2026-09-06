"""Execution quality regressions and differential tests against the rule oracle."""
import copy
from dataclasses import replace
import unittest

from kaggle_environments.envs.kaggriculture import kaggriculture as official
from src.kaggriculture_agent import rules
from src.kaggriculture_agent.intraday import IntradaySession, farm_key, search_day
from src.kaggriculture_agent.operating import DailyPlanningSession
from src.kaggriculture_agent.planner import PlannerConfig, make_plan
from src.kaggriculture_agent.realization import ExecutionChoices, legacy_choices
from src.kaggriculture_agent.state import TileState, WorkerState, reconstruct
from src.kaggriculture_eval.intraday_benchmark import (
    oracle_environment, oracle_step, scenarios, run_scenario,
)


class IntradayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = {name: (state, plan) for name, state, plan in scenarios()}
        cls.results = {name: run_scenario(name, state, plan)
                       for name, (state, plan) in cls.fixtures.items()}

    def test_clusters_complete_chains_instead_of_repeated_shed_visits(self):
        r = self.results["clustered-harvest"]
        self.assertEqual(r["search"]["maintenance_debt"], 0)
        self.assertEqual(r["search"]["crops"], 0)
        self.assertGreater(r["search"]["economic_value"], r["greedy"]["economic_value"] + 1000)
        self.assertLess(r["search"]["movement"], r["greedy"]["movement"])

    def test_distributed_work_and_competing_routes_improve_reachable_state(self):
        for name in ("dispersed-maintenance", "competing-routes"):
            with self.subTest(name=name):
                r = self.results[name]
                self.assertEqual(r["search"]["maintenance_debt"], 0)
                # The shared reactive-market repair also improves the weak
                # benchmark. These legacy cases are sanity checks, not the
                # reference-driven maturity standard.
                self.assertGreater(r["search"]["economic_value"], r["greedy"]["economic_value"])
                self.assertLess(r["search"]["movement"], r["greedy"]["movement"])

    def test_mixed_work_and_carry_divisions_reduce_waste_at_equal_value(self):
        for name in ("mixed-maintenance", "carry-chains"):
            r = self.results[name]
            self.assertGreaterEqual(r["search"]["economic_value"], r["greedy"]["economic_value"])
            self.assertLessEqual(r["search"]["movement"], r["greedy"]["movement"] - 5)
            self.assertEqual(r["search"]["maintenance_debt"], 0)

    def test_terminal_search_banks_more_cash_not_just_carried_output(self):
        r = self.results["terminal-sale"]
        self.assertGreater(r["search"]["cash"], r["greedy"]["cash"] + 100)
        self.assertEqual(r["search"]["carried"] + r["search"]["shed_units"], 0)

    def test_staffing_is_chosen_by_full_day_outcome_before_work(self):
        state, plan = self.fixtures["hiring-unlocks-work"]
        result = search_day(state, plan)
        self.assertGreater(result.choices.hire_count, 0)
        self.assertLess(result.choices.hire_count, plan.max_hands)
        self.assertEqual(sum(order[0] == "HIRE" for order in result.executions[0].market_orders), result.choices.hire_count)
        from src.kaggriculture_agent.planner import hiring_commitments
        hires = hiring_commitments(state, result.choices.hire_count)
        self.assertEqual(len(hires), result.choices.hire_count)
        self.assertTrue(all(p.actions.capacity_supplied[state.day] == 23 for p in hires))
        self.assertEqual(self.results["hiring-unlocks-work"]["search"]["maintenance_debt"], 0)
        # Extra hands cannot repay their cost when there is no work.
        empty = replace(state, tiles=tuple(TileState(t.position, None) for t in state.tiles))
        empty_plan = make_plan(empty, PlannerConfig(cash_reserve=10**9, max_daily_hands=3))
        self.assertEqual(search_day(empty, empty_plan).choices.hire_count, 0)

    def test_persistent_sites_are_execution_choices_not_daily_targets(self):
        state, plan = self.fixtures["persistent-placement"]
        self.assertFalse(hasattr(plan, "crop_targets"))
        for p in (*plan.selected, *plan.obligations):
            if p.identifier in plan.placement_domains:
                self.assertIsNone(p.target)
                self.assertTrue(all(i.position is None for i in p.land.intervals))
                self.assertTrue(all(w.position is None for w in p.actions.work))
        before = copy.deepcopy(plan)
        a, b = legacy_choices(state, plan, 0), legacy_choices(state, plan, 1)
        self.assertNotEqual(a.crop_targets(plan), b.crop_targets(plan))
        self.assertEqual(plan, before)
        r = self.results["persistent-placement"]
        self.assertGreaterEqual(r["search"]["economic_value"], r["greedy"]["economic_value"])
        self.assertLess(r["search"]["movement"], r["greedy"]["movement"])
        chosen = search_day(state, plan).choices
        animal = next(p for p in plan.obligations if p.kind == "ANIMAL_PLACEMENT")
        self.assertEqual(rules.distance_to_shed(chosen.target(animal)), 0)

    def test_production_count_is_not_capped_at_three(self):
        state, _ = self.fixtures["persistent-placement"]
        plan = make_plan(replace(state, shed={}, seeds={}))
        self.assertGreater(len(plan.selected), 3)

    def test_trajectory_is_retained_and_repaired_without_rewriting_intent(self):
        state, plan = self.fixtures["carry-chains"]
        session = IntradaySession()
        first = session.execution_for(state, plan)
        self.assertEqual(first, session.execution_for(copy.deepcopy(state), plan))
        self.assertEqual(len(session.diagnostics), 1)
        following = rules.advance_owned(state, first.worker_actions, first.market_orders)
        session.execution_for(following, plan)
        self.assertEqual(len(session.diagnostics), 1)
        moved = replace(following, workers=(replace(following.workers[0], position=(9, 9)), *following.workers[1:]))
        session.execution_for(moved, plan)
        self.assertEqual(len(session.diagnostics), 2)
        self.assertEqual(session.diagnostics[-1]["reason"], "divergence")
        self.assertIs(session.trajectories[state.player][2], plan)
        _, _, retained = session.trajectories[state.player]
        self.assertEqual(retained, plan)

    def test_cash_invalidation_is_repaired_before_failed_input_purchase(self):
        state, plan = self.fixtures["carry-chains"]
        session = IntradaySession()
        session.execution_for(state, plan)
        session.execution_for(replace(state, money=0), plan)
        self.assertEqual(session.diagnostics[-1]["reason"], "divergence")
        self.assertEqual(len(session.diagnostics), 2)

    def test_failed_unbuilt_location_can_be_rebound_without_changing_goals(self):
        state, plan = self.fixtures["persistent-placement"]
        bound = legacy_choices(state, plan)
        target = next(iter(bound.crop_targets(plan)))
        tiles = list(state.tiles)
        tiles[target[1] * 10 + target[0]] = TileState(target, {"kind": "WEED"})
        repaired = legacy_choices(replace(state, tiles=tuple(tiles)), plan, previous=bound)
        self.assertNotIn(target, repaired.crop_targets(plan))
        self.assertEqual(sorted(repaired.crop_targets(plan).values()), sorted(bound.crop_targets(plan).values()))
        self.assertEqual(set(repaired.placements), set(bound.placements))

    def test_next_day_has_fresh_economic_and_intraday_plans(self):
        state, _ = self.fixtures["carry-chains"]
        session = DailyPlanningSession(config=PlannerConfig(cash_reserve=10**9, max_daily_hands=1))
        daily = session.plan_for(state)
        session.execution_for(state, daily)
        next_day = rules.advance_owned(replace(state, step=state.day * 24 + 23, hour=23), ())
        following = session.plan_for(next_day)
        self.assertIsNot(following, daily)
        session.execution_for(next_day, following)
        self.assertEqual(session._intraday.diagnostics[-1]["reason"], "daily")

    def test_repeated_fresh_search_is_action_deterministic(self):
        state, plan = self.fixtures["clustered-harvest"]
        a, b = search_day(state, plan), search_day(copy.deepcopy(state), copy.deepcopy(plan))
        self.assertEqual(a.executions, b.executions)
        self.assertEqual(a.value, b.value)


class TransitionParityTests(unittest.TestCase):
    def check(self, state, actions, orders=()):
        before = copy.deepcopy(state)
        predicted = rules.advance_owned(state, actions, orders)
        actual = oracle_step(oracle_environment(state), actions, orders)
        self.assertEqual(state, before, "search must not mutate its observation or siblings")
        self.assertEqual(farm_key(actual), farm_key(predicted))
        self.assertEqual(actual.money, predicted.money)
        self.assertEqual(actual.market_inventory, predicted.market_inventory)
        self.assertEqual(actual.market_prices, predicted.market_prices)

    def test_atomic_plant_and_unit_before_purchase_and_hire(self):
        _, state, _ = next(scenarios())
        state = replace(state, step=0, day=0, hour=0, seeds={"WHEAT": 2},
            workers=(WorkerState(0, (4, 4), {}), WorkerState(1, (4, 3), {})), hires_today=1)
        self.check(state, (["PLANT", "WHEAT"], ["PLANT", "WHEAT"]), (["HIRE"], ["BUY_SEED", "WHEAT", 1]))
        self.check(replace(state, seeds={"WHEAT": 1}), (["PLANT", "WHEAT"], ["PLANT", "WHEAT"]), (["BUY_SEED", "WHEAT", 1],))

    def test_daily_overflow_inventory_order_hands_and_spawn(self):
        _, state, _ = next(scenarios())
        state = replace(state, step=311, day=12, hour=23, shed={"WHEAT": 98},
            workers=(WorkerState(0, (0, 0), {"CARROT": 3, "MELON": 4}),
                     WorkerState(1, (9, 9), {"EGG": 3})), hires_today=1)
        self.check(state, (["PASS"], ["PASS"]))

    def test_feed_care_production_fertilizer_and_decay(self):
        fixtures = {name: (s, p) for name, s, p in scenarios()}
        state, _ = fixtures["mixed-maintenance"]
        state = replace(state, hour=23, step=311,
            workers=(WorkerState(0, (2, 3), {"WHEAT": 1}), WorkerState(1, (3, 2), {"FERTILIZER": 1})))
        self.check(state, (["FEED"], ["FERTILIZE"]))
        state = replace(state, step=312, day=13, hour=0)
        self.check(state, (["CARE"], ["WATER"]))

    def test_terminal_does_not_drop_carried_inventory(self):
        _, state, _ = next(scenarios())
        self.check(replace(state, step=718, day=29, hour=22,
            workers=(WorkerState(0, (0, 0), {"MELON": 2}),)), (["PASS"],))

    def test_land_known_duplicate_shop_demand_and_negative_inventory(self):
        _, state, _ = next(scenarios())
        tiles = tuple(TileState(t.position, None if rules.quadrant(t.position) == "NW" else "LOCKED") for t in state.tiles)
        state = replace(state, step=24, day=1, hour=0, tiles=tiles,
            unlocked_quadrants=("NW",), unlocked_shops=("PET_CAFE", "PET_CAFE"),
            market_inventory={item: -10 for item in rules.PRODUCTS})
        self.check(state, (["PASS"],), (["BUY_LAND"],))
        reconstructed = reconstruct(oracle_environment(state).state[0].observation)
        self.assertEqual(reconstructed.market_inventory["WHEAT"], -10)


if __name__ == "__main__":
    unittest.main()
