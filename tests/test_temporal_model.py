"""Formulation/transition correctness, not planner-quality benchmarks."""
from dataclasses import asdict, replace
from importlib.util import find_spec
import unittest

from src.kaggriculture_agent import rules
from src.kaggriculture_agent.economics import (
    ActionDimension, CashDimension, EconomicCommitment, LandDimension,
    PhysicalDimension, RevenueDimension, TimeDimension, WorkAmount,
)
from src.kaggriculture_agent.intent import compile_intent, unit_event, EntityWork, IntentProblem, ServiceGoal
from src.kaggriculture_agent.market import realization_orders
from src.kaggriculture_agent.planner import Plan
from src.kaggriculture_agent.state import OwnedState, TileState, WorkerState
from src.kaggriculture_eval.plan_io import plan_from_dict


def state_at(hour=22, day=2, money=0, tiles=None, workers=None, shed=None, seeds=None):
    return OwnedState(day*24+hour, day, hour, 0, money, 10,
        tuple(TileState((x, y), (tiles or {}).get((x, y))) for y in range(10) for x in range(10)),
        tuple(WorkerState(i, p, inv) for i, (p, inv) in enumerate(workers or [((4, 4), {})])),
        ("NW", "NE", "SW", "SE"), len(workers or [0])-1, shed or {}, seeds or {},
        {i: 10000 for i in rules.PRODUCTS}, {i: rules.market_price(i, 10000) for i in rules.PRODUCTS}, ())


def crop(kind="CARROT", day=0, quantity=1):
    return dict(kind="PLANT", crop=kind, planted_day=day, watered_today=False,
                consecutive_unwatered=0, yield_units=quantity, fertilized_until_day=-1,
                max_lifespan_step=(day + rules.CROPS[kind].max_yield_day + 1)*24)


def work_plan(state, specifications, domains=None):
    projects = []
    for identifier, position, kinds, metadata in specifications:
        projects.append(EconomicCommitment(identifier, "WORK", position, position is not None,
            CashDimension(), TimeDimension(state.step, min(state.step+state.turns_left_today-1, 718), 718),
            LandDimension(), ActionDimension(tuple(WorkAmount(state.day, k, 1, position=position) for k in kinds)),
            PhysicalDimension(), RevenueDimension(), metadata))
    return Plan(tuple(projects), (), (), {}, frozenset(), (), False, 0, 0,
                day=state.day, formed_step=state.step, max_hands=None, placement_domains=domains or {})


class IntentTests(unittest.TestCase):
    def test_unit_projection_does_not_advance_clock_or_refresh(self):
        s = state_at(hour=23, tiles={(4, 4): crop()}, workers=[((4, 4), {"FERTILIZER": 1})])
        after, _, raw, delta = unit_event(s, 0, ["FERTILIZE"])
        self.assertEqual(after.step, s.step)
        self.assertEqual(after.market_inventory, s.market_inventory)
        self.assertEqual(raw["fertilized_until_day"], 4)
        self.assertEqual(delta, {"FERTILIZER": -1})
        self.assertEqual(s.workers[0].inventory, {"FERTILIZER": 1})

    def test_local_orders_are_causal_not_a_fixed_template(self):
        s = state_at(tiles={(4, 4): crop()})
        p = work_plan(s, [("crop", (4, 4), ("FERTILIZE", "WATER", "HARVEST"), {})])
        paths = compile_intent(s, p).entities[0].paths
        complete = {tuple(e.action[0] for e in path) for path in paths if len(path) == 3}
        self.assertEqual(complete, {("WATER", "FERTILIZE", "HARVEST"), ("FERTILIZE", "WATER", "HARVEST")})
        self.assertEqual(plan_from_dict(asdict(p)), p)

    def test_market_protects_inputs_and_reserves_required_entries(self):
        s = state_at(shed={i: 4 for i in rules.PRODUCTS})
        orders = realization_orders(s, (["PASS"],), [["BUY_SEED", "CARROT", 1]], 2,
                                    remaining_inputs={"WHEAT": 4})
        self.assertEqual(len(orders), 10)
        self.assertNotIn(["SELL", "WHEAT", 4], orders)
        self.assertEqual(orders[-3:], (["HIRE"], ["HIRE"], ["BUY_SEED", "CARROT", 1]))

    def test_effect_evaluator_does_not_double_credit_a_service(self):
        from src.kaggriculture_eval.realization_benchmark import score_effects
        s = state_at(tiles={(4, 4): crop()})
        p = work_plan(s, [])
        goals = tuple(ServiceGoal(k, "e", ("WATER",), {}, {}, 71) for k in ("a", "b"))
        problem = IntentProblem(s, p, (EntityWork("e", ((4, 4),), crop(), True, goals, ()),), ())
        following, before, after, delta = unit_event(s, 0, ["WATER"])
        effect = dict(achieved=True, position=(4, 4), step=s.step, before=before, after=after,
                      physical_delta=delta, action=["WATER"])
        result = score_effects(problem, [effect], {}, following)
        self.assertEqual(result["completed"], 1)
        self.assertEqual(len(result["unfulfilled"]), 1)

    def test_effect_evaluator_rejects_out_of_domain_placement(self):
        from src.kaggriculture_eval.realization_benchmark import score_effects
        s = state_at()
        p = work_plan(s, [("a", None, ("PLANT",), {"crop": "CARROT"})], {"a": ((3, 4),)})
        with self.assertRaisesRegex(ValueError, "outside fixed Plan"):
            score_effects(compile_intent(s, p), [], {"new:a": (4, 4)}, s)


@unittest.skipUnless(find_spec("ortools"), "optional constraint research runtime")
class TemporalCorrectnessTests(unittest.TestCase):
    def solve(self, s, p, **kwargs):
        from src.kaggriculture_agent.temporal_model import solve_temporal, TemporalConfig
        r = solve_temporal(s, p, TemporalConfig(deterministic_time=1, **kwargs))
        self.assertFalse(r.diagnostics.get("planning_failure"), r.diagnostics)
        self.assertEqual(r.diagnostics["refinement_failures"], [])
        return r

    def test_watering_harvest_and_automatic_drop(self):
        s = state_at(tiles={(4, 4): crop()})
        p = work_plan(s, [("crop", (4, 4), ("WATER", "HARVEST"), {})])
        r = self.solve(s, p)
        self.assertEqual(len(r.completed), 2)
        self.assertEqual([e.worker_actions[0] for e in r.executions], [["WATER"], ["HARVEST"]])
        self.assertEqual(r.final_state.shed.get("CARROT"), 2)

    def test_global_atomic_seed_constraint(self):
        positions = [(4, 4), (3, 4), (2, 4)]
        s = state_at(hour=23, workers=[(p, {}) for p in positions], seeds={"CARROT": 1})
        p = work_plan(s, [(str(n), pos, ("PLANT",), {"crop": "CARROT"}) for n, pos in enumerate(positions)])
        r = self.solve(s, p)
        self.assertEqual(len(r.completed), 1)
        self.assertEqual(len(r.unfulfilled), 2)

    def test_shared_pickup_and_same_turn_feed(self):
        animal = dict(kind="COOP", animal="GOOSE", placed_day=0, fed_today=False,
            cared_today=False, consecutive_unfed=0, fertilizer_available=False, pending_care_bonus=0, yield_units=0)
        s = state_at(hour=22, tiles={(4, 4): animal, (5, 4): animal},
            workers=[((4, 4), {}), ((5, 4), {})], shed={"WHEAT": 2})
        p = work_plan(s, [(str(n), pos, ("FEED",), {}) for n, pos in enumerate(((4, 4), (5, 4)))])
        r = self.solve(s, p)
        self.assertEqual(len(r.completed), 2)
        self.assertEqual(r.final_state.owned_total("WHEAT"), 0)

    def test_terminal_has_no_automatic_drop(self):
        s = state_at(hour=22, day=29, workers=[((0, 0), {"CARROT": 3})])
        r = self.solve(s, work_plan(s, []))
        self.assertEqual(r.final_state.step, 719)
        self.assertEqual(r.final_state.workers[0].inventory, {"CARROT": 3})
        self.assertEqual(r.final_state.money, 0)

    def test_drop_overflow_retains_original_insertion_order(self):
        s = state_at(hour=23, shed={"GOOSE": 99}, workers=[((0, 0), {"CARROT": 3, "WHEAT": 2})])
        r = self.solve(s, work_plan(s, []))
        self.assertEqual(r.final_state.shed.get("CARROT"), 1)
        self.assertEqual(r.final_state.shed.get("WHEAT", 0), 0)

    def test_hiring_is_joint_with_pickup_and_service_time(self):
        animal = dict(kind="COOP", animal="GOOSE", placed_day=0, fed_today=False,
            cared_today=False, consecutive_unfed=1, fertilizer_available=False, pending_care_bonus=0, yield_units=0)
        s = state_at(hour=21, money=2, tiles={(4, 4): animal}, workers=[((0, 0), {})], shed={"WHEAT": 1})
        p = work_plan(s, [("animal", (4, 4), ("FEED", "CARE"), {})])
        r = self.solve(s, p)
        self.assertEqual(len(r.completed), 2)
        self.assertIn(["HIRE"], r.executions[0].market_orders)
        self.assertEqual(r.final_state.tile_at((4, 4)).raw["pending_care_bonus"], 1)

    def test_open_placements_and_worker_ordered_build_place(self):
        s = state_at(hour=22, workers=[((3, 4), {}), ((3, 4), {"GOOSE": 1})])
        p = work_plan(s, [("new", None, ("BUILD", "PICKUP_PLACE"), {"structure": "COOP", "animal": "GOOSE"})],
                      {"new": ((0, 0), (3, 4), (4, 4))})
        r = self.solve(s, p)
        self.assertEqual(len(r.completed), 2)
        position = r.placements["new:new"]
        self.assertIn(position, ((3, 4), (4, 4)))
        self.assertEqual(r.final_state.tile_at(position).animal, "GOOSE")

    def test_fertilizer_yield_and_clock_sensitive_decay(self):
        s = state_at(hour=21, tiles={(4, 4): crop("WHEAT", quantity=2)},
                     workers=[((4, 4), {"FERTILIZER": 1})])
        p = work_plan(s, [("crop", (4, 4), ("FERTILIZE", "WATER", "HARVEST"), {})])
        r = self.solve(s, p)
        self.assertEqual(len(r.completed), 3)
        self.assertEqual(r.final_state.shed.get("WHEAT"), 4)
        dying = crop("WHEAT", quantity=3)
        dying["max_lifespan_step"] = 140
        s = state_at(day=5, hour=21, tiles={(4, 4): dying}, workers=[((3, 4), {})])
        r = self.solve(s, work_plan(s, [("crop", (4, 4), ("HARVEST",), {})]))
        self.assertEqual(len(r.completed), 1)
        # Move at 141, harvest at 142 before that turn's next decay.
        self.assertEqual(r.final_state.shed.get("WHEAT"), 3)

    def test_land_is_a_costed_goal_with_next_turn_availability(self):
        s = state_at(hour=22, money=1000, tiles={(5, 4): "LOCKED"}, workers=[((5, 4), {})])
        s = replace(s, unlocked_quadrants=("NW",))
        p = work_plan(s, [("structure", None, ("BUILD",), {"structure": "COOP"})], {"structure": ((5, 4),)})
        land = EconomicCommitment("land:NE", "LAND", None, False, CashDimension(),
            TimeDimension(s.step, s.step+1, s.step+1), LandDimension(capacity_created=25),
            ActionDimension(), PhysicalDimension(), RevenueDimension(), {"quadrant": "NE"})
        p = replace(p, support=(land,), buy_land=True)
        r = self.solve(s, p)
        self.assertIn("NE", r.final_state.unlocked_quadrants)
        self.assertEqual(r.final_state.money, 0)
        self.assertEqual(r.final_state.tile_at((5, 4)).kind, "COOP")

    def test_official_trajectory_parity(self):
        from src.kaggriculture_eval.intraday_benchmark import oracle_environment, oracle_step
        from src.kaggriculture_agent.intraday import farm_key
        s = state_at(hour=20, tiles={(4, 4): crop()}, workers=[((4, 4), {"FERTILIZER": 1})])
        p = work_plan(s, [("crop", (4, 4), ("FERTILIZE", "WATER", "HARVEST"), {})])
        r = self.solve(s, p)
        env = oracle_environment(s)
        expected = s
        for e in r.executions:
            expected = rules.advance_owned(expected, e.worker_actions, e.market_orders)
            actual = oracle_step(env, e.worker_actions, e.market_orders)
            self.assertEqual(farm_key(actual), farm_key(expected))
            self.assertEqual(actual.money, expected.money)
            self.assertEqual(actual.market_inventory, expected.market_inventory)

    def test_retention_and_repair_preserve_plan_and_completed_work(self):
        from src.kaggriculture_agent.temporal_session import TemporalSession
        from src.kaggriculture_agent.temporal_model import TemporalConfig
        s = state_at(hour=21, tiles={(4, 4): crop()})
        p = work_plan(s, [("crop", (4, 4), ("WATER", "HARVEST"), {})])
        frozen = asdict(p)
        session = TemporalSession(TemporalConfig(deterministic_time=1))
        a = session.execution_for(s, p)
        self.assertEqual(a, session.execution_for(s, p))
        nxt = rules.advance_owned(s, a.worker_actions, a.market_orders)
        session.execution_for(nxt, p)
        self.assertEqual(len(session.diagnostics), 1)
        # A routing perturbation must not rewrite production or retry WATER.
        altered = replace(nxt, workers=(replace(nxt.workers[0], position=(3, 4)),))
        session.execution_for(altered, p)
        self.assertEqual(session.diagnostics[-1]["reason"], "repair")
        self.assertEqual(asdict(p), frozen)

    def test_seed_witness_respects_equivalent_asset_label_symmetry(self):
        s = state_at(hour=20, workers=[((4, 4), {}), ((3, 4), {})], seeds={"CARROT": 2})
        p = work_plan(s, [(key, None, ("PLANT", "WATER"), {"crop": "CARROT"}) for key in ("a", "b")],
                      {key: ((3, 4), (4, 4)) for key in ("a", "b")})
        r = self.solve(s, p)
        self.assertTrue(r.diagnostics["initialization"]["validated"])
        self.assertEqual(len(r.completed), 4)

    def test_sale_truncation_does_not_delay_required_acquisitions(self):
        s = state_at(hour=22, workers=[((4, 4), {}), ((3, 4), {})], shed={i: 1 for i in rules.PRODUCTS})
        p = work_plan(s, [("a", (4, 4), ("PLANT",), {"crop": "CARROT"}),
                          ("b", (3, 4), ("PLANT",), {"crop": "WHEAT"})])
        r = self.solve(s, p)
        self.assertEqual(len(r.completed), 2)
        first = r.executions[0].market_orders
        self.assertIn(["BUY_SEED", "CARROT", 1], first)
        self.assertIn(["BUY_SEED", "WHEAT", 1], first)
        self.assertLessEqual(len(first), 10)

    def test_acquisition_is_not_misreported_as_task_completion(self):
        from src.kaggriculture_agent.temporal_model import TemporalModel, validate_temporal_solution
        animal = dict(kind="COOP", animal="GOOSE", placed_day=0, fed_today=False,
            cared_today=False, consecutive_unfed=0, fertilizer_available=False, pending_care_bonus=0, yield_units=0)
        s = state_at(hour=23, money=100, tiles={(0, 0): animal})
        p = work_plan(s, [("a", (0, 0), ("FEED",), {})])
        m = TemporalModel(s, p)
        m.model.add(m.buy["WHEAT", 0] == 1)
        solver = m.cp.CpSolver()
        solver.parameters.num_search_workers = 1
        self.assertIn(solver.solve(m.model), (m.cp.OPTIMAL, m.cp.FEASIBLE))
        result, failure = validate_temporal_solution(m, solver, m.decode(solver))
        self.assertIsNone(failure)
        self.assertEqual(result.final_state.shed.get("WHEAT"), 1)
        self.assertEqual(len(result.unfulfilled), 1)


if __name__ == "__main__":
    unittest.main()
