"""Correctness checks for event-route representation and compilation."""
from dataclasses import replace
from random import Random
from types import SimpleNamespace
import unittest

from src.kaggriculture_agent import rules
from src.kaggriculture_agent.economics import (
    ActionDimension, CashDimension, EconomicCommitment, LandDimension,
    PhysicalDimension, RevenueDimension, TimeDimension, WorkAmount,
)
from src.kaggriculture_agent.planner import Plan
from src.kaggriculture_agent.route_compiler import compile_skeleton
from src.kaggriculture_agent.route_structure import (
    LogisticsEvent, ResourceLink, RouteSkeleton, SyncBundle, TileLease, build_route_problem,
    initial_skeleton, route_precedence_feasible, selected_paths,
    validate_skeleton, with_initial_logistics,
)
from src.kaggriculture_agent.route_search import (
    RouteSearchConfig, _complete_initial_placement, _compress_workforce_frontier,
    _mutate, _reconstruct_open_placement_matching,
    _ruin_recreate, normalize, result_score, solve_routes,
)
from src.kaggriculture_agent.state import OwnedState, TileState, WorkerState


def state_at(hour=20, day=2, money=0, tiles=None, workers=None, shed=None, seeds=None):
    workers = workers or [((4, 4), {})]
    return OwnedState(day*24+hour, day, hour, 0, money, 10,
        tuple(TileState((x, y), (tiles or {}).get((x, y))) for y in range(10) for x in range(10)),
        tuple(WorkerState(i, p, inv) for i, (p, inv) in enumerate(workers)),
        ("NW", "NE", "SW", "SE"), len(workers)-1, shed or {}, seeds or {},
        {i: 10000 for i in rules.PRODUCTS}, {i: rules.market_price(i, 10000) for i in rules.PRODUCTS}, ())


def crop(kind="WHEAT", planted=0, quantity=1):
    rule = rules.CROPS[kind]
    return dict(kind="PLANT", crop=kind, planted_day=planted, watered_today=False,
        consecutive_unwatered=0, yield_units=quantity, fertilized_until_day=-1,
        max_lifespan_step=(planted+rule.max_yield_day+1)*24)


def work_plan(state, specifications, domains=None):
    projects = []
    for identifier, position, kinds, metadata in specifications:
        projects.append(EconomicCommitment(identifier, "WORK", position, position is not None,
            CashDimension(), TimeDimension(state.step, min(state.step+state.turns_left_today-1, 718), 718),
            LandDimension(), ActionDimension(tuple(WorkAmount(state.day, kind, 1, position=position) for kind in kinds)),
            PhysicalDimension(), RevenueDimension(), metadata))
    return Plan(tuple(projects), (), (), {}, frozenset(), (), False, 0, 0,
        day=state.day, formed_step=state.step, max_hands=None, placement_domains=domains or {})


def action_goals(problem, entity):
    path = problem.complete_paths[entity][0]
    return {event.action[0]: event.goal for event in path}


class RouteCompilerTests(unittest.TestCase):
    def test_market_ledger_attributes_real_hire_input_land_and_sale_cash(self):
        state = replace(state_at(hour=20, money=2000, shed={"FERTILIZER": 1}),
                        unlocked_quadrants=("NW",))
        orders = (["SELL", "FERTILIZER", 1], ["HIRE"], ["HIRE"],
                  ["BUY_SEED", "WHEAT", 2], ["BUY_LAND"])
        ledger = rules.solo_market_ledger(state, orders)
        following = rules.advance_owned(state, (["PASS"],), orders)
        self.assertEqual(rules.hire_expenditure(0, 2), 2)
        self.assertEqual(rules.hire_expenditure(3, 2), 8)
        self.assertEqual(ledger["hire_expenditure"], 2)
        self.assertEqual(ledger["input_expenditure"], 20)
        self.assertEqual(ledger["land_expenditure"], 1000)
        self.assertEqual(ledger["sale_revenue"], 100)
        self.assertEqual(ledger["executed_hires"], 2)
        self.assertEqual(ledger["ending_cash"], following.money)

    def test_same_turn_ordered_workers_can_build_then_place(self):
        state = state_at(hour=23, workers=[((3, 4), {}), ((3, 4), {"GOOSE": 1})])
        plan = work_plan(state, [("asset", (3, 4), ("BUILD", "PICKUP_PLACE"),
                                  {"structure": "COOP", "animal": "GOOSE"})])
        problem = build_route_problem(state, plan)
        entity = problem.intent.entities[0].identifier
        goals = action_goals(problem, entity)
        skeleton = RouteSkeleton(2, ((goals["BUILD_COOP"],), (goals["PLACE"],)),
            {entity: 0}, {entity: (3, 4)},
            (ResourceLink(goals["PLACE"], "GOOSE", 1, "CARRY", source_worker=1),),
            (TileLease(entity, (3, 4)),),
            (SyncBundle((goals["BUILD_COOP"], goals["PLACE"])),))
        result = compile_skeleton(problem, skeleton)
        self.assertEqual(len(result.completed), 2)
        self.assertEqual(result.executions[0].worker_actions,
                         (["BUILD_COOP"], ["PLACE", "GOOSE"]))
        self.assertEqual(result.final_state.tile_at((3, 4)).animal, "GOOSE")

    def test_temporal_lease_allows_harvest_then_replant(self):
        state = state_at(hour=20, money=20, tiles={(4, 4): crop()})
        plan = work_plan(state, [
            ("old", (4, 4), ("HARVEST",), {}),
            ("new", None, ("PLANT", "WATER"), {"crop": "WHEAT"}),
        ], {"new": ((4, 4),)})
        old, new = plan.obligations
        plan = replace(plan, obligations=(old,), selected=(replace(new, kind="CROP"),))
        problem = build_route_problem(state, plan)
        skeleton = initial_skeleton(problem, workforce=1)
        result = compile_skeleton(problem, skeleton)
        self.assertEqual(len(result.completed), 3, result.diagnostics)
        self.assertEqual(result.final_state.tile_at((4, 4)).raw["crop"], "WHEAT")
        self.assertEqual(result.final_state.tile_at((4, 4)).raw["consecutive_unwatered"], 0)
        leases = {lease.entity: lease for lease in skeleton.leases}
        new_entity = next(e.identifier for e in problem.intent.entities if not e.existing)
        self.assertIsNotNone(leases[new_entity].available_after)

    def test_cross_worker_output_uses_explicit_shed_transfer(self):
        animal = dict(kind="COOP", animal="GOOSE", placed_day=0, fed_today=False,
            cared_today=False, consecutive_unfed=0, fertilizer_available=False,
            pending_care_bonus=0, yield_units=0)
        state = state_at(hour=17, tiles={(4, 4): crop(), (6, 4): animal},
                         workers=[((4, 4), {}), ((6, 4), {})])
        plan = work_plan(state, [("crop", (4, 4), ("HARVEST",), {}),
                                 ("animal", (6, 4), ("FEED",), {})])
        problem = build_route_problem(state, plan)
        entities = {e.positions[0]: e.identifier for e in problem.intent.entities}
        producer = action_goals(problem, entities[(4, 4)])["HARVEST"]
        consumer = action_goals(problem, entities[(6, 4)])["FEED"]
        skeleton = RouteSkeleton(2, ((producer,), (consumer,)),
            {entity: 0 for entity in entities.values()},
            {entity: pos for pos, entity in entities.items()},
            (ResourceLink(consumer, "WHEAT", 1, "EVENT", producer=producer, via_shed=True),),
            tuple(TileLease(entity, pos) for pos, entity in entities.items()))
        skeleton = with_initial_logistics(problem, skeleton)
        result = compile_skeleton(problem, skeleton)
        self.assertEqual(len(result.completed), 2, result.diagnostics)
        self.assertGreaterEqual(result.diagnostics["logistics"], 2)

    def test_unroutable_goal_is_reported_not_removed_from_plan(self):
        state = state_at(hour=23, tiles={(0, 0): crop()}, workers=[((9, 9), {})])
        plan = work_plan(state, [("crop", (0, 0), ("WATER",), {})])
        problem = build_route_problem(state, plan)
        entity = problem.intent.entities[0].identifier
        skeleton = RouteSkeleton(1, ((),), {entity: 0}, {entity: (0, 0)}, (),
                                 (TileLease(entity, (0, 0)),))
        result = compile_skeleton(problem, skeleton)
        self.assertEqual(len(result.completed), 0)
        self.assertEqual(result.unfulfilled, frozenset(problem.goals))

    def test_search_is_deterministic_and_does_not_overhire_equivalent_capacity(self):
        state = state_at(hour=20, money=20, tiles={(4, 4): crop()})
        plan = work_plan(state, [("old", (4, 4), ("HARVEST",), {}),
                                 ("new", None, ("PLANT", "WATER"), {"crop": "WHEAT"})],
                         {"new": ((4, 4),)})
        old, new = plan.obligations
        plan = replace(plan, obligations=(old,), selected=(replace(new, kind="CROP"),))
        config = RouteSearchConfig(iterations=30, population=6)
        first, second = solve_routes(state, plan, config), solve_routes(state, plan, config)
        self.assertEqual(first.executions, second.executions)
        self.assertEqual(first.final_state, second.final_state)
        self.assertEqual(first.diagnostics["workforce"], 1)
        self.assertEqual(first.diagnostics["hire_expenditure"], 0)
        self.assertEqual(len(first.completed), 3)
        basins = first.diagnostics["search_workforce_basins"]
        self.assertEqual([basin["workforce"] for basin in basins], [1, 2, 3])
        self.assertEqual(first.diagnostics["search_global_iterations"]
                         + sum(basin["structural_iterations"] for basin in basins), 30)
        self.assertTrue(all(basin["structural_iterations"] >= 3 for basin in basins))
        self.assertTrue(all(basin["exact_compilations"] >= 1 for basin in basins))

    def test_exact_score_values_reached_state_without_ledger_double_counting(self):
        state = state_at(hour=20, money=20)
        diagnostics = {"hire_expenditure": 0, "used_worker_turns": 2,
                       "movement": 1, "logistics": 0, "workforce": 1}
        baseline = SimpleNamespace(completed=frozenset({"goal"}), final_state=state,
                                   diagnostics=diagnostics)
        misleading_ledger = SimpleNamespace(
            completed=baseline.completed, final_state=state,
            diagnostics={**diagnostics, "hire_expenditure": 10_000, "workforce": 20})
        self.assertEqual(result_score(baseline, state),
                         result_score(misleading_ledger, state))
        richer = SimpleNamespace(completed=baseline.completed,
                                 final_state=replace(state, money=state.money+1),
                                 diagnostics={**diagnostics, "hire_expenditure": 1})
        self.assertGreater(result_score(richer, state), result_score(baseline, state))

    def test_hire_spawn_layout_can_defer_logistics_without_losing_it(self):
        animal = dict(kind="COOP", animal="GOOSE", placed_day=0, fed_today=False,
            cared_today=False, consecutive_unfed=0, fertilizer_available=False,
            pending_care_bonus=0, yield_units=0)
        state = state_at(hour=20, money=20, shed={"WHEAT": 1},
                         tiles={(4, 4): animal})
        plan = work_plan(state, [("animal", (4, 4), ("FEED",), {})])
        problem = build_route_problem(state, plan)
        entity = problem.intent.entities[0].identifier
        goal = action_goals(problem, entity)["FEED"]
        pickup = LogisticsEvent("pickup", "PICKUP", "WHEAT", 1)
        skeleton = RouteSkeleton(2, (("pickup", goal), ()), {entity: 0},
            {entity: (4, 4)}, (ResourceLink(goal, "WHEAT", 1, "SHED"),),
            (TileLease(entity, (4, 4)),),
            spawn_preferences=((4, 4),), market_priority=("HIRE:0",),
            logistics={"pickup": pickup}, required_entry_caps=(1,))
        result = compile_skeleton(problem, skeleton)
        first = result.executions[0]
        self.assertIn(first.worker_actions[0][0], {"NORTH", "SOUTH", "EAST", "WEST"})
        self.assertEqual(result.expected[1].workers[1].position, (4, 4))
        self.assertTrue(any(execution.worker_actions[0][0] == "PICKUP"
                            for execution in result.executions[1:]))

    def test_required_market_cap_preserves_sale_slot_before_hiring(self):
        state = state_at(hour=20, money=1000, shed={"FERTILIZER": 1})
        problem = build_route_problem(state, work_plan(state, []))
        skeleton = RouteSkeleton(10, tuple(() for _ in range(10)), {}, {}, (), (),
            spawn_preferences=tuple(rules.shed_access(10)[n % 4] for n in range(9)),
            market_priority=tuple(f"HIRE:{n}" for n in range(9)),
            required_entry_caps=(8,))
        result = compile_skeleton(problem, skeleton)
        orders = result.executions[0].market_orders
        self.assertEqual(sum(order[0] == "HIRE" for order in orders), 8)
        self.assertEqual(sum(order[0] == "SELL" for order in orders), 1)
        self.assertEqual(result.diagnostics["executed_hires"], 9)
        self.assertEqual(result.diagnostics["hire_expenditure"], 88)
        self.assertEqual(result.diagnostics["hire_timing"][0],
                         {"step": state.step, "count": 8, "expenditure": 54})
        staged = compile_skeleton(problem, replace(skeleton, hire_caps=(3,)))
        self.assertEqual(sum(order[0] == "HIRE"
                             for order in staged.executions[0].market_orders), 3)

    def test_ruin_recreate_changes_several_assignments_without_losing_work(self):
        positions = ((0, 0), (0, 4), (4, 0), (4, 4), (8, 0),
                     (8, 4), (0, 8), (4, 8), (8, 8))
        state = state_at(hour=8, tiles={position: crop() for position in positions},
                         workers=[((4, 4), {}), ((5, 4), {}), ((4, 5), {})])
        plan = work_plan(state, [(f"crop-{n}", position, ("WATER",), {})
                                 for n, position in enumerate(positions)])
        problem = build_route_problem(state, plan)
        opening = initial_skeleton(problem, workforce=3)
        first = normalize(problem, _ruin_recreate(problem, opening, Random(16)))
        second = normalize(problem, _ruin_recreate(problem, opening, Random(16)))
        self.assertEqual(first.routes, second.routes)
        before = {goal: worker for worker, route in enumerate(opening.routes) for goal in route}
        after = {goal: worker for worker, route in enumerate(first.routes)
                 for goal in route if goal in before}
        self.assertEqual(set(after), set(before))
        self.assertGreaterEqual(sum(before[goal] != after[goal] for goal in before), 2)

    def test_normalize_preserves_first_class_realization_choices(self):
        animal = dict(kind="COOP", animal="GOOSE", placed_day=0, fed_today=False,
            cared_today=False, consecutive_unfed=0, fertilizer_available=False,
            pending_care_bonus=0, yield_units=0)
        state = state_at(hour=18, money=20, tiles={(4, 4): animal})
        plan = work_plan(state, [("animal", (4, 4), ("FEED",), {})])
        problem = build_route_problem(state, plan)
        opening = initial_skeleton(problem, workforce=1)
        pickup = next(token for token in opening.logistics
                      if opening.logistics[token].operation == "PICKUP")
        custom = replace(
            opening,
            acquisitions={"custom-batch": ("BUY_PRODUCT", "WHEAT", 1)},
            market_priority=("BUY:custom-batch",),
            required_entry_caps=(7,), hire_caps=(2,),
            logistics={**opening.logistics,
                       "pause": LogisticsEvent("pause", "PASS")},
            routes=((pickup, "pause", *tuple(token for token in opening.routes[0]
                                               if token != pickup)),),
        )
        normalized = normalize(problem, custom)
        for field in ("resources", "logistics", "acquisitions", "market_priority",
                      "required_entry_caps", "hire_caps", "routes"):
            self.assertEqual(getattr(normalized, field), getattr(custom, field), field)

    def test_initial_constructor_keeps_a_natural_spatial_district(self):
        positions = ((0, 0), (0, 1), (1, 0), (1, 1))
        state = state_at(hour=8, tiles={position: crop() for position in positions},
                         workers=[((0, 0), {}), ((9, 9), {})])
        plan = work_plan(state, [(f"crop-{n}", position, ("WATER",), {})
                                 for n, position in enumerate(positions)])
        skeleton = initial_skeleton(build_route_problem(state, plan), workforce=2)
        self.assertEqual(len(skeleton.routes[0]), 4)
        self.assertEqual(skeleton.routes[1], ())

    def test_search_mutations_preserve_causal_route_feasibility(self):
        state = state_at(hour=10, money=100, tiles={(4, 4): crop()},
                         workers=[((4, 4), {}), ((3, 4), {})], seeds={"WHEAT": 1})
        plan = work_plan(state, [
            ("old", (4, 4), ("HARVEST",), {}),
            ("new", None, ("PLANT", "WATER"), {"crop": "WHEAT"}),
        ], {"new": ((4, 4),)})
        old, new = plan.obligations
        plan = replace(plan, obligations=(old,), selected=(replace(new, kind="CROP"),))
        problem = build_route_problem(state, plan)
        skeleton = initial_skeleton(problem, workforce=2)
        rng = Random(17)
        for _ in range(200):
            skeleton = _mutate(problem, skeleton, rng, ruin_probability=.5,
                               fixed_workforce=True)
            self.assertTrue(route_precedence_feasible(problem, skeleton))
            self.assertTrue(validate_skeleton(problem, skeleton))

    def test_ruin_recreate_preserves_unplaced_goal_as_reported_infeasibility(self):
        positions = ((0, 0), (4, 4), (8, 8))
        state = state_at(hour=8, tiles={position: crop() for position in positions},
                         workers=[((4, 4), {}), ((5, 4), {})])
        plan = work_plan(state, [(f"crop-{n}", position, ("WATER",), {})
                                 for n, position in enumerate(positions)])
        problem = build_route_problem(state, plan)
        opening = initial_skeleton(problem, workforce=2)
        entity = problem.goal_entity[next(iter(problem.goals))]
        placement = dict(opening.placements); placement.pop(entity)
        incomplete = replace(opening, placements=placement)
        goal = next(goal for goal, owner in problem.goal_entity.items() if owner == entity)
        rebuilt = _ruin_recreate(problem, incomplete, Random(0), focus_goals=(goal,))
        self.assertEqual(sum(goal in route for route in rebuilt.routes), 1)

    def test_open_placement_reconstruction_uses_augmenting_displacement(self):
        state = state_at(hour=20, money=100, seeds={"WHEAT": 3})
        plan = work_plan(state, [
            ("broad", None, ("PLANT", "WATER"), {"crop": "WHEAT"}),
            ("medium", None, ("PLANT", "WATER"), {"crop": "WHEAT"}),
            ("narrow", None, ("PLANT", "WATER"), {"crop": "WHEAT"}),
        ], {"broad": ((4, 4), (3, 4), (5, 4)),
            "medium": ((4, 4), (3, 4)), "narrow": ((4, 4),)})
        plan = replace(plan, selected=tuple(replace(item, kind="CROP")
                                            for item in plan.obligations), obligations=())
        problem = build_route_problem(state, plan)
        skeleton = initial_skeleton(problem, workforce=2)
        self.assertLess(len(skeleton.placements), len(problem.intent.entities))
        rebuilt = _reconstruct_open_placement_matching(problem, skeleton, Random(0))
        self.assertEqual(len(rebuilt.placements), len(problem.intent.entities))
        self.assertEqual(len(set(rebuilt.placements.values())), 3)

    def test_search_initialization_routes_every_matchable_placement_goal(self):
        state = state_at(hour=18, money=100, seeds={"WHEAT": 3})
        plan = work_plan(state, [
            ("broad", None, ("PLANT", "WATER"), {"crop": "WHEAT"}),
            ("medium", None, ("PLANT", "WATER"), {"crop": "WHEAT"}),
            ("narrow", None, ("PLANT", "WATER"), {"crop": "WHEAT"}),
        ], {"broad": ((4, 4), (3, 4), (5, 4)),
            "medium": ((4, 4), (3, 4)), "narrow": ((4, 4),)})
        plan = replace(plan, selected=tuple(replace(item, kind="CROP")
                                            for item in plan.obligations), obligations=())
        problem = build_route_problem(state, plan)
        incomplete = initial_skeleton(problem, workforce=2)
        self.assertLess(len(incomplete.placements), len(problem.intent.entities))
        initialized = _complete_initial_placement(problem, incomplete, seed=0)
        routed = {goal for route in initialized.routes for goal in route}
        self.assertEqual(len(initialized.placements), len(problem.intent.entities))
        self.assertEqual(set(problem.goals), routed & set(problem.goals))

    def test_staffing_compression_preserves_identical_goal_set(self):
        positions = ((4, 4), (4, 5), (5, 4))
        state = state_at(hour=18, money=20,
                         tiles={position: crop() for position in positions})
        plan = work_plan(state, [(f"crop-{n}", position, ("WATER",), {})
                                 for n, position in enumerate(positions)])
        problem = build_route_problem(state, plan)
        skeleton = initial_skeleton(problem, workforce=2)
        original = compile_skeleton(problem, skeleton)
        compressed = [compile_skeleton(problem, candidate)
                      for candidate in _compress_workforce_frontier(
                          problem, skeleton, limit=6)]
        self.assertTrue(any(result.completed == original.completed
                            and result.diagnostics["workforce"] == 1
                            and result.diagnostics["hire_expenditure"] == 0
                            for result in compressed))


if __name__ == "__main__":
    unittest.main()
