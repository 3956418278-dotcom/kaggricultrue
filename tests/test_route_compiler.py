"""Correctness checks for event-route representation and compilation."""
from dataclasses import replace
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
    initial_skeleton, selected_paths, with_initial_logistics,
)
from src.kaggriculture_agent.route_search import RouteSearchConfig, solve_routes
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
        self.assertEqual(len(first.completed), 3)

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
        staged = compile_skeleton(problem, replace(skeleton, hire_caps=(3,)))
        self.assertEqual(sum(order[0] == "HIRE"
                             for order in staged.executions[0].market_orders), 3)


if __name__ == "__main__":
    unittest.main()
