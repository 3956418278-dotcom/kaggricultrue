"""Focused contracts for route structure, conditional support and compilation."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.kaggriculture_agent import rules
from src.kaggriculture_agent.economics import (
    ActionDimension, CashDimension, EconomicCommitment, LandDimension,
    PhysicalDimension, RevenueDimension, TimeDimension, WorkAmount,
)
from src.kaggriculture_agent.planner import Plan
from src.kaggriculture_agent.route_compiler import compile_skeleton
from src.kaggriculture_agent.route_search import (
    RouteSearchConfig, result_score, solve_routes,
)
from src.kaggriculture_agent.route_structure import (
    LogisticsEvent, ResourceLink, RouteSkeleton, RouteStructure, SyncBundle,
    TileLease, build_route_problem, with_initial_logistics,
)
from src.kaggriculture_agent.route_support import solve_support
from src.kaggriculture_agent import route_support as route_support_module
from src.kaggriculture_agent.route_support import SupportConfig
from src.kaggriculture_agent.state import OwnedState, TileState, WorkerState


def state_at(hour=20, day=2, money=0, tiles=None, workers=None, shed=None,
             seeds=None):
    workers = workers or [((4, 4), {})]
    return OwnedState(
        day*24+hour, day, hour, 0, money, 10,
        tuple(TileState((x, y), (tiles or {}).get((x, y)))
              for y in range(10) for x in range(10)),
        tuple(WorkerState(index, position, inventory)
              for index, (position, inventory) in enumerate(workers)),
        ("NW", "NE", "SW", "SE"), len(workers)-1, shed or {}, seeds or {},
        {item: 10000 for item in rules.PRODUCTS},
        {item: rules.market_price(item, 10000) for item in rules.PRODUCTS}, (),
    )


def crop(kind="WHEAT", planted=0, quantity=1):
    rule = rules.CROPS[kind]
    return dict(
        kind="PLANT", crop=kind, planted_day=planted, watered_today=False,
        consecutive_unwatered=0, yield_units=quantity,
        fertilized_until_day=-1,
        max_lifespan_step=(planted+rule.max_yield_day+1)*24,
    )


def animal(kind="GOOSE"):
    return dict(
        kind="COOP", animal=kind, placed_day=0, fed_today=False,
        cared_today=False, consecutive_unfed=0,
        fertilizer_available=False, pending_care_bonus=0, yield_units=0,
    )


def work_plan(state, specifications, domains=None):
    projects = []
    for identifier, position, kinds, metadata in specifications:
        projects.append(EconomicCommitment(
            identifier, "WORK", position, position is not None,
            CashDimension(),
            TimeDimension(state.step,
                          min(state.step+state.turns_left_today-1, 718), 718),
            LandDimension(),
            ActionDimension(tuple(WorkAmount(
                state.day, kind, 1, position=position) for kind in kinds)),
            PhysicalDimension(), RevenueDimension(), metadata,
        ))
    return Plan(tuple(projects), (), (), {}, frozenset(), (), False, 0, 0,
                day=state.day, formed_step=state.step, max_hands=None,
                placement_domains=domains or {})


def action_goals(problem, entity):
    path = problem.complete_paths[entity][0]
    return {event.action[0]: event.goal for event in path}


def structure_for(problem, workforce, routes, placements=None):
    placements = placements or {
        entity.identifier: entity.positions[0]
        for entity in problem.intent.entities if entity.positions
    }
    return RouteStructure(workforce, tuple(tuple(route) for route in routes),
                          placements)


class RouteCompilerTests(unittest.TestCase):
    def test_market_ledger_attributes_real_hire_and_input_cost(self):
        state = replace(state_at(hour=20, money=2000,
                                 shed={"FERTILIZER": 1}),
                        unlocked_quadrants=("NW",))
        orders = (["SELL", "FERTILIZER", 1], ["HIRE"], ["HIRE"],
                  ["BUY_SEED", "WHEAT", 2], ["BUY_LAND"])
        ledger = rules.solo_market_ledger(state, orders)
        following = rules.advance_owned(state, (["PASS"],), orders)
        self.assertEqual(ledger["hire_expenditure"], 2)
        self.assertEqual(ledger["input_expenditure"], 20)
        self.assertEqual(ledger["land_expenditure"], 1000)
        self.assertEqual(ledger["sale_revenue"], 100)
        self.assertEqual(ledger["ending_cash"], following.money)

    def test_same_turn_ordered_workers_can_build_then_place(self):
        state = state_at(hour=23,
                         workers=[((3, 4), {}), ((3, 4), {"GOOSE": 1})])
        plan = work_plan(state, [
            ("asset", (3, 4), ("BUILD", "PICKUP_PLACE"),
             {"structure": "COOP", "animal": "GOOSE"}),
        ])
        problem = build_route_problem(state, plan)
        entity = problem.intent.entities[0].identifier
        goals = action_goals(problem, entity)
        skeleton = RouteSkeleton(
            2, ((goals["BUILD_COOP"],), (goals["PLACE"],)), {entity: 0},
            {entity: (3, 4)},
            (ResourceLink(goals["PLACE"], "GOOSE", 1, "CARRY",
                          source_worker=1),),
            (TileLease(entity, (3, 4)),),
            (SyncBundle((goals["BUILD_COOP"], goals["PLACE"])),),
        )
        result = compile_skeleton(problem, skeleton)
        self.assertEqual(len(result.completed), 2)
        self.assertEqual(result.executions[0].worker_actions,
                         (["BUILD_COOP"], ["PLACE", "GOOSE"]))

    def test_cross_worker_output_has_explicit_transfer(self):
        state = state_at(
            hour=17, tiles={(4, 4): crop(), (6, 4): animal()},
            workers=[((4, 4), {}), ((6, 4), {})],
        )
        inventory = dict(state.market_inventory); inventory["WHEAT"] = 0
        prices = dict(state.market_prices)
        prices["WHEAT"] = rules.market_price("WHEAT", 0)
        state = replace(state, market_inventory=inventory, market_prices=prices)
        plan = work_plan(state, [
            ("crop", (4, 4), ("HARVEST",), {}),
            ("animal", (6, 4), ("FEED",), {}),
        ])
        problem = build_route_problem(state, plan)
        entities = {entity.positions[0]: entity.identifier
                    for entity in problem.intent.entities}
        producer = action_goals(problem, entities[(4, 4)])["HARVEST"]
        consumer = action_goals(problem, entities[(6, 4)])["FEED"]
        support = solve_support(problem, structure_for(
            problem, 2, ((producer,), (consumer,))))
        self.assertTrue(support.feasible, support.conflict)
        self.assertTrue(any(link.kind == "EVENT" and link.via_shed
                            for link in support.skeleton.resources))
        operations = {event.operation
                      for event in support.skeleton.logistics.values()}
        self.assertTrue({"PLACE", "PICKUP"} <= operations)

    def test_manual_transfer_compiles_under_exact_transition(self):
        state = state_at(
            hour=17, tiles={(4, 4): crop(), (6, 4): animal()},
            workers=[((4, 4), {}), ((6, 4), {})],
        )
        plan = work_plan(state, [
            ("crop", (4, 4), ("HARVEST",), {}),
            ("animal", (6, 4), ("FEED",), {}),
        ])
        problem = build_route_problem(state, plan)
        entities = {entity.positions[0]: entity.identifier
                    for entity in problem.intent.entities}
        producer = action_goals(problem, entities[(4, 4)])["HARVEST"]
        consumer = action_goals(problem, entities[(6, 4)])["FEED"]
        skeleton = RouteSkeleton(
            2, ((producer,), (consumer,)),
            {entity: 0 for entity in entities.values()},
            {entity: position for position, entity in entities.items()},
            (ResourceLink(consumer, "WHEAT", 1, "EVENT",
                          producer=producer, via_shed=True),),
            tuple(TileLease(entity, position)
                  for position, entity in entities.items()),
        )
        result = compile_skeleton(problem, with_initial_logistics(problem,
                                                                   skeleton))
        self.assertEqual(len(result.completed), 2, result.diagnostics)
        self.assertGreaterEqual(result.diagnostics["logistics"], 2)


class ConditionalSupportTests(unittest.TestCase):
    def _feed_problem(self, count=1):
        positions = tuple((4+index, 4) for index in range(count))
        state = state_at(
            hour=18, money=200,
            tiles={position: animal() for position in positions},
            shed={"WHEAT": count},
        )
        plan = work_plan(state, [
            (f"animal-{index}", position, ("FEED",), {})
            for index, position in enumerate(positions)
        ])
        problem = build_route_problem(state, plan)
        goals = tuple(problem.goals)
        return state, problem, structure_for(problem, 1, (goals,))

    @staticmethod
    def _compiled(state, completed, unfulfilled=(), money_delta=0):
        return SimpleNamespace(
            completed=frozenset(completed),
            unfulfilled=frozenset(unfulfilled),
            final_state=replace(state, money=state.money+money_delta),
            diagnostics={
                "blockers": ({"missing": {"WHEAT": 1}},),
                "unbought_inputs": (), "unlocked_land_missing": (),
            },
        )

    def test_schedule_alternative_repairs_first_compiler_failure(self):
        state, problem, structure = self._feed_problem(count=2)
        failed = self._compiled(state, (), problem.goals)
        succeeded = self._compiled(state, problem.goals)
        config = SupportConfig(
            max_material_alternatives=1,
            max_schedule_alternatives=2,
            max_exact_compilations=2,
        )
        with patch.object(route_support_module, "compile_skeleton",
                          side_effect=(failed, succeeded)) as compiler:
            support = solve_support(problem, structure, config)
        self.assertTrue(support.feasible, support.conflict)
        self.assertIsNone(support.conflict)
        self.assertEqual(compiler.call_count, 2)
        self.assertEqual(support.diagnostics["selected_schedule_policy"],
                         "split-early-deferred-transfer")
        self.assertEqual(support.diagnostics["compiler_rejections"],
                         {"capacity-or-timing": 1})

    def test_exact_value_overrules_material_proxy_order(self):
        state, problem, structure = self._feed_problem()
        entity = problem.goal_entity[next(iter(problem.goals))]
        goal = next(iter(problem.goals))
        first_signature = route_support_module._SupportSignature(
            ((entity, 0),), (0,), (), ())
        second_signature = route_support_module._SupportSignature(
            ((entity, 0),), (1,), (), ())
        first_links = (ResourceLink(goal, "WHEAT", 1, "SHED"),)
        second_links = (ResourceLink(goal, "WHEAT", 1, "PURCHASE"),)
        material_results = (
            ({entity: 0}, first_links, {}, first_signature,
             {"material_objective": 1}, None),
            ({entity: 0}, second_links, {}, second_signature,
             {"material_objective": 10}, None),
        )
        proxy_winner = self._compiled(state, problem.goals, money_delta=1)
        exact_winner = self._compiled(state, problem.goals, money_delta=10)
        config = SupportConfig(
            max_material_alternatives=2,
            max_schedule_alternatives=1,
            max_exact_compilations=2,
        )
        with patch.object(route_support_module, "_solve_material_network",
                          side_effect=material_results), patch.object(
                              route_support_module, "compile_skeleton",
                              side_effect=(proxy_winner, exact_winner)):
            support = solve_support(problem, structure, config)
        self.assertTrue(support.feasible, support.conflict)
        self.assertEqual(support.diagnostics["proxy_objectives"], (1, 10))
        self.assertEqual(support.diagnostics[
            "selected_material_alternative"], 1)
        self.assertEqual(support.compilation.final_state.money,
                         state.money+10)

    def test_compiler_rejection_adds_internal_material_no_good(self):
        state, problem, structure = self._feed_problem()
        entity = problem.goal_entity[next(iter(problem.goals))]
        goal = next(iter(problem.goals))
        first_signature = route_support_module._SupportSignature(
            ((entity, 0),), (0,), (), ())
        second_signature = route_support_module._SupportSignature(
            ((entity, 0),), (1,), (), ())
        material_results = (
            ({entity: 0}, (ResourceLink(goal, "WHEAT", 1, "SHED"),), {},
             first_signature, {"material_objective": 1}, None),
            ({entity: 0}, (ResourceLink(goal, "WHEAT", 1, "PURCHASE"),), {},
             second_signature, {"material_objective": 2}, None),
        )
        failed = self._compiled(state, (), problem.goals)
        succeeded = self._compiled(state, problem.goals)
        config = SupportConfig(
            max_material_alternatives=2,
            max_schedule_alternatives=1,
            max_exact_compilations=2,
        )
        with patch.object(route_support_module, "_solve_material_network",
                          side_effect=material_results) as network, patch.object(
                              route_support_module, "compile_skeleton",
                              side_effect=(failed, succeeded)):
            support = solve_support(problem, structure, config)
        self.assertTrue(support.feasible, support.conflict)
        self.assertIsNone(support.conflict)
        self.assertEqual(network.call_count, 2)
        exclusions = network.call_args_list[1].args[3]
        self.assertEqual(exclusions, (first_signature,))
        self.assertEqual(support.diagnostics["material_alternatives"], 2)

    def test_temporal_lease_allows_harvest_then_replant(self):
        state = state_at(hour=20, money=20, tiles={(4, 4): crop()})
        plan = work_plan(state, [
            ("old", (4, 4), ("HARVEST",), {}),
            ("new", None, ("PLANT", "WATER"), {"crop": "WHEAT"}),
        ], {"new": ((4, 4),)})
        old, new = plan.obligations
        plan = replace(plan, obligations=(old,),
                       selected=(replace(new, kind="CROP"),))
        problem = build_route_problem(state, plan)
        goals = []
        for entity in problem.intent.entities:
            goals.extend(event.goal
                         for event in problem.complete_paths[entity.identifier][0])
        support = solve_support(problem, structure_for(problem, 1, (tuple(goals),)))
        self.assertTrue(support.feasible, support.conflict)
        leases = {lease.entity: lease for lease in support.skeleton.leases}
        new_entity = next(entity.identifier for entity in problem.intent.entities
                          if not entity.existing)
        self.assertIsNotNone(leases[new_entity].available_after)

    def test_support_never_drops_an_unrouted_plan_goal(self):
        state = state_at(hour=20, tiles={(4, 4): crop()})
        problem = build_route_problem(
            state, work_plan(state, [("crop", (4, 4), ("WATER",), {})]))
        support = solve_support(problem, structure_for(problem, 1, ((),)))
        self.assertFalse(support.feasible)
        self.assertEqual(support.conflict.kind, "structure")
        self.assertEqual(set(support.conflict.goals), set(problem.goals))

    def test_flow_conservation_covers_every_mandatory_input(self):
        state = state_at(hour=18, money=100, tiles={(4, 4): animal()},
                         shed={"WHEAT": 1})
        problem = build_route_problem(
            state, work_plan(state, [("animal", (4, 4), ("FEED",), {})]))
        goal = next(iter(problem.goals))
        support = solve_support(problem, structure_for(problem, 1, ((goal,),)))
        self.assertTrue(support.feasible, support.conflict)
        supplied = sum(link.quantity for link in support.skeleton.resources
                       if link.consumer == goal and link.item == "WHEAT")
        self.assertEqual(supplied, 1)
        self.assertEqual(
            sum(event.operation == "PICKUP"
                for event in support.skeleton.logistics.values()), 1)

    def test_full_realization_capacity_returns_explicit_conflict(self):
        state = state_at(hour=23, tiles={(0, 0): crop()},
                         workers=[((9, 9), {})])
        problem = build_route_problem(
            state, work_plan(state, [("crop", (0, 0), ("WATER",), {})]))
        goal = next(iter(problem.goals))
        support = solve_support(problem, structure_for(problem, 1, ((goal,),)))
        self.assertFalse(support.feasible)
        self.assertEqual(support.conflict.kind, "capacity-or-timing")
        self.assertEqual(set(support.conflict.goals), set(problem.goals))

    def test_score_ignores_labor_and_workforce_diagnostics(self):
        state = state_at(hour=20, money=20)
        baseline = SimpleNamespace(
            completed=frozenset({"goal"}), final_state=state,
            diagnostics={"workforce": 1, "movement": 0},
        )
        changed = SimpleNamespace(
            completed=baseline.completed, final_state=state,
            diagnostics={"workforce": 20, "movement": 10_000},
        )
        self.assertEqual(result_score(baseline, state),
                         result_score(changed, state))
        self.assertEqual(len(result_score(baseline, state)), 2)

    def test_search_uses_only_new_solver_path_and_is_deterministic(self):
        state = state_at(hour=20, money=20, tiles={(4, 4): crop()})
        plan = work_plan(state, [("old", (4, 4), ("HARVEST",), {})])
        config = RouteSearchConfig(
            structural_evaluations=8, basin_probe_evaluations=3,
            basin_refinement_evaluations=3, max_workforce_challengers=1,
        )
        first = solve_routes(state, plan, config)
        second = solve_routes(state, plan, config)
        self.assertEqual(first.executions, second.executions)
        self.assertEqual(first.final_state, second.final_state)
        self.assertEqual(first.diagnostics["selection_objective"],
                         "(Plan fulfillment, V(S_end))")
        self.assertIn("workforce_basins", first.diagnostics)
        for removed in ("population", "search_exact_candidates",
                        "staffing_compression_evaluations",
                        "search_feedback_evaluations"):
            self.assertNotIn(removed, first.diagnostics)


if __name__ == "__main__":
    unittest.main()
