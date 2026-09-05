from __future__ import annotations

import copy
from dataclasses import replace
import unittest

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official

from main import agent
from src.kaggriculture_agent import rules
from src.kaggriculture_agent.economics import (
    ActionDimension,
    CashDimension,
    LandDimension,
    PhysicalDimension,
    RevenueDimension,
    TimeDimension,
)
from src.kaggriculture_agent.execution import WorkTask, execute, generate_tasks, schedule
from src.kaggriculture_agent.operating import DailyPlanningSession
from src.kaggriculture_agent.realization import bind_plan
from src.kaggriculture_agent.planner import (
    PlannerConfig,
    _fertilizer_marginal_outputs,
    enumerate_projects,
    make_plan,
)
from src.kaggriculture_agent.state import TileState, WorkerState, reconstruct


def initial_state():
    environment = make("kaggriculture", configuration={"seed": 41})
    return reconstruct(environment.reset(2)[0].observation)


class BaselineModelTests(unittest.TestCase):
    def test_local_market_rule_matches_pinned_implementation(self):
        for item in rules.MARKET:
            for inventory in (8_000, 9_500, 9_999, 10_000, 10_001, 10_500, 12_000):
                self.assertEqual(
                    rules.market_price(item, inventory),
                    official.market_price(item, inventory),
                    (item, inventory),
                )

    def test_projects_retain_all_six_dimensions(self):
        state = initial_state()
        projects = enumerate_projects(state, PlannerConfig())
        # Exercise both production families before selection suppresses dominated work.
        self.assertIn("CROP", {project.kind for project in projects})
        self.assertIn("ANIMAL", {project.kind for project in projects})
        for project in projects:
            self.assertIsInstance(project.cash, CashDimension)
            self.assertIsInstance(project.time, TimeDimension)
            self.assertIsInstance(project.land, LandDimension)
            self.assertIsInstance(project.actions, ActionDimension)
            self.assertIsInstance(project.physical, PhysicalDimension)
            self.assertIsInstance(project.revenue, RevenueDimension)

    def test_existing_seed_is_sunk_not_charged_again(self):
        state = replace(initial_state(), seeds={"MELON": 1})
        plan = make_plan(state)
        melon = next(item for item in plan.selected if item.metadata.get("crop") == "MELON")
        self.assertEqual(melon.cash.upfront, 0)
        self.assertEqual(melon.cash.sunk_cost, rules.CROPS["MELON"].seed_cost)

    def test_support_mechanisms_use_commitment_model(self):
        state = initial_state()
        tiles = list(state.tiles)
        position = (3, 4)
        tiles[position[1] * state.board_size + position[0]] = TileState(
            position,
            {
                "kind": "PLANT", "crop": "TOMATO", "planted_day": state.day - 7,
                "watered_today": False, "fertilized_until_day": -1, "yield_units": 0,
            },
        )
        fertilized_state = replace(state, tiles=tuple(tiles), shed={"FERTILIZER": 1, "TOMATO": 2})
        kinds = {item.kind for item in make_plan(fertilized_state).support}
        self.assertIn("FERTILIZER", kinds)
        self.assertIn("LIQUIDATION", kinds)

        weed_tiles = tuple(
            TileState(tile.position, {"kind": "WEED"}) if tile.is_empty else tile
            for tile in state.tiles
        )
        labor_plan = make_plan(replace(state, tiles=weed_tiles))
        self.assertIn("HIRE", {item.kind for item in labor_plan.support})
        self.assertIn("LAND", {item.kind for item in labor_plan.support})
        market_operations = [order[0] for order in execute(replace(state, tiles=weed_tiles), labor_plan).market_orders]
        self.assertIn("HIRE", market_operations)
        self.assertIn("BUY_LAND", market_operations)

    def test_scheduler_never_overcommits_atomic_seeds(self):
        state = replace(
            initial_state(),
            seeds={"WHEAT": 1},
            workers=(
                WorkerState(0, (4, 4), {}),
                WorkerState(1, (5, 4), {}),
                WorkerState(2, (4, 5), {}),
            ),
        )
        tasks = tuple(
            WorkTask(f"plant:{index}", "PLANT", ("PLANT", "WHEAT"), 1, 23, target, shared_item="WHEAT_SEED")
            for index, target in enumerate(((4, 4), (5, 4), (4, 5)))
        )
        actions, _ = schedule(state, tasks)
        self.assertLessEqual(sum(action == ["PLANT", "WHEAT"] for action in actions), 1)

    def test_one_time_crop_waters_before_final_same_day_harvest(self):
        state = initial_state()
        position = (3, 4)

        def crop_state(day: int):
            tiles = list(state.tiles)
            tiles[position[1] * state.board_size + position[0]] = TileState(
                position,
                {
                    "kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
                    "watered_today": False, "consecutive_unwatered": 0,
                    "yield_units": 1, "max_lifespan_step": 120,
                    "fertilized_until_day": -1,
                },
            )
            return replace(state, step=day * 24, day=day, hour=0, tiles=tuple(tiles))

        early = crop_state(2)
        early_kinds = {task.kind for task in generate_tasks(early, make_plan(early))}
        self.assertNotIn("HARVEST", early_kinds)
        self.assertIn("WATER", early_kinds)

        mature = crop_state(4)
        mature_kinds = {task.kind for task in generate_tasks(mature, make_plan(mature))}
        self.assertIn("WATER", mature_kinds)
        self.assertNotIn("HARVEST", mature_kinds)

        watered_tiles = list(mature.tiles)
        watered_raw = dict(mature.tile_at(position).raw)
        watered_raw.update({"watered_today": True, "yield_units": 2})
        watered_tiles[position[1] * mature.board_size + position[0]] = TileState(
            position, watered_raw
        )
        watered = replace(mature, tiles=tuple(watered_tiles))
        watered_kinds = {
            task.kind for task in generate_tasks(watered, make_plan(watered))
        }
        self.assertIn("HARVEST", watered_kinds)
        self.assertNotIn("WATER", watered_kinds)

    def test_animal_projection_uses_feed_and_accumulated_care_rules(self):
        state = initial_state()
        animals = {
            str(project.metadata["animal"]): project
            for project in enumerate_projects(state, PlannerConfig())
            if project.kind == "ANIMAL"
        }
        cow_outputs = [
            output.quantity
            for output in animals["COW"].physical.outputs
            if output.item == "MILK"
        ]
        self.assertEqual(cow_outputs[:3], [6, 3, 3])
        official_tile = official._new_animal("COW", 0)
        official_farm = {"tiles": [[official_tile]]}
        observed = []
        for service_day in range(10):
            official_tile["fed_today"] = True
            official_tile["cared_today"] = True
            official._daily_refresh_animals(official_farm, service_day)
            if official_tile["yield_units"]:
                observed.append(official_tile["yield_units"])
                official_tile["yield_units"] = 0
        self.assertEqual(observed, cow_outputs[: len(observed)])
        goose_outputs = [
            output.quantity
            for output in animals["GOOSE"].physical.outputs
            if output.item == "EGG"
        ]
        self.assertEqual(goose_outputs[:3], [4, 2, 2])

    def test_fertilizer_uses_actual_marginal_realizable_output(self):
        state = initial_state()
        position = (3, 4)
        tiles = list(state.tiles)
        tiles[position[1] * state.board_size + position[0]] = TileState(
            position,
            {
                "kind": "PLANT", "crop": "TOMATO", "planted_day": -7,
                "watered_today": False, "consecutive_unwatered": 0,
                "yield_units": 0, "max_lifespan_step": -1,
                "fertilized_until_day": -1,
            },
        )
        state = replace(state, tiles=tuple(tiles), shed={"FERTILIZER": 1})
        marginal = _fertilizer_marginal_outputs(state, state.tile_at(position))
        self.assertEqual(
            [(item.item, item.quantity) for item in marginal],
            [("TOMATO", 1), ("TOMATO", 1), ("TOMATO", 1)],
        )
        support = next(item for item in make_plan(state).support if item.kind == "FERTILIZER")
        self.assertEqual(support.physical.outputs, marginal)
        self.assertEqual(support.metadata["marginal_units"], 3)
        carried = replace(
            state,
            shed={},
            workers=(WorkerState(0, position, {"FERTILIZER": 1}),),
        )
        self.assertEqual(
            execute(carried, make_plan(carried)).worker_actions[0],
            ["FERTILIZE"],
        )
        too_late = replace(carried, step=23, hour=23)
        self.assertNotIn(
            "FERTILIZER", {item.kind for item in make_plan(too_late).support}
        )

        baseline = official._new_plant("TOMATO", -7, 24)
        treated = dict(baseline)
        treated["fertilized_until_day"] = 2
        realized_marginal = []
        for day in range(3):
            baseline["watered_today"] = True
            treated["watered_today"] = True
            official._daily_refresh_plants({"tiles": [[baseline]]}, day, 24)
            official._daily_refresh_plants({"tiles": [[treated]]}, day, 24)
            realized_marginal.append(treated["yield_units"] - baseline["yield_units"])
            baseline["yield_units"] = 0
            treated["yield_units"] = 0
        self.assertEqual(realized_marginal, [1, 1, 1])

    def test_terminal_liquidation_recovers_crop_and_animal_tile_yield(self):
        state = initial_state()
        crop_position = (4, 4)
        animal_position = (3, 4)
        tiles = list(state.tiles)
        tiles[crop_position[1] * 10 + crop_position[0]] = TileState(
            crop_position,
            {
                "kind": "PLANT", "crop": "WHEAT", "planted_day": 25,
                "watered_today": True, "consecutive_unwatered": 0,
                "yield_units": 4, "max_lifespan_step": 720,
                "fertilized_until_day": -1,
            },
        )
        tiles[animal_position[1] * 10 + animal_position[0]] = TileState(
            animal_position,
            {
                "kind": "COOP", "animal": "GOOSE", "placed_day": 20,
                "yield_units": 3, "consecutive_unfed": 0,
                "fed_today": False, "cared_today": False,
                "fertilizer_available": False, "pending_care_bonus": 0,
            },
        )
        terminal = replace(
            state,
            step=707,
            day=29,
            hour=11,
            tiles=tuple(tiles),
            workers=(WorkerState(0, crop_position, {}),),
        )
        plan = make_plan(terminal)
        liquidation = next(item for item in plan.support if item.kind == "LIQUIDATION")
        projected = {(item.item, item.quantity) for item in liquidation.physical.outputs}
        self.assertEqual(projected, {("WHEAT", 4), ("EGG", 3)})
        terminal_tasks = generate_tasks(terminal, plan)
        self.assertEqual(
            {task.identifier for task in terminal_tasks if task.kind == "HARVEST"},
            {
                f"terminal-harvest:{crop_position}",
                f"terminal-animal-harvest:{animal_position}",
            },
        )

    def test_market_budget_keeps_required_purchases_when_sales_fill_budget(self):
        state = initial_state()
        full_shed = {item: 1 for item in rules.SELLABLE_PRODUCTS}
        state = replace(state, shed=full_shed)
        plan = replace(
            bind_plan(state, make_plan(state)),
            crop_targets={(0, 0): "WHEAT", (1, 0): "CARROT", (2, 0): "TOMATO"},
            hire_count=5,
            feed_reserve=2,
            fertilizer_reserve=2,
            animal_purchases=(),
            buy_land=False,
        )
        orders = execute(state, plan).market_orders
        self.assertEqual(len(orders), rules.MAX_MARKET_ORDERS)
        self.assertEqual(sum(order[0] == "HIRE" for order in orders), 5)
        self.assertEqual(
            {tuple(order[:2]) for order in orders if order[0].startswith("BUY_")},
            {
                ("BUY_SEED", "WHEAT"),
                ("BUY_SEED", "CARROT"),
                ("BUY_SEED", "TOMATO"),
                ("BUY_PRODUCT", "FERTILIZER"),
                ("BUY_PRODUCT", "WHEAT"),
            },
        )

    def test_same_observation_is_deterministic_and_json_safe(self):
        environment = make("kaggriculture", configuration={"seed": 43})
        observation = environment.reset(2)[0].observation
        first = agent(copy.deepcopy(observation))
        second = agent(copy.deepcopy(observation))
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"farmer", "hands", "market"})

    def test_daily_commitments_and_staffing_survive_ordinary_intraday_state(self):
        session = DailyPlanningSession()
        opening = initial_state()
        plan = session.plan_for(opening)
        intraday = replace(
            opening,
            step=6,
            hour=6,
            market_prices={**opening.market_prices, "MELON": 1, "WHEAT": 500},
            market_inventory={**opening.market_inventory, "MELON": 12_000},
        )

        same_plan = session.plan_for(intraday)
        self.assertIs(same_plan, plan)
        self.assertEqual(
            tuple(item.identifier for item in same_plan.selected),
            tuple(item.identifier for item in plan.selected),
        )
        self.assertEqual(same_plan.hire_count, plan.hire_count)

    def test_daily_staffing_is_hired_at_day_start_and_not_repeated(self):
        opening = initial_state()
        weed_tiles = tuple(
            TileState(tile.position, {"kind": "WEED"}) if tile.is_empty else tile
            for tile in opening.tiles
        )
        opening = replace(opening, tiles=weed_tiles)
        session = DailyPlanningSession()
        plan = session.plan_for(opening)

        self.assertGreater(plan.hire_count, 0)
        self.assertIn("HIRE", [order[0] for order in execute(opening, plan).market_orders])

        hired = replace(
            opening,
            step=1,
            hour=1,
            hires_today=plan.hire_count,
            workers=(
                *opening.workers,
                *tuple(
                    WorkerState(index + 1, (4, 4), {})
                    for index in range(plan.hire_count)
                ),
            ),
        )
        self.assertIs(session.plan_for(hired), plan)
        self.assertNotIn("HIRE", [order[0] for order in execute(hired, plan).market_orders])

    def test_staffing_target_does_not_grow_only_from_elapsed_hours(self):
        opening = initial_state()
        recovery_tiles = tuple(
            TileState(tile.position, {"kind": "WEED"}) if index < 47 else tile
            for index, tile in enumerate(opening.tiles)
        )
        opening = replace(opening, tiles=recovery_tiles)
        one_hour_later = replace(opening, step=1, hour=1)

        self.assertEqual(make_plan(opening).hire_count, make_plan(one_hour_later).hire_count)

    def test_fixed_daily_plan_does_not_repeat_fulfilled_commitment_orders(self):
        opening = initial_state()
        crop_plan = bind_plan(opening, make_plan(opening))
        self.assertTrue(crop_plan.crop_targets)
        tiles = list(opening.tiles)
        for target, crop in crop_plan.crop_targets.items():
            tiles[target[1] * opening.board_size + target[0]] = TileState(
                target,
                {
                    "kind": "PLANT", "crop": crop, "planted_day": opening.day,
                    "watered_today": False, "consecutive_unwatered": 0,
                    "yield_units": 1, "max_lifespan_step": 240,
                    "fertilized_until_day": -1,
                },
            )
        planted = replace(opening, step=1, hour=1, tiles=tuple(tiles))
        self.assertNotIn(
            "BUY_SEED", [order[0] for order in execute(planted, crop_plan).market_orders]
        )

        animal_plan = replace(crop_plan, animal_purchases=("COW",))
        acquired = replace(opening, step=1, hour=1, shed={"COW": 1})
        self.assertNotIn(
            "BUY_ANIMAL", [order[0] for order in execute(acquired, animal_plan).market_orders]
        )

        weed_tiles = tuple(
            TileState(tile.position, {"kind": "WEED"}) if tile.is_empty else tile
            for tile in opening.tiles
        )
        land_state = replace(opening, tiles=weed_tiles)
        land_plan = make_plan(land_state)
        self.assertTrue(land_plan.buy_land)
        unlocked = replace(
            land_state,
            step=1,
            hour=1,
            unlocked_quadrants=("NW", "NE"),
        )
        self.assertNotIn(
            "BUY_LAND", [order[0] for order in execute(unlocked, land_plan).market_orders]
        )

        fertilizer_target = (3, 4)
        fertilizer_tiles = list(opening.tiles)
        fertilizer_tiles[fertilizer_target[1] * 10 + fertilizer_target[0]] = TileState(
            fertilizer_target,
            {
                "kind": "PLANT", "crop": "TOMATO", "planted_day": -7,
                "watered_today": False, "consecutive_unwatered": 0,
                "yield_units": 0, "max_lifespan_step": -1,
                "fertilized_until_day": -1,
            },
        )
        fertilizer_state = replace(
            opening, tiles=tuple(fertilizer_tiles), shed={"FERTILIZER": 1}
        )
        fertilizer_plan = make_plan(fertilizer_state)
        applied_raw = dict(fertilizer_state.tile_at(fertilizer_target).raw)
        applied_raw["fertilized_until_day"] = opening.day + 2
        fertilizer_tiles[fertilizer_target[1] * 10 + fertilizer_target[0]] = TileState(
            fertilizer_target, applied_raw
        )
        applied = replace(
            fertilizer_state,
            step=1,
            hour=1,
            tiles=tuple(fertilizer_tiles),
            shed={},
        )
        self.assertNotIn(
            ("BUY_PRODUCT", "FERTILIZER"),
            {
                tuple(order[:2])
                for order in execute(applied, fertilizer_plan).market_orders
            },
        )

    def test_intraday_routing_adapts_without_replacing_production_plan(self):
        opening = initial_state()
        target = (3, 4)
        tiles = list(opening.tiles)
        tiles[target[1] * opening.board_size + target[0]] = TileState(
            target,
            {
                "kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
                "watered_today": False, "consecutive_unwatered": 0,
                "yield_units": 1, "max_lifespan_step": 120,
                "fertilized_until_day": -1,
            },
        )
        opening = replace(opening, tiles=tuple(tiles))
        session = DailyPlanningSession()
        plan = session.plan_for(opening)
        first_action = execute(opening, plan).worker_actions[0]

        arrived = replace(
            opening,
            step=1,
            hour=1,
            workers=(WorkerState(0, target, {}),),
        )
        same_plan = session.plan_for(arrived)
        arrived_action = execute(arrived, same_plan).worker_actions[0]

        self.assertIs(same_plan, plan)
        self.assertEqual(first_action, ["WEST"])
        self.assertEqual(arrived_action, ["WATER"])

    def test_intraday_sales_react_without_replacing_daily_plan(self):
        opening = replace(
            initial_state(),
            shed={"CARROT": 2, "MELON": 2},
            market_prices={
                **initial_state().market_prices,
                "CARROT": 500,
                "MELON": 1,
            },
        )
        session = DailyPlanningSession()
        plan = session.plan_for(opening)
        first_sale = next(
            order for order in execute(opening, plan).market_orders if order[0] == "SELL"
        )
        changed_market = replace(
            opening,
            step=1,
            hour=1,
            market_prices={**opening.market_prices, "CARROT": 1, "MELON": 500},
        )
        same_plan = session.plan_for(changed_market)
        second_sale = next(
            order
            for order in execute(changed_market, same_plan).market_orders
            if order[0] == "SELL"
        )

        self.assertIs(same_plan, plan)
        self.assertEqual(first_sale[1], "CARROT")
        self.assertEqual(second_sale[1], "MELON")

    def test_next_day_forms_a_fresh_plan_from_reset_state(self):
        session = DailyPlanningSession()
        opening = initial_state()
        day_zero = session.plan_for(opening)
        next_day = replace(
            opening,
            step=24,
            day=1,
            hour=0,
            workers=(WorkerState(0, (4, 4), {}),),
            hires_today=0,
            shed={"WHEAT": 7},
        )

        day_one = session.plan_for(next_day)
        self.assertIsNot(day_one, day_zero)
        self.assertEqual(day_one.day, 1)
        self.assertEqual(day_one.formed_step, 24)
        self.assertEqual(day_one.starting_animals, {
            animal: next_day.owned_animals(animal) for animal in rules.ANIMALS
        })

    def test_intraday_replan_is_material_and_bounded(self):
        session = DailyPlanningSession()
        opening = initial_state()
        original = session.plan_for(opening)
        # New-production sites belong to the intraday realization now. An
        # existing crop is a fixed physical premise of daily economic intent.
        position = (3, 4)
        tiles = list(opening.tiles)
        tiles[43] = TileState(position, official._new_plant("MELON", 0, 24))
        opening = replace(opening, tiles=tuple(tiles))
        session.reset()
        original = session.plan_for(opening)
        first_target = original.obligations[0].target
        self.assertIsNotNone(first_target)
        tiles = list(opening.tiles)
        tiles[first_target[1] * opening.board_size + first_target[0]] = TileState(
            first_target, {"kind": "WEED"}
        )
        invalidated = replace(opening, step=1, hour=1, tiles=tuple(tiles))

        repaired = session.plan_for(invalidated)
        self.assertEqual(repaired.revision, 1)
        self.assertIn("disappeared", repaired.replan_reason)

        second_target = bind_plan(invalidated, repaired).selected[0].target
        tiles = list(invalidated.tiles)
        tiles[second_target[1] * opening.board_size + second_target[0]] = TileState(
            second_target, {"kind": "WEED"}
        )
        invalidated_again = replace(invalidated, step=2, hour=2, tiles=tuple(tiles))
        self.assertIs(session.plan_for(invalidated_again), repaired)

    def test_late_day_carry_relies_on_official_refresh_drop(self):
        state = initial_state()
        late = replace(
            state,
            step=22,
            hour=22,
            workers=(WorkerState(0, state.workers[0].position, {"CARROT": 1}),),
        )
        tasks = generate_tasks(late, make_plan(late))
        self.assertNotIn("drop:0", {task.identifier for task in tasks})


if __name__ == "__main__":
    unittest.main()
