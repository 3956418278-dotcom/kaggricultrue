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
from src.kaggriculture_agent.planner import PlannerConfig, enumerate_projects, make_plan
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
                "kind": "PLANT", "crop": "MELON", "planted_day": state.day - 6,
                "watered_today": False, "fertilized_until_day": -1, "yield_units": 1,
            },
        )
        fertilized_state = replace(state, tiles=tuple(tiles), shed={"FERTILIZER": 1, "MELON": 2})
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

    def test_one_time_crop_waits_for_its_bonus_window(self):
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
        self.assertIn("HARVEST", mature_kinds)

    def test_same_observation_is_deterministic_and_json_safe(self):
        environment = make("kaggriculture", configuration={"seed": 43})
        observation = environment.reset(2)[0].observation
        first = agent(copy.deepcopy(observation))
        second = agent(copy.deepcopy(observation))
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"farmer", "hands", "market"})


if __name__ == "__main__":
    unittest.main()
