"""Day-level Plan remains the economic contract, not an execution schedule."""

from dataclasses import fields, replace
import unittest

from kaggle_environments import make

from src.kaggriculture_agent import rules
from src.kaggriculture_agent.intraday import _expand_plan
from src.kaggriculture_agent.planner import Plan, PlannerConfig, make_plan
from src.kaggriculture_agent.state import TileState, reconstruct


class PlannerModelTests(unittest.TestCase):
    def test_plan_keeps_economic_dimensions_without_intraday_fields(self):
        state = reconstruct(make("kaggriculture", configuration={"seed": 4}).reset(2)[0].observation)
        plan = make_plan(state, PlannerConfig(cash_reserve=0))
        self.assertIsInstance(plan, Plan)
        for project in (*plan.obligations, *plan.selected, *plan.support):
            self.assertIsNotNone(project.cash)
            self.assertIsNotNone(project.time)
            self.assertIsNotNone(project.land)
            self.assertIsNotNone(project.actions)
            self.assertIsNotNone(project.physical)
            self.assertIsNotNone(project.revenue)
        self.assertFalse(hasattr(plan, "routes"))
        self.assertFalse(hasattr(plan, "resource_links"))
        self.assertNotIn("placement_domains", {field.name for field in fields(Plan)})
        self.assertIn("economic_windows", {field.name for field in fields(Plan)})
        for project in (*plan.obligations, *plan.selected, *plan.support):
            if project.kind != "LAND" and project.required_state:
                self.assertIsNotNone(project.target)

    def test_empty_plan_stays_empty(self):
        state = reconstruct(make("kaggriculture", configuration={"seed": 4}).reset(2)[0].observation)
        base = make_plan(state)
        empty = replace(base, obligations=(), selected=(), support=())
        self.assertEqual(empty.obligations + empty.selected + empty.support, ())

    def test_one_time_crop_harvest_requires_same_day_water_gain(self):
        state = reconstruct(make("kaggriculture", configuration={"seed": 4}).reset(2)[0].observation)
        crop = "WHEAT"
        rule = rules.CROPS[crop]
        target = state.tiles[0].position
        day = rule.max_yield_day
        opening_yield = rule.max_yield - 1
        crop_tile = TileState(target, {
            "kind": "PLANT",
            "crop": crop,
            "planted_day": 0,
            "watered_today": False,
            "consecutive_unwatered": 0,
            "yield_units": opening_yield,
            "max_lifespan_step": (rule.max_yield_day + 1) * rules.TURNS_PER_DAY,
            "fertilized_until_day": -1,
        })
        state = replace(
            state,
            step=day * rules.TURNS_PER_DAY,
            day=day,
            hour=0,
            tiles=tuple(
                crop_tile if tile.position == target else tile
                for tile in state.tiles
            ),
        )

        formed = make_plan(state, PlannerConfig(cash_reserve=10**9))
        harvest = next(
            project for project in formed.obligations
            if project.identifier == f"existing-crop:{crop}:{target[0]}:{target[1]}"
        )
        plan = replace(
            formed,
            obligations=(harvest,),
            selected=(),
            support=(),
            economic_windows=(),
        )

        self.assertEqual(harvest.required_outputs[crop], rule.max_yield)
        projects, _, _ = _expand_plan(state, plan)
        self.assertEqual(
            tuple(event.action[0] for event in projects[0].events),
            ("WATER", "HARVEST"),
        )


if __name__ == "__main__":
    unittest.main()
