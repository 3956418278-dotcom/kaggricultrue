"""Day-level Plan remains the economic contract, not an execution schedule."""

from dataclasses import fields, replace
import unittest

from kaggle_environments import make

from src.kaggriculture_agent.planner import Plan, PlannerConfig, make_plan
from src.kaggriculture_agent.state import reconstruct


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


if __name__ == "__main__":
    unittest.main()
