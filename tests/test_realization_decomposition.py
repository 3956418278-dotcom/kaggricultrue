"""Regression checks for witness/search/efficiency benchmark semantics."""
import unittest

from src.kaggriculture_eval.realization_benchmark import (
    compare_efficiency, goal_set_gap,
)


class RealizationDecompositionTests(unittest.TestCase):
    def test_representation_and_search_gaps_are_set_valued(self):
        strong_witness = {"completed_ids": ["a", "b", "c"]}
        compiler_witness = {"completed_ids": ["a", "c"]}
        gap = goal_set_gap(strong_witness, compiler_witness)
        self.assertEqual(gap["missing_from_realization"], ["b"])
        self.assertEqual(gap["additional_in_realization"], [])
        self.assertFalse(gap["target_is_covered"])

    def test_efficiency_is_not_interpreted_across_different_goal_sets(self):
        target = {"completed_ids": ["a", "b"]}
        faster_but_incomplete = {"completed_ids": ["a"]}
        target_effort = {"used_worker_turns": 10, "movement": 5,
                         "pickup_drop_place": 1, "available_worker_turns": 20,
                         "peak_workforce": 2, "executed_hires": 1,
                         "hire_expenditure": 1, "input_expenditure": 10,
                         "land_expenditure": 0, "sale_revenue": 20,
                         "ending_cash": 100}
        candidate_effort = {"used_worker_turns": 1, "movement": 0,
                            "pickup_drop_place": 0, "available_worker_turns": 20,
                            "peak_workforce": 1, "executed_hires": 0,
                            "hire_expenditure": 0, "input_expenditure": 8,
                            "land_expenditure": 0, "sale_revenue": 5,
                            "ending_cash": 90}
        comparison = compare_efficiency(
            target, target_effort, faster_but_incomplete, candidate_effort)
        self.assertEqual(comparison["comparison_status"],
                         "not-comparable-different-goal-set")
        self.assertEqual(comparison["used_worker_turn_difference"], -9)
        self.assertEqual(comparison["peak_workforce_difference"], -1)
        self.assertEqual(comparison["hire_expenditure_difference"], -1)
        self.assertEqual(comparison["ending_cash_difference"], -10)


if __name__ == "__main__":
    unittest.main()
