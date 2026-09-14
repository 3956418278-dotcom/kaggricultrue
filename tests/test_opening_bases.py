from __future__ import annotations

import unittest

from kaggle_environments import make

from src.kaggriculture_agent.opening_bases import OPENING_BASES, make_opening_base


class OpeningBaseTests(unittest.TestCase):
    def test_two_bases_have_exact_six_day_prefixes(self):
        self.assertEqual(set(OPENING_BASES), {12, 14})
        for record in OPENING_BASES.values():
            self.assertEqual(record["handoff_step"], 144)
            self.assertEqual(len(record["actions"]), 144)

    def test_recorded_prefix_runs_and_handoff_starts_at_day_seven(self):
        for rank in (12, 14):
            env = make("kaggriculture", configuration={"seed": 713 + rank})
            env.reset(2)
            candidate = make_opening_base(rank)
            opponent = {"farmer": ["PASS"], "hands": [], "market": []}
            while env.state[0].observation.step < 144:
                step = env.state[0].observation.step
                self.assertEqual(candidate(env.state[0].observation),
                                 OPENING_BASES[rank]["actions"][step])
                env.step([OPENING_BASES[rank]["actions"][step], opponent])
            self.assertEqual(env.state[0].observation.day, 6)
            action = candidate(env.state[0].observation)
            self.assertEqual(set(action), {"farmer", "hands", "market"})
            self.assertEqual(candidate.session.plans[0].formed_step, 144)


if __name__ == "__main__":
    unittest.main()
