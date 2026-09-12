"""Semantic extraction and the corrected fixed-Plan boundary."""
from copy import deepcopy
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

from kaggle_environments import make
from src.kaggriculture_eval.player_days import (
    observation, reconstruct_day, unit_effects, validate_replay, write_shard, read_samples,
)
from src.kaggriculture_eval.reference_pipeline import qualify_sides
from src.kaggriculture_eval.reference_audit import audit_sample


class PlayerDayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = make("kaggriculture", configuration={"seed": 123})
        def demonstration(obs):
            actions = {0: {"farmer": ["BUILD_PASTURE"], "market": [["BUY_ANIMAL", "COW", 1], ["BUY_PRODUCT", "WHEAT", 2], ["HIRE"]]},
                       1: {"farmer": ["PICKUP", "COW", 1]}, 2: {"farmer": ["PLACE", "COW"]},
                       3: {"farmer": ["PICKUP", "WHEAT", 1]}, 4: {"farmer": ["FEED"]},
                       5: {"farmer": ["CARE"], "market": [["SELL", "WHEAT", 1]]}}
            return actions.get(obs.step, {"farmer": ["PASS"]})
        env.run([demonstration, "pass"])
        cls.replay = env.toJSON()
        cls.replay["info"]["EpisodeId"] = 123
        cls.sample = reconstruct_day(cls.replay, 0, 0, {}, "test")

    def test_full_official_round_trip_and_action_alignment(self):
        result = validate_replay(self.replay)
        self.assertEqual(result["official_joint_transitions"], 719)
        self.assertTrue(result["terminal_rewards_match"])
        self.assertEqual(self.sample["demonstrated_realization"][0]["action"]["farmer"], ["BUILD_PASTURE"])

    def test_missing_shared_clock_and_last_day(self):
        replay = deepcopy(self.replay)
        for frame in replay["steps"]:
            frame[1]["observation"].pop("step", None)
        self.assertEqual(observation(replay, 696, 1)["step"], 696)
        last = reconstruct_day(replay, 1, 29, {}, "test")
        self.assertEqual(last["actionable_turns"], 23)
        self.assertEqual(last["demonstrated_realization"][-1]["step"], 718)

    def test_plan_erases_worker_logistics_and_selling_not_asset_effects(self):
        plan = self.sample["plan"]
        self.assertNotIn("hire_count", plan)
        goals = plan["selected"] + plan["obligations"]
        self.assertEqual(len(goals), 1)  # one fixed-place daily animal outcome
        self.assertTrue(all(g["kind"] == "STATE_EFFECT" for g in goals))
        self.assertTrue(all(g["target"] is not None for g in goals))
        self.assertTrue(all("worker" not in g["metadata"] and "action" not in g["metadata"] for g in goals))
        self.assertIsNone(plan["max_hands"])
        self.assertEqual(len({g["metadata"]["entity"] for g in goals}), 1)
        self.assertNotIn("placement_domains", plan)
        self.assertTrue(all(not g["actions"]["work"] for g in goals))

    def test_atomic_seed_failure_is_an_attempt_not_an_achievement(self):
        obs = observation(self.replay, 0, 0)
        obs["farms"][0]["hands"] = [[3, 4]]
        obs["private"]["inventories"] = [{}, {}]
        obs["private"]["seeds"] = {"WHEAT": 1}
        events, noops = unit_effects(obs, {"farmer": ["PLANT", "WHEAT"], "hands": [["PLANT", "WHEAT"]]})
        self.assertEqual(len(events), 2)
        self.assertTrue(all(not e["achieved"] for e in events))
        self.assertTrue(all(n["reason"] == "atomic seeds" for n in noops))

    def test_side_filter_uses_pregame_score_not_winner_or_other_side(self):
        snapshot = {"observed_at": "test", "teams": [{"teamId": 1, "score": "2800"}]}
        episode = {"agents": [{"teamId": 1, "submissionId": 10, "initialScore": 2801, "reward": 1},
                              {"index": 1, "teamId": 2, "submissionId": 20, "initialScore": 3000, "reward": 999}]}
        self.assertEqual(set(qualify_sides(episode, snapshot)), {0})
        episode["agents"][0].update(initialScore=2700, updatedScore=3000)
        self.assertFalse(qualify_sides(episode, snapshot))

    def test_shard_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.jsonl.gz"
            first = write_shard(path, [self.sample])
            self.assertEqual(first, write_shard(path, [self.sample]))
            self.assertEqual(list(read_samples(path)), [self.sample])

    def test_repeated_failed_planting_is_one_eventual_asset(self):
        env = make("kaggriculture", configuration={"seed": 123})
        def retry(obs):
            if obs.step < 3:
                return {"farmer": ["PLANT", "WHEAT"],
                        "market": [["BUY_SEED", "WHEAT", 1]] if obs.step == 1 else []}
            return {}
        env.run([retry, "pass"])
        replay = env.toJSON()
        replay["info"]["EpisodeId"] = 456
        sample = reconstruct_day(replay, 0, 0, {}, "test")
        goals = sample["plan"]["selected"]
        self.assertEqual(len(goals), 1)
        self.assertNotIn("attempt_observed", goals[0]["metadata"])
        self.assertIsNotNone(goals[0]["target"])
        self.assertEqual(audit_sample(sample)["attempt_only_goals"], 0)

    def test_audit_rejects_goal_effect_and_boundary_corruption(self):
        self.assertEqual(audit_sample(self.sample)["player_days"], 1)
        for mutate in (
            lambda s: s["plan"].update(hire_count=3),
            lambda s: s["plan"]["selected"][0]["required_state"].update({"$tile": None}),
            lambda s: s["day_end_state"].update(step=23),
        ):
            sample = deepcopy(self.sample)
            mutate(sample)
            with self.assertRaises(ValueError):
                audit_sample(sample)


if __name__ == "__main__":
    unittest.main()
