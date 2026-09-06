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
from src.kaggriculture_agent.planner import make_plan, PlannerConfig
from src.kaggriculture_agent.realization import ExecutionChoices, legacy_choices
from src.kaggriculture_agent.state import reconstruct, TileState
from src.kaggriculture_agent.execution import generate_tasks


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
        self.assertEqual(len(goals), 4)  # structure, animal placement, feed, care
        self.assertTrue(all(g["kind"] == "STATE_EFFECT" for g in goals))
        self.assertTrue(all(g["target"] is None for g in goals))
        self.assertTrue(all("worker" not in g["metadata"] and "action" not in g["metadata"] for g in goals))
        self.assertIsNone(plan["max_hands"])
        self.assertEqual(len({g["metadata"]["entity"] for g in goals}), 1)
        self.assertGreater(len(next(iter(plan["placement_domains"].values()))), 1)

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


class FixedBoundaryTests(unittest.TestCase):
    def test_existing_structure_domain_does_not_become_empty_land(self):
        state = reconstruct(make("kaggriculture", configuration={"seed": 4}).reset(2)[0].observation)
        tiles = list(state.tiles)
        for pos in ((0, 0), (0, 1)):
            tiles[pos[1]*10+pos[0]] = TileState(pos, {"kind": "PASTURE"})
        state = replace(state, tiles=tuple(tiles), shed={"COW": 1})
        plan = make_plan(state, PlannerConfig(cash_reserve=100000))
        project = next(p for p in plan.obligations if p.kind == "ANIMAL_PLACEMENT")
        self.assertEqual(set(plan.placement_domains[project.identifier]), {(0, 0), (0, 1)})
        saved = deepcopy(plan)
        choices = legacy_choices(state, plan)
        self.assertNotIsInstance(choices, type(plan))
        self.assertEqual(plan, saved)

    def test_no_opportunistic_work_outside_fixed_daily_intent(self):
        state = reconstruct(make("kaggriculture", configuration={"seed": 4}).reset(2)[0].observation)
        plan = replace(make_plan(state), obligations=(), selected=(), support=(), placement_domains={})
        tiles = tuple(TileState(t.position, {"kind": "WEED"}) if t.is_empty else t for t in state.tiles)
        self.assertFalse(generate_tasks(replace(state, tiles=tiles), plan, ExecutionChoices()))


if __name__ == "__main__":
    unittest.main()
