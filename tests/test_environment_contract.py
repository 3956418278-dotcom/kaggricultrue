"""Executable checks for the pinned official Kaggriculture contract.

These tests deliberately use pass/probe actions only. They validate the environment
and submission boundary; they do not define a project agent or strategy.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import unittest
from importlib.metadata import distributions, version
from pathlib import Path

from kaggle_environments import make
from packaging.utils import canonicalize_name


PINNED_PACKAGE_VERSION = "1.32.7"
PINNED_SCHEMA_VERSION = "0.1.0"
PINNED_SEED = 20260904
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}
KEY_FILE_SHA256 = {
    "AGENTS.md": "e1a80501a7b02a212eaac9370ada4129a64e0ee6cb3cbc790f3d77d22863fe22",
    "README.md": "3081e52baf8eb2da5d861acc63a3636ce29425f6bdb79a67036ba234ac4ade00",
    "kaggriculture.json": "a82c89c1a2315b93f39775d8e025471a01b738647c9772658368ee6b1b6f4867",
    "kaggriculture.py": "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e",
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def pass_agent(_observation):
    return PASS_ACTION


def _make_env(
    *, episode_steps: int = 720, seed: int = PINNED_SEED, **configuration
):
    return make(
        "kaggriculture",
        configuration={
            "episodeSteps": episode_steps,
            "seed": seed,
            **configuration,
        },
        debug=True,
    )


def _replay_state_digest(env) -> str:
    """Hash deterministic game state while excluding runtime-duration metadata."""
    rows = []
    for step in env.steps:
        step_rows = []
        for agent_state in step:
            step_rows.append(
                {
                    "action": agent_state.action,
                    "observation": agent_state.observation,
                    "reward": agent_state.reward,
                    "status": agent_state.status,
                }
            )
        rows.append(step_rows)
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class EnvironmentIdentityTests(unittest.TestCase):
    def test_python_runtime_is_pinned(self):
        self.assertEqual(sys.version_info[:3], (3, 12, 3))

    def test_installed_python_distributions_match_lock(self):
        expected = {}
        for raw_line in (REPOSITORY_ROOT / "requirements.lock").read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            name, pinned_version = line.split("==", maxsplit=1)
            expected[canonicalize_name(name)] = pinned_version
        actual = {
            canonicalize_name(distribution.metadata["Name"]): distribution.version
            for distribution in distributions()
        }
        self.assertEqual(actual, expected)

    def test_distribution_schema_and_contract_file_hashes(self):
        self.assertEqual(version("kaggle-environments"), PINNED_PACKAGE_VERSION)

        implementation = importlib.import_module(
            "kaggle_environments.envs.kaggriculture.kaggriculture"
        )
        self.assertEqual(implementation.specification["version"], PINNED_SCHEMA_VERSION)
        source_dir = Path(implementation.__file__).resolve().parent
        for relative_path, expected in KEY_FILE_SHA256.items():
            actual = hashlib.sha256((source_dir / relative_path).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative_path)

    def test_explicit_seed_is_resolved_outside_agent_observation(self):
        env = _make_env(episode_steps=4)
        self.assertEqual(env.info["seed"], PINNED_SEED)
        self.assertIsNone(env.configuration.seed)
        self.assertNotIn("seed", env.state[0].observation)
        self.assertNotIn("seed", env.state[1].observation)


class GameContractTests(unittest.TestCase):
    def test_private_inventory_is_not_part_of_public_farm_state(self):
        env = _make_env(episode_steps=4)
        buy_seed = {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["BUY_SEED", "WHEAT", 1]],
        }
        env.step([buy_seed, PASS_ACTION])

        player_zero = env.state[0].observation
        player_one = env.state[1].observation
        self.assertEqual(player_zero.private["seeds"]["WHEAT"], 1)
        self.assertEqual(player_one.private["seeds"]["WHEAT"], 0)
        for public_farm in player_one.farms:
            self.assertNotIn("shed", public_farm)
            self.assertNotIn("seeds", public_farm)
            self.assertNotIn("inventories", public_farm)

    def test_product_buys_are_lockstep_and_town_consumes_at_step_zero(self):
        env = _make_env(episode_steps=4)
        buy_wheat = {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["BUY_PRODUCT", "WHEAT", 1]],
        }
        env.step([buy_wheat, buy_wheat])

        observation = env.state[0].observation
        self.assertEqual(observation.farms[0]["money"], observation.farms[1]["money"])
        self.assertEqual(env.state[0].observation.private["shed"]["WHEAT"], 1)
        self.assertEqual(env.state[1].observation.private["shed"]["WHEAT"], 1)
        # Two purchases plus the town-center tick on interpreter step zero.
        self.assertEqual(observation.market["inventory"]["WHEAT"], 9_997)
        # A product untouched by the agents still receives the town-center tick.
        self.assertEqual(observation.market["inventory"]["CARROT"], 9_999)

    def test_market_purchase_arrives_after_unit_actions(self):
        env = _make_env(episode_steps=5)
        buy_and_plant = {
            "farmer": ["PLANT", "WHEAT"],
            "hands": [],
            "market": [["BUY_SEED", "WHEAT", 1]],
        }
        env.step([buy_and_plant, PASS_ACTION])

        player = env.state[0].observation
        x, y = player.farms[0]["farmer"]
        self.assertIsNone(player.farms[0]["tiles"][y][x])
        self.assertEqual(player.private["seeds"]["WHEAT"], 1)

    def test_excess_plant_requests_are_rejected_atomically(self):
        env = _make_env(episode_steps=5)
        prepare = {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["HIRE"], ["BUY_SEED", "WHEAT", 1]],
        }
        env.step([prepare, PASS_ACTION])
        main_x, main_y = env.state[0].observation.farms[0]["farmer"]

        over_request = {
            "farmer": ["PLANT", "WHEAT"],
            "hands": [["PLANT", "WHEAT"]],
            "market": [],
        }
        env.step([over_request, PASS_ACTION])
        player = env.state[0].observation
        self.assertIsNone(player.farms[0]["tiles"][main_y][main_x])
        self.assertEqual(player.private["seeds"]["WHEAT"], 1)

    def test_nonexistent_hand_plant_still_counts_in_atomic_demand(self):
        env = _make_env(episode_steps=5)
        buy_seed = {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["BUY_SEED", "WHEAT", 1]],
        }
        env.step([buy_seed, PASS_ACTION])
        main_x, main_y = env.state[0].observation.farms[0]["farmer"]

        request_with_extra_hand = {
            "farmer": ["PLANT", "WHEAT"],
            "hands": [["PLANT", "WHEAT"]],
            "market": [],
        }
        env.step([request_with_extra_hand, PASS_ACTION])
        player = env.state[0].observation
        self.assertEqual(player.farms[0]["hands"], [])
        self.assertIsNone(player.farms[0]["tiles"][main_y][main_x])
        self.assertEqual(player.private["seeds"]["WHEAT"], 1)

    def test_malformed_pickup_or_place_quantity_raises(self):
        for operation in ("PICKUP", "PLACE"):
            with self.subTest(operation=operation):
                env = _make_env(episode_steps=4)
                malformed = {
                    "farmer": [operation, "WHEAT", "not-an-integer"],
                    "hands": [],
                    "market": [],
                }
                with self.assertRaises(ValueError):
                    env.step([malformed, PASS_ACTION])

    def test_daily_refresh_removes_hands_and_unlocks_shop_on_schedule(self):
        env = _make_env(
            episode_steps=5,
            turnsPerDay=2,
            townShopUnlockInterval=1,
            weedSpawnChance=0,
        )
        hire = {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]}
        env.step([hire, PASS_ACTION])
        self.assertEqual(len(env.state[0].observation.farms[0]["hands"]), 1)
        env.step([PASS_ACTION, PASS_ACTION])

        observation = env.state[0].observation
        self.assertEqual(observation.day, 1)
        self.assertEqual(observation.hour, 0)
        self.assertEqual(observation.farms[0]["hands"], [])
        self.assertEqual(observation.farms[0]["hires_today"], 0)
        self.assertEqual(len(observation.town["unlocked_shops"]), 1)

    def test_daily_refresh_drops_inventory_with_overflow_and_resets_farmer(self):
        env = _make_env(episode_steps=5, turnsPerDay=2, weedSpawnChance=0)
        hire = {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]}
        env.step([hire, PASS_ACTION])
        observation = env.state[0].observation
        observation.farms[0]["farmer"] = [0, 0]
        observation.private["shed"]["WHEAT"] = 97
        observation.private["inventories"][0]["CARROT"] = 2
        observation.private["inventories"][1]["MELON"] = 3

        env.step([PASS_ACTION, PASS_ACTION])

        refreshed = env.state[0].observation
        self.assertEqual(refreshed.farms[0]["farmer"], [4, 4])
        self.assertEqual(refreshed.farms[0]["hands"], [])
        self.assertEqual(sum(refreshed.private["shed"].values()), 100)
        self.assertEqual(refreshed.private["shed"]["CARROT"], 2)
        self.assertEqual(refreshed.private["shed"]["MELON"], 1)
        self.assertEqual(refreshed.private["inventories"], [{}])

    def test_market_order_entry_limit_discards_later_entries(self):
        env = _make_env(episode_steps=4, maxMarketOrdersPerTurn=1)
        orders = {
            "farmer": ["PASS"],
            "hands": [],
            "market": [
                ["BUY_SEED", "WHEAT", 1],
                ["BUY_SEED", "CARROT", 1],
            ],
        }
        env.step([orders, PASS_ACTION])
        seeds = env.state[0].observation.private["seeds"]
        self.assertEqual(seeds["WHEAT"], 1)
        self.assertEqual(seeds["CARROT"], 0)

    def test_drop_discards_only_shed_overflow(self):
        env = _make_env(episode_steps=4)
        private = env.state[0].observation.private
        private["shed"]["WHEAT"] = 99
        private["inventories"][0]["MELON"] = 3
        drop = {"farmer": ["DROP"], "hands": [], "market": []}
        env.step([drop, PASS_ACTION])

        result = env.state[0].observation.private
        self.assertEqual(sum(result["shed"].values()), 100)
        self.assertEqual(result["shed"]["MELON"], 1)
        self.assertNotIn("MELON", result["inventories"][0])

    def test_animal_placement_and_fertilizer_lifetime(self):
        env = _make_env(episode_steps=5, weedSpawnChance=0)
        observation = env.state[0].observation
        x, y = observation.farms[0]["farmer"]
        observation.farms[0]["tiles"][y][x] = {"kind": "PASTURE"}
        observation.private["inventories"][0]["COW"] = 1
        place = {"farmer": ["PLACE", "COW"], "hands": [], "market": []}
        env.step([place, PASS_ACTION])
        self.assertEqual(env.state[0].observation.farms[0]["tiles"][y][x]["animal"], "COW")

        # A separate fresh environment avoids DIG's prohibition on placed animals.
        crop_env = _make_env(episode_steps=5, weedSpawnChance=0)
        crop_observation = crop_env.state[0].observation
        cx, cy = crop_observation.farms[0]["farmer"]
        crop_observation.farms[0]["tiles"][cy][cx] = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 0,
            "watered_today": False,
            "consecutive_unwatered": 0,
            "fertilized_until_day": -1,
            "yield_units": 1,
            "max_lifespan_step": 120,
        }
        crop_observation.private["inventories"][0]["FERTILIZER"] = 1
        fertilize = {"farmer": ["FERTILIZE"], "hands": [], "market": []}
        crop_env.step([fertilize, PASS_ACTION])
        tile = crop_env.state[0].observation.farms[0]["tiles"][cy][cx]
        self.assertEqual(tile["fertilized_until_day"], 2)
        self.assertNotIn("FERTILIZER", crop_env.state[0].observation.private["inventories"][0])

    def test_full_pass_episode_has_720_states_and_final_money_rewards(self):
        env = _make_env()
        calls = [0, 0]

        def player_zero(_observation):
            calls[0] += 1
            return PASS_ACTION

        def player_one(_observation):
            calls[1] += 1
            return PASS_ACTION

        env.run([player_zero, player_one])

        self.assertEqual(len(env.steps), 720)
        self.assertEqual(calls, [719, 719])
        self.assertEqual([state.status for state in env.state], ["DONE", "DONE"])
        self.assertEqual([state.reward for state in env.state], [3_000.0, 3_000.0])
        self.assertEqual(env.state[0].observation.step, 719)
        self.assertEqual(env.state[0].observation.day, 29)
        self.assertEqual(env.state[0].observation.hour, 23)

    def test_step_718_action_is_applied_before_done(self):
        env = _make_env()

        def final_turn_buyer(observation):
            market = []
            if observation.step == 718:
                market = [["BUY_SEED", "WHEAT", 1]]
            return {"farmer": ["PASS"], "hands": [], "market": market}

        env.run([final_turn_buyer, pass_agent])
        self.assertEqual(env.state[0].reward, 2_990.0)
        self.assertEqual(env.state[0].observation.private["seeds"]["WHEAT"], 1)
        self.assertEqual(env.state[0].observation.step, 719)
        self.assertEqual(env.state[0].status, "DONE")

    def test_same_seed_and_trajectory_reproduce_replay_state(self):
        first = _make_env()
        second = _make_env()
        first.run([pass_agent, pass_agent])
        second.run([pass_agent, pass_agent])
        self.assertEqual(_replay_state_digest(first), _replay_state_digest(second))


class SubmissionContractTests(unittest.TestCase):
    def test_root_main_py_agent_loads_through_official_file_loader(self):
        env = _make_env(episode_steps=4)
        env.run([str(REPOSITORY_ROOT / "main.py"), pass_agent])

        self.assertEqual([state.status for state in env.state], ["DONE", "DONE"])


if __name__ == "__main__":
    unittest.main()
