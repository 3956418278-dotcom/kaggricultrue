from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from kaggle_environments import make

from src.kaggriculture_eval.replay import (
    ReplayError,
    load_replay,
    render_replay_html,
    replay_summary,
    write_replay,
)


def _pass(_observation):
    return {"farmer": ["PASS"], "hands": [], "market": []}


def _short_replay() -> dict:
    environment = make(
        "kaggriculture",
        configuration={"seed": 123, "episodeSteps": 4},
        debug=True,
    )
    environment.run([_pass, _pass])
    return environment.toJSON()


class ReplayViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = _short_replay()

    def test_official_replay_round_trip_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "game.json"
            write_replay(self.replay, replay_path)
            loaded = load_replay(replay_path)

        summary = replay_summary(loaded)
        self.assertEqual(summary.environment, "kaggriculture")
        self.assertEqual(summary.environment_version, "0.1.0")
        self.assertEqual(summary.module_version, summary.local_module_version)
        self.assertEqual(summary.steps, 4)
        self.assertEqual(summary.agents, 2)
        self.assertEqual(summary.statuses, ("DONE", "DONE"))
        self.assertFalse(summary.warnings)

    def test_render_uses_official_visualizer_and_embeds_timeline(self):
        html = render_replay_html(self.replay, initial_step=2)

        self.assertIn("<title>Kaggriculture Visualizer</title>", html)
        self.assertIn("window.kaggle =", html)
        self.assertIn('"name": "kaggriculture"', html)
        self.assertIn('"step": 2', html)
        self.assertIn('"controls": true', html)

    def test_wrapped_window_payload_is_accepted(self):
        wrapped = {"playing": False, "environment": self.replay}
        summary = replay_summary(wrapped)
        self.assertEqual(summary.steps, 4)

    def test_wrong_environment_is_rejected(self):
        replay = dict(self.replay)
        replay["name"] = "connectx"
        with self.assertRaisesRegex(ReplayError, "expected environment"):
            replay_summary(replay)

    def test_invalid_initial_step_is_rejected(self):
        with self.assertRaisesRegex(ReplayError, "initial step"):
            render_replay_html(self.replay, initial_step=4)

    def test_malformed_file_has_reader_error(self):
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "bad.json"
            replay_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with self.assertRaisesRegex(ReplayError, "root must be"):
                load_replay(replay_path)

    def test_malformed_agent_state_is_rejected(self):
        replay = dict(self.replay)
        replay["steps"] = [["not an agent state"]]
        with self.assertRaisesRegex(ReplayError, "agent 0 state"):
            replay_summary(replay)


if __name__ == "__main__":
    unittest.main()
