"""One-run collection/reconstruction, checkpoint and provenance contracts."""
import hashlib
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official
from src.kaggriculture_eval.reference_pipeline import run_collection, write_json, discover_metadata
from src.kaggriculture_eval.reference_audit import audit_collection
from src.kaggriculture_eval.reference_storage import restore_dataset


class CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = make("kaggriculture", configuration={"seed": 17})
        env.run(["pass", "pass"])
        cls.replay = env.toJSON()
        cls.replay["info"]["EpisodeId"] = 987

    def prepare(self, root):
        source, output = root / "raw", root / "output"
        source.mkdir()
        write_json(source / "987.json", self.replay)
        (source / "manifest.csv").write_text("episode_id,size_bytes\n987," + str((source / "987.json").stat().st_size) + "\n")
        snapshot = {"observed_at": "2026-09-05T13:44:52+00:00", "teams": [{"teamId": 1, "score": 2800}]}
        metadata = {"episodes": {"987": {"id": 987, "state": "COMPLETED", "agents": [
            {"teamId": 1, "submissionId": 10, "initialScore": 2900, "reward": 3000}]}}}
        config = {"official_source_sha256": hashlib.sha256(Path(official.__file__).read_bytes()).hexdigest(),
                  "daily_manifest_sha256": hashlib.sha256((source / "manifest.csv").read_bytes()).hexdigest(),
                  "sampling_salt": "test", "candidate_limit": 1, "source_hashes": {},
                  "leaderboard": snapshot, "metadata_queries": 0}
        return source, output, config, metadata

    def test_single_run_creates_samples_audit_and_hash_compatible_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, config, metadata = self.prepare(root)
            first = run_collection(source, output, config, metadata, cache=root / "cache")
            self.assertEqual(first["player_days"], 30)
            self.assertTrue(json.loads((output / "audit.json").read_text())["passed"])
            with patch("src.kaggriculture_eval.player_days.validate_replay", side_effect=AssertionError("must reuse")):
                second = run_collection(source, root / "resumed", config, metadata, cache=root / "cache", resume=output)
            self.assertEqual(first["episodes"], second["episodes"])
            self.assertEqual(audit_collection(root / "resumed")["totals"]["qualified_sides"], 1)
            # Kaggle expands .gz on Dataset ingestion. Restore the exact gzip
            # bytes, not a new reconstruction or weakened content-only identity.
            expanded = root / "expanded"
            expanded.mkdir()
            for name in ("extraction-manifest.json", "selection.json", "episode-metadata.json", "collection.json"):
                (expanded / name).write_bytes((output / name).read_bytes())
            shard = first["episodes"]["987"]["shard"]
            (expanded / Path(shard).with_suffix("")).write_bytes(gzip.decompress((output / shard).read_bytes()))
            result = restore_dataset(expanded, root / "restored")
            self.assertEqual(result["restored_shards"], 1)
            self.assertEqual(audit_collection(root / "restored")["totals"]["player_days"], 30)
            with self.assertRaisesRegex(ValueError, "identity differs"):
                run_collection(source, root / "incompatible", {**config, "sampling_salt": "changed"},
                               metadata, cache=root / "cache", resume=output)

    def test_missing_metadata_is_unqualified_not_invented(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, config, _ = self.prepare(root)
            result = run_collection(source, output, config, {"episodes": {}}, cache=root / "cache")
            self.assertEqual(result["status_counts"], {"unqualified": 1})
            self.assertEqual(result["player_days"], 0)

    def test_deadline_preserves_resumable_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, config, metadata = self.prepare(root)
            result = run_collection(source, output, {**config, "extraction_seconds": 0}, metadata, cache=root / "cache")
            self.assertFalse(result["complete"])
            self.assertEqual(result["episodes"], {})
            self.assertTrue((output / "collection.json").exists())
            self.assertFalse(json.loads((output / "audit.json").read_text())["complete"])

    def test_source_change_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, config, metadata = self.prepare(root)
            with self.assertRaisesRegex(ValueError, "source version changed"):
                run_collection(source, output, {**config, "daily_manifest_sha256": "wrong"}, metadata, cache=root / "cache")

    def test_metadata_denial_stops_requests_and_keeps_usable_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            response = unittest.mock.Mock(status_code=403)
            with patch("src.kaggriculture_eval.reference_pipeline.requests.post", return_value=response) as post:
                result = discover_metadata(directory, {"observed_at": "2026-09-05T00:00:00+00:00", "teams": []}, limit=20)
            self.assertEqual(post.call_count, 1)
            self.assertEqual(result["episodes"], {})
