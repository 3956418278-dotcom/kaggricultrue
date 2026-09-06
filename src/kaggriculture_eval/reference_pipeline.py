"""Bounded, checkpointed CPU extraction; final artifacts are Kaggle Dataset files."""
from __future__ import annotations

import csv
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

import requests

EPISODE_SERVICE = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def discover_metadata(directory, snapshot, seed_submission=55897276, limit=24):
    """Metadata discovery only, NOT a method of determining leaderboard rank.

    Query once per submission, cache responses, stop on any denial/rate limit.
    Every resulting reference must still join the official index and qualify
    against the separately frozen leaderboard. Missing coverage is reported.
    """
    directory = Path(directory)
    top = {row["teamId"] for row in snapshot["teams"]}
    queue = {seed_submission: (False, 0.0, "")}
    queried, episodes, provenance = set(), {}, []
    recent = (datetime.fromisoformat(snapshot["observed_at"]).date() - timedelta(days=1)).isoformat()
    def ingest(payload):
        for episode in payload.get("episodes", []):
            if episode.get("state") != "COMPLETED":
                continue
            episodes[str(episode["id"])] = episode
            for side in episode.get("agents", []):
                sid = side.get("submissionId")
                if sid:
                    stamp = episode.get("endTime", "")
                    priority = (stamp >= recent, float(side.get("initialScore", 0)), stamp)
                    queue[sid] = max(queue.get(sid, (False, 0.0, "")), priority)
    # Reuse every earlier discovery response, not just the path followed by a
    # previous queue order. This is metadata caching, never a ranking definition.
    for cached in sorted(directory.glob("submission-*.json")):
        sid = int(cached.stem.split("-")[1])
        queried.add(sid)
        ingest(json.loads(cached.read_text()))
        provenance.append({"submission_id": sid, "response_sha256": hashlib.sha256(cached.read_bytes()).hexdigest()})
    for _ in range(limit):
        candidates = [s for s in queue if s not in queried]
        if not candidates:
            break
        submission = max(candidates, key=lambda s: (queue[s], s))
        path = directory / f"submission-{submission}.json"
        if path.exists():
            payload = json.loads(path.read_text())
        else:
            response = requests.post(EPISODE_SERVICE, json={"submissionId": submission}, timeout=30)
            if response.status_code != 200 or "application/json" not in response.headers.get("Content-Type", ""):
                raise RuntimeError(f"metadata access stopped: HTTP {response.status_code}; no retry or bypass")
            payload = response.json()
            write_json(path, payload)
            time.sleep(1)
        queried.add(submission)
        provenance.append({"submission_id": submission, "response_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        ingest(payload)
        print(f"metadata request {_+1}/{limit}: submission {submission}, {len(episodes)} episodes", flush=True)
    result = {"source": EPISODE_SERVICE, "queried_submissions": sorted(queried),
              "response_provenance": provenance, "episodes": episodes}
    write_json(directory / "joined-metadata.json", result)
    return result


def qualify_sides(episode, snapshot):
    top = {row["teamId"] for row in snapshot["teams"]}
    threshold = float(snapshot["teams"][-1]["score"])
    result = {}
    for agent in episode.get("agents", []):
        index = agent.get("index", 0)  # protobuf omits default index 0
        score = agent.get("initialScore")
        if (index in (0, 1) and agent.get("teamId") in top and agent.get("submissionId")
                and isinstance(score, (int, float)) and score >= threshold):
            if index in result:
                raise ValueError("ambiguous side index")
            result[index] = {"team_id": agent["teamId"], "submission_id": agent["submissionId"],
                "pre_game_score": score, "post_game_score": agent.get("updatedScore"),
                "score_threshold": threshold, "snapshot_utc": snapshot["observed_at"],
                "rule": "snapshot-top10-team-and-pre-game-score-at-least-snapshot-cutoff",
                "competitive_claim": "snapshot-qualified high-rating side; not historical rank"}
    return result


def run_pilot(input_root, output, config, metadata):
    from .player_days import digest, observation, reconstruct_day, validate_replay, write_shard
    from kaggle_environments.envs.kaggriculture import kaggriculture as official
    if importlib.metadata.version("kaggle-environments") != "1.32.7":
        raise RuntimeError("extractor requires kaggle-environments==1.32.7")
    if hashlib.sha256(Path(official.__file__).read_bytes()).hexdigest() != config["official_source_sha256"]:
        raise RuntimeError("official executable source differs from the verified local contract")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(input_root)
    daily_manifest = source / "manifest.csv"
    if hashlib.sha256(daily_manifest.read_bytes()).hexdigest() != config["daily_manifest_sha256"]:
        raise ValueError("daily source version changed")
    rows = list(csv.DictReader(daily_manifest.open()))
    # Deterministic sample of candidate pool, independent of outcomes/winners.
    rows.sort(key=lambda r: hashlib.sha256((config["sampling_salt"] + r["episode_id"]).encode()).hexdigest())
    rows = rows[:config["candidate_limit"]]
    identity = digest({"config": config, "metadata": metadata,
                       "extractor_sha256": config["source_hashes"]})
    write_json(output / "selection.json", config)
    write_json(output / "episode-metadata.json", metadata)
    report_path = output / "extraction-manifest.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        "schema_version": "player-day-v1", "identity": identity, "episodes": {},
        "environment": {"python": platform.python_version(), "kaggle_environments": "1.32.7",
            "official_source_sha256": hashlib.sha256(Path(official.__file__).read_bytes()).hexdigest()},
        "selection": "player-side; no automatic qualification of the other side",
        "cloud_cpu": True, "complete": False}
    if report["identity"] != identity:
        raise ValueError("checkpoint identity differs; use a new version/output")
    for n, row in enumerate(rows):
        eid = row["episode_id"]
        prior = report["episodes"].get(eid)
        if prior and prior.get("shard"):
            shard = output / prior["shard"]
            if shard.exists() and hashlib.sha256(shard.read_bytes()).hexdigest() == prior["sha256"]:
                continue
        started = time.perf_counter()
        episode = metadata["episodes"].get(eid)
        if not episode:
            report["episodes"][eid] = {"status": "unqualified", "reason": "missing side metadata"}
        else:
            sides = qualify_sides(episode, config["leaderboard"])
            if not sides:
                report["episodes"][eid] = {"status": "unqualified", "reason": "no qualifying side"}
            else:
                path = source / f"{eid}.json"
                try:
                    raw = path.read_bytes()
                    replay = json.loads(raw)
                    replay_hash = hashlib.sha256(raw).hexdigest()
                    if replay["info"]["EpisodeId"] != int(eid):
                        raise ValueError("episode id mismatch")
                    if len(raw) != int(row["size_bytes"]):
                        raise ValueError("source manifest size mismatch")
                    # Reward/seat joins are checked in addition to ID. Team IDs
                    # come from episode metadata, never inferred from names.
                    for side in sides:
                        agent = next(a for a in episode["agents"] if a.get("index", 0) == side)
                        if replay["rewards"][side] != agent["reward"]:
                            raise ValueError("side metadata reward mismatch")
                    validation = validate_replay(replay)
                    if not validation["terminal_rewards_match"]:
                        raise ValueError("terminal reward mismatch")
                    shard = f"episode-{eid}.jsonl.gz"
                    samples = [reconstruct_day(replay, side, day, qualification, replay_hash)
                               for side, qualification in sorted(sides.items()) for day in range(30)]
                    sha = write_shard(output / shard, samples)
                    report["episodes"][eid] = {"status": "extracted", "sides": sides, "player_days": len(samples),
                        "shard": shard, "sha256": sha, "replay_sha256": replay_hash, "validation": validation,
                        "attempt_only_goals": sum(s["diagnostics"]["attempt_only_goals"] for s in samples),
                        "seconds": time.perf_counter()-started}
                    del samples, replay, raw
                except Exception as exc:
                    report["episodes"][eid] = {"status": "quarantined", "reason": str(exc), "type": type(exc).__name__}
        write_json(report_path, report)
        print(f"candidate {n+1}/{len(rows)} {eid}: {report['episodes'][eid]['status']}", flush=True)
    report["complete"] = True
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    report["player_days"] = sum(r.get("player_days", 0) for r in report["episodes"].values())
    report["status_counts"] = dict(__import__("collections").Counter(r["status"] for r in report["episodes"].values()))
    write_json(report_path, report)
    print(json.dumps({k: report[k] for k in ("player_days", "status_counts", "complete")}), flush=True)
    return report
