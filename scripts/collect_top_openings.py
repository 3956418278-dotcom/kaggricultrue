#!/usr/bin/env python3
"""Freeze current Kaggle top openings and their official replay variants.

This is evaluation infrastructure.  It reads only Kaggle's official leaderboard,
team-submission, episode-list and replay APIs.  Raw and derived artifacts belong
under ignored ``runs/`` and never enter the submitted policy automatically.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi


COMPETITION = "kaggriculture"
OPENING_DAYS = 8
OPENING_TURNS = OPENING_DAYS * 24


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_gzip_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    temporary = path.with_suffix(path.suffix + ".partial")
    with gzip.GzipFile(filename=str(temporary), mode="wb", compresslevel=9, mtime=0) as handle:
        handle.write(payload)
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _agent_index(agent: dict[str, Any]) -> int:
    # Protobuf JSON omits the default value for player zero.
    return int(agent.get("index", 0) or 0)


def _opening_record(replay: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    if replay.get("name") != COMPETITION or replay.get("module_version") != "1.32.7":
        raise ValueError("official replay environment identity mismatch")
    if len(replay.get("steps", ())) != 720 or replay.get("statuses") != ["DONE", "DONE"]:
        raise ValueError("official replay is not a complete default episode")
    seed = replay.get("info", {}).get("seed")
    if not isinstance(seed, int):
        raise ValueError("official replay has no integer environment seed")
    if replay.get("info", {}).get("EpisodeId") != int(episode["id"]):
        raise ValueError("episode ID differs between metadata and replay")

    sides: list[dict[str, Any]] = []
    for side in range(2):
        metadata = next(a for a in episode["agents"] if _agent_index(a) == side)
        actions = [
            replay["steps"][step + 1][side].get("action") or {}
            for step in range(OPENING_TURNS)
        ]
        daily = []
        for day in range(OPENING_DAYS + 1):
            frame = day * 24
            observation = replay["steps"][frame][side].get("observation") or {}
            shared = replay["steps"][frame][0].get("observation") or {}
            farm = (observation.get("farms") or shared.get("farms"))[side]
            town = observation.get("town") or shared.get("town") or {}
            market = observation.get("market") or shared.get("market") or {}
            private = observation.get("private") or {}
            daily.append({
                "day": day + 1,
                "step": frame,
                "money": farm.get("money"),
                "worker_positions": [farm.get("farmer"), *farm.get("hands", [])],
                "owned_land": farm.get("unlocked_quadrants", []),
                "shops": town.get("unlocked_shops", []),
                "market_inventory": market.get("inventory", {}),
                "shed": private.get("shed", {}),
                "seeds": private.get("seeds", {}),
                "tiles": farm.get("tiles"),
            })
        sides.append({
            "side": side,
            "team_id": metadata.get("teamId"),
            "team_name": metadata.get("teamName"),
            "submission_id": metadata.get("submissionId"),
            "reward": metadata.get("reward"),
            "actions": actions,
            "daily_state": daily,
        })
    first_shop = sides[0]["daily_state"][3]["shops"]
    return {
        "schema_version": "official-top-opening-v1",
        "episode_id": int(episode["id"]),
        "environment_seed": seed,
        "module_version": replay["module_version"],
        "configuration": replay["configuration"],
        "statuses": replay["statuses"],
        "terminal_rewards": replay["rewards"],
        "first_shop": first_shop,
        "sides": sides,
    }


def _episode_relation(episode: dict[str, Any], submission_id: int) -> dict[str, Any] | None:
    own = next((a for a in episode.get("agents", ()) if a.get("submissionId") == submission_id), None)
    opponent = next((a for a in episode.get("agents", ()) if a.get("submissionId") != submission_id), None)
    if own is None or opponent is None or episode.get("state") != "COMPLETED":
        return None
    return {
        "episode_id": int(episode["id"]),
        "create_time": episode.get("createTime"),
        "end_time": episode.get("endTime"),
        "own_seat": _agent_index(own),
        "own_reward": own.get("reward"),
        "opponent_submission_id": opponent.get("submissionId"),
        "opponent_team_id": opponent.get("teamId"),
        "opponent_team_name": opponent.get("teamName"),
        "opponent_reward": opponent.get("reward"),
    }


def _candidate_pool(relations: list[dict[str, Any]], same_count: int, cross_count: int) -> dict[str, Any]:
    by_same: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    by_opponent: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for relation in relations:
        opponent = int(relation["opponent_submission_id"])
        by_same[(opponent, int(relation["own_seat"]))].append(relation)
        by_opponent[opponent].append(relation)
    repeated = [
        (key, rows) for key, rows in by_same.items() if len(rows) >= 2
    ]
    repeated.sort(key=lambda item: (-len(item[1]), -max(row["episode_id"] for row in item[1]), item[0]))
    if not repeated:
        raise ValueError("submission has no same-opponent, same-seat repeated episodes")
    same_key, same_rows = repeated[0]
    same_rows = sorted(same_rows, key=lambda row: row["episode_id"], reverse=True)[:same_count]

    newest_per_opponent = [
        max(rows, key=lambda row: row["episode_id"])
        for opponent, rows in by_opponent.items()
        if opponent != same_key[0]
    ]
    newest_per_opponent.sort(key=lambda row: row["episode_id"], reverse=True)
    cross_rows = newest_per_opponent[:cross_count]
    selected = {row["episode_id"]: row for row in (*same_rows, *cross_rows)}
    return {
        "same_opponent_submission_id": same_key[0],
        "same_own_seat": same_key[1],
        "same_opponent_candidates": same_rows,
        "cross_opponent_candidates": cross_rows,
        "selected_episode_ids": sorted(selected, reverse=True),
    }


def collect(output: Path, same_count: int, cross_count: int) -> None:
    api = KaggleApi()
    api.authenticate()
    selection_path = output / "leaderboard-and-selection.json"
    metadata_path = output / "episode-metadata.json"
    if selection_path.exists() or metadata_path.exists():
        if not selection_path.exists() or not metadata_path.exists():
            raise RuntimeError("partial frozen selection; use a new output directory")
        snapshot = json.loads(selection_path.read_text(encoding="utf-8"))
        episode_metadata = {
            int(key): value
            for key, value in json.loads(metadata_path.read_text(encoding="utf-8")).items()
        }
        print(f"resuming frozen leaderboard observed at {snapshot['observed_at']}", flush=True)
    else:
        observed_at = datetime.now(timezone.utc).isoformat()
        leaderboard = api.competition_leaderboard_view(COMPETITION, page_size=15)
        if len(leaderboard) != 15:
            raise RuntimeError(f"expected 15 leaderboard rows, received {len(leaderboard)}")

        strategies: list[dict[str, Any]] = []
        episode_metadata: dict[int, dict[str, Any]] = {}
        for rank, row in enumerate(leaderboard, 1):
            submissions = api.competition_team_submissions(int(row.team_id))
            if not submissions:
                raise RuntimeError(f"top-{rank} team {row.team_id} has no active submission")
            primary = max(submissions, key=lambda submission: float(submission.public_score))
            episodes = [_jsonable(episode) for episode in api.competition_list_episodes(int(primary.id))]
            relations = [relation for episode in episodes
                         if (relation := _episode_relation(episode, int(primary.id))) is not None]
            pool = _candidate_pool(relations, same_count, cross_count)
            for episode in episodes:
                if int(episode["id"]) in pool["selected_episode_ids"]:
                    episode_metadata[int(episode["id"])] = episode
            strategies.append({
                "rank": rank,
                "team_id": int(row.team_id),
                "team_name": row.team_name,
                "leaderboard_score": row.score,
                "leaderboard_submission_date": _jsonable(row.submission_date),
                "active_submissions": [_jsonable(submission) for submission in submissions],
                "primary_submission_id": int(primary.id),
                "primary_selection": "highest publicScore among the team's two active submissions",
                "episode_count": len(relations),
                "candidate_pool": pool,
            })
            print(
                f"metadata {rank:02d}/15 {row.team_name}: submission={primary.id} "
                f"episodes={len(relations)} selected={len(pool['selected_episode_ids'])}",
                flush=True,
            )

        snapshot = {
            "schema_version": "official-top-opening-collection-v1",
            "competition": COMPETITION,
            "observed_at": observed_at,
            "opening_days": OPENING_DAYS,
            "selection": {
                "leaderboard": "current official top 15",
                "primary_submission": "maximum active publicScore",
                "same_opponent": f"{same_count} newest episodes from the most frequent same-opponent/same-seat group",
                "cross_opponent": f"newest episode for each of {cross_count} different opponent submissions",
                "outcome_independent": True,
                "evidence_class": "exploratory replay census; leaderboard rank is an official timestamped snapshot",
            },
            "strategies": strategies,
        }
        _write_json(selection_path, snapshot)
        _write_json(metadata_path, {str(k): v for k, v in sorted(episode_metadata.items())})

    replay_hashes: dict[str, Any] = {}
    total = len(episode_metadata)
    for index, (episode_id, episode) in enumerate(sorted(episode_metadata.items()), 1):
        replay_path = output / "replays" / f"episode-{episode_id}.json.gz"
        opening_path = output / "openings" / f"episode-{episode_id}.opening.json.gz"
        if replay_path.exists() and opening_path.exists():
            with gzip.open(replay_path, "rb") as handle:
                official_replay_sha = hashlib.sha256(handle.read()).hexdigest()
            with gzip.open(opening_path, "rt", encoding="utf-8") as handle:
                cached_opening = json.load(handle)
            replay_hashes[str(episode_id)] = {
                "official_replay_sha256": official_replay_sha,
                "replay_gzip_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
                "opening_gzip_sha256": hashlib.sha256(opening_path.read_bytes()).hexdigest(),
                "environment_seed": cached_opening["environment_seed"],
                "first_shop": cached_opening["first_shop"],
            }
            continue
        with tempfile.TemporaryDirectory(prefix=f"kaggriculture-episode-{episode_id}-") as temporary:
            api.competition_episode_replay(episode_id, path=temporary, quiet=True)
            downloaded = Path(temporary) / f"episode-{episode_id}-replay.json"
            raw = downloaded.read_bytes()
            replay = json.loads(raw)
            opening = _opening_record(replay, episode)
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            partial_replay = replay_path.with_suffix(replay_path.suffix + ".partial")
            with downloaded.open("rb") as source, gzip.GzipFile(
                filename=str(partial_replay), mode="wb", compresslevel=9, mtime=0
            ) as target:
                shutil.copyfileobj(source, target)
            partial_replay.replace(replay_path)
            opening_sha = _write_gzip_json(opening_path, opening)
            replay_hashes[str(episode_id)] = {
                "official_replay_sha256": hashlib.sha256(raw).hexdigest(),
                "replay_gzip_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
                "opening_gzip_sha256": opening_sha,
                "environment_seed": opening["environment_seed"],
                "first_shop": opening["first_shop"],
            }
        _write_json(output / "artifact-hashes.json", replay_hashes)
        print(f"replay {index:03d}/{total} episode={episode_id}", flush=True)

    snapshot_sha = hashlib.sha256((output / "leaderboard-and-selection.json").read_bytes()).hexdigest()
    metadata_sha = hashlib.sha256((output / "episode-metadata.json").read_bytes()).hexdigest()
    _write_json(output / "collection.json", {
        "complete": len(replay_hashes) == total,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "leaderboard_and_selection_sha256": snapshot_sha,
        "episode_metadata_sha256": metadata_sha,
        "episodes": total,
        "strategies": 15,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--same-opponent", type=int, default=3)
    parser.add_argument("--cross-opponents", type=int, default=8)
    args = parser.parse_args()
    collect(args.output, args.same_opponent, args.cross_opponents)


if __name__ == "__main__":
    main()
