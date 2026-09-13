#!/usr/bin/env python3
"""Turn an official top-opening replay census into 15 auditable strategies."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any


FIRST_SHOP_STEP = 72
OPENING_TURNS = 192


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_gzip_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    temporary = path.with_suffix(path.suffix + ".partial")
    with gzip.GzipFile(filename=str(temporary), mode="wb", compresslevel=9, mtime=0) as handle:
        handle.write(payload)
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _side(opening: dict[str, Any], submission_id: int) -> dict[str, Any]:
    return next(side for side in opening["sides"] if side["submission_id"] == submission_id)


def _normalized_action(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "farmer": action.get("farmer") or ["PASS"],
        "hands": action.get("hands") or [],
        "market": action.get("market") or [],
    }


def _difference(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any] | None:
    for step, (raw_left, raw_right) in enumerate(zip(left, right)):
        a, b = _normalized_action(raw_left), _normalized_action(raw_right)
        if a == b:
            continue
        components = [name for name in ("farmer", "hands", "market") if a[name] != b[name]]
        return {
            "step": step,
            "day": step // 24 + 1,
            "hour": step % 24,
            "before_first_shop": step < FIRST_SHOP_STEP,
            "components": components,
            "left_action": raw_left,
            "right_action": raw_right,
        }
    return None


def _tile_counts(tiles: list[list[Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in tiles:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                counts[f"CROP:{tile.get('crop')}"] += 1
            elif tile.get("animal"):
                counts[f"ANIMAL:{tile.get('animal')}"] += 1
            elif tile.get("kind"):
                counts[f"TILE:{tile.get('kind')}"] += 1
    return dict(sorted(counts.items()))


def _daily_summary(side: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    actions = side["actions"]
    for day in range(8):
        start, end = day * 24, (day + 1) * 24
        market = [
            {"step": step, "orders": action.get("market", [])}
            for step, action in enumerate(actions[start:end], start)
            if action.get("market")
        ]
        units = Counter()
        for action in actions[start:end]:
            for unit in [action.get("farmer", ["PASS"]), *(action.get("hands") or [])]:
                units[str(unit[0]) if unit else "PASS"] += 1
        ending = side["daily_state"][day + 1]
        result.append({
            "day": day + 1,
            "market_actions": market,
            "unit_action_counts": dict(sorted(units.items())),
            "end_money": ending["money"],
            "end_owned_land": ending["owned_land"],
            "end_assets": _tile_counts(ending["tiles"]),
            "end_shed": {k: v for k, v in ending["shed"].items() if v},
            "shops_visible_next_day": ending["shops"],
        })
    return result


def _first_difference_text(difference: dict[str, Any] | None) -> str:
    if difference is None:
        return "same through D8"
    return (
        f"step {difference['step']} (D{difference['day']} h{difference['hour']:02d}; "
        f"{'+'.join(difference['components'])})"
    )


def analyze(root: Path) -> None:
    selection = _load_json(root / "leaderboard-and-selection.json")
    hashes = _load_json(root / "artifact-hashes.json")
    audit_path = root / "opening-transition-audit.json"
    audit = _load_json(audit_path) if audit_path.exists() else None
    opening_cache: dict[int, dict[str, Any]] = {}

    def opening(episode_id: int) -> dict[str, Any]:
        if episode_id not in opening_cache:
            opening_cache[episode_id] = _load_gzip_json(
                root / "openings" / f"episode-{episode_id}.opening.json.gz"
            )
        return opening_cache[episode_id]

    analyses: list[dict[str, Any]] = []
    strategy_hashes: dict[str, str] = {}
    report = [
        "# Official Kaggle top-15 opening census",
        "",
        f"Leaderboard frozen at `{selection['observed_at']}`. D1-D8 means official steps 0-191. ",
        "A difference across opponent and seed is observational; it does not by itself identify which one caused it.",
        "" if audit is None else (
            f"Pinned official-transition audit: `{len(audit['episodes'])}` episodes, "
            f"`{sum(row['observation_checks'] for row in audit['episodes'].values())}` observation checks, "
            f"passed=`{audit.get('passed', False)}`."
        ),
        "",
        "| Rank | Team / submission | Canonical episode | Same opponent, other seeds | Same first shop, different opponents |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for strategy in selection["strategies"]:
        rank = int(strategy["rank"])
        submission_id = int(strategy["primary_submission_id"])
        pool = strategy["candidate_pool"]
        same_rows = pool["same_opponent_candidates"]
        canonical_relation = same_rows[0]
        canonical_opening = opening(int(canonical_relation["episode_id"]))
        canonical_side = _side(canonical_opening, submission_id)
        canonical_actions = canonical_side["actions"]

        same_comparisons = []
        for relation in same_rows[1:]:
            variant_opening = opening(int(relation["episode_id"]))
            variant_side = _side(variant_opening, submission_id)
            same_comparisons.append({
                "episode_id": int(relation["episode_id"]),
                "environment_seed": variant_opening["environment_seed"],
                "first_shop": variant_opening["first_shop"],
                "difference_from_canonical": _difference(canonical_actions, variant_side["actions"]),
            })

        by_shop: dict[tuple[str, ...], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for relation in pool["cross_opponent_candidates"]:
            item = opening(int(relation["episode_id"]))
            by_shop[tuple(item["first_shop"])].append((relation, item))
        eligible = [rows for rows in by_shop.values() if len(rows) >= 2]
        if not eligible:
            raise ValueError(f"rank {rank}: no same-shop cross-opponent pair")
        pair_group = max(
            eligible,
            key=lambda rows: (len(rows), max(int(row[0]["episode_id"]) for row in rows)),
        )
        left_relation, left_opening = pair_group[0]
        right_relation, right_opening = pair_group[1]
        left_side = _side(left_opening, submission_id)
        right_side = _side(right_opening, submission_id)
        cross_comparison = {
            "first_shop": left_opening["first_shop"],
            "left": {
                "episode_id": int(left_relation["episode_id"]),
                "environment_seed": left_opening["environment_seed"],
                "opponent_submission_id": left_relation["opponent_submission_id"],
                "opponent_team_name": left_relation["opponent_team_name"],
            },
            "right": {
                "episode_id": int(right_relation["episode_id"]),
                "environment_seed": right_opening["environment_seed"],
                "opponent_submission_id": right_relation["opponent_submission_id"],
                "opponent_team_name": right_relation["opponent_team_name"],
            },
            "difference": _difference(left_side["actions"], right_side["actions"]),
        }

        all_relations = {
            int(relation["episode_id"]): relation
            for relation in (*pool["same_opponent_candidates"], *pool["cross_opponent_candidates"])
        }
        variants = []
        for episode_id, relation in sorted(all_relations.items(), reverse=True):
            item = opening(episode_id)
            own = _side(item, submission_id)
            opponent = next(side for side in item["sides"] if side["submission_id"] != submission_id)
            variants.append({
                "episode_id": episode_id,
                "environment_seed": item["environment_seed"],
                "first_shop": item["first_shop"],
                "own_side": own["side"],
                "opponent_team_id": opponent["team_id"],
                "opponent_team_name": opponent["team_name"],
                "opponent_submission_id": opponent["submission_id"],
                "actions": own["actions"],
            })

        same_pre_shop = any(
            comparison["difference_from_canonical"] is not None
            and comparison["difference_from_canonical"]["before_first_shop"]
            for comparison in same_comparisons
        )
        different_shop_pairs = [
            comparison for comparison in same_comparisons
            if comparison["first_shop"] != canonical_opening["first_shop"]
        ]
        shop_only_observed = bool(different_shop_pairs) and all(
            comparison["difference_from_canonical"] is None
            or not comparison["difference_from_canonical"]["before_first_shop"]
            for comparison in different_shop_pairs
        )
        analysis = {
            "rank": rank,
            "team_id": strategy["team_id"],
            "team_name": strategy["team_name"],
            "leaderboard_score": strategy["leaderboard_score"],
            "submission_id": submission_id,
            "canonical": {
                "episode_id": int(canonical_relation["episode_id"]),
                "environment_seed": canonical_opening["environment_seed"],
                "own_side": canonical_side["side"],
                "opponent_submission_id": canonical_relation["opponent_submission_id"],
                "opponent_team_name": canonical_relation["opponent_team_name"],
                "first_shop": canonical_opening["first_shop"],
                "daily_summary": _daily_summary(canonical_side),
            },
            "same_opponent_same_seat": {
                "opponent_submission_id": pool["same_opponent_submission_id"],
                "own_seat": pool["same_own_seat"],
                "comparisons": same_comparisons,
                "pre_shop_variation_observed": same_pre_shop,
                "different_first_shop_only_changes_at_or_after_reveal": shop_only_observed,
            },
            "same_first_shop_different_opponents_and_seeds": cross_comparison,
            "evidence_note": "Different-opponent comparisons also use different environment seeds; observed variation is not causal attribution.",
        }
        analyses.append(analysis)

        strategy_record = {
            "schema_version": "official-top-opening-strategy-v1",
            "leaderboard_observed_at": selection["observed_at"],
            "rank": rank,
            "team_id": strategy["team_id"],
            "team_name": strategy["team_name"],
            "leaderboard_score": strategy["leaderboard_score"],
            "submission_id": submission_id,
            "canonical": {
                **analysis["canonical"],
                "actions": canonical_actions,
                "daily_state": canonical_side["daily_state"],
            },
            "variants": variants,
            "comparisons": {
                "same_opponent_same_seat": analysis["same_opponent_same_seat"],
                "same_first_shop_different_opponents_and_seeds": cross_comparison,
            },
            "source_artifacts": {
                str(episode_id): hashes[str(episode_id)] for episode_id in all_relations
            },
        }
        safe = re.sub(r"[^a-z0-9]+", "-", strategy["team_name"].lower()).strip("-") or "team"
        filename = f"{rank:02d}-{safe}-{submission_id}.json.gz"
        strategy_hashes[filename] = _write_gzip_json(root / "strategies" / filename, strategy_record)

        same_text = "; ".join(
            f"{item['episode_id']}/{item['environment_seed']}: "
            f"{_first_difference_text(item['difference_from_canonical'])}"
            for item in same_comparisons
        )
        cross = cross_comparison
        cross_text = (
            f"{','.join(cross['first_shop'])}: {cross['left']['episode_id']} vs "
            f"{cross['right']['episode_id']}, {_first_difference_text(cross['difference'])}"
        )
        report.append(
            f"| {rank} | {strategy['team_name']} / `{submission_id}` | "
            f"`{canonical_relation['episode_id']}` seed `{canonical_opening['environment_seed']}`; "
            f"{','.join(canonical_opening['first_shop'])} | {same_text} | {cross_text} |"
        )

    _write_json(root / "opening-analysis.json", {
        "schema_version": "official-top-opening-analysis-v1",
        "leaderboard_observed_at": selection["observed_at"],
        "first_shop_visible_at_step": FIRST_SHOP_STEP,
        "opening_turns": OPENING_TURNS,
        "transition_audit": None if audit is None else {
            "complete": audit.get("complete"),
            "passed": audit.get("passed"),
            "episodes": len(audit.get("episodes", {})),
            "observation_checks": sum(
                row["observation_checks"] for row in audit.get("episodes", {}).values()
            ),
        },
        "strategies": analyses,
    })
    _write_json(root / "strategy-artifact-hashes.json", strategy_hashes)
    report.extend((
        "",
        "## Interpretation boundary",
        "",
        "The canonical trace is the newest episode in the prespecified most-frequent same-opponent/same-seat group; outcome was not used. "
        "Each strategy gzip contains the exact submitted D1-D8 actions for the canonical episode and every sampled variant. "
        "The table records exact first divergence, not inferred intent or causal opponent response.",
        "",
    ))
    (root / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("collection", type=Path)
    args = parser.parse_args()
    analyze(args.collection)


if __name__ == "__main__":
    main()
