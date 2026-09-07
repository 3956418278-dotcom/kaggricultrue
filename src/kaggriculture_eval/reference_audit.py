"""Streaming structural/semantic checks for collected player-day references.

This is an extraction audit, not a claim that demonstrated execution is optimal.
Official transition parity is established separately before a shard is written.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from .player_days import digest, read_samples
from .reference_pipeline import qualify_sides


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit_sample(sample):
    day, side = sample["day"], sample["side"]
    start, end = day * 24, min((day + 1) * 24, 719)
    trace, plan = sample["demonstrated_realization"], sample["plan"]
    check(sample["actionable_turns"] == end - start == len(trace), "day length")
    check(sample["day_start_state"]["step"] == start, "opening clock")
    check(sample["day_end_state"]["step"] == end, "ending clock")
    check(trace[0]["state_before"] == sample["day_start_state"], "opening state")
    check(plan["day"] == day and plan["formed_step"] == start, "Plan clock")
    check("hire_count" not in plan and plan["max_hands"] is None, "staffing leaked into Plan")
    goals = {p["identifier"]: p for p in plan["obligations"] + plan["selected"]}
    check(len(goals) == len(plan["obligations"]) + len(plan["selected"]), "duplicate goals")
    counts, attempts, seen, pending = Counter(), set(), set(), {}
    metrics = Counter(player_days=1)
    for offset, turn in enumerate(trace):
        check(turn["step"] == start + offset == turn["state_before"]["step"], "trace clock")
        check(turn["state_before"]["player"] == side, "trace side")
        for event in turn["effects"]:
            key = event["goal"]
            check(key in goals, "effect has no goal")
            goal = goals[key]
            metadata = goal["metadata"]
            check(goal["kind"] == "STATE_EFFECT", "nonsemantic work goal")
            check(metadata["entity"] == event["entity"], "entity mismatch")
            check(metadata["required_effect"] == event["fields"], "required effect mismatch")
            check(not ({"worker", "action", "hire_count", "market"} & metadata.keys()), "execution in Plan")
            check(key == digest({"entity": event["entity"], "fields": event["fields"],
                                 "physical_delta": event["physical_delta"]}), "goal content identity")
            if goal["target"] is not None:
                check(goal["target"] == event["position"], "existing position changed")
            else:
                check(event["position"] in plan["placement_domains"][key], "placement outside domain")
            retry = (tuple(event["position"]), digest(event["fields"]))
            if retry in pending:
                check(pending[retry] == event["entity"], "retry created a phantom entity")
            if event["achieved"]:
                counts[key] += 1
                pending.pop(retry, None)
                if event["after"] is None:
                    pending = {k: v for k, v in pending.items() if k[0] != tuple(event["position"])}
            else:
                attempts.add(key)
                pending[retry] = event["entity"]
            seen.add(key)
        action = turn["action"]
        workers = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
        metrics["available_worker_turns"] += 1 + len(turn["state_before"]["farms"][side]["hands"])
        for operation in workers:
            if isinstance(operation, list) and operation:
                metrics["action_" + str(operation[0])] += 1
        metrics["unresolved_noops"] += sum(not n.get("input_repair_exposes_economic_effect", False)
                                             for n in turn["noops"])
    check(seen == goals.keys(), "goal has no demonstrated effect or attempt")
    for key, goal in goals.items():
        metadata = goal["metadata"]
        check(metadata["demonstrated_count"] == counts[key], "effect multiplicity")
        check(metadata["completion_observed"] == bool(counts[key]), "achievement flag")
        check(metadata["attempt_observed"] == (key in attempts), "attempt flag")
        metrics["attempt_only_goals"] += not counts[key]
    check(all(p["kind"] == "LAND" for p in plan["support"]), "non-land support in reconstructed Plan")
    actual_land = set(sample["day_end_state"]["farms"][side]["unlocked_quadrants"]) - set(
        sample["day_start_state"]["farms"][side]["unlocked_quadrants"])
    check({p["metadata"]["quadrant"] for p in plan["support"]} == actual_land, "land effects")
    check(metrics["attempt_only_goals"] == sample["diagnostics"]["attempt_only_goals"], "attempt summary")
    metrics["goals"] = len(goals)
    metrics["empty_plans"] = not goals and not plan["support"]
    return metrics


def audit_collection(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "extraction-manifest.json").read_text())
    selection = json.loads((directory / "selection.json").read_text())
    metadata = json.loads((directory / "episode-metadata.json").read_text())
    totals, teams, submissions = Counter(), Counter(), Counter()
    samples_seen = set()
    for eid, record in sorted(manifest["episodes"].items()):
        if record["status"] != "extracted":
            continue
        shard = directory / record["shard"]
        check(shard.resolve().is_relative_to(directory.resolve()), "unsafe shard path")
        check(hashlib.sha256(shard.read_bytes()).hexdigest() == record["sha256"], "shard hash")
        check(record["validation"] == {"official_joint_transitions": 719, "observation_checks": 1440,
                                       "terminal_rewards_match": True}, "missing official parity")
        qualified = qualify_sides(metadata["episodes"][eid], selection["leaderboard"])
        expected = {f"{eid}:{side}:{day}" for side in qualified for day in range(30)}
        actual, ends, per_side = set(), {}, Counter()
        for sample in read_samples(shard):
            sid, side = sample["sample_id"], sample["side"]
            check(sid not in samples_seen and sid in expected, "duplicate or unqualified sample")
            check(sample["provenance"]["qualification"] == qualified[side], "qualification mismatch")
            check(sample["provenance"]["replay_sha256"] == record["replay_sha256"], "replay hash")
            if side in ends:
                check(ends[side] == sample["day_start_state"], "day boundary discontinuity")
            ends[side] = sample["day_end_state"]
            totals.update(audit_sample(sample))
            samples_seen.add(sid)
            actual.add(sid)
            per_side[side] += 1
        check(actual == expected and len(actual) == record["player_days"], "incomplete qualified side")
        for side in per_side:
            teams[str(qualified[side]["team_id"])] += 1
            submissions[str(qualified[side]["submission_id"])] += 1
        totals["episodes"] += 1
        totals["qualified_sides"] += len(per_side)
    check(totals["player_days"] == manifest["player_days"], "collection count")
    return {"passed": True, "complete": manifest["complete"], "identity": manifest["identity"],
            "totals": dict(totals), "team_sides": dict(teams), "submission_sides": dict(submissions),
            "scope": "hashes, qualification, complete day chains, effect/goal agreement, placement and Plan boundary; not optimality",
            "limitations": ["unresolved no-ops are not hidden-intent goals", "standalone stock accumulation is not a work goal",
                            "placement prerequisites and state-effect equivalence need benchmark-evaluator review",
                            "action-type counts are diagnostics, never a waste score"]}
