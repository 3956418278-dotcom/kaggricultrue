"""Official replay -> fixed daily intent + demonstrated execution.

Effects are obtained from the official unit transition, not a verb taxonomy.
Locations of existing assets are constraints; newly created entities have
placement domains and stable logical identities, independent of worker routes.
No future replay actions are supplied to the planner input.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official

from src.kaggriculture_agent.economics import (
    ActionDimension, CashDimension, EconomicCommitment, LandDimension,
    OccupancyInterval, PhysicalDimension, RevenueDimension, TimeDimension,
    TimedAmount, WorkAmount,
)
from src.kaggriculture_agent.planner import Plan

SCHEMA_VERSION = "player-day-v1"


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [jsonable(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def observation(replay, frame, side):
    """Kaggle omits shared step on side 1; never default it to zero."""
    obs = deepcopy(replay["steps"][frame][side]["observation"])
    shared = replay["steps"][frame][0]["observation"]
    for key in ("farms", "market", "town", "step", "day", "hour"):
        if key not in obs and key in shared:
            obs[key] = deepcopy(shared[key])
    obs.setdefault("step", frame)
    obs.setdefault("day", frame // 24)
    obs.setdefault("hour", frame % 24)
    obs["player"] = side
    obs.pop("remainingOverageTime", None)
    if (obs["step"], obs["day"], obs["hour"]) != (frame, frame // 24, frame % 24):
        raise ValueError(f"inconsistent replay clock at {frame}/{side}")
    return obs


def validate_replay(replay):
    """Reexecute every joint action with the original seed and official engine."""
    if replay.get("name") != "kaggriculture" or replay.get("module_version") != "1.32.7":
        raise ValueError("incompatible recorded environment")
    if len(replay.get("steps", [])) != 720 or replay.get("statuses") != ["DONE", "DONE"]:
        raise ValueError("not a complete default-horizon episode")
    config = deepcopy(replay["configuration"])
    seed = replay.get("info", {}).get("seed")
    if not isinstance(seed, int):
        raise ValueError("missing reproducible seed")
    config["seed"] = seed
    env = make("kaggriculture", configuration=config)
    env.reset(2)
    for frame in range(720):
        if frame:
            env.step([deepcopy(s.get("action") or {}) for s in replay["steps"][frame]])
        for side in range(2):
            expected = observation(replay, frame, side)
            actual = deepcopy(dict(env.state[side].observation))
            actual.pop("remainingOverageTime", None)
            actual["step"] = frame
            if actual != expected:
                keys = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
                raise ValueError(f"official transition mismatch frame={frame} side={side} fields={keys}")
    return {"official_joint_transitions": 719, "observation_checks": 1440,
            "terminal_rewards_match": [s.reward for s in env.state] == replay["rewards"]}


def _physical(private):
    result = Counter(private.get("shed", {}))
    for inventory in private.get("inventories", []):
        result.update(inventory)
    result.update({f"{k}_SEED": v for k, v in private.get("seeds", {}).items()})
    return result


def _changes(before, after):
    """Presence-aware field changes; no operation-name classification."""
    a = before if isinstance(before, dict) else {"$tile": before}
    b = after if isinstance(after, dict) else {"$tile": after}
    return {k: {"before_present": k in a, "before": a.get(k),
                "after_present": k in b, "after": b.get(k)}
            for k in sorted(set(a) | set(b)) if (k in a, a.get(k)) != (k in b, b.get(k))}


def unit_effects(obs, action):
    """Ordered unit effects with official atomic-seed validation.

    Aggregate inventory erases pickup/drop/carry implementation. Only direct
    asset changes qualify as economic work. No-op attempts are retained for
    inspection, but are not guessed into fulfilled goals.
    """
    farm, private = deepcopy(obs["farms"][obs["player"]]), deepcopy(obs["private"])
    units = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    demand = Counter(a[1] for a in units if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT")
    blocked = {c for c, n in demand.items() if n > private.get("seeds", {}).get(c, 0)}
    events, noops = [], []
    for worker, submitted in enumerate(units):
        if worker > len(farm["hands"]):
            noops.append({"worker": worker, "action": submitted, "reason": "nonexistent worker"})
            continue
        pos = tuple(farm["farmer"] if worker == 0 else farm["hands"][worker - 1])
        before = deepcopy(farm["tiles"][pos[1]][pos[0]])
        physical = _physical(private)
        pre_farm, pre_private = deepcopy(farm), deepcopy(private)
        allowed = ["PASS"] if (isinstance(submitted, list) and len(submitted) >= 2
                    and submitted[0] == "PLANT" and submitted[1] in blocked) else submitted
        official._apply_unit_action(farm, private, worker, allowed, len(farm["tiles"]), obs["day"], 24, 100)
        after = deepcopy(farm["tiles"][pos[1]][pos[0]])
        fields = _changes(before, after)
        amounts = _physical(private)
        delta = {k: amounts[k] - physical[k] for k in sorted(set(amounts) | set(physical))
                 if amounts[k] != physical[k]}
        if fields:
            events.append({"position": pos, "before": before, "after": after,
                           "fields": fields, "physical_delta": delta, "worker": worker,
                           "action": submitted, "achieved": True})
        elif farm == pre_farm and private == pre_private and submitted not in (["PASS"], None, []):
            # Only promote a failed attempt if supplying a missing input makes
            # the SAME official action create an economic effect. No hidden
            # goals are inferred from movement, pass, or an unknown operation.
            probe_farm, probe_private = deepcopy(pre_farm), deepcopy(pre_private)
            inv = probe_private["inventories"][worker]
            for item in (*official.ANIMALS, "WHEAT", "FERTILIZER"):
                inv[item] = max(1, inv.get(item, 0))
            for crop in official.CROPS:
                probe_private["seeds"][crop] = max(1, probe_private["seeds"].get(crop, 0))
            probe_before = _physical(probe_private)
            official._apply_unit_action(probe_farm, probe_private, worker, submitted,
                                       len(farm["tiles"]), obs["day"], 24, 100)
            probe_after = probe_farm["tiles"][pos[1]][pos[0]]
            changed = _changes(before, probe_after)
            if changed:
                probe_amounts = _physical(probe_private)
                events.append({"position": pos, "before": before, "after": probe_after,
                    "fields": changed, "physical_delta": {k: probe_amounts[k] - probe_before[k]
                        for k in set(probe_amounts) | set(probe_before) if probe_amounts[k] != probe_before[k]},
                    "worker": worker, "action": submitted, "achieved": False})
            noops.append({"worker": worker, "action": submitted,
                          "reason": "atomic seeds" if allowed != submitted else "no effect",
                          "input_repair_exposes_economic_effect": bool(changed)})
    return events, noops


def reconstruct_day(replay, side, day, qualification, replay_sha256):
    start, end = day * 24, min((day + 1) * 24, 719)
    opening = observation(replay, start, side)
    ending = observation(replay, end, side)
    initial_tiles = opening["farms"][side]["tiles"]
    new_land = [q for q in ending["farms"][side]["unlocked_quadrants"]
                if q not in opening["farms"][side]["unlocked_quadrants"]]
    expansion = {(x, y) for y, row in enumerate(initial_tiles) for x, tile in enumerate(row)
                 if tile == "LOCKED" and official._quadrant_of(x, y, len(initial_tiles)) in new_land}
    entities = {(x, y): f"existing:{x}:{y}" for y, row in enumerate(initial_tiles)
                for x, tile in enumerate(row) if isinstance(tile, dict)}
    entity_domains, goals, trace = {}, {}, []
    cleared = set()
    for frame in range(start, end):
        obs = observation(replay, frame, side)
        action = deepcopy(replay["steps"][frame + 1][side].get("action") or {})
        events, noops = unit_effects(obs, action)
        step_events = []
        for event in events:
            pos, before, after = event["position"], event["before"], event["after"]
            occupies_structure = (isinstance(before, dict) and isinstance(after, dict)
                                  and "animal" not in before and "animal" in after)
            if not isinstance(before, dict) or (occupies_structure and not entities.get(pos, "").startswith("new:")):
                entity = f"new:{len(entity_domains)}"
                # All initially identical terrain is execution freedom. Other
                # terrain is not silently assumed available or unlocked.
                entity_domains[entity] = tuple(sorted({(x, y) for y, row in enumerate(initial_tiles)
                                              for x, tile in enumerate(row) if tile == before}
                                              | ((cleared | expansion) if before is None else set())))
                if event["achieved"]:
                    entities[pos] = entity
            else:
                entity = entities.get(pos, f"existing:{pos[0]}:{pos[1]}")
            fixed = None if entity.startswith("new:") else pos
            requirement = {"entity": entity, "fields": event["fields"], "physical_delta": event["physical_delta"]}
            key = digest(requirement)
            if key not in goals:
                goals[key] = {**requirement, "position": fixed, "achieved": False,
                              "attempted": False, "count": 0}
            goal = goals[key]
            goal["achieved"] |= event["achieved"]
            goal["attempted"] |= not event["achieved"]
            goal["count"] += int(event["achieved"])
            step_events.append({**event, "goal": key, "entity": entity})
            if after is None and event["achieved"]:
                entities.pop(pos, None)
                cleared.add(pos)
        trace.append({"step": frame, "state_before": obs, "action": action, "effects": step_events, "noops": noops,
                      "opponent_market": deepcopy((replay["steps"][frame + 1][1-side].get("action") or {}).get("market", []))})
    commitments, domains = [], {}
    for identifier, goal in sorted(goals.items()):
        position, entity = goal["position"], goal["entity"]
        delta = goal["physical_delta"]
        inputs = tuple(TimedAmount(start, k, -v) for k, v in delta.items() if v < 0)
        outputs = tuple(TimedAmount(end, k, v) for k, v in delta.items() if v > 0)
        commitments.append(EconomicCommitment(identifier=identifier, kind="STATE_EFFECT", target=position,
            existing=position is not None, cash=CashDimension(),
            time=TimeDimension(start, end - 1, end - 1, (end - 1,)),
            land=LandDimension((OccupancyInterval(position, start, end - 1),)),
            actions=ActionDimension((WorkAmount(day, "STATE_EFFECT", max(1, goal["count"]), position=position,
                                               deadline_step=end - 1),)),
            physical=PhysicalDimension(inputs=inputs, outputs=outputs), revenue=RevenueDimension(),
            metadata={"entity": entity, "required_effect": goal["fields"],
                      "demonstrated_count": goal["count"], "attempt_observed": goal["attempted"],
                      "completion_observed": goal["achieved"]}))
        if entity in entity_domains:
            domains[identifier] = entity_domains[entity]
    land = tuple(EconomicCommitment(identifier=f"land:{q}", kind="LAND", target=None,
        existing=False, cash=CashDimension(), time=TimeDimension(start, end-1, end-1),
        land=LandDimension(capacity_created=25), actions=ActionDimension(),
        physical=PhysicalDimension(outputs=(TimedAmount(end-1, "LAND_TILE_CAPACITY", 25),)),
        revenue=RevenueDimension(), metadata={"quadrant": q}) for q in new_land)
    plan = Plan(obligations=tuple(p for p in commitments if p.existing),
        selected=tuple(p for p in commitments if not p.existing), support=land, rejected={},
        fertilize_targets=frozenset(), animal_purchases=(), buy_land=bool(new_land),
        feed_reserve=0, fertilizer_reserve=0, day=day, formed_step=start,
        max_hands=None, placement_domains=domains,
        diagnostics={"source": "official-transition-effects", "economic_valuation": "not inferred",
                     "staffing_constraint": "actual cash and Fibonacci costs; no reference staffing cap",
                     "terminal_day": day == 29})
    episode_id = replay["info"]["EpisodeId"]
    for turn in trace:
        for event in turn["effects"]:
            if event["achieved"] and event["entity"] in entity_domains:
                if tuple(event["position"]) not in entity_domains[event["entity"]]:
                    raise ValueError("demonstrated placement violates reconstructed domain")
    return jsonable({"schema_version": SCHEMA_VERSION, "sample_id": f"{episode_id}:{side}:{day}",
        "episode_id": episode_id, "side": side, "day": day, "actionable_turns": end-start,
        "environment": {"module_version": replay["module_version"], "configuration": replay["configuration"],
                        "seed": replay["info"]["seed"]},
        "provenance": {"replay_sha256": replay_sha256, "qualification": qualification},
        "day_start_state": opening, "plan": jsonable(asdict(plan)),
        "demonstrated_realization": trace, "day_end_state": ending,
        "diagnostics": {"goals": len(goals), "attempt_only_goals": sum(not g["achieved"] for g in goals.values()),
                        "selling_in_plan": False, "financing_control_required": "not assumed; assess on executable divergence"}})


def write_shard(path, samples):
    """Atomic, content-addressable episode checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as output:
            for sample in samples:
                output.write((json.dumps(sample, separators=(",", ":")) + "\n").encode())
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_samples(path):
    with gzip.open(path, "rt") as source:
        for line in source:
            yield json.loads(line)
