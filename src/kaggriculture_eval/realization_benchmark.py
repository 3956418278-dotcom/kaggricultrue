"""Same-state/fixed-Plan development comparisons, independent of solver scores.

The candidate receives only opening state and Plan. Demonstrated actions are
used exclusively on the reference/evaluation side of this boundary.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from src.kaggriculture_agent.intent import compile_intent, matches
from src.kaggriculture_agent.state import reconstruct
from src.kaggriculture_agent.temporal_model import solve_temporal
from .intraday_benchmark import oracle_environment, oracle_step
from .plan_io import plan_from_dict
from .player_days import unit_effects


def episode_partition(episode_id):
    # Frozen label, shared by every day and side of an episode.
    score = int(sha256(f"temporal-reference-split-20260906:{episode_id}".encode()).hexdigest(), 16)
    return "held-out" if score % 5 == 0 else "development"


def source_identity():
    roots = (Path("src/kaggriculture_agent"), Path("src/kaggriculture_eval"))
    paths = [p for root in roots for p in sorted(root.glob("*.py"))]
    paths += [Path("scripts/benchmark_realizations.py"), Path("requirements-planning.txt"), Path("requirements.lock")]
    return {str(p): sha256(p.read_bytes()).hexdigest() for p in paths}


def score_effects(problem, events, placements, final_state):
    """Maximum semantic matching; never trust a planner's self-reported credit.

    One effect cannot satisfy several copies of the same requested operation.
    Augmenting paths avoid a greedy evaluator accidentally losing feasible
    matches between overlapping effect requirements.
    """
    goals = problem.goals
    positions = {e.identifier: (e.positions[0] if e.existing and len(e.positions) == 1 else placements.get(e.identifier))
                 for e in problem.entities}
    for e in problem.entities:
        if positions[e.identifier] is not None and tuple(positions[e.identifier]) not in e.positions:
            raise ValueError(f"realization placement outside fixed Plan domain: {e.identifier}")
    edges = []
    for goal in goals:
        edges.append([i for i, event in enumerate(events) if event["achieved"]
            and tuple(event["position"]) == positions[goal.entity]
            and event["step"] <= goal.deadline
            and matches(goal, event["before"], event["after"], event["physical_delta"], event["action"])])
    matching = {}

    def augment(goal, seen):
        for event in edges[goal]:
            if event in seen:
                continue
            seen.add(event)
            if event not in matching or augment(matching[event], seen):
                matching[event] = goal
                return True
        return False

    for g in range(len(goals)):
        augment(g, set())
    completed = {goals[g].identifier for g in matching.values()}
    completed.update(f"land:{q}" for q in problem.land if q in final_state.unlocked_quadrants)
    all_goals = {g.identifier for g in goals} | {f"land:{q}" for q in problem.land}
    return {"goal_count": len(all_goals), "completed": len(completed),
            "completed_ids": sorted(completed), "unfulfilled": sorted(all_goals - completed)}


def execute_reference_controlled(start, actions):
    """Official deterministic farm replay, with the opponent passing.

    Own demonstrated sales/acquisitions stay in the reference realization, not
    Plan. The recorded reference is scored separately so any background-induced
    loss of demonstrated work is visible rather than silently blamed on routing.
    """
    env = oracle_environment(start)
    events, trace, operations = [], [], Counter()
    capacity = 0
    for action in actions:
        obs = deepcopy(dict(env.state[0].observation))
        obs["player"] = 0
        step_events, noops = unit_effects(obs, action)
        events.extend({**e, "step": obs["step"]} for e in step_events)
        units = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
        current = reconstruct(obs)
        capacity += len(current.workers)
        operations.update(a[0] for a in units[:len(current.workers)])
        result = oracle_step(env, units, action.get("market", []))
        trace.append({"state_before": obs, "action": action, "effects": step_events, "noops": noops})
    return result, events, trace, {"operations": dict(operations), "available_worker_turns": capacity,
        "used_worker_turns": sum(n for op, n in operations.items() if op != "PASS"),
        "movement": sum(operations[o] for o in ("NORTH", "SOUTH", "EAST", "WEST")),
        "pickup_drop_place": sum(operations[o] for o in ("PICKUP", "DROP", "PLACE"))}


def compare_sample(sample, config, solver=solve_temporal):
    # Explicitly reject nondefault physics; do not silently change the game.
    expected = {"boardSize": 10, "turnsPerDay": 24, "shedCapacity": 100,
                "episodeSteps": 720, "farmHandCostMult": 1, "maxMarketOrdersPerTurn": 10,
                "townShopSellInterval": 4, "townCenterSellInterval": 24}
    if sample["environment"]["module_version"] != "1.32.7":
        raise ValueError("reference environment identity mismatch")
    for key, value in expected.items():
        if sample["environment"]["configuration"].get(key, value) != value:
            raise ValueError(f"unsupported reference configuration: {key}")
    start, plan = reconstruct(sample["day_start_state"]), plan_from_dict(sample["plan"])
    problem = compile_intent(start, plan)
    candidate = solver(start, plan, config)
    # No demonstrated action, staffing, assignment or placement entered solve.
    actions = [{"farmer": e.worker_actions[0], "hands": list(e.worker_actions[1:]),
                "market": list(e.market_orders)} for e in candidate.executions]
    ours_end, ours_events, ours_trace, ours_effort = execute_reference_controlled(start, actions)
    reference = sample["demonstrated_realization"]
    ref_end, ref_events, ref_trace, ref_effort = execute_reference_controlled(start, [t["action"] for t in reference])
    recorded_events = [{**e, "step": t["step"]} for t in reference for e in t["effects"]]
    placements = {e["entity"]: tuple(e["position"]) for e in recorded_events if e["achieved"]}
    original = score_effects(problem, recorded_events, placements, reconstruct(sample["day_end_state"]))
    controlled = score_effects(problem, ref_events, placements, ref_end)
    ours = score_effects(problem, ours_events, candidate.placements, ours_end)
    # The independent oracle must agree with the predicted physical realization.
    # Random weeds were disabled, and random shop additions occur only after the
    # measured day; these are not claimed to be future predictions.
    for key in ("money", "tiles", "workers", "shed", "seeds", "hires_today", "unlocked_quadrants", "market_inventory"):
        if getattr(ours_end, key) != getattr(candidate.final_state, key):
            raise AssertionError(f"candidate official parity mismatch: {key}")
    return {"sample_id": sample["sample_id"], "episode_id": sample["episode_id"],
        "partition": episode_partition(sample["episode_id"]), "configuration": asdict(config),
        "qualification": sample["provenance"]["qualification"],
        "recorded_reference": original,
        "reference": {**controlled, **ref_effort, "final_state": asdict(ref_end), "trace": ref_trace},
        "candidate": {**ours, **ours_effort, "final_state": asdict(ours_end), "trace": ours_trace,
                      "diagnostics": candidate.diagnostics, "placements": candidate.placements},
        "background_materially_changes_reference_completion": original["completed_ids"] != controlled["completed_ids"],
        "comparison": {"completion_difference": ours["completed"] - controlled["completed"],
            "equal_completed_goals": ours["completed_ids"] == controlled["completed_ids"],
            "used_worker_turn_difference": ours_effort["used_worker_turns"] - ref_effort["used_worker_turns"]}}
