"""Frozen development evaluation for the public intraday solver contract.

The evaluator gives the candidate only ``OwnedState`` and ``Plan``. A candidate
row is either an exactly validated full Realization or PlanningFailure; there is
no partial-completion score emitted by the production planner.
"""

from hashlib import sha256
from pathlib import Path
from time import perf_counter

from src.kaggriculture_agent import rules
from src.kaggriculture_agent.execution import execute_realization
from src.kaggriculture_agent.intraday import solve_intraday
from src.kaggriculture_agent.realization import PlanningFailure
from src.kaggriculture_agent.state import reconstruct
from src.kaggriculture_agent.valuation import end_value

from .plan_io import plan_from_sample


def episode_partition(episode_id):
    score = int(sha256(
        f"temporal-reference-split-20260906:{episode_id}".encode()
    ).hexdigest(), 16)
    return "held-out" if score % 5 == 0 else "development"


def source_identity():
    roots = (Path("src/kaggriculture_agent"), Path("src/kaggriculture_eval"))
    paths = [path for root in roots for path in sorted(root.glob("*.py"))]
    paths.append(Path("scripts/benchmark_intraday.py"))
    return {str(path): sha256(path.read_bytes()).hexdigest()
            for path in paths if path.exists()}


def _requirement_ids(plan):
    return tuple(project.identifier for project in
                 (*plan.obligations, *plan.selected, *plan.support)
                 if project.kind == "LAND"
                 or project.required_state or project.required_outputs)


def _effort(initial, turns):
    hires_today = initial.hires_today
    workforce = len(initial.workers)
    hire_expenditure = 0
    movement = logistics = used = 0
    for worker_actions, market_orders in turns:
        for action in worker_actions:
            operation = action[0] if action else "PASS"
            used += operation != "PASS"
            movement += operation in ("NORTH", "SOUTH", "EAST", "WEST")
            logistics += operation in ("PICKUP", "DROP") or (
                operation == "PLACE" and len(action) > 2
            )
        for order in market_orders:
            if order and order[0] == "HIRE":
                hire_expenditure += rules.fibonacci_hire_cost(hires_today)
                hires_today += 1
                workforce += 1
    return {
        "workforce": workforce,
        "hire_expenditure": hire_expenditure,
        "worker_actions_used": used,
        "movement": movement,
        "logistics": logistics,
    }


def compare_intraday_sample(sample, solver=solve_intraday):
    opening = reconstruct(sample["day_start_state"])
    plan = plan_from_sample(sample)
    reference_end = reconstruct(sample["day_end_state"])
    requirements = _requirement_ids(plan)
    reference_turns = tuple(
        (
            tuple(tuple(action) for action in (
                [entry["action"].get("farmer", ["PASS"])]
                + entry["action"].get("hands", [])
            )),
            tuple(tuple(order) for order in entry["action"].get("market", [])),
        )
        for entry in sample["demonstrated_realization"]
    )
    begun = perf_counter()
    try:
        realization = solver(opening, plan)
        ending = execute_realization(opening, plan, realization)
        elapsed = perf_counter() - begun
        candidate_turns = tuple((turn.worker_actions, turn.market_orders)
                                for turn in realization.turns)
        candidate = {
            "status": "complete",
            "completed": len(requirements),
            "required": len(requirements),
            "same_goal_set": True,
            "end_value": end_value(opening, ending),
            "runtime_seconds": elapsed,
            **_effort(opening, candidate_turns),
        }
    except PlanningFailure as failure:
        elapsed = perf_counter() - begun
        candidate = {
            "status": "PlanningFailure",
            "completed": 0,
            "required": len(requirements),
            "same_goal_set": False,
            "end_value": None,
            "runtime_seconds": elapsed,
            "workforce": None,
            "hire_expenditure": None,
            "diagnostics": failure.diagnostics,
        }
    reference = {
        "status": "complete",
        "completed": len(requirements),
        "required": len(requirements),
        "end_value": end_value(opening, reference_end),
        **_effort(opening, reference_turns),
    }
    return {
        "sample_id": sample["sample_id"],
        "reference": reference,
        "candidate": candidate,
    }


def goal_set_gap(target, candidate):
    target_ids = set(target["completed_ids"])
    candidate_ids = set(candidate["completed_ids"])
    return {
        "missing_from_realization": sorted(target_ids - candidate_ids),
        "additional_in_realization": sorted(candidate_ids - target_ids),
        "target_is_covered": target_ids <= candidate_ids,
    }


def compare_efficiency(target, target_effort, candidate, candidate_effort,
                       target_economic_value=None,
                       realization_economic_value=None):
    same = set(target["completed_ids"]) == set(candidate["completed_ids"])
    result = {
        "comparison_status": ("comparable-same-goal-set" if same
                              else "not-comparable-different-goal-set")
    }
    for key in set(target_effort) & set(candidate_effort):
        if isinstance(target_effort[key], (int, float)) and isinstance(candidate_effort[key], (int, float)):
            result[f"{key}_difference"] = candidate_effort[key] - target_effort[key]
    if target_economic_value is not None and realization_economic_value is not None:
        result["economic_state_value_difference"] = (
            realization_economic_value - target_economic_value
        )
    return result
