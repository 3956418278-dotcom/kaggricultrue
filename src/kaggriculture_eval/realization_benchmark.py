"""Same-state/fixed-Plan development comparisons, independent of solver scores.

The candidate receives only opening state and Plan. Demonstrated actions are
used exclusively on the reference/evaluation side of this boundary.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from src.kaggriculture_agent import rules
from src.kaggriculture_agent.intent import compile_intent, matches
from src.kaggriculture_agent.route_structure import selected_paths
from src.kaggriculture_agent.state import OwnedState, TileState, WorkerState, reconstruct
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
    paths += [Path("scripts/benchmark_realizations.py"),
              Path("scripts/benchmark_route_realizations.py"),
              Path("scripts/check_route_representation.py"),
              Path("scripts/summarize_route_benchmark.py"),
              Path("requirements-planning.txt"), Path("requirements.lock")]
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
    transactions = Counter()
    transaction_breakdowns = {
        "input_expenditure_by_kind": Counter(),
        "input_expenditure_by_item": Counter(),
        "purchased_quantity_by_item": Counter(),
        "sale_revenue_by_item": Counter(),
        "sold_quantity_by_item": Counter(),
    }
    hire_timing = []
    capacity = 0
    opening_workforce = len(start.workers)
    peak_workforce = opening_workforce
    for action in actions:
        obs = deepcopy(dict(env.state[0].observation))
        obs["player"] = 0
        step_events, noops = unit_effects(obs, action)
        events.extend({**e, "step": obs["step"]} for e in step_events)
        units = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
        current = reconstruct(obs)
        capacity += len(current.workers)
        peak_workforce = max(peak_workforce, len(current.workers))
        operations.update(a[0] for a in units[:len(current.workers)])
        worker_actions = units[:len(current.workers)]
        after_units = rules.advance_owned(current, worker_actions, unit_only=True)
        turn_ledger = rules.solo_market_ledger(after_units, action.get("market", []))
        result = oracle_step(env, units, action.get("market", []))
        if turn_ledger["ending_cash"] != result.money:
            raise AssertionError("controlled replay market-ledger parity mismatch")
        for key in ("hire_expenditure", "input_expenditure", "land_expenditure",
                    "sale_revenue", "executed_hires", "executed_land_purchases"):
            transactions[key] += turn_ledger[key]
        for key, totals in transaction_breakdowns.items():
            totals.update(turn_ledger[key])
        if turn_ledger["executed_hires"]:
            hire_timing.append({"step": current.step,
                                "count": turn_ledger["executed_hires"],
                                "expenditure": turn_ledger["hire_expenditure"]})
        peak_workforce = max(peak_workforce,
                             len(current.workers) + turn_ledger["executed_hires"])
        trace.append({"state_before": obs, "action": action, "effects": step_events, "noops": noops})
    return result, events, trace, {"operations": dict(operations), "available_worker_turns": capacity,
        "used_worker_turns": sum(n for op, n in operations.items() if op != "PASS"),
        "movement": sum(operations[o] for o in ("NORTH", "SOUTH", "EAST", "WEST")),
        "pickup_drop_place": sum(operations[o] for o in ("PICKUP", "DROP", "PLACE")),
        "opening_workforce": opening_workforce,
        "peak_workforce": peak_workforce,
        "ending_workforce": len(result.workers),
        "executed_hires": transactions["executed_hires"],
        "hire_timing": hire_timing,
        "hire_expenditure": transactions["hire_expenditure"],
        "input_expenditure": transactions["input_expenditure"],
        "land_expenditure": transactions["land_expenditure"],
        "sale_revenue": transactions["sale_revenue"],
        "executed_land_purchases": transactions["executed_land_purchases"],
        "ending_cash": result.money,
        **{key: dict(sorted(value.items()))
           for key, value in transaction_breakdowns.items()}}


def goal_set_gap(target, realization):
    """Set-valued completion gap, with no implication that target is optimal."""
    target_ids = set(target["completed_ids"])
    realized_ids = set(realization["completed_ids"])
    return {
        "target_completed": len(target_ids),
        "realized_completed": len(realized_ids),
        "missing_from_realization": sorted(target_ids-realized_ids),
        "additional_in_realization": sorted(realized_ids-target_ids),
        "same_goal_set": target_ids == realized_ids,
        "target_is_covered": target_ids <= realized_ids,
    }


def _physical_inventory(state):
    result = Counter(state.shed)
    for worker in state.workers:
        result.update(worker.inventory)
    result.update({f"{item}_SEED": quantity for item, quantity in state.seeds.items()})
    return {item: quantity for item, quantity in sorted(result.items()) if quantity}


def _asset_counts(state):
    result = Counter()
    for tile in state.tiles:
        raw = tile.raw
        if not isinstance(raw, dict):
            if raw is not None:
                result[str(raw)] += 1
            continue
        if raw.get("kind") == "PLANT":
            result[f"CROP:{raw.get('crop')}"] += 1
        elif raw.get("animal"):
            result[f"ANIMAL:{raw.get('animal')}"] += 1
        else:
            result[f"STRUCTURE:{raw.get('kind')}"] += 1
    return dict(sorted(result.items()))


def _owned_from_mapping(value):
    fields = dict(value)
    fields["tiles"] = tuple(TileState(tuple(tile["position"]), tile["raw"])
                            for tile in fields["tiles"])
    fields["workers"] = tuple(WorkerState(worker["index"], tuple(worker["position"]),
                                          worker["inventory"])
                              for worker in fields["workers"])
    fields["unlocked_quadrants"] = tuple(fields["unlocked_quadrants"])
    fields["unlocked_shops"] = tuple(fields["unlocked_shops"])
    return OwnedState(**fields)


def compare_farm_states(target, realization):
    """Exact, interpretable state delta; it is not an invented utility score."""
    target_inventory, realized_inventory = _physical_inventory(target), _physical_inventory(realization)
    inventory_items = set(target_inventory) | set(realized_inventory)
    inventory_delta = {item: realized_inventory.get(item, 0)-target_inventory.get(item, 0)
                       for item in sorted(inventory_items)
                       if realized_inventory.get(item, 0) != target_inventory.get(item, 0)}
    tile_differences = []
    for expected, actual in zip(target.tiles, realization.tiles):
        if expected.raw != actual.raw:
            tile_differences.append({"position": expected.position,
                                     "target": expected.raw, "realization": actual.raw})
    target_assets, realized_assets = _asset_counts(target), _asset_counts(realization)
    asset_keys = set(target_assets) | set(realized_assets)
    return {
        "same_persistent_farm_state": all((
            target.money == realization.money,
            target.shed == realization.shed,
            target.seeds == realization.seeds,
            target.tiles == realization.tiles,
            target.unlocked_quadrants == realization.unlocked_quadrants,
        )),
        "money_difference": realization.money-target.money,
        "same_physical_inventory": target_inventory == realized_inventory,
        "inventory_difference": inventory_delta,
        "same_asset_multiset": target_assets == realized_assets,
        "same_tile_layout": not tile_differences,
        "asset_count_difference": {key: realized_assets.get(key, 0)-target_assets.get(key, 0)
                                   for key in sorted(asset_keys)
                                   if realized_assets.get(key, 0) != target_assets.get(key, 0)},
        "differing_tile_count": len(tile_differences),
        "differing_tiles": tile_differences,
        "target_unlocked": target.unlocked_quadrants,
        "realization_unlocked": realization.unlocked_quadrants,
    }


def compare_efficiency(target_score, target_effort, realization_score, realization_effort):
    """Compare capacity only when the economic result makes it meaningful."""
    target_ids = set(target_score["completed_ids"])
    realized_ids = set(realization_score["completed_ids"])
    if target_ids == realized_ids:
        status = "same-goal-set"
    elif target_ids < realized_ids:
        status = "realization-covers-target-with-additional-goals"
    else:
        status = "not-comparable-different-goal-set"
    comparison = {
        "comparison_status": status,
        "used_worker_turn_difference": (realization_effort["used_worker_turns"]
                                        - target_effort["used_worker_turns"]),
        "movement_difference": realization_effort["movement"]-target_effort["movement"],
        "logistics_difference": (realization_effort["pickup_drop_place"]
                                 - target_effort["pickup_drop_place"]),
        "available_worker_turn_difference": (realization_effort["available_worker_turns"]
                                             - target_effort["available_worker_turns"]),
    }
    for key in ("peak_workforce", "executed_hires", "hire_expenditure",
                "input_expenditure", "land_expenditure", "sale_revenue",
                "ending_cash"):
        comparison[f"{key}_difference"] = (realization_effort.get(key, 0)
                                              - target_effort.get(key, 0))
    return comparison


def realization_economic_summary(score, effort, state):
    """Inspectable realization outcome; meaningful comparisons require equal goals."""
    return {
        "completed_goal_count": score["completed"],
        "completed_goal_ids": score["completed_ids"],
        "workforce": {
            "opening": effort["opening_workforce"],
            "peak": effort["peak_workforce"],
            "ending": effort["ending_workforce"],
            "hires": effort["executed_hires"],
            "hire_timing": effort["hire_timing"],
        },
        "transactions": {
            "hire_expenditure": effort["hire_expenditure"],
            "input_expenditure": effort["input_expenditure"],
            "input_expenditure_by_kind": effort["input_expenditure_by_kind"],
            "input_expenditure_by_item": effort["input_expenditure_by_item"],
            "land_expenditure": effort["land_expenditure"],
            "sale_revenue": effort["sale_revenue"],
            "sale_revenue_by_item": effort["sale_revenue_by_item"],
        },
        "ending_cash": state.money,
        "ending_inventory": _physical_inventory(state),
        "ending_assets": _asset_counts(state),
        "labor": {
            "available_worker_turns": effort["available_worker_turns"],
            "used_worker_turns": effort["used_worker_turns"],
            "movement": effort["movement"],
            "logistics": effort["pickup_drop_place"],
        },
    }


def demonstrated_structural_patterns(sample, problem, skeleton):
    """Describe witnessed route structure without declaring it necessary."""
    workers_by_entity = {}
    for entity, path in selected_paths(problem, skeleton).items():
        goals = {event.goal for event in path}
        workers_by_entity[entity] = {worker for worker, route in enumerate(skeleton.routes)
                                     if goals & set(route)}
    purchase_entries = Counter()
    hire_turns = 0
    for turn in sample["demonstrated_realization"]:
        orders = turn["action"].get("market", [])
        if any(order[0] == "HIRE" for order in orders):
            hire_turns += 1
        for order in orders:
            if order[0].startswith("BUY_") and order[0] != "BUY_LAND" and len(order) > 1:
                purchase_entries[(order[0], order[1])] += 1
    counts = {
        "entities_split_across_workers": sum(len(workers) > 1 for workers in workers_by_entity.values()),
        "same_turn_ordered_bundles": len(skeleton.synchronizations),
        "temporal_tile_reuses": sum(lease.available_after is not None for lease in skeleton.leases),
        "cross_worker_shed_material_links": sum(link.kind == "EVENT" and link.via_shed
                                                for link in skeleton.resources),
        "open_placement_entities": sum(not entity.existing and len(entity.positions) > 1
                                       for entity in problem.intent.entities),
        "repeated_purchase_item_entries": sum(quantity-1 for quantity in purchase_entries.values()
                                              if quantity > 1),
        "hiring_turns": hire_turns,
    }
    return {"counts": counts, "present": sorted(key for key, value in counts.items() if value)}


def _goal_description(problem, identifier):
    goal = problem.goals[identifier]
    actions = sorted({tuple(event.action) for path in problem.complete_paths[goal.entity]
                      for event in path if event.goal == identifier})
    entity = next(entity for entity in problem.intent.entities
                  if entity.identifier == goal.entity)
    return {"identifier": identifier, "entity": goal.entity,
            "actions": actions, "position_domain": entity.positions,
            "deadline": goal.deadline}


def compare_route_structures(problem, reference_skeleton, candidate_skeleton, missing_goals):
    """Assignment/order diagnostics; different worker labels are not a score."""
    reference_routes = reference_skeleton.routes
    candidate_routes = tuple(tuple(route) for route in candidate_skeleton["routes"])
    reference_assignment = {goal: worker for worker, route in enumerate(reference_routes)
                            for goal in route if goal in problem.goals}
    candidate_assignment = {goal: worker for worker, route in enumerate(candidate_routes)
                            for goal in route if goal in problem.goals}
    common = set(reference_assignment) & set(candidate_assignment)
    context = []
    for goal in sorted(missing_goals):
        reference_worker = reference_assignment.get(goal)
        candidate_worker = candidate_assignment.get(goal)
        reference_route = reference_routes[reference_worker] if reference_worker is not None else ()
        candidate_route = candidate_routes[candidate_worker] if candidate_worker is not None else ()
        context.append({**_goal_description(problem, goal),
            "reference_worker": reference_worker,
            "reference_route_index": (reference_route.index(goal) if goal in reference_route else None),
            "reference_route_length": len(reference_route),
            "candidate_worker": candidate_worker,
            "candidate_route_index": (candidate_route.index(goal) if goal in candidate_route else None),
            "candidate_route_length": len(candidate_route),
        })
    return {
        "reference_workforce": reference_skeleton.workforce,
        "candidate_workforce": len(candidate_routes),
        "goals_assigned_to_different_worker_index": sum(
            reference_assignment[goal] != candidate_assignment[goal] for goal in common),
        "common_assigned_goals": len(common),
        "reference_route_goal_counts": [sum(goal in problem.goals for goal in route)
                                        for route in reference_routes],
        "candidate_route_goal_counts": [sum(goal in problem.goals for goal in route)
                                        for route in candidate_routes],
        "missing_goal_context": context,
    }


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


def compare_route_sample(sample, config, solver):
    """Decompose representation, blind search, reached-state, and efficiency gaps.

    The demonstrated realization is a strong feasible witness. It is never
    described or used as an optimum, and its route structure never enters the
    blind solver call made by ``compare_sample``.
    """
    from src.kaggriculture_agent.route_compiler import compile_skeleton
    from .route_reference import demonstrated_skeleton

    result = compare_sample(sample, config, solver=solver)
    problem, skeleton = demonstrated_skeleton(sample)
    witness = compile_skeleton(problem, skeleton)
    witness_actions = [{"farmer": execution.worker_actions[0],
                        "hands": list(execution.worker_actions[1:]),
                        "market": list(execution.market_orders)}
                       for execution in witness.executions]
    start = reconstruct(sample["day_start_state"])
    witness_end, witness_events, witness_trace, witness_effort = execute_reference_controlled(
        start, witness_actions)
    witness_score = score_effects(problem.intent, witness_events, witness.placements, witness_end)
    if witness_score["completed_ids"] != sorted(witness.completed):
        raise AssertionError("representation witness official parity mismatch: completed goals")

    controlled = result["reference"]
    candidate = result["candidate"]
    controlled_score = {key: controlled[key] for key in
                        ("goal_count", "completed", "completed_ids", "unfulfilled")}
    candidate_score = {key: candidate[key] for key in
                       ("goal_count", "completed", "completed_ids", "unfulfilled")}
    reference_end = _owned_from_mapping(controlled["final_state"])
    candidate_end = _owned_from_mapping(candidate["final_state"])
    demonstration_witness_gap = goal_set_gap(controlled_score, witness_score)
    search_gap = goal_set_gap(witness_score, candidate_score)
    reference_gap = goal_set_gap(controlled_score, candidate_score)
    controlled_ids = set(controlled_score["completed_ids"])
    witness_ids = set(witness_score["completed_ids"])
    candidate_ids = set(candidate_score["completed_ids"])
    recovered_witness_goals = sorted((controlled_ids-witness_ids) & candidate_ids)
    unresolved_representation_or_search = sorted(controlled_ids-(witness_ids | candidate_ids))
    patterns = demonstrated_structural_patterns(sample, problem, skeleton)
    result["representation_witness"] = {
        **witness_score, **witness_effort,
        "final_state": asdict(witness_end), "trace": witness_trace,
        "compiler_diagnostics": witness.diagnostics,
    }
    result["structural_patterns"] = patterns
    result["gaps"] = {
        "representation": {
            # A failed demonstration compilation is not by itself proof that
            # the route model cannot express the Plan. Blind search may produce
            # another skeleton that realizes the same goals, as occurs in the
            # development corpus. Unsupported patterns require an explicit
            # structural diagnosis, not absence from a finite search.
            "established_unsupported_patterns": [],
            "established_missing_goals": [],
            "demonstration_translation_or_compilation": demonstration_witness_gap,
            "witness_missing_goals_recovered_by_blind_search": recovered_witness_goals,
            "unresolved_representation_or_search_goals": unresolved_representation_or_search,
        },
        "search": search_gap,
        "candidate_vs_strong_witness": reference_gap,
        "demonstration_witness_gap_associated_patterns": (
            patterns["present"] if demonstration_witness_gap["missing_from_realization"] else []),
    }
    result["resulting_state"] = {
        "representation_witness_vs_controlled_reference": compare_farm_states(
            reference_end, witness_end),
        "candidate_vs_representation_witness": compare_farm_states(witness_end, candidate_end),
        "candidate_vs_controlled_reference": compare_farm_states(reference_end, candidate_end),
    }
    effort_keys = ("available_worker_turns", "used_worker_turns", "movement",
                   "pickup_drop_place", "opening_workforce", "peak_workforce",
                   "ending_workforce", "executed_hires", "hire_timing",
                   "hire_expenditure", "input_expenditure", "land_expenditure",
                   "sale_revenue", "ending_cash", "input_expenditure_by_kind",
                   "input_expenditure_by_item", "sale_revenue_by_item")
    controlled_effort = {key: controlled[key] for key in effort_keys}
    candidate_effort = {key: candidate[key] for key in effort_keys}
    result["efficiency"] = {
        "representation_witness_vs_controlled_reference": compare_efficiency(
            controlled_score, controlled_effort, witness_score, witness_effort),
        "candidate_vs_representation_witness": compare_efficiency(
            witness_score, witness_effort, candidate_score, candidate_effort),
        "candidate_vs_controlled_reference": compare_efficiency(
            controlled_score, controlled_effort, candidate_score, candidate_effort),
    }
    result["equal_goal_set_realization_comparison"] = {
        "comparison_status": result["efficiency"][
            "candidate_vs_controlled_reference"]["comparison_status"],
        "interpretation": ("Workforce and economic-efficiency differences are directly "
                           "comparable only when achieved goal sets are equal."),
        "reference": realization_economic_summary(
            controlled_score, controlled_effort, reference_end),
        "candidate": realization_economic_summary(
            candidate_score, candidate_effort, candidate_end),
    }
    result["route_structure_comparison"] = compare_route_structures(
        problem, skeleton, candidate["diagnostics"]["skeleton"],
        reference_gap["missing_from_realization"])
    result["comparison"].update({
        "representation_missing_goals": 0,
        "demonstration_witness_missing_goals": len(
            demonstration_witness_gap["missing_from_realization"]),
        "witness_missing_goals_recovered_by_blind_search": len(recovered_witness_goals),
        "unresolved_representation_or_search_goals": len(unresolved_representation_or_search),
        "search_missing_goals": len(search_gap["missing_from_realization"]),
        "candidate_additional_goals": len(reference_gap["additional_in_realization"]),
        "efficiency_comparison_status": result["efficiency"][
            "candidate_vs_controlled_reference"]["comparison_status"],
    })
    return result
