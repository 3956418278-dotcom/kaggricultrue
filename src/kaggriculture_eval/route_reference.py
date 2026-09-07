"""Evaluation-only conversion of demonstrations to event-route structure.

This module tests representation/compiler expressiveness. It is never imported
by the agent and demonstrated actions are never supplied to route search.
"""
from collections import Counter, defaultdict
from dataclasses import replace

from src.kaggriculture_agent.route_search import rebuild_leases
from src.kaggriculture_agent.route_structure import (
    LogisticsEvent, RouteSkeleton, SyncBundle, build_route_problem, initial_resources,
)
from src.kaggriculture_agent.state import reconstruct
from .plan_io import plan_from_dict


def demonstrated_skeleton(sample):
    state = reconstruct(sample["day_start_state"])
    plan = plan_from_dict(sample["plan"])
    problem = build_route_problem(state, plan)
    by_entity, routes, placements = defaultdict(list), defaultdict(list), {}
    same_turn = defaultdict(list)
    workforce = 1
    first_positions = {}
    market_priority, bought, hire_number, land_number = [], set(), 0, 0
    logistics = {}
    acquisitions, acquisition_kind = Counter(), {}
    required_entry_caps = []
    hire_caps = []

    for turn in sample["demonstrated_realization"]:
        farm = turn["state_before"]["farms"][sample["side"]]
        positions = [farm["farmer"], *farm.get("hands", [])]
        workforce = max(workforce, len(positions))
        for worker, position in enumerate(positions):
            first_positions.setdefault(worker, tuple(position))
        effects_by_worker = defaultdict(list)
        for effect in turn["effects"]:
            if not effect["achieved"] or effect.get("goal") not in problem.goals:
                continue
            goal, entity, worker = effect["goal"], effect["entity"], effect["worker"]
            by_entity[entity].append(goal); effects_by_worker[worker].append(goal)
            placements.setdefault(entity, tuple(effect["position"]))
            same_turn[turn["step"], entity].append((worker, goal))
        unit_actions = [turn["action"].get("farmer", ["PASS"]), *turn["action"].get("hands", [])]
        inventories = turn["state_before"]["private"]["inventories"]
        market_orders = turn["action"].get("market", [])
        hire_caps.append(sum(order[0] == "HIRE" for order in market_orders))
        # Only a hire wave needs an explicit reservation for same-turn sales:
        # after staffing is complete, purchase timing remains free to follow
        # the compiled route's actual material demand.
        required_entry_caps.append(
            sum(order[0] != "SELL" for order in market_orders)
            if any(order[0] == "HIRE" for order in market_orders)
            else 10)
        for worker, action in enumerate(unit_actions[:len(inventories)]):
            if effects_by_worker[worker]:
                routes[worker].extend(effects_by_worker[worker])
                continue
            identifier = f"reference:{turn['step']}:{worker}"
            if action[0] == "PICKUP":
                quantity = min(int(action[2]) if len(action) > 2 else 1,
                               turn["state_before"]["private"]["shed"].get(action[1], 0))
                event = LogisticsEvent(identifier, "PICKUP", action[1], quantity,
                                       purpose="demonstrated-input") if quantity else None
            elif action[0] == "PLACE" and len(action) > 2:
                quantity = min(int(action[2]), inventories[worker].get(action[1], 0))
                event = LogisticsEvent(identifier, "PLACE", action[1], quantity,
                                       purpose="demonstrated-storage") if quantity else None
            elif action[0] == "DROP" and any(inventories[worker].values()):
                event = LogisticsEvent(identifier, "DROP", purpose="demonstrated-storage")
            else:
                event = None
            if event:
                logistics[identifier] = event; routes[worker].append(identifier)
        for order in market_orders:
            if order[0] == "HIRE":
                market_priority.append(f"HIRE:{hire_number}"); hire_number += 1
            elif order[0].startswith("BUY_") and order[0] != "BUY_LAND":
                item = order[1]+"_SEED" if order[0] == "BUY_SEED" else order[1]
                acquisitions[item] += int(order[2]) if len(order) > 2 else 1
                acquisition_kind[item] = order[0]
                if item not in bought:
                    market_priority.append(f"BUY:{item}"); bought.add(item)
            elif order[0] == "BUY_LAND" and land_number < len(problem.intent.land):
                market_priority.append(f"LAND:{problem.intent.land[land_number]}"); land_number += 1

    path_choices = {}
    for entity in problem.intent.entities:
        sequence = tuple(by_entity[entity.identifier])
        choice = next((n for n, path in enumerate(problem.complete_paths[entity.identifier])
                       if tuple(event.goal for event in path) == sequence), None)
        if choice is None:
            raise ValueError(f"demonstrated effects do not select a complete local path: {entity.identifier}")
        path_choices[entity.identifier] = choice
        if entity.existing and entity.identifier not in placements and entity.positions:
            placements[entity.identifier] = entity.positions[0]

    route_tuple = tuple(tuple(routes[worker]) for worker in range(workforce))
    synchronizations = []
    for effects in same_turn.values():
        effects.sort()
        if len(effects) > 1 and len({worker for worker, _ in effects}) == len(effects):
            synchronizations.append(SyncBundle(tuple(goal for _, goal in effects)))
    preferences = tuple(first_positions[worker] for worker in range(len(state.workers), workforce))
    acquisition_orders = {item: (acquisition_kind[item], item[:-5] if item.endswith("_SEED") else item, quantity)
                          for item, quantity in acquisitions.items()}
    shell = RouteSkeleton(workforce, route_tuple, path_choices, placements, (), (),
                          tuple(synchronizations), preferences, tuple(market_priority), logistics,
                          acquisition_orders, tuple(required_entry_caps), tuple(hire_caps))
    shell = replace(shell, leases=rebuild_leases(problem, shell))
    shell = replace(shell, resources=initial_resources(problem, shell))
    present = set(shell.market_priority)
    required = [*(f"BUY:{link.item}" for link in shell.resources if link.kind == "PURCHASE"),
                *(f"LAND:{q}" for q in problem.intent.land),
                *(f"HIRE:{n}" for n in range(max(0, workforce-len(state.workers))))]
    return problem, replace(shell, market_priority=tuple([*shell.market_priority,
        *(token for token in required if token not in present)]))
