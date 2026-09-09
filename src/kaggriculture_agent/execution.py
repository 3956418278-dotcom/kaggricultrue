"""Exact execution and validation of an intraday ``Realization``.

This module never chooses actions. It replays submitted turns through the
single rule owner and verifies that every requirement in the fixed Plan was
actually achieved.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping

from . import rules
from .planner import Plan
from .realization import Realization
from .state import OwnedState


class InvalidRealization(ValueError):
    pass


def _tile_fields(raw: object) -> Mapping[str, object]:
    return raw if isinstance(raw, Mapping) else {"$tile": raw}


def _effect_matches(requirement: Mapping[str, object], before: object, after: object) -> bool:
    fields = _tile_fields(after)
    for name, condition in requirement.items():
        expected_present = bool(condition["after_present"])
        if (name in fields) != expected_present:
            return False
        if not expected_present:
            continue
        expected, actual = condition.get("after"), fields[name]
        if name in ("yield_units", "fertilized_until_day") and isinstance(expected, int) and expected > 0:
            if not isinstance(actual, int) or actual < expected:
                return False
        elif actual != expected:
            return False
    return before != after


def _operation(project, kind: str) -> tuple[object, ...]:
    if kind == "PLANT":
        return ("PLANT", project.metadata["crop"])
    if kind == "PLACE":
        return ("PLACE", project.metadata["animal"])
    if kind == "BUILD":
        return ("BUILD_" + str(project.metadata["structure"]),)
    return (kind,)


def _required(plan: Plan, placements: Mapping[str, tuple[int, int]]):
    """Yield validation records directly from Plan; this is not a planner IR."""
    aliases = {
        "PICKUP_PLACE": ("PLACE",),
        "HARVEST_TRANSPORT": ("HARVEST",),
        "WATER_HARVEST_TRANSPORT": ("WATER", "HARVEST"),
    }
    records = []
    for project in (*plan.obligations, *plan.selected, *plan.support):
        if project.kind == "LAND":
            records.append((project.identifier, None, None, None, 1, str(project.metadata["quadrant"])))
            continue
        target = project.target if project.target is not None else placements.get(project.identifier)
        if project.kind == "STATE_EFFECT":
            count = max(1, int(project.metadata.get("demonstrated_count", 1)))
            records.append((project.identifier, target, None,
                            project.metadata["required_effect"], count, None))
            continue
        for index, work in enumerate(project.actions.work):
            if work.day != plan.day:
                continue
            position = work.position if work.position is not None else target
            for kind in aliases.get(work.kind, (work.kind,)):
                records.append((f"{project.identifier}:{index}:{kind}", position,
                                _operation(project, kind), None, 1, None))
    return records


def execute_realization(state: OwnedState, plan: Plan, realization: Realization) -> OwnedState:
    """Replay and validate a complete realization, returning its exact end state."""
    requirements = _required(plan, realization.placements)
    missing_positions = [identifier for identifier, position, operation, effect, _, land in requirements
                         if land is None and position is None and (operation is not None or effect is not None)]
    if missing_positions:
        raise InvalidRealization(f"unbound Plan placements: {missing_positions[:3]}")
    achieved = Counter()
    by_position = defaultdict(list)
    for record in requirements:
        if record[1] is not None:
            by_position[record[1]].append(record)

    current = state
    for offset, decision in enumerate(realization.turns):
        if len(decision.worker_actions) != len(current.workers):
            raise InvalidRealization(
                f"turn {offset}: expected {len(current.workers)} worker actions, got {len(decision.worker_actions)}"
            )
        if len(decision.market_orders) > rules.MAX_MARKET_ORDERS:
            raise InvalidRealization(f"turn {offset}: too many market orders")
        micro = current
        requested_plants = Counter(
            str(action[1]) for action in decision.worker_actions
            if action and action[0] == "PLANT" and len(action) > 1
        )
        if any(quantity > current.seeds.get(crop, 0) for crop, quantity in requested_plants.items()):
            raise InvalidRealization(f"turn {offset}: atomic seed shortfall")
        for worker, action in enumerate(decision.worker_actions):
            before_position = micro.workers[worker].position
            before_tile = micro.tile_at(before_position).raw
            unit_actions = tuple(action if index == worker else ("PASS",)
                                 for index in range(len(micro.workers)))
            after_micro = rules.advance_owned(micro, unit_actions, unit_only=True)
            after_tile = after_micro.tile_at(before_position).raw
            op = tuple(action)
            changed = before_tile != after_tile
            is_move = bool(action and action[0] in ("NORTH", "SOUTH", "EAST", "WEST"))
            is_logistics = bool(action and action[0] in ("PICKUP", "DROP"))
            if action and action[0] != "PASS" and not (changed or is_move or is_logistics):
                raise InvalidRealization(f"turn {offset}, worker {worker}: illegal/no-op action {op!r}")
            for identifier, _, operation, effect, count, _ in by_position.get(before_position, ()):
                if achieved[identifier] >= count:
                    continue
                if operation is not None and op == operation and changed:
                    achieved[identifier] += 1
                    break
                if effect is not None and _effect_matches(effect, before_tile, after_tile):
                    achieved[identifier] += 1
                    break
            micro = after_micro
        current = rules.advance_owned(current, decision.worker_actions, decision.market_orders)

    missing = []
    for identifier, _, _, _, count, land in requirements:
        if land is not None:
            if land not in current.unlocked_quadrants:
                missing.append(identifier)
        elif achieved[identifier] < count:
            missing.append(identifier)
    if missing:
        raise InvalidRealization(f"Daily Plan incomplete: {len(missing)} requirement(s): {missing[:5]}")
    return current


validate_realization = execute_realization
