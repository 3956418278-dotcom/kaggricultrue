"""Exact execution and full outcome validation for an intraday Realization."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping

from . import rules
from .planner import Plan
from .realization import Realization
from .state import OwnedState


class InvalidRealization(ValueError):
    pass


def _stock(state: OwnedState) -> Counter:
    result = Counter(state.shed)
    for worker in state.workers:
        result.update(worker.inventory)
    result.update({f"{crop}_SEED": quantity
                   for crop, quantity in state.seeds.items()})
    return result


def _state_matches(raw: object, required: Mapping[str, object]) -> bool:
    if "$tile" in required:
        return raw == required["$tile"]
    if not isinstance(raw, Mapping):
        return False
    for field, expected in required.items():
        actual = raw.get(field)
        if isinstance(expected, Mapping) and set(expected) == {"at_least"}:
            if not isinstance(actual, int) or actual < int(expected["at_least"]):
                return False
        elif field not in raw or actual != expected:
            return False
    return True


def _outputs_match(produced, required):
    return all(produced.get(item, 0) >= int(quantity)
               for item, quantity in required.items())


def _window_ready(after_units, window):
    return (after_units.money >= window.minimum_cash
        and all(after_units.shed.get(item, 0) >= quantity
                for item, quantity in window.required_shed.items()))


def _contains_orders(actual, required):
    actual_counts = Counter(tuple(order) for order in actual)
    required_counts = Counter(tuple(order) for order in required)
    return all(actual_counts[order] >= count
               for order, count in required_counts.items())


def execute_realization(
    state: OwnedState, plan: Plan, realization: Realization
) -> OwnedState:
    """Replay real turns and require every canonical Plan outcome."""
    commitments = tuple(
        commitment for commitment
        in (*plan.obligations, *plan.selected, *plan.support)
        if commitment.kind != "LAND"
        and (commitment.required_state or commitment.required_outputs)
    )
    for commitment in commitments:
        if commitment.target is None:
            raise InvalidRealization(
                f"Plan commitment {commitment.identifier!r} has no fixed placement"
            )
        supplied = realization.placements.get(commitment.identifier)
        if supplied is not None and tuple(supplied) != tuple(commitment.target):
            raise InvalidRealization(
                f"Realization changed Plan placement {commitment.identifier!r}"
            )

    produced = defaultdict(Counter)
    achieved = set()
    day_end = min((plan.day + 1) * rules.TURNS_PER_DAY - 1,
                  rules.TERMINAL_ACTION_STEP)
    deadlines = {
        commitment.identifier: min(
            (int(value) for value in commitment.time.deadlines
             if state.step <= int(value) <= day_end),
            default=day_end,
        )
        for commitment in commitments
    }
    for commitment in commitments:
        raw = state.tile_at(tuple(commitment.target)).raw
        if (_state_matches(raw, commitment.required_state)
                and _outputs_match(produced[tuple(commitment.target)],
                                   commitment.required_outputs)):
            achieved.add(commitment.identifier)

    completed_windows = set()
    current = state
    for offset, decision in enumerate(realization.turns):
        if len(decision.worker_actions) != len(current.workers):
            raise InvalidRealization(
                f"turn {offset}: expected {len(current.workers)} worker actions, "
                f"got {len(decision.worker_actions)}"
            )
        if len(decision.market_orders) > rules.MAX_MARKET_ORDERS:
            raise InvalidRealization(f"turn {offset}: too many market orders")
        requested_plants = Counter(
            str(action[1]) for action in decision.worker_actions
            if action and action[0] == "PLANT" and len(action) > 1
        )
        if any(quantity > current.seeds.get(crop, 0)
               for crop, quantity in requested_plants.items()):
            raise InvalidRealization(f"turn {offset}: atomic seed shortfall")

        micro = current
        for worker, action in enumerate(decision.worker_actions):
            before_position = micro.workers[worker].position
            before_tile = micro.tile_at(before_position).raw
            before_stock = _stock(micro)
            unit_actions = tuple(action if index == worker else ("PASS",)
                                 for index in range(len(micro.workers)))
            after_micro = rules.advance_owned(micro, unit_actions, unit_only=True)
            after_tile = after_micro.tile_at(before_position).raw
            after_stock = _stock(after_micro)
            operation = action[0] if action else "PASS"
            changed = before_tile != after_tile
            moved = operation in ("NORTH", "SOUTH", "EAST", "WEST")
            logistics = operation in ("PICKUP", "DROP") or (
                operation == "PLACE"
                and before_position in rules.shed_access(current.board_size)
            )
            if operation != "PASS" and not (changed or moved or logistics):
                raise InvalidRealization(
                    f"turn {offset}, worker {worker}: illegal/no-op action {tuple(action)!r}"
                )
            for item in before_stock.keys() | after_stock.keys():
                gain = after_stock[item] - before_stock[item]
                if gain > 0:
                    produced[before_position][item] += gain
            absolute_step = state.step + offset
            for commitment in commitments:
                if commitment.identifier in achieved:
                    continue
                if tuple(commitment.target) != before_position:
                    continue
                if absolute_step > deadlines[commitment.identifier]:
                    continue
                if (_state_matches(after_tile, commitment.required_state)
                        and _outputs_match(produced[before_position],
                                           commitment.required_outputs)):
                    achieved.add(commitment.identifier)
            micro = after_micro

        after_units = rules.advance_owned(
            current, decision.worker_actions, unit_only=True
        )
        for index, window in enumerate(plan.economic_windows):
            if index in completed_windows:
                continue
            if (window.start_turn <= current.hour <= window.end_turn
                    and _window_ready(after_units, window)
                    and _contains_orders(decision.market_orders,
                                         window.market_orders)):
                completed_windows.add(index)
        current = rules.advance_owned(
            current, decision.worker_actions, decision.market_orders
        )

    missing = [commitment.identifier for commitment in commitments
               if commitment.identifier not in achieved]
    missing.extend(
        commitment.identifier
        for commitment in (*plan.obligations, *plan.selected, *plan.support)
        if commitment.kind == "LAND"
        and str(commitment.metadata["quadrant"])
        not in current.unlocked_quadrants
    )
    if len(completed_windows) != len(plan.economic_windows):
        missing.extend(
            f"economic-window:{index}"
            for index in range(len(plan.economic_windows))
            if index not in completed_windows
        )
    if missing:
        raise InvalidRealization(
            f"Daily Plan incomplete: {len(missing)} requirement(s): {missing[:5]}"
        )
    return current


validate_realization = execute_realization
