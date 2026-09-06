"""Reactive market owner. Selling is not part of daily economic intent."""
from collections import Counter
from . import rules
from .planner import Plan
from .state import OwnedState


def build_market_orders(state: OwnedState, plan: Plan, worker_actions: tuple[list[object], ...], choices=None) -> tuple[list[object], ...]:
    """Construct ordered sales and purchases using post-unit-action inventory."""
    from .realization import legacy_choices
    choices = choices or legacy_choices(state, plan)
    projected_shed = dict(state.shed)
    used_carried: dict[str, int] = {}
    for worker, action in zip(state.workers, worker_actions):
        op = action[0] if action else "PASS"
        if op == "DROP":
            for item, quantity in worker.inventory.items():
                projected_shed[item] = projected_shed.get(item, 0) + quantity
        elif op == "PLACE" and len(action) > 1:
            item = str(action[1])
            used_carried[item] = used_carried.get(item, 0) + 1
        elif op == "FEED":
            used_carried["WHEAT"] = used_carried.get("WHEAT", 0) + 1
        elif op == "FERTILIZE":
            used_carried["FERTILIZER"] = used_carried.get("FERTILIZER", 0) + 1
    terminal = state.step >= 707
    applied = {w.position for w, a in zip(state.workers, worker_actions) if a and a[0] == "FERTILIZE"}
    pending_fertilizer = (
        sum(
            state.tile_at(target).kind == "PLANT"
            and int(state.tile_at(target).raw.get("fertilized_until_day", -1)) < state.day + 2
            for target in plan.fertilize_targets if target not in applied
        )
        if plan.fertilize_targets
        else plan.fertilizer_reserve
    )
    reserves = {
        "WHEAT": 0 if terminal else plan.feed_reserve,
        "FERTILIZER": 0 if terminal else pending_fertilizer,
    }
    sales: list[list[object]] = []
    for item in sorted(rules.SELLABLE_PRODUCTS, key=lambda name: (-state.market_prices.get(name, 0), name)):
        quantity = max(0, projected_shed.get(item, 0) - reserves.get(item, 0))
        if quantity:
            sales.append(["SELL", item, quantity])
    if terminal:
        return tuple(sales[: rules.MAX_MARKET_ORDERS])

    commitment_orders: list[list[object]] = []
    for crop in rules.CROPS:
        remaining_targets = sum(
            planned_crop == crop and state.tile_at(target).is_empty
            for target, planned_crop in choices.crop_targets(plan).items()
        )
        shortfall = max(0, remaining_targets - state.seeds.get(crop, 0))
        if shortfall:
            commitment_orders.append(["BUY_SEED", crop, shortfall])
    animal_counts = Counter(plan.animal_purchases)
    for animal in rules.ANIMALS:
        acquired_today = max(
            0,
            state.owned_animals(animal) - plan.starting_animals.get(animal, 0),
        )
        shortfall = max(0, animal_counts[animal] - acquired_today)
        if shortfall:
            commitment_orders.append(
                ["BUY_ANIMAL", animal, shortfall]
            )
    fertilizer_after_actions = max(0, state.owned_total("FERTILIZER") - used_carried.get("FERTILIZER", 0))
    fertilizer_shortfall = max(0, pending_fertilizer - fertilizer_after_actions)
    if fertilizer_shortfall:
        commitment_orders.append(
            ["BUY_PRODUCT", "FERTILIZER", fertilizer_shortfall]
        )
    wheat_after_actions = max(0, state.owned_total("WHEAT") - used_carried.get("WHEAT", 0))
    wheat_shortfall = max(0, plan.feed_reserve - wheat_after_actions)
    if wheat_shortfall:
        commitment_orders.append(["BUY_PRODUCT", "WHEAT", wheat_shortfall])
    land_target = next(
        (
            str(commitment.metadata.get("quadrant"))
            for commitment in plan.support
            if commitment.kind == "LAND"
        ),
        None,
    )
    if plan.buy_land and land_target not in state.unlocked_quadrants:
        commitment_orders.append(["BUY_LAND"])

    # Reserve the finite entry budget for commitment inputs.  Sales still execute
    # first (and can finance later entries), but only occupy genuinely spare slots.
    if len(commitment_orders) > rules.MAX_MARKET_ORDERS:
        raise ValueError("planner commitments exceed the market-entry budget")
    hire_slots = max(
        0, rules.MAX_MARKET_ORDERS - len(commitment_orders)
    )
    outstanding_hires = max(0, choices.hire_count - state.hires_today)
    hire_orders = [["HIRE"] for _ in range(min(outstanding_hires, hire_slots))]
    required_orders = [*hire_orders, *commitment_orders]
    sale_slots = rules.MAX_MARKET_ORDERS - len(required_orders)
    return tuple([*sales[:sale_slots], *required_orders])
