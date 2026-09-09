"""Private market-order materialization used by the intraday solver."""
from . import rules


def realization_orders(state, worker_actions, purchases=(), hires=0, land=0,
                       remaining_inputs=None, reserve_is_shed=False,
                       required_sequence=None):
    """Reactive selling around explicit realization-owned acquisition decisions.

    Input reserve is for the unchanged economic target, minus inputs already
    carried. The intraday solver controls purchases/hiring/logistics, not sales.
    """
    after_units = rules.advance_owned(state, worker_actions, unit_only=True)
    required = ([list(order) for order in required_sequence]
                if required_sequence is not None else
                [["HIRE"] for _ in range(hires)] + [["BUY_LAND"] for _ in range(land)]
                + [list(p) for p in purchases])
    if len(required) > rules.MAX_MARKET_ORDERS:
        raise ValueError("realization exceeds market entry capacity")
    remaining_inputs = remaining_inputs or {}
    sales = []
    for item in sorted(rules.SELLABLE_PRODUCTS, key=lambda i: (-state.market_prices.get(i, 0), i)):
        reserve = remaining_inputs.get(item, 0)
        if not reserve_is_shed:
            reserve = max(0, reserve-after_units.carried_total(item))
        amount = max(0, after_units.shed.get(item, 0) - reserve)
        if amount:
            sales.append(["SELL", item, amount])
    return tuple(sales[:rules.MAX_MARKET_ORDERS - len(required)] + required)
