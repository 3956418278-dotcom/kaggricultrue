"""Mechanical market-order materialization for a fixed Daily Plan."""
from . import rules


def realization_orders(state, worker_actions, *, required_sequence=()):
    """Return only Plan-required or mechanically required orders.

    There is intentionally no opportunistic selling or market-timing search.
    The state/action arguments keep this boundary ready for same-turn legality
    checks without giving it economic selection authority.
    """
    del state, worker_actions
    required = [tuple(order) for order in required_sequence]
    if len(required) > rules.MAX_MARKET_ORDERS:
        raise ValueError("realization exceeds market entry capacity")
    return tuple(required)
