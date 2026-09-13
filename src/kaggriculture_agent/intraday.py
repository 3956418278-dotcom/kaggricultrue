"""D11+ zonal executor integration point.

The former global INNER/OUTER router has been removed.  The fixed template
library is validated independently before the new route kernel is connected.
"""


class PlanningFailure(RuntimeError):
    pass


def clear_route_cache():
    pass


def solve_intraday(*_args, **_kwargs):
    raise PlanningFailure("zonal route kernel is not connected yet")
