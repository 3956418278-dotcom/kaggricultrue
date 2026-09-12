"""Validation replay for a complete frozen Programme."""
from __future__ import annotations

from .programme import Programme
from .simulation import SimulationResult, simulate_programme
from .state import State


class InvalidProgramme(ValueError):
    pass


def execute_programme(state: State, programme: Programme) -> SimulationResult:
    result = simulate_programme(state, programme)
    if not result.feasible:
        raise InvalidProgramme(result.failure or "programme is infeasible")
    return result
