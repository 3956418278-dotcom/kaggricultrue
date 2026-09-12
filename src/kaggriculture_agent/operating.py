"""Lifecycle owner for the daily frozen macro programme."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from . import rules
from .market import known_demand_events, opponent_pressure, optimize_sales
from .planner import _arrivals, make_plan
from .programme import Programme
from .realization import TurnDecision
from .state import State


def programme_invalidation(state: State, programme: Programme) -> str | None:
    if state.step < programme.formed_step:
        return "new episode"
    if tuple(state.shops) != tuple(programme.shops):
        return "shop list changed"
    for asset in programme.assets:
        if asset.decision != "KEEP" or not asset.existing:
            continue
        tile = state.tile_at(asset.tile)
        if asset.asset_type in rules.ANIMALS and tile.animal != asset.asset_type:
            return f"kept {asset.asset_id} no longer exists"
        if asset.asset_type in rules.CROPS and tile.crop != asset.asset_type:
            # A completed one-time harvest intentionally releases its tile.
            if not any(e.step < state.step for e in asset.harvest_schedule):
                return f"kept {asset.asset_id} no longer exists"
    return None


def _decision(state: State, programme: Programme) -> TurnDecision:
    actions = [("PASS",) for _ in state.workers]
    for route in programme.routes:
        if route.worker < len(actions) and state.step in route.actions:
            actions[route.worker] = tuple(route.actions[state.step])
    orders = [("SELL", item, quantity)
              for item, quantity in sorted(programme.planned_sale.get(state.step, {}).items()) if quantity]
    for event in programme.events_at(state.step):
        if event.kind == "HIRE": orders.extend(("HIRE",) for _ in range(event.quantity))
        elif event.kind == "BUY_LAND": orders.append(("BUY_LAND",))
        elif event.kind in {"BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT"}:
            orders.append((event.kind, event.item, event.quantity))
    return TurnDecision(tuple(actions), tuple(orders[:rules.MAX_MARKET_ORDERS]))


def _refresh_sales(state: State, programme: Programme) -> Programme:
    # All non-sale commitments, placements, services and staffing remain frozen.
    sale = optimize_sales(state, _arrivals(state, programme),
                          known_demand_events(state), opponent_pressure(state))
    future = {step: amounts for step, amounts in sale.planned_sale.items() if step >= state.step}
    past = {step: amounts for step, amounts in programme.planned_sale.items() if step < state.step}
    return replace(programme, planned_sale={**past, **future},
                   market_inventory=sale.market_inventory)


@dataclass
class DailyPlanningSession:
    _plans: dict[int, Programme] = field(default_factory=dict, init=False)
    _last_steps: dict[int, int] = field(default_factory=dict, init=False)

    def reset(self):
        self._plans.clear(); self._last_steps.clear()

    def plan_for(self, state: State) -> Programme:
        prior = self._plans.get(state.player)
        last = self._last_steps.get(state.player, -1)
        reason = None if prior is None else programme_invalidation(state, prior)
        if prior is None or state.step < last or prior.day != state.day or reason is not None:
            prior = make_plan(state)
        self._plans[state.player] = prior
        self._last_steps[state.player] = state.step
        return prior

    def execution_for(self, state: State, programme: Programme) -> TurnDecision:
        if programme.planned_sale.get(state.step):
            programme = _refresh_sales(state, programme)
            self._plans[state.player] = programme
        return _decision(state, programme)

    @property
    def plans(self) -> Mapping[int, Programme]:
        return dict(self._plans)

    @property
    def planning_diagnostics(self):
        return tuple(plan.diagnostics for plan in self._plans.values())
