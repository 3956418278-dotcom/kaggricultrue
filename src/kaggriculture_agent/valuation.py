"""Economic value of an exactly reached owned state."""

from __future__ import annotations

from dataclasses import replace

from . import rules
from .planner import _animal_commitment, _existing_crop_obligation


class EndValue:
    """Cash plus reachable inventory and marginal surviving-asset value."""

    def __init__(self, initial):
        self.initial = initial
        self.cache = {}

    def __call__(self, state):
        if state.step > rules.TERMINAL_ACTION_STEP:
            return (state.money, 0, 0)
        settled = state
        if state.day == self.initial.day:
            if state.day == 29:
                return (state.money, 0, 0)
            settled = rules.advance_owned(
                replace(state, hour=23, step=state.day * 24 + 23), ()
            )
        value = settled.money
        for item in rules.PRODUCTS:
            value += rules.projected_sale_revenue(
                item, settled.owned_total(item), settled.market_inventory[item],
                settled.step, settled.step, (),
            )
        value += sum(quantity * rules.CROPS[crop].seed_cost
                     for crop, quantity in settled.seeds.items())
        value += sum(settled.owned_total(animal) * rule.cost
                     for animal, rule in rules.ANIMALS.items())
        debt = servicing = 0
        for tile in settled.tiles:
            if tile.kind != "PLANT" and not tile.animal:
                continue
            key = (settled.day, tile.position, repr(tile.raw))
            if key not in self.cache:
                valued = replace(
                    settled,
                    market_inventory=self.initial.market_inventory,
                    market_prices=self.initial.market_prices,
                    unlocked_shops=self.initial.unlocked_shops,
                )
                if tile.animal:
                    project = _animal_commitment(
                        valued, tile.animal, tile.position, existing=True,
                        placed_day=tile.raw.get("placed_day", settled.day),
                        current_yield=tile.raw.get("yield_units", 0),
                        fertilizer_available=tile.raw.get("fertilizer_available", False),
                        pending_care_bonus=tile.raw.get("pending_care_bonus", 0),
                    )
                else:
                    project = _existing_crop_obligation(valued, tile)
                marginal = max(0, project.terminal_profit)
                if tile.kind == "PLANT" and tile.raw.get("max_lifespan_step", -1) >= 0:
                    arrival = settled.step + rules.distance_to_shed(tile.position)
                    lifespan = tile.raw["max_lifespan_step"]
                    if arrival >= lifespan:
                        lost = max(0, (arrival - 1 - lifespan) // 2 + 1)
                        realizable = max(0, tile.raw.get("yield_units", 0) - lost)
                        marginal = min(
                            marginal,
                            rules.projected_sale_revenue(
                                tile.raw["crop"], realizable,
                                valued.market_inventory[tile.raw["crop"]],
                                arrival, settled.step, settled.unlocked_shops,
                            ),
                        )
                self.cache[key] = (marginal, len({w.day for w in project.actions.work}))
            marginal, work = self.cache[key]
            value += marginal
            debt += tile.raw.get(
                "consecutive_unwatered", tile.raw.get("consecutive_unfed", 0)
            )
            servicing += work * rules.distance_to_shed(tile.position)
        return (value, -debt, -servicing)


def end_value(opening_state, ending_state) -> int:
    return EndValue(opening_state)(ending_state)[0]
