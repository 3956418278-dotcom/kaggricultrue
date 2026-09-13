"""Regression checks for lightweight, attach-anytime current-asset handling."""
from dataclasses import replace
from unittest.mock import patch
import unittest

from kaggle_environments import make

from src.kaggriculture_agent import current_assets as current_assets_module
from src.kaggriculture_agent import rules
from src.kaggriculture_agent.current_assets import (
    AnimalDecisionInputs,
    EXIT,
    MAINTAIN,
    animal_locality_bonus,
    current_asset_programmes,
    read_current_assets,
)
from src.kaggriculture_agent.planner import (
    _crop_programme,
    _programme_from_assets,
    _unused_tiles,
)
from src.kaggriculture_agent.operating import programme_invalidation
from src.kaggriculture_agent.programme import (
    LandProgramme,
    Programme,
    ProgrammeEvent,
)
from src.kaggriculture_agent.state import AssetState, TileState, reconstruct


class MidgameAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        observation = make(
            "kaggriculture", configuration={"seed": 17}
        ).reset(2)[0].observation
        cls.base = reconstruct(observation)

    def animal_state(
        self,
        *,
        inventory=10_000,
        day=11,
        held=1,
        consecutive=0,
        fed=False,
        fertilizer=False,
        pending_care=0,
        placed_day=3,
        shops=(),
        prior_assets=(),
        prior_shops=(),
        replacement=0,
        base_incremental_hire_cost=0,
        care_incremental_hire_cost=0,
    ):
        position = (4, 4)
        raw = {
            "kind": "PASTURE",
            "animal": "COW",
            "placed_day": placed_day,
            "yield_units": held,
            "consecutive_unfed": consecutive,
            "fed_today": fed,
            "cared_today": False,
            "fertilizer_available": fertilizer,
            "pending_care_bonus": pending_care,
        }
        tiles = list(self.base.own.tiles)
        tiles[position[1] * self.base.board_size + position[0]] = TileState(
            position, raw
        )
        animal = AssetState("COW", position, raw)
        market_inventory = dict(self.base.market.inventory)
        market_inventory.update(
            {"MILK": inventory, "WHEAT": 10_000, "FERTILIZER": 10_000}
        )
        state = replace(
            self.base,
            step=day * 24,
            day=day,
            turn=0,
            shops=tuple(shops),
            market=replace(self.base.market, inventory=market_inventory),
            own=replace(
                self.base.own,
                tiles=tuple(tiles),
                animals=(animal,),
                crops=(),
            ),
        )
        identifier = "current:COW:4:4"
        inputs = AnimalDecisionInputs(
            replacement_advantages={position: replacement},
            base_incremental_hire_costs={
                identifier: base_incremental_hire_cost
            },
            care_incremental_hire_costs={
                identifier: care_incremental_hire_cost
            },
        )
        current = read_current_assets(
            state,
            prior_assets=prior_assets,
            prior_shops=prior_shops,
            inputs=inputs,
        )[0]
        return state, current

    def test_no_key_event_preserves_decision_without_revaluing_cycle(self):
        _, prior = self.animal_state(held=5, placed_day=11)
        self.assertEqual(prior.animal_mode, MAINTAIN)

        with (
            patch.object(
                current_assets_module,
                "_cycle_value",
                wraps=current_assets_module._cycle_value,
            ) as cycle_value,
            patch.object(
                current_assets_module,
                "known_demand_events",
                wraps=current_assets_module.known_demand_events,
            ) as demand_events,
        ):
            _, frozen = self.animal_state(
                inventory=10_100,
                day=12,
                held=5,
                consecutive=1,
                placed_day=11,
                prior_assets=(prior,),
                prior_shops=(),
            )

        cycle_value.assert_not_called()
        demand_events.assert_not_called()
        self.assertIsNone(frozen.decision_event)
        self.assertEqual(frozen.animal_mode, prior.animal_mode)
        self.assertEqual(frozen.base_gain, prior.base_gain)
        self.assertEqual(
            frozen.next_production_step, prior.next_production_step
        )

    def test_shop_reveal_revalues_next_cycle(self):
        _, prior = self.animal_state(held=5, placed_day=11)
        with patch.object(
            current_assets_module,
            "_cycle_value",
            wraps=current_assets_module._cycle_value,
        ) as cycle_value:
            _, current = self.animal_state(
                inventory=10_100,
                day=12,
                held=5,
                consecutive=1,
                placed_day=11,
                shops=("YARN_STORE",),
                prior_assets=(prior,),
                prior_shops=(),
            )
        self.assertEqual(cycle_value.call_count, 1)
        self.assertEqual(current.decision_event, "SHOP_REVEAL")
        self.assertEqual(current.animal_mode, EXIT)

    def test_completed_production_revalues_next_cycle(self):
        _, prior = self.animal_state(day=11, held=0, placed_day=4)
        self.assertEqual(prior.next_production_step, 12 * 24)
        with patch.object(
            current_assets_module,
            "_cycle_value",
            wraps=current_assets_module._cycle_value,
        ) as cycle_value:
            _, current = self.animal_state(
                day=12,
                held=1,
                consecutive=1,
                placed_day=4,
                prior_assets=(prior,),
                prior_shops=(),
            )
        self.assertEqual(cycle_value.call_count, 1)
        self.assertEqual(current.decision_event, "PRODUCTION")
        self.assertGreater(
            current.next_production_step, prior.next_production_step
        )

    def test_harvest_capacity_change_invalidates_and_revalues(self):
        state, prior = self.animal_state(held=1, placed_day=11)
        shell = current_asset_programmes((prior,))
        programme = Programme(
            state.step,
            state.day,
            state.shops,
            assets=shell,
            current_assets=(prior,),
        )
        harvested_state, _ = self.animal_state(held=0, placed_day=11)
        self.assertIn(
            "held capacity changed",
            programme_invalidation(harvested_state, programme),
        )
        with patch.object(
            current_assets_module,
            "_cycle_value",
            wraps=current_assets_module._cycle_value,
        ) as cycle_value:
            current = read_current_assets(
                harvested_state,
                prior_assets=(prior,),
                prior_shops=(),
            )[0]
        self.assertEqual(cycle_value.call_count, 1)
        self.assertEqual(current.decision_event, "HELD_CAPACITY_CHANGE")

    def test_positive_blocked_replacement_revalues_and_can_exit(self):
        _, prior = self.animal_state(held=5, placed_day=11)
        replacement = prior.base_gain + 1
        with patch.object(
            current_assets_module,
            "_cycle_value",
            wraps=current_assets_module._cycle_value,
        ) as cycle_value:
            _, current = self.animal_state(
                held=5,
                placed_day=11,
                prior_assets=(prior,),
                prior_shops=(),
                replacement=replacement,
            )
        self.assertEqual(cycle_value.call_count, 1)
        self.assertEqual(current.decision_event, "REPLACEMENT")
        self.assertEqual(current.replacement_advantage, replacement)
        self.assertEqual(current.animal_mode, EXIT)

    def test_long_cycle_cow_values_its_next_batch_instead_of_exiting(self):
        _, current = self.animal_state(day=11, held=0, placed_day=11)
        self.assertEqual(current.next_production_step, 19 * 24)
        self.assertGreater(current.base_gain, 0)
        self.assertEqual(current.animal_mode, MAINTAIN)

    def test_proven_incremental_hire_cost_enters_base_gain_directly(self):
        _, baseline = self.animal_state(day=11, held=0, placed_day=11)
        _, burdened = self.animal_state(
            day=11,
            held=0,
            placed_day=11,
            base_incremental_hire_cost=baseline.base_gain,
        )
        self.assertEqual(burdened.base_gain, 0)
        self.assertEqual(burdened.animal_mode, EXIT)

    def test_current_asset_machine_can_attach_before_day_11(self):
        _, current = self.animal_state(day=5, held=0, placed_day=5)
        self.assertEqual(current.decision_event, "ATTACH")
        self.assertEqual(current.next_production_step, 13 * 24)
        self.assertIn(current.animal_mode, {MAINTAIN, EXIT})

    def test_care_is_separate_and_rejected_when_bonus_would_overflow(self):
        _, open_capacity = self.animal_state(
            day=11, held=0, placed_day=11
        )
        _, insufficient_capacity = self.animal_state(
            day=11, held=5, placed_day=11
        )
        self.assertEqual(open_capacity.animal_mode, MAINTAIN)
        self.assertTrue(open_capacity.care_approved)
        self.assertGreater(open_capacity.care_gain, 0)
        self.assertFalse(insufficient_capacity.care_approved)

    def test_exit_first_night_keeps_official_base_production(self):
        state, current = self.animal_state(
            inventory=10_100,
            held=0,
            consecutive=0,
            placed_day=4,
            replacement=1,
        )
        self.assertEqual(current.animal_mode, EXIT)
        output = [
            event for event in current.next_events if event.kind == "OUTPUT"
        ]
        self.assertEqual(
            [(event.step, event.item, event.quantity) for event in output],
            [(12 * 24, "MILK", 1)],
        )
        self.assertEqual(current.physical_release_step, 13 * 24)

        closing = replace(state, step=12 * 24 - 1, day=11, turn=23)
        after_first_night = rules.advance_owned(closing, (("PASS",),))
        first_raw = after_first_night.tile_at((4, 4)).raw
        self.assertEqual(first_raw["yield_units"], 1)
        self.assertEqual(first_raw["consecutive_unfed"], 1)
        self.assertTrue(first_raw["fertilizer_available"])
        current_state = after_first_night
        for _ in range(24):
            current_state = rules.advance_owned(
                current_state, (("PASS",),)
            )
        self.assertIsNone(current_state.tile_at((4, 4)).animal)
        self.assertEqual(current_state.tile_at((4, 4)).kind, "PASTURE")

    def test_exit_second_day_takes_last_product_and_f_on_same_visit(self):
        _, current = self.animal_state(
            inventory=10_100,
            held=1,
            consecutive=1,
            fertilizer=True,
        )
        self.assertEqual(current.animal_mode, EXIT)
        kinds = {event.kind for event in current.today_events}
        self.assertIn("HARVEST", kinds)
        self.assertIn("COLLECT_F", kinds)
        self.assertNotIn("FEED", kinds)
        self.assertEqual(current.physical_release_step, 12 * 24)

    def test_held_product_below_capacity_is_not_automatically_harvested(self):
        _, current = self.animal_state(
            inventory=10_000, held=1, placed_day=3
        )
        self.assertEqual(current.animal_mode, MAINTAIN)
        self.assertNotIn(
            "HARVEST", {event.kind for event in current.today_events}
        )

    def test_harvest_occurs_before_next_refresh_would_overflow_max_held(self):
        _, current = self.animal_state(
            inventory=10_000, held=6, placed_day=4
        )
        harvest = [
            event
            for event in current.today_events
            if event.kind == "HARVEST"
        ]
        self.assertEqual(current.animal_mode, MAINTAIN)
        self.assertEqual(len(harvest), 1)
        self.assertEqual(harvest[0].source, "HELD_OVERFLOW")

    def test_bought_land_joins_old_and_new_tiles_without_predig(self):
        land = LandProgramme(
            "NE",
            self.base.step,
            rules.LAND_PRICES[0],
            (5, 4),
            "UNBLOCK:COW",
        )
        buy = ProgrammeEvent(
            self.base.step, -30, "land:NE", "BUY_LAND"
        )
        programme = _programme_from_assets(self.base, (), (buy,), (land,))
        pool = set(_unused_tiles(self.base, programme))
        self.assertTrue(
            any(rules.quadrant(tile) == "NW" for tile in pool)
        )
        self.assertTrue(
            any(rules.quadrant(tile) == "NE" for tile in pool)
        )
        new_tile = min(
            tile for tile in pool if rules.quadrant(tile) == "NE"
        )
        crop = _crop_programme(
            self.base,
            "TOMATO",
            new_tile,
            existing=False,
            start_step=self.base.step + 1,
        )
        self.assertNotIn(
            "DIG", {event.kind for event in crop.service_schedule}
        )

    def test_animal_locality_bonus_is_early_only(self):
        near = animal_locality_bonus("COW", (4, 4), 11)
        far = animal_locality_bonus("COW", (0, 0), 11)
        late = animal_locality_bonus("COW", (4, 4), 24)
        before_handoff = animal_locality_bonus("COW", (4, 4), 5)
        self.assertGreater(near, far)
        self.assertEqual(near, before_handoff)
        self.assertEqual(late, 0.0)

    def test_current_snapshot_keeps_official_fields_and_stays_short(self):
        _, current = self.animal_state(inventory=10_000)
        self.assertIn("placed_day", current.official)
        self.assertIn("pending_care_bonus", current.official)
        self.assertTrue(
            all(event.step // 24 <= 12 for event in current.next_events)
        )
        self.assertTrue(
            all(event.step // 24 == 11 for event in current.today_events)
        )
        shell = current_asset_programmes((current,))[0]
        self.assertEqual(shell.output_schedule, ())
        self.assertTrue(
            all(
                event.step // 24 == 11
                for event in (
                    *shell.service_schedule,
                    *shell.harvest_schedule,
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
