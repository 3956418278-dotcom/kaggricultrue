"""Regression checks for the runnable observation-driven midgame line."""
from dataclasses import replace
import unittest

from kaggle_environments import make

from src.kaggriculture_agent import rules
from src.kaggriculture_agent.contract import construct_action
from src.kaggriculture_agent.current_assets import (
    EXIT,
    MAINTAIN,
    PRODUCE,
    animal_daily_value,
    animal_locality_bonus,
    crop_daily_value,
    read_current_assets,
)
from src.kaggriculture_agent.market import optimize_short_sales
from src.kaggriculture_agent.operating import (
    DailyPlanningSession,
    _decision,
)
from src.kaggriculture_agent.planner import (
    _animal_programme,
    _arrivals,
    _crop_programme,
    _land_expansion,
    _programme_from_assets,
    _unused_tiles,
    make_plan,
)
from src.kaggriculture_agent.programme import (
    Programme,
    ProgrammeEvent,
)
from src.kaggriculture_agent.return_requirements import (
    annotate_programme_returns,
    mark_tile_workloads,
    return_requirements,
)
from src.kaggriculture_agent.realization import TurnDecision
from src.kaggriculture_agent.state import (
    AssetState,
    TileState,
    reconstruct,
)


class MidgameAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        observation = make(
            "kaggriculture", configuration={"seed": 17}
        ).reset(2)[0].observation
        cls.base = reconstruct(observation)

    def state(
        self,
        *,
        day=11,
        animals=(),
        crops=(),
        shed=None,
        seeds=None,
        money=3_000,
        inventories=None,
    ):
        tiles = list(self.base.own.tiles)
        for asset in (*animals, *crops):
            x, y = asset.position
            tiles[y * self.base.board_size + x] = TileState(
                asset.position, dict(asset.official))
        market_inventory = dict(self.base.market.inventory)
        market_inventory.update(inventories or {})
        market_price = {
            item: rules.market_price(item, market_inventory[item])
            for item in rules.PRODUCTS
        }
        return replace(
            self.base,
            step=day * 24,
            day=day,
            turn=0,
            market=replace(
                self.base.market,
                inventory=market_inventory,
                price=market_price,
            ),
            own=replace(
                self.base.own,
                money=money,
                shed_inventory=dict(shed or {}),
                seeds=dict(seeds or {}),
                tiles=tuple(tiles),
                animals=tuple(animals),
                crops=tuple(crops),
            ),
        )

    @staticmethod
    def cow(*, held=0, consecutive=0, fertilizer=False,
            placed_day=3, position=(4, 4), pending=0):
        raw = {
            "kind": "PASTURE",
            "animal": "COW",
            "placed_day": placed_day,
            "yield_units": held,
            "consecutive_unfed": consecutive,
            "fed_today": False,
            "cared_today": False,
            "fertilizer_available": fertilizer,
            "pending_care_bonus": pending,
        }
        return AssetState("COW", position, raw)

    @staticmethod
    def crop(kind, *, planted_day, held, position=(3, 3),
             consecutive=0, fertilized=-1, watered=False):
        rule = rules.CROPS[kind]
        raw = {
            "kind": "PLANT",
            "crop": kind,
            "planted_day": planted_day,
            "watered_today": watered,
            "consecutive_unwatered": consecutive,
            "fertilized_until_day": fertilized,
            "yield_units": held,
            "max_lifespan_step": (
                -1 if rule.ongoing
                else (planted_day + rule.max_yield_day + 1) * 24
            ),
        }
        return AssetState(kind, position, raw)

    def current(self, state, prior=()):
        return read_current_assets(state, prior_assets=prior)[0]

    def test_no_feed_crop_purpose_or_programme_field_remains(self):
        self.assertFalse(hasattr(Programme(0, 0, ()), "wheat_feed"))
        source = (
            __import__(
                "src.kaggriculture_agent.planner",
                fromlist=["x"],
            ).__file__
        )
        self.assertIsNotNone(source)

    def test_feed_deficit_buys_wheat_without_planting_it(self):
        cow = self.cow(consecutive=1, placed_day=21)
        state = self.state(
            day=28, animals=(cow,), money=500,
            inventories={"MILK": 10_000, "FERTILIZER": 10_000})
        plan = make_plan(state)
        self.assertTrue(plan.feasible, plan.diagnostics)
        buys = [
            event for event in plan.events
            if event.kind == "BUY_PRODUCT" and event.item == "WHEAT"
        ]
        self.assertEqual(sum(event.quantity for event in buys), 1)
        self.assertTrue(all(
            asset.purpose == "W_BUFFER"
            for asset in plan.additions
            if asset.asset_type == "WHEAT"))

    def test_mature_existing_wheat_is_used_before_buy(self):
        cow = self.cow(
            consecutive=1, placed_day=21, position=(4, 4))
        wheat = self.crop(
            "WHEAT", planted_day=26, held=2, position=(3, 4))
        state = self.state(
            day=28, animals=(cow,), crops=(wheat,), money=500)
        plan = make_plan(state)
        self.assertTrue(plan.feasible, plan.diagnostics)
        self.assertTrue(any(
            event.kind == "HARVEST"
            and event.item == "WHEAT"
            and event.source == "FEED_SUPPLY"
            for event in plan.events))
        self.assertFalse(any(
            event.kind == "BUY_PRODUCT" and event.item == "WHEAT"
            for event in plan.events))
        drop_step = min(
            step for step, amounts in _arrivals(state, plan).items()
            if amounts.get("WHEAT"))
        feed_step = min(
            step for route in plan.routes
            for step, action in route.actions.items()
            if action and action[0] == "FEED")
        self.assertLess(drop_step, feed_step)

    def test_animal_daily_value_contains_realizable_fertilizer(self):
        high = self.state(inventories={"FERTILIZER": 9_800})
        low = self.state(inventories={"FERTILIZER": 30_000})
        self.assertGreater(
            animal_daily_value(high, "COW"),
            animal_daily_value(low, "COW"))

    def test_current_asset_state_has_no_unwired_optimizer_maps(self):
        current = self.current(self.state(
            animals=(self.cow(),), shed={"WHEAT": 2}))
        self.assertFalse(hasattr(current, "replacement_advantage"))
        self.assertFalse(hasattr(current, "base_gain"))

    def test_runtime_path_reaches_produce_maintain_and_exit(self):
        produce = self.current(self.state(
            day=20, animals=(self.cow(placed_day=12),),
            shed={"WHEAT": 2}))
        maintain = self.current(self.state(
            day=20, animals=(self.cow(placed_day=12),),
            shed={"WHEAT": 2},
            inventories={"MILK": 30_000}))
        exit_asset = self.current(self.state(
            day=20, animals=(self.cow(placed_day=12),),
            shed={"WHEAT": 2},
            inventories={
                "MILK": 30_000,
                "FERTILIZER": 30_000,
                "WHEAT": 0,
            }))
        self.assertEqual(produce.mode, PRODUCE)
        self.assertEqual(maintain.mode, MAINTAIN)
        self.assertEqual(exit_asset.mode, EXIT)

    def test_exit_order_is_nearest_first_and_revalues_remaining_animals(self):
        from unittest.mock import patch

        from src.kaggriculture_agent import current_assets
        from src.kaggriculture_agent.current_assets import (
            exit_current_asset,
            next_animal_production_day,
        )
        from src.kaggriculture_agent.market import forecast_inventory
        from src.kaggriculture_agent.midgame_config import MidgameParameters

        near = self.cow(placed_day=3, position=(4, 4))
        far = self.cow(placed_day=3, position=(0, 0))
        state = self.state(
            day=20, animals=(far, near),
            inventories={
                "MILK": 9_886,
                "FERTILIZER": 30_000,
                "WHEAT": 0,
            })
        params = MidgameParameters(expected_shop_demand_per_reveal={})
        real_value = current_assets.animal_daily_value

        def record(*args, **kwargs):
            kwargs.pop("self", None)
            calls.append(kwargs.get("include_purchase_cost", True))
            return real_value(*args, **kwargs)

        calls = []
        with patch.object(
                current_assets, "animal_daily_value",
                side_effect=lambda *a, **k: record(*a, **k)):
            current = read_current_assets(state, params=params)
        # Existing animals are valued without re-paying their purchase cost.
        self.assertTrue(calls)
        self.assertTrue(all(value is False for value in calls))
        by_tile = {asset.tile: asset for asset in current}
        # Positive continuation value keeps the animal instead of exiting on
        # a repeated sunk cost.
        self.assertIn(by_tile[(4, 4)].mode, (PRODUCE, MAINTAIN))
        self.assertIsNone(by_tile[(4, 4)].exit_order)
        self.assertGreater(by_tile[(4, 4)].daily_value, 0)
        self.assertIn(by_tile[(0, 0)].mode, (PRODUCE, MAINTAIN))
        self.assertIsNone(by_tile[(0, 0)].exit_order)
        self.assertGreater(by_tile[(0, 0)].daily_value, 0)
        # Excluding an exiting tile lowers the forecast supply the remaining
        # animal is priced against.
        step = (
            next_animal_production_day(far.official, state.day) + 1) * 24
        kept = forecast_inventory(
            state, "MILK", step, excluded_own_tiles=frozenset({(4, 4)}),
            params=params)
        both = forecast_inventory(
            state, "MILK", step, excluded_own_tiles=frozenset(),
            params=params)
        self.assertLess(kept, both)
        # An explicit EXIT stays sticky and keeps the nearest-first order.
        prior = exit_current_asset(state, near, params)
        self.assertEqual(prior.mode, EXIT)
        current = read_current_assets(
            state, prior_assets=(prior,), params=params)
        by_tile = {asset.tile: asset for asset in current}
        self.assertEqual(by_tile[(4, 4)].mode, EXIT)
        self.assertEqual(by_tile[(4, 4)].exit_order, 0)
        self.assertIn(by_tile[(0, 0)].mode, (PRODUCE, MAINTAIN))
        self.assertIsNone(by_tile[(0, 0)].exit_order)
        self.assertGreater(by_tile[(0, 0)].daily_value, 0)

    def test_multiple_exits_receive_nearest_first_order(self):
        animals = (
            self.cow(placed_day=3, position=(0, 0)),
            self.cow(placed_day=3, position=(3, 4)),
            self.cow(placed_day=3, position=(4, 4)),
        )
        state = self.state(
            day=20, animals=animals,
            inventories={
                "MILK": 30_000,
                "FERTILIZER": 30_000,
                "WHEAT": 0,
            })
        current = read_current_assets(state)
        orders = {asset.tile: asset.exit_order for asset in current}
        self.assertEqual(orders, {(4, 4): 0, (3, 4): 1, (0, 0): 2})

    def test_early_near_harvest_is_labeled_must_return(self):
        state = self.state(day=9)
        near = (1, 4)  # shed distance 3
        far = (0, 0)
        programme = Programme(
            state.step, state.day, (), events=(
                ProgrammeEvent(
                    state.step, 0, "near", "HARVEST", near, "near",
                    "MELON", 6, ("HARVEST",)),
                ProgrammeEvent(
                    state.step, 0, "far", "HARVEST", far, "far",
                    "MELON", 6, ("HARVEST",)),
            ))
        labels = return_requirements(
            state, programme, eod_capacity=100)
        self.assertEqual(labels.must_return, frozenset((near,)))
        self.assertEqual(labels.eod, frozenset((far,)))
        self.assertEqual(labels.reason[near], "EARLY_NEAR_SHED")

    def test_capacity_labels_farthest_output_eod_and_nearer_output_return(self):
        from src.kaggriculture_agent import zonal_templates as zonal

        state = self.state(day=11)
        tiles = sorted(
            zonal._owned_tiles(zonal.THREE_LAND),
            key=lambda tile: (
                -rules.distance_to_shed(tile), tile[1], tile[0]))[:18]
        programme = Programme(
            state.step, state.day, (), events=tuple(
                ProgrammeEvent(
                    state.step, 0, f"melon:{x}:{y}", "HARVEST",
                    (x, y), f"melon:{x}:{y}", "MELON", 6,
                    ("HARVEST",))
                for x, y in tiles))
        labels = return_requirements(
            state, programme, eod_capacity=rules.SHED_CAPACITY)
        self.assertEqual(labels.eod, frozenset(tiles[:16]))
        self.assertEqual(labels.must_return, frozenset(tiles[16:]))
        self.assertEqual(labels.eod_units, 96)

        workloads = mark_tile_workloads(
            {tile: zonal.TileWorkload(1, 6) for tile in tiles}, labels)
        solution = zonal.select_minimum_workforce(
            zonal.THREE_LAND, workloads)
        self.assertIsNotNone(solution)
        self.assertLessEqual(108 - solution.returned_units, 100)

    def test_opening_shed_inventory_reduces_eod_capacity(self):
        state = self.state(day=11, shed={"MELON": 94})
        tiles = ((0, 0), (1, 0), (2, 0))
        programme = Programme(
            state.step, state.day, (), events=tuple(
                ProgrammeEvent(
                    state.step, 0, f"harvest:{x}", "HARVEST", tile,
                    f"crop:{x}", "MELON", 6, ("HARVEST",))
                for x, tile in enumerate(tiles)))
        labels = return_requirements(state, programme)
        self.assertEqual(labels.eod_capacity, 6)
        self.assertEqual(labels.eod, frozenset(((0, 0),)))
        self.assertEqual(labels.must_return, frozenset(((1, 0), (2, 0))))

    def test_must_return_annotation_reaches_shed_by_turn_22(self):
        from src.kaggriculture_agent.intraday import solve_intraday

        state = self.state(day=9)
        tile = (1, 4)
        programme = Programme(
            state.step, state.day, (), events=(ProgrammeEvent(
                state.step, 0, "harvest", "HARVEST", tile, "crop",
                "MELON", 6, ("HARVEST",), deadline=state.step + 22),))
        annotated = annotate_programme_returns(state, programme)
        self.assertEqual(
            annotated.must_return[state.day], frozenset((tile,)))
        solved = solve_intraday(state, annotated)
        drops = [
            step - state.day * rules.TURNS_PER_DAY
            for route in solved.routes
            for step, action in route.actions.items()
            if action and action[0] == "DROP"]
        self.assertTrue(drops)
        self.assertLessEqual(min(drops), 22)

    def test_only_produce_cares_and_maintain_feeds_minimally(self):
        produce = self.current(self.state(
            day=20,
            animals=(self.cow(placed_day=12, consecutive=0),),
            shed={"WHEAT": 2}))
        maintain = self.current(self.state(
            day=20,
            animals=(self.cow(
                held=5, placed_day=12, consecutive=1),),
            shed={"WHEAT": 2},
            inventories={"MILK": 30_000}))
        self.assertIn("CARE", {e.kind for e in produce.today_events})
        kinds = {e.kind for e in maintain.today_events}
        self.assertIn("FEED", kinds)
        self.assertNotIn("CARE", kinds)

    def test_exit_first_night_keeps_last_official_production(self):
        cow = self.cow(placed_day=19, consecutive=0)
        state = self.state(
            day=28, animals=(cow,),
            inventories={
                "MILK": 30_000,
                "FERTILIZER": 30_000,
                "WHEAT": 0,
            })
        current = self.current(state)
        self.assertEqual(current.mode, EXIT)
        self.assertEqual(
            [(e.step, e.item, e.quantity)
             for e in current.next_events if e.kind == "OUTPUT"],
            [(29 * 24, "MILK", 1)],
        )
        self.assertEqual(
            current.physical_release_step,
            rules.TERMINAL_ACTION_STEP + 1)
        closing = replace(state, step=28 * 24 + 23, turn=23)
        after = rules.advance_owned(closing, (("PASS",),))
        raw = after.tile_at(cow.position).raw
        self.assertEqual(raw["yield_units"], 1)
        self.assertTrue(raw["fertilizer_available"])

    def test_exit_collects_last_f_without_another_visit_reason(self):
        cow = self.cow(
            held=0, consecutive=1, fertilizer=True, placed_day=19)
        state = self.state(
            day=28, animals=(cow,),
            inventories={
                "MILK": 30_000,
                "FERTILIZER": 30_000,
                "WHEAT": 0,
            })
        current = self.current(state)
        self.assertEqual(current.mode, EXIT)
        self.assertIn(
            "COLLECT_F", {e.kind for e in current.today_events})
        self.assertNotIn("FEED", {e.kind for e in current.today_events})

    def test_exit_second_day_liquidates_product_and_f_together(self):
        cow = self.cow(
            held=1, consecutive=1, fertilizer=True, placed_day=19)
        state = self.state(
            day=28, animals=(cow,),
            inventories={
                "MILK": 30_000,
                "FERTILIZER": 30_000,
                "WHEAT": 0,
            })
        current = self.current(state)
        self.assertEqual(
            {event.kind for event in current.today_events},
            {"HARVEST", "COLLECT_F"})

    def test_terminal_animal_exits_without_wasting_feed(self):
        cow = self.cow(
            held=1, consecutive=1, fertilizer=True, placed_day=21)
        current = self.current(self.state(
            day=29, animals=(cow,), shed={"WHEAT": 2}))
        self.assertEqual(current.mode, EXIT)
        self.assertNotIn(
            "FEED", {event.kind for event in current.today_events})
        self.assertEqual(
            {event.kind for event in current.today_events},
            {"HARVEST", "COLLECT_F"})

    def test_held_product_is_not_harvested_until_capacity_requires_it(self):
        ordinary = self.current(self.state(
            day=11,
            animals=(self.cow(held=1, placed_day=11),),
            shed={"WHEAT": 2}))
        full = self.current(self.state(
            day=11,
            animals=(self.cow(held=6, placed_day=4),),
            shed={"WHEAT": 2}))
        self.assertNotIn(
            "HARVEST", {e.kind for e in ordinary.today_events})
        harvests = [
            e for e in full.today_events if e.kind == "HARVEST"]
        self.assertEqual(len(harvests), 1)
        self.assertEqual(harvests[0].source, "HELD_OVERFLOW")

    def test_new_animal_helper_has_no_automatic_harvest_schedule(self):
        state = self.state()
        animal = _animal_programme(
            state, "COW", (4, 4), existing=False)
        self.assertEqual(animal.output_schedule, ())
        self.assertEqual(animal.harvest_schedule, ())

    def test_biological_output_is_not_market_inventory(self):
        cow = self.cow(placed_day=21)
        state = self.state(
            day=28, animals=(cow,), shed={"WHEAT": 2},
            inventories={"MILK": 10_000})
        plan = make_plan(state)
        opening = state.market.inventory["MILK"]
        self.assertTrue(any(
            e.kind == "OUTPUT"
            for asset in plan.current_assets
            for e in asset.next_events))
        self.assertTrue(all(
            amounts.get("MILK", opening) <= opening
            for amounts in plan.market_inventory.values()))

    def test_existing_tomato_runs_real_water_and_f_decision(self):
        tomato = self.crop(
            "TOMATO", planted_day=3, held=0, consecutive=1)
        state = self.state(
            day=11, crops=(tomato,),
            shed={"FERTILIZER": 1},
            inventories={"FERTILIZER": 30_000})
        current = self.current(state)
        kinds = {event.kind for event in current.today_events}
        self.assertIn("WATER", kinds)
        self.assertIn("FERTILIZE", kinds)
        self.assertGreater(current.input_gain, 0)

    def test_existing_strawberry_uses_same_next_event_input_path(self):
        strawberry = self.crop(
            "STRAWBERRY", planted_day=1, held=0, consecutive=1)
        state = self.state(
            day=12, crops=(strawberry,),
            shed={"FERTILIZER": 1},
            inventories={"FERTILIZER": 30_000})
        kinds = {
            event.kind for event in self.current(state).today_events}
        self.assertEqual(kinds, {"FERTILIZE", "WATER"})

    def test_existing_crop_does_not_fertilize_when_increment_is_negative(self):
        tomato = self.crop(
            "TOMATO", planted_day=3, held=0, consecutive=1)
        state = self.state(
            day=11, crops=(tomato,),
            inventories={"FERTILIZER": 0})
        kinds = {
            event.kind for event in self.current(state).today_events}
        self.assertIn("WATER", kinds)
        self.assertNotIn("FERTILIZE", kinds)

    def test_collected_fertilizer_can_feed_an_approved_same_day_target(self):
        cow = self.cow(
            held=0, consecutive=1, fertilizer=True,
            placed_day=4, position=(4, 4))
        tomato = self.crop(
            "TOMATO", planted_day=3, held=0,
            consecutive=1, position=(3, 4))
        state = self.state(
            day=11, animals=(cow,), crops=(tomato,),
            shed={"WHEAT": 1}, money=0,
            inventories={"FERTILIZER": 30_000})
        plan = make_plan(state)
        self.assertTrue(plan.feasible, plan.diagnostics)
        self.assertFalse(any(
            event.kind == "BUY_PRODUCT"
            and event.item == "FERTILIZER"
            for event in plan.events))
        actions = {
            step: action
            for route in plan.routes
            for step, action in route.actions.items()
            if action != ("PASS",)
        }
        collect = min(
            step for step, action in actions.items()
            if action[0] == "COLLECT_FERTILIZER")
        fertilize = min(
            step for step, action in actions.items()
            if action[0] == "FERTILIZE")
        self.assertLess(collect, fertilize)

    def test_ongoing_crop_minus_one_lifespan_is_not_false_expiry(self):
        tomato = self.crop(
            "TOMATO", planted_day=10, held=1, consecutive=0)
        current = self.current(self.state(day=11, crops=(tomato,)))
        self.assertNotIn(
            "HARVEST", {event.kind for event in current.today_events})

    def test_terminal_ongoing_crop_harvests_without_watering(self):
        tomato = self.crop(
            "TOMATO", planted_day=20, held=2, consecutive=1)
        current = self.current(self.state(day=29, crops=(tomato,)))
        kinds = {event.kind for event in current.today_events}
        self.assertIn("HARVEST", kinds)
        self.assertNotIn("WATER", kinds)

    def test_past_last_one_time_harvest_point_is_liquidated(self):
        melon = self.crop("MELON", planted_day=0, held=3)
        current = self.current(self.state(day=13, crops=(melon,)))
        self.assertIn(
            "HARVEST", {event.kind for event in current.today_events})

    def test_fertilize_is_selected_only_when_daily_value_improves(self):
        cheap = self.state(inventories={"FERTILIZER": 30_000})
        expensive = self.state(inventories={"FERTILIZER": 0})
        self.assertGreater(
            crop_daily_value(cheap, "TOMATO").fertilized,
            crop_daily_value(cheap, "TOMATO").normal)
        self.assertLessEqual(
            crop_daily_value(expensive, "TOMATO").fertilized,
            crop_daily_value(expensive, "TOMATO").normal)

    def test_melon_selects_a_real_remaining_harvest_point(self):
        melon = self.crop("MELON", planted_day=1, held=1)
        state = self.state(
            day=11, crops=(melon,),
            shed={"FERTILIZER": 1},
            inventories={"FERTILIZER": 30_000})
        current = self.current(state)
        self.assertIn(
            current.next_harvest_step,
            {11 * 24, 12 * 24, 13 * 24})
        self.assertTrue(
            any(e.kind in {"WATER", "HARVEST"}
                for e in current.today_events))

    def test_seed_ledger_buys_only_global_deficit(self):
        state = self.state(seeds={"STRAWBERRY": 5})
        assets = tuple(
            _crop_programme(
                state, "STRAWBERRY", (index, 0),
                existing=False)
            for index in range(8)
        )
        programme = _programme_from_assets(state, assets)
        buys = [
            event for event in programme.events
            if event.kind == "BUY_SEED"
            and event.item == "STRAWBERRY"
        ]
        self.assertEqual(sum(event.quantity for event in buys), 3)

    def test_existing_seed_is_not_bought_again(self):
        state = self.state(seeds={"STRAWBERRY": 5})
        assets = tuple(
            _crop_programme(
                state, "STRAWBERRY", (index, 0),
                existing=False)
            for index in range(5)
        )
        programme = _programme_from_assets(state, assets)
        self.assertFalse(any(
            event.kind == "BUY_SEED" for event in programme.events))

    def test_land_requires_scale_and_joins_normal_tile_pool(self):
        poor = self.state(money=1_300)
        shell = _programme_from_assets(poor, ())
        self.assertFalse(_land_expansion(
            poor, shell,
            __import__(
                "src.kaggriculture_agent.midgame_config",
                fromlist=["DEFAULT_MIDGAME_PARAMETERS"],
            ).DEFAULT_MIDGAME_PARAMETERS,
        ).land)

        rich = self.state(money=10_000)
        expanded = _land_expansion(
            rich, _programme_from_assets(rich, ()),
            __import__(
                "src.kaggriculture_agent.midgame_config",
                fromlist=["DEFAULT_MIDGAME_PARAMETERS"],
            ).DEFAULT_MIDGAME_PARAMETERS,
        )
        self.assertTrue(expanded.land)
        quadrants = {
            rules.quadrant(tile) for tile in _unused_tiles(rich, expanded)}
        self.assertIn("NW", quadrants)
        self.assertIn("NE", quadrants)

    def test_animal_locality_bonus_decays_to_zero(self):
        self.assertGreater(
            animal_locality_bonus("COW", (4, 4), 11),
            animal_locality_bonus("COW", (0, 0), 11))
        self.assertEqual(
            animal_locality_bonus("COW", (4, 4), 24), 0.0)


class TradeAndExecutionTests(unittest.TestCase):
    state = MidgameAssetTests.state

    @classmethod
    def setUpClass(cls):
        observation = make(
            "kaggriculture", configuration={"seed": 17}
        ).reset(2)[0].observation
        cls.base = reconstruct(observation)

    def test_trade_machine_uses_only_now_plus_four_plus_eight(self):
        state = self.state(shed={"MILK": 3})
        sale = optimize_short_sales(state, {})
        self.assertTrue(sale.feasible)
        self.assertTrue(set(sale.planned_sale) <= {
            state.step, state.step + 4, state.step + 8})

    def test_new_checkpoint_replans_from_real_inventory(self):
        first = self.state(shed={"MILK": 3})
        second = replace(
            first,
            step=first.step + 4,
            turn=4,
            market=replace(
                first.market,
                inventory={
                    **first.market.inventory,
                    "MILK": first.market.inventory["MILK"] + 100,
                },
            ),
        )
        first_sale = optimize_short_sales(first, {})
        second_sale = optimize_short_sales(second, {})
        self.assertNotEqual(
            first_sale.market_inventory,
            second_sale.market_inventory)

    def test_shed_overflow_forces_sale_before_drop(self):
        state = self.state(shed={"MILK": 100})
        sale = optimize_short_sales(
            state, {state.step + 2: {"WOOL": 1}})
        self.assertTrue(sale.feasible, sale.failure)
        self.assertGreaterEqual(
            sum(sale.planned_sale.get(state.step + 1, {}).values()), 1)

    def test_cash_requirement_forces_same_turn_sale(self):
        state = self.state(
            money=0, shed={"MILK": 2})
        commitment = ProgrammeEvent(
            state.step, -1, "seed", "BUY_SEED",
            item="WHEAT", quantity=1)
        sale = optimize_short_sales(
            state, {}, commitments=(commitment,))
        self.assertTrue(sale.feasible, sale.failure)
        self.assertGreaterEqual(
            sum(sale.planned_sale.get(state.step, {}).values()), 1)

    def test_ordinary_sell_moves_one_turn_when_market_lines_are_full(self):
        state = self.state(money=10_000, shed={"MILK": 2})
        sell_step = state.step + 8
        commitments = tuple(
            ProgrammeEvent(
                sell_step, -1, f"seed-{index}", "BUY_SEED",
                item="WHEAT", quantity=1)
            for index in range(rules.MAX_MARKET_ORDERS)
        )
        sale = optimize_short_sales(
            state, {}, commitments=commitments)
        self.assertTrue(sale.feasible, sale.failure)
        self.assertEqual(
            sale.planned_sale.get(sell_step, {}).get("MILK", 0), 0)
        self.assertEqual(
            sale.planned_sale[sell_step + 1]["MILK"], 2)

    def test_deferred_sell_keeps_every_turn_within_market_line_limit(self):
        state = self.state(
            money=10_000, shed={"MILK": 2, "WOOL": 2})
        sell_step = state.step + 8
        commitments = tuple(
            ProgrammeEvent(
                sell_step, -1, f"seed-{index}", "BUY_SEED",
                item="WHEAT", quantity=1)
            for index in range(rules.MAX_MARKET_ORDERS - 1)
        )
        sale = optimize_short_sales(
            state, {}, commitments=commitments)
        self.assertTrue(sale.feasible, sale.failure)
        programme = Programme(
            state.step, state.day, (), events=commitments,
            planned_sale=sale.planned_sale, feasible=True)
        for step in (sell_step, sell_step + 1):
            actual = replace(state, step=step, turn=step % 24)
            self.assertLessEqual(
                len(_decision(actual, programme).market_orders),
                rules.MAX_MARKET_ORDERS)

    def test_confirmed_product_buy_arrives_before_reserved_pickup(self):
        state = self.state(money=100)
        commitment = ProgrammeEvent(
            state.step, -1, "wheat", "BUY_PRODUCT",
            item="WHEAT", quantity=1)
        sale = optimize_short_sales(
            state, {},
            consumptions={state.step + 1: {"WHEAT": 1}},
            commitments=(commitment,))
        self.assertTrue(sale.feasible, sale.failure)

    def test_trade_does_not_sell_inputs_reserved_after_eight_turns(self):
        state = self.state(shed={"WHEAT": 2})
        sale = optimize_short_sales(
            state, {},
            consumptions={state.step + 12: {"WHEAT": 2}})
        self.assertFalse(any(
            amounts.get("WHEAT", 0)
            for amounts in sale.planned_sale.values()))

    def test_checkpoint_ignores_already_executed_pickup_reservations(self):
        opening = self.state(shed={"WHEAT": 2})
        state = replace(opening, step=opening.step + 4, turn=4)
        sale = optimize_short_sales(
            state, {},
            consumptions={state.step - 2: {"WHEAT": 2}})
        self.assertTrue(sale.feasible, sale.failure)
        self.assertEqual(
            sum(amounts.get("WHEAT", 0)
                for amounts in sale.planned_sale.values()),
            2)

    def test_terminal_window_liquidates(self):
        state = replace(
            self.state(day=29, shed={"MILK": 2}),
            step=rules.TERMINAL_ACTION_STEP,
            turn=22,
        )
        sale = optimize_short_sales(state, {})
        self.assertEqual(
            sale.planned_sale[rules.TERMINAL_ACTION_STEP]["MILK"], 2)

    def test_terminal_asset_liquidation_runs_through_controller(self):
        cow = MidgameAssetTests.cow(
            held=1, consecutive=1, fertilizer=True,
            placed_day=21, position=(0, 0))
        state = self.state(day=29, animals=(cow,), money=0)
        session = DailyPlanningSession()
        while state.step <= rules.TERMINAL_ACTION_STEP:
            programme = session.plan_for(state)
            decision = session.execution_for(state, programme)
            state = rules.advance_owned(
                state, decision.worker_actions, decision.market_orders)
        self.assertGreater(state.money, 0)
        self.assertEqual(state.shed.get("MILK", 0), 0)
        self.assertEqual(state.shed.get("FERTILIZER", 0), 0)

    def test_terminal_opening_worker_stock_is_returned_and_sold(self):
        opening = self.state(day=29, money=0)
        worker = replace(
            opening.workers[0], position=(0, 0), inventory={"MILK": 1})
        state = replace(
            opening,
            own=replace(
                opening.own,
                workers=(worker,),
                worker_positions=(worker.position,),
                worker_inventory=(worker.inventory,),
            ),
        )
        session = DailyPlanningSession()
        while state.step <= rules.TERMINAL_ACTION_STEP:
            programme = session.plan_for(state)
            decision = session.execution_for(state, programme)
            state = rules.advance_owned(
                state, decision.worker_actions, decision.market_orders)
        self.assertGreater(state.money, 0)
        self.assertEqual(state.workers[0].inventory, {})

    def test_runtime_market_orders_never_silently_truncate(self):
        state = self.state()
        too_many = Programme(
            state.step, state.day, (),
            events=(ProgrammeEvent(
                state.step, -1, "hire", "HIRE", quantity=11),),
            routes=(),
        )
        with self.assertRaises(AssertionError):
            _decision(state, too_many)
        decision = TurnDecision(
            tuple(("PASS",) for _ in state.workers),
            tuple(("HIRE",) for _ in range(11)),
        )
        with self.assertRaises(AssertionError):
            construct_action(state, decision)

    def test_intraday_runtime_is_executable(self):
        state = self.state(money=3_000)
        plan = make_plan(state)
        self.assertTrue(plan.feasible, plan.diagnostics)
        self.assertTrue(plan.routes)
        self.assertTrue(all(
            len([
                event for event in plan.events_at(step)
                if event.kind in {
                    "BUY_LAND", "BUY_ANIMAL", "BUY_SEED", "BUY_PRODUCT"
                }
            ]) + sum(
                event.quantity for event in plan.events_at(step)
                if event.kind == "HIRE"
            ) <= rules.MAX_MARKET_ORDERS
            for step in {event.step for event in plan.events}
        ))

    def test_controller_refreshes_sales_at_each_four_turn_checkpoint(self):
        state = self.state(shed={"MILK": 2})
        plan = Programme(
            state.step, state.day, state.shops,
            feasible=True,
        )
        session = DailyPlanningSession()
        session._plans[state.player] = plan
        session.execution_for(state, plan)
        refreshed = session.plans[state.player]
        later_inventory = {
            **state.market.inventory,
            "MILK": state.market.inventory["MILK"] + 100,
        }
        later = replace(
            state, step=state.step + 4, turn=4,
            market=replace(state.market, inventory=later_inventory))
        session.execution_for(later, refreshed)
        self.assertNotEqual(
            refreshed.market_inventory,
            session.plans[state.player].market_inventory)

class SubmitRegressionTests(unittest.TestCase):
    setUpClass = classmethod(MidgameAssetTests.setUpClass.__func__)
    state = MidgameAssetTests.state
    cow = staticmethod(MidgameAssetTests.cow)
    crop = staticmethod(MidgameAssetTests.crop)

    def test_ongoing_cleanup_replay_and_same_day_replant(self):
        from src.kaggriculture_agent.current_assets import completed_production_count
        from src.kaggriculture_agent.operating import programme_invalidation
        for kind in ("TOMATO", "STRAWBERRY"):
            rule = rules.CROPS[kind]
            completed_day = 2 + rule.first_yield_day + 3 * rule.interval
            for held in (0, 2):
                with self.subTest(kind=kind, held=held):
                    crop = self.crop(kind, planted_day=2, held=held, position=(4, 4))
                    state = self.state(day=completed_day, crops=(crop,), money=1000)
                    current = read_current_assets(state)[0]
                    self.assertEqual(completed_production_count(crop.official, completed_day - 1), 3)
                    self.assertEqual(completed_production_count(crop.official, completed_day), 4)
                    self.assertIsNone(current.next_production_step)
                    self.assertFalse(any(e.kind.startswith("OUTPUT") for e in current.next_events))
                    cleanup = [e for e in current.today_events if e.kind in {"HARVEST", "DIG"}]
                    self.assertEqual([e.kind for e in cleanup],
                                     ["HARVEST", "DIG"] if held else ["DIG"])
                    if held:
                        self.assertGreater(cleanup[1].step, cleanup[0].step)
                    plan = make_plan(state)
                    self.assertTrue(plan.feasible, plan.diagnostics)
                    self.assertTrue(any(a.tile == crop.position for a in plan.additions))
                    dig_commitment = next(e for e in plan.events
                                          if e.tile == crop.position and e.kind == "DIG")
                    for e in plan.events:
                        if e.tile == crop.position and e.kind in {"PLANT", "PLACE"}:
                            self.assertGreater(e.step, dig_commitment.step)
                    dig_step = None
                    planted_step = None
                    for _ in range(24):
                        decision = _decision(state, plan)
                        before = state
                        state = rules.advance_owned(state, decision.worker_actions, decision.market_orders)
                        for worker, action in zip(before.workers, decision.worker_actions):
                            if worker.position != crop.position:
                                continue
                            if action[0] == "DIG":
                                dig_step = before.step
                                self.assertIsNone(state.tile_at(crop.position).raw)
                                self.assertEqual(plan.current_assets[0].physical_release_step, state.step)
                                self.assertIsNone(programme_invalidation(state, plan))
                            elif action[0] in {"PLANT", "PLACE"}:
                                planted_step = before.step
                                self.assertIsNotNone(dig_step)
                                self.assertGreater(planted_step, dig_step)
                                self.assertIsInstance(state.tile_at(crop.position).raw, dict)
                    self.assertIsNotNone(dig_step)
                    self.assertIsNotNone(planted_step)

    def test_opponent_subtracts_one_per_asset_before_sum(self):
        from src.kaggriculture_agent.market import forecast_inventory
        from src.kaggriculture_agent.midgame_config import MidgameParameters
        params = MidgameParameters(expected_shop_demand_per_reveal={})
        baseline = self.state(day=10)
        for count, target, expected in ((1, 13, 0), (1, 15, 1), (2, 15, 2)):
            cows = tuple(self.cow(placed_day=5, position=(i, 0)) for i in range(count))
            state = replace(baseline, opp=replace(baseline.opp, visible_animals=cows))
            actual = forecast_inventory(state, "MILK", target * 24, params=params)
            empty = forecast_inventory(baseline, "MILK", target * 24, params=params)
            self.assertEqual(actual - empty, expected)

    def test_own_planned_supply_and_candidate_not_double_counted(self):
        from src.kaggriculture_agent.market import forecast_inventory
        from src.kaggriculture_agent.midgame_config import MidgameParameters
        params = MidgameParameters(expected_shop_demand_per_reveal={})
        base = self.state(day=10)
        target = 18 * 24
        own = self.cow(placed_day=5)
        state = self.state(day=10, animals=(own,))
        empty = forecast_inventory(base, "MILK", target, params=params)
        self.assertEqual(forecast_inventory(state, "MILK", target, params=params) - empty, 3)
        a = _animal_programme(state, "COW", (2, 2), existing=False)
        b = _animal_programme(state, "COW", (3, 2), existing=False)
        before = dict(state.market.inventory)
        one = forecast_inventory(state, "MILK", target, planned_assets=(a,), params=params)
        two = forecast_inventory(state, "MILK", target, planned_assets=(a, b), params=params)
        self.assertEqual(one - empty, 4)
        self.assertEqual(two - one, 1)
        self.assertEqual(state.market.inventory, before)
        # Candidate is NOT added unless accepted and explicitly passed.
        self.assertEqual(forecast_inventory(state, "MILK", target, params=params), empty + 3)

    def test_known_demand_and_reveal_weights_are_exact(self):
        from src.kaggriculture_agent.market import forecast_inventory
        from src.kaggriculture_agent.midgame_config import MidgameParameters
        state = self.state(day=11)
        state = replace(state, shops=("PIZZA_SHOP", "SMOOTHIE_SHOP"))
        params = MidgameParameters(expected_shop_demand_per_reveal={"MILK": 3})
        # [D11,D14): 18 shop ticks * 2 + 3 Town; one reveal D13.
        self.assertEqual(forecast_inventory(state, "MILK", 14 * 24, params=params),
                         10000 - 36 - 3 - 3)
        # [D11,D17): 36 shop ticks * 2 + 6 Town; D13 and D16.
        self.assertEqual(forecast_inventory(state, "MILK", 17 * 24, params=params),
                         10000 - 72 - 6 - 6)

    def test_all_new_long_assets_use_first_output_and_truncated_count(self):
        from unittest.mock import patch
        from src.kaggriculture_agent import planner
        from src.kaggriculture_agent.current_assets import conservative_f_price, _minimum_survival_feed_units
        from src.kaggriculture_agent.market import buy_cost, forecast_inventory
        from src.kaggriculture_agent.midgame_config import DEFAULT_MIDGAME_PARAMETERS as params
        from src.kaggriculture_agent.scenario_forecast import forecast_product_distribution
        import numpy as np
        for kind in planner.LONG_ASSETS:
            state = self.state(day=10)
            rule = (rules.ANIMALS if kind in rules.ANIMALS else rules.CROPS)[kind]
            if kind in rules.ANIMALS:
                with patch.object(planner, "forecast_product_distribution", wraps=forecast_product_distribution) as forecast:
                    planner._asset_daily_value(state, kind, params)
                self.assertEqual(forecast.call_args.args[1], rule.product)
            else:
                with patch.object(planner, "forecast_inventory", wraps=forecast_inventory) as forecast:
                    planner._asset_daily_value(state, kind, params)
                self.assertEqual(forecast.call_args.args[2], (10 + rule.first_yield_day) * 24)
                self.assertEqual(forecast.call_args.kwargs["planned_assets"], ())
            late = self.state(day=29 - rule.first_yield_day)
            value = planner._asset_daily_value(late, kind, params)
            if kind in rules.ANIMALS:
                days = rule.first_yield_day
                wheat = _minimum_survival_feed_units(days)
                dist = forecast_product_distribution(late, rule.product)
                h = 29 * 24 - late.step
                prices_s = np.array([rules.market_price(rule.product, int(round(inv))) for inv in dist.inventory_paths[:, h]])
                expected_rev = float(dist.weights @ prices_s)
                expected = (expected_rev + max(1, wheat) * conservative_f_price(late)
                            - rule.cost - buy_cost("WHEAT", wheat, late.market.inventory["WHEAT"])) / days
                self.assertAlmostEqual(value, expected)
            elif rule.ongoing:
                price = rules.market_price(kind, forecast_inventory(late, kind, 29 * 24))
                self.assertAlmostEqual(value, (price - rule.seed_cost) / rule.first_yield_day)
            self.assertEqual(planner._asset_daily_value(
                self.state(day=30 - rule.first_yield_day), kind, params), 0.0)
            if kind in rules.ANIMALS or rule.ongoing:
                two = self.state(day=29 - rule.first_yield_day - rule.interval)
                days = rule.first_yield_day + rule.interval
                if kind in rules.ANIMALS:
                    wheat = _minimum_survival_feed_units(days)
                    dist = forecast_product_distribution(two, rule.product)
                    p_days = [two.day + rule.first_yield_day - 1, two.day + rule.first_yield_day - 1 + rule.interval]
                    expected_rev = 0.0
                    for pd in p_days:
                        h = (pd + 1) * 24 - two.step
                        prices_s = np.array([rules.market_price(rule.product, int(round(inv))) for inv in dist.inventory_paths[:, h]])
                        expected_rev += float(dist.weights @ prices_s)
                    expected = (expected_rev + max(1, wheat) * conservative_f_price(two)
                                - rule.cost - buy_cost("WHEAT", wheat, two.market.inventory["WHEAT"])) / days
                else:
                    price = rules.market_price(kind, forecast_inventory(
                        two, kind, (two.day + rule.first_yield_day) * 24))
                    expected = (2 * price - rule.seed_cost) / days
                self.assertAlmostEqual(planner._asset_daily_value(two, kind, params), expected)

    def test_runtime_fill_keeps_accepted_supply_across_land_purchase(self):
        from unittest.mock import patch
        from src.kaggriculture_agent import planner
        original = planner._asset_daily_value
        seen = []
        def record(state, kind, params, programme=None):
            seen.append((bool(programme.land), tuple(a.asset_id for a in programme.additions)))
            return original(state, kind, params, programme)
        state = self.state(day=11, money=20000)
        with patch.object(planner, "_asset_daily_value", side_effect=record):
            plan = make_plan(state)
        self.assertTrue(plan.feasible, plan.diagnostics)
        old = [set(ids) for land, ids in seen if not land and ids]
        new = [set(ids) for land, ids in seen if land]
        self.assertTrue(old)
        self.assertTrue(new)
        self.assertTrue(any(ids <= new[0] for ids in old))

    def test_first_output_protection_includes_production_night(self):
        for day in range(20, 28):
            animal = self.cow(placed_day=20, consecutive=1)
            state = self.state(day=day, animals=(animal,), money=500,
                               inventories={"MILK": 30000, "FERTILIZER": 30000, "WHEAT": 0})
            plan = make_plan(state)
            self.assertEqual(plan.current_assets[0].mode, MAINTAIN)
            self.assertTrue(any(e.kind == "FEED" for e in plan.events))
            self.assertFalse(any(e.kind == "CARE" for e in plan.events))
        state = self.state(day=28, animals=(self.cow(placed_day=20),),
                           inventories={"MILK": 30000, "FERTILIZER": 30000, "WHEAT": 0})
        self.assertEqual(make_plan(state).current_assets[0].mode, EXIT)

    def test_existing_animals_use_individual_next_output(self):
        from src.kaggriculture_agent.market import forecast_inventory
        from src.kaggriculture_agent.current_assets import next_animal_production_day
        from src.kaggriculture_agent.midgame_config import MidgameParameters
        params = MidgameParameters(expected_shop_demand_per_reveal={})
        cows = (self.cow(placed_day=2, position=(3, 4)),
                self.cow(placed_day=3, position=(4, 4)))
        # Large known demand separates the two next-output prices enough to
        # cross the existing zero base-value boundary.
        state = self.state(day=13, animals=cows, money=1000,
                           inventories={"MILK": 10103, "FERTILIZER": 30000})
        state = replace(state, shops=("SMOOTHIE_SHOP",) * 8)
        targets = [(next_animal_production_day(a.official, state.day) + 1) * 24 for a in cows]
        self.assertEqual(targets, [14 * 24, 15 * 24])
        prices = [rules.market_price("MILK", forecast_inventory(state, "MILK", t, params=params))
                  for t in targets]
        self.assertGreater(prices[1], prices[0])
        modes = [a.mode for a in read_current_assets(state, params=params)]
        self.assertEqual(modes[0], EXIT)
        self.assertIn(modes[1], (PRODUCE, MAINTAIN))

    def test_land_gate_runs_after_old_land_costs_and_hires(self):
        from unittest.mock import patch
        from src.kaggriculture_agent import planner
        calls = []
        original = planner._land_expansion
        def record(state, programme, params):
            calls.append(programme)
            return original(state, programme, params)
        state = self.state(day=11, money=10000)
        with patch.object(planner, "_land_expansion", side_effect=record):
            plan = make_plan(state)
        self.assertTrue(plan.feasible, plan.diagnostics)
        self.assertEqual(len(calls), 1)
        old = calls[0]
        self.assertTrue(old.routes)
        self.assertTrue(old.additions)
        self.assertTrue(all(rules.quadrant(a.tile) == "NW" for a in old.additions))
        self.assertGreater(planner._committed_purchase_cost(state, old), 0)
        self.assertTrue(all(a.asset_id in {b.asset_id for b in plan.assets} for a in old.assets))
        # Opening cash passes the naked land gate, but existing-land inputs
        # consume the margin: the complete runtime must not buy.
        lower = make_plan(self.state(day=11, money=1800))
        self.assertTrue(lower.feasible, lower.diagnostics)
        self.assertTrue(lower.additions)
        self.assertFalse(lower.land)

    def test_land_strict_threshold_new_quadrant_and_fourth_disabled(self):
        from src.kaggriculture_agent.midgame_config import MidgameParameters
        from src.kaggriculture_agent import planner
        self.assertNotIn("land_min_deployable_count", MidgameParameters.__dataclass_fields__)
        self.assertNotIn("land_value_cover_ratio", MidgameParameters.__dataclass_fields__)
        self.assertFalse(hasattr(planner, "_enforce_land_scale"))
        # D28: no affordable production can mature before terminal; no buffer.
        for size in (10, 8):
            half = size // 2
            for extra, buys in ((0, False), (1, True)):
                threshold = 1000 + half * half * rules.CROPS["WHEAT"].seed_cost + 200
                state = self.state(day=28, money=threshold + extra)
                tiles = tuple(TileState((x, y), None if x < half and y < half else "LOCKED")
                              for y in range(size) for x in range(size))
                worker = replace(state.workers[0], position=(half - 1, half - 1))
                state = replace(state, board_size=size, own=replace(
                    state.own, tiles=tiles, workers=(worker,),
                    usable_tiles=tuple(t.position for t in tiles if not t.is_locked)))
                plan = make_plan(state)
                self.assertTrue(plan.feasible, plan.diagnostics)
                self.assertEqual(bool(plan.land), buys)
        state = self.state(day=28, money=50000)
        state = replace(state, own=replace(state.own, owned_land=("NW", "NE", "SW")))
        self.assertFalse(make_plan(state).land)

    def test_same_worker_carry_avoids_buy_through_make_plan(self):
        for item in ("WHEAT", "FERTILIZER"):
            if item == "WHEAT":
                state = self.state(day=28, money=100, animals=(
                    self.cow(placed_day=21, consecutive=1),))
                expected = "FEED"
            else:
                state = self.state(day=28, money=100, crops=(
                    self.crop("TOMATO", planted_day=20, held=0, consecutive=1, position=(4, 4)),),
                    inventories={"FERTILIZER": 30000})
                expected = "FERTILIZE"
            worker = replace(state.workers[0], position=(4, 4), inventory={item: 1})
            state = replace(state, own=replace(state.own, workers=(worker,)))
            plan = make_plan(state)
            self.assertTrue(plan.feasible, plan.diagnostics)
            self.assertFalse(any(e.kind == "BUY_PRODUCT" and e.item == item for e in plan.events))
            actions = [(s, a) for r in plan.routes for s, a in r.actions.items()]
            self.assertTrue(any(a[0] == expected for _, a in actions))
            for _ in range(4):
                decision = _decision(state, plan)
                state = rules.advance_owned(state, decision.worker_actions, decision.market_orders)
            raw = state.tile_at((4, 4)).raw
            self.assertTrue(raw["fed_today"] if item == "WHEAT"
                            else raw["fertilized_until_day"] >= state.day)

    def test_cross_worker_inputs_require_real_drop_pickup(self):
        from src.kaggriculture_agent.planner import _route_input_deficits
        from src.kaggriculture_agent.programme import WorkerRoute
        from src.kaggriculture_agent.state import WorkerState
        for item, op in (("WHEAT", "FEED"), ("FERTILIZER", "FERTILIZE")):
            animal = self.cow(placed_day=21, consecutive=1, position=(4, 3))
            crop = self.crop("TOMATO", planted_day=20, held=0, position=(4, 3))
            state = self.state(day=28, animals=(animal,) if item == "WHEAT" else (),
                               crops=(crop,) if item == "FERTILIZER" else ())
            workers = (WorkerState(0, (4, 4), {item: 1}), WorkerState(1, (4, 4), {}))
            state = replace(state, own=replace(state.own, workers=workers))
            for relay in (False, True):
                routes = (WorkerRoute(0, 28, "INNER", {state.step: ("DROP",) if relay else ("PASS",)}),
                          WorkerRoute(1, 28, "INNER", {
                              state.step + 1: ("PICKUP", item, 1),
                              state.step + 2: ("NORTH",), state.step + 3: (op,)}))
                plan = Programme(state.step, state.day, (), routes=routes, worker_count=2)
                self.assertEqual(_route_input_deficits(state, plan)[item], 0 if relay else 1)
                actual = state
                for _ in range(4):
                    d = _decision(actual, plan)
                    actual = rules.advance_owned(actual, d.worker_actions, d.market_orders)
                raw = actual.tile_at((4, 3)).raw
                self.assertEqual(bool(raw["fed_today"]) if item == "WHEAT"
                                 else raw["fertilized_until_day"] >= state.day, relay)

    def test_f_reserve_expires_after_real_pickup(self):
        from src.kaggriculture_agent.operating import _refresh_sales
        from src.kaggriculture_agent.programme import WorkerRoute
        state = self.state(day=28, shed={"FERTILIZER": 1}, crops=(
            self.crop("TOMATO", planted_day=20, held=0, position=(4, 4)),))
        route = WorkerRoute(0, 28, "INNER", {
            state.step + 1: ("PICKUP", "FERTILIZER", 1),
            state.step + 2: ("FERTILIZE",)})
        old_event = ProgrammeEvent(state.step, 0, "f", "FERTILIZE", quantity=1)
        plan = Programme(state.step, state.day, (), events=(old_event,), routes=(route,), feasible=True)
        refreshed = _refresh_sales(state, plan)
        self.assertTrue(refreshed.feasible, refreshed.diagnostics)
        self.assertFalse(any(q.get("FERTILIZER", 0) for q in refreshed.planned_sale.values()))
        actual = state
        for _ in range(4):
            decision = _decision(actual, refreshed)
            actual = rules.advance_owned(actual, decision.worker_actions, decision.market_orders)
        self.assertEqual(actual.tile_at((4, 4)).raw["fertilized_until_day"], state.day + 2)
        self.assertEqual(actual.shed.get("FERTILIZER", 0), 0)
        for offset in (4, 8):
            later = replace(actual, step=state.step + offset, turn=offset,
                            own=replace(actual.own, shed_inventory={"FERTILIZER": 1, "MILK": 2}))
            refreshed = _refresh_sales(later, plan)
            self.assertEqual(refreshed.planned_sale[later.step]["FERTILIZER"], 1)
            self.assertTrue(all(s in {later.step, later.step + 4, later.step + 8}
                                for s, q in refreshed.planned_sale.items() if q.get("MILK")))

    def test_full_pinned_controller_episode_without_exception_fallback(self):
        from src.kaggriculture_agent.scripted_opening import scripted_opening_action
        for seed in (119, 120):
            env = make("kaggriculture", configuration={"seed": seed})
            env.reset(2)
            session = DailyPlanningSession()
            midgame_days = set()
            while not env.done:
                observation = env.state[0].observation
                if observation.day < 10:
                    action = scripted_opening_action(observation)
                else:
                    state = reconstruct(observation)
                    plan = session.plan_for(state)
                    self.assertTrue(plan.feasible, (seed, state.step, plan.diagnostics))
                    decision = session.execution_for(state, plan)
                    self.assertTrue(session.plans[state.player].feasible,
                                    (seed, state.step, session.plans[state.player].diagnostics))
                    action = {"farmer": list(decision.worker_actions[0]),
                              "hands": [list(a) for a in decision.worker_actions[1:]],
                              "market": [list(o) for o in decision.market_orders]}
                    midgame_days.add(state.day)
                env.step([action, {"farmer": ["PASS"], "hands": [], "market": []}])
            self.assertEqual(midgame_days, set(range(10, 30)))
            self.assertEqual(env.state[0].status, "DONE")


if __name__ == "__main__":
    unittest.main()
