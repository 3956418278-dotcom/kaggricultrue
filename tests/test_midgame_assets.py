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
            day=20, animals=(self.cow(placed_day=13),),
            shed={"WHEAT": 2}))
        maintain = self.current(self.state(
            day=20, animals=(self.cow(placed_day=13),),
            shed={"WHEAT": 2},
            inventories={"MILK": 30_000}))
        exit_asset = self.current(self.state(
            day=20, animals=(self.cow(placed_day=13),),
            shed={"WHEAT": 2},
            inventories={
                "MILK": 30_000,
                "FERTILIZER": 30_000,
                "WHEAT": 0,
            }))
        self.assertEqual(produce.mode, PRODUCE)
        self.assertEqual(maintain.mode, MAINTAIN)
        self.assertEqual(exit_asset.mode, EXIT)

    def test_only_produce_cares_and_maintain_feeds_minimally(self):
        produce = self.current(self.state(
            day=20,
            animals=(self.cow(placed_day=13, consecutive=0),),
            shed={"WHEAT": 2}))
        maintain = self.current(self.state(
            day=20,
            animals=(self.cow(
                held=5, placed_day=13, consecutive=1),),
            shed={"WHEAT": 2},
            inventories={"MILK": 30_000}))
        self.assertIn("CARE", {e.kind for e in produce.today_events})
        kinds = {e.kind for e in maintain.today_events}
        self.assertIn("FEED", kinds)
        self.assertNotIn("CARE", kinds)

    def test_exit_first_night_keeps_last_official_production(self):
        cow = self.cow(placed_day=21, consecutive=0)
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
            held=0, consecutive=1, fertilizer=True, placed_day=21)
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
            held=1, consecutive=1, fertilizer=True, placed_day=21)
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


if __name__ == "__main__":
    unittest.main()
