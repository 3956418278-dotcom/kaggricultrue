"""Unit tests for market inventory prediction failure protection patch."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from kaggle_environments import make

from src.kaggriculture_agent import rules
from src.kaggriculture_agent.market import optimize_short_sales
from src.kaggriculture_agent.operating import (
    DailyPlanningSession,
    MARKET_FAILURE_SELL_WINDOW,
    _refresh_sales,
)
from src.kaggriculture_agent.programme import Programme
from src.kaggriculture_agent.state import reconstruct


class MarketFailurePatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        observation = make(
            "kaggriculture", configuration={"seed": 17}
        ).reset(2)[0].observation
        cls.base = reconstruct(observation)

    def state(self, *, step=241, shed=None, market_inv=None, money=3_000, player=0):
        market_inventory = dict(self.base.market.inventory)
        if market_inv:
            market_inventory.update(market_inv)
        market_price = {
            item: rules.market_price(item, market_inventory[item])
            for item in rules.PRODUCTS
        }
        return replace(
            self.base,
            step=step,
            day=step // 24,
            turn=step % 24,
            player=player,
            market=replace(
                self.base.market,
                inventory=market_inventory,
                price=market_price,
            ),
            own=replace(
                self.base.own,
                money=money,
                shed_inventory=dict(shed or {}),
            ),
        )

    def test_failure_recording(self):
        """At step t, deviation of non-excluded product records failure step."""
        t = 241  # step % 4 != 0
        state = self.state(step=t, market_inv={"MILK": 10, "WOOL": 10, "WHEAT": 10})
        # programme predicts MILK: 5 (mismatch), WOOL: 10 (match), WHEAT: 5 (mismatch but excluded)
        programme = Programme(
            t, t // 24, (),
            market_inventory={t: {"MILK": 5, "WOOL": 10, "WHEAT": 5}}
        )
        session = DailyPlanningSession()
        session.execution_for(state, programme)

        self.assertIn(state.player, session._market_failure_steps)
        self.assertEqual(session._market_failure_steps[state.player].get("MILK"), t)
        self.assertNotIn("WOOL", session._market_failure_steps[state.player])
        self.assertNotIn("WHEAT", session._market_failure_steps[state.player])

    def test_immediate_sale_on_arrival(self):
        """Failure at t causes immediate sale of arrival at t+1."""
        t = 241
        session = DailyPlanningSession()
        # Step t: trigger MILK failure
        state_t = self.state(step=t, market_inv={"MILK": 10})
        prog_t = Programme(t, t // 24, (), market_inventory={t: {"MILK": 5}})
        session.execution_for(state_t, prog_t)
        self.assertEqual(session._market_failure_steps[state_t.player]["MILK"], t)

        # Step t+1: MILK arrival = 3
        t1 = t + 1
        state_t1 = self.state(step=t1, shed={"MILK": 3}, market_inv={"MILK": 10})
        prog_t1 = Programme(t, t // 24, (), market_inventory={t: {"MILK": 10}})
        with patch("src.kaggriculture_agent.operating._arrivals", return_value={t1: {"MILK": 3}}):
            decision = session.execution_for(state_t1, prog_t1)

        # Verify sell orders in decision
        milk_sold = sum(qty for action, item, qty in decision.market_orders if action == "SELL" and item == "MILK")
        self.assertGreaterEqual(milk_sold, 3)

    def test_only_new_inventory_sold(self):
        """Existing shed stock 8 + arrival 3 forces minimum 3, not 11."""
        step = 241
        state_shed8 = self.state(step=step, shed={"MILK": 8}, market_inv={"MILK": 10})
        sale = optimize_short_sales(
            state_shed8,
            {step: {"MILK": 3}},
            forced_sales={step: {"MILK": 3}},
        )
        self.assertTrue(sale.feasible)
        self.assertGreaterEqual(sale.planned_sale.get(step, {}).get("MILK", 0), 3)

        session = DailyPlanningSession()
        session._market_failure_steps[state_shed8.player] = {"MILK": step}
        prog_exec = Programme(step, step // 24, (), market_inventory={step: {"MILK": 10}})
        with patch("src.kaggriculture_agent.operating._arrivals", return_value={step: {"MILK": 3}}):
            decision = session.execution_for(state_shed8, prog_exec)

        milk_orders = [qty for action, item, qty in decision.market_orders if action == "SELL" and item == "MILK"]
        self.assertTrue(milk_orders)
        self.assertGreaterEqual(sum(milk_orders), 3)

    def test_window_boundary(self):
        """Failure at t forces t+1..t+4, but t+5 is not forced by this failure."""
        t = 240
        session = DailyPlanningSession()
        session._market_failure_steps[0] = {"MILK": t}

        for offset in range(0, MARKET_FAILURE_SELL_WINDOW + 1):
            s = t + offset
            self.assertTrue(0 <= s - t <= MARKET_FAILURE_SELL_WINDOW)
            st = self.state(step=s, shed={"MILK": 2}, market_inv={"MILK": 10})
            pr = Programme(s, s // 24, (), market_inventory={s: {"MILK": 10}})
            with patch("src.kaggriculture_agent.operating._arrivals", return_value={s: {"MILK": 2}}):
                dec = session.execution_for(st, pr)
            milk_sold = sum(q for a, i, q in dec.market_orders if a == "SELL" and i == "MILK")
            self.assertGreaterEqual(milk_sold, 2, f"Failed to force sell at offset {offset}")

        s_exp = t + 5
        self.assertFalse(0 <= s_exp - t <= MARKET_FAILURE_SELL_WINDOW)
        with patch("src.kaggriculture_agent.operating._refresh_sales", wraps=_refresh_sales) as mock_refresh:
            st = self.state(step=s_exp, shed={"MILK": 2}, market_inv={"MILK": 10})
            pr = Programme(s_exp, s_exp // 24, (), market_inventory={s_exp: {"MILK": 10}})
            with patch("src.kaggriculture_agent.operating._arrivals", return_value={s_exp: {"MILK": 2}}):
                session.execution_for(st, pr)
            for call in mock_refresh.call_args_list:
                self.assertIsNone(call.kwargs.get("forced_sales"))

    def test_refresh_window(self):
        """Failure at t, again at t+3, resets failure step and extends window to t+7."""
        t = 240
        session = DailyPlanningSession()
        session._market_failure_steps[0] = {"MILK": t}

        t_new = t + 3
        st3 = self.state(step=t_new, market_inv={"MILK": 20})
        pr3 = Programme(t_new, t_new // 24, (), market_inventory={t_new: {"MILK": 10}})
        session.execution_for(st3, pr3)
        self.assertEqual(session._market_failure_steps[0]["MILK"], t_new)

        t_test = t + 6
        st6 = self.state(step=t_test, shed={"MILK": 2}, market_inv={"MILK": 20})
        pr6 = Programme(t_test, t_test // 24, (), market_inventory={t_test: {"MILK": 20}})
        with patch("src.kaggriculture_agent.operating._arrivals", return_value={t_test: {"MILK": 2}}):
            dec = session.execution_for(st6, pr6)
        milk_sold = sum(q for a, i, q in dec.market_orders if a == "SELL" and i == "MILK")
        self.assertGreaterEqual(milk_sold, 2)

    def test_per_product_isolation(self):
        """MILK failed, WOOL did not: only MILK gets forced sale."""
        t = 241
        session = DailyPlanningSession()
        session._market_failure_steps[0] = {"MILK": t}

        t1 = t + 1
        st = self.state(step=t1, shed={"MILK": 3, "WOOL": 4}, market_inv={"MILK": 10, "WOOL": 10})
        pr = Programme(t1, t1 // 24, (), market_inventory={t1: {"MILK": 10, "WOOL": 10}})

        with patch("src.kaggriculture_agent.operating._refresh_sales", wraps=_refresh_sales) as mock_refresh:
            with patch("src.kaggriculture_agent.operating._arrivals", return_value={t1: {"MILK": 3, "WOOL": 4}}):
                session.execution_for(st, pr)
            self.assertTrue(mock_refresh.called)
            forced_sales = mock_refresh.call_args.kwargs.get("forced_sales")
            self.assertIsNotNone(forced_sales)
            self.assertIn(t1, forced_sales)
            self.assertEqual(forced_sales[t1].get("MILK"), 3)
            self.assertNotIn("WOOL", forced_sales[t1])

    def test_no_arrival(self):
        """Product in failure window with arrival=0 does not force sale."""
        t = 241
        session = DailyPlanningSession()
        session._market_failure_steps[0] = {"MILK": t}

        t1 = t + 1
        st = self.state(step=t1, shed={"MILK": 5}, market_inv={"MILK": 10})
        pr = Programme(t1, t1 // 24, (), market_inventory={t1: {"MILK": 10}})
        with patch("src.kaggriculture_agent.operating._refresh_sales", wraps=_refresh_sales) as mock_refresh:
            with patch("src.kaggriculture_agent.operating._arrivals", return_value={t1: {}}):
                session.execution_for(st, pr)
            for call in mock_refresh.call_args_list:
                forced_sales = call.kwargs.get("forced_sales")
                if forced_sales:
                    self.assertNotIn("MILK", forced_sales.get(t1, {}))

    def test_episode_reset(self):
        """New episode (state.step < last) and session.reset() clears failure history."""
        session = DailyPlanningSession()
        session._market_failure_steps[0] = {"MILK": 200, "WOOL": 201}
        session._market_failure_steps[1] = {"EGG": 200}
        session._last_steps[0] = 250

        # Case A: session.reset()
        session.reset()
        self.assertEqual(session._market_failure_steps, {})

        # Case B: new episode state.step < last in plan_for
        session._market_failure_steps[0] = {"MILK": 200}
        session._last_steps[0] = 250
        st = self.state(step=0, player=0)
        session.plan_for(st)
        self.assertNotIn(0, session._market_failure_steps)

        # Case C: new episode state.step < last in execution_for
        session._market_failure_steps[0] = {"MILK": 200}
        session._last_steps[0] = 250
        st = self.state(step=0, player=0)
        pr = Programme(0, 0, ())
        session.execution_for(st, pr)
        self.assertNotIn(0, session._market_failure_steps)
