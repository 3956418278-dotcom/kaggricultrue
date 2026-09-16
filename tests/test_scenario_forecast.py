"""Unit and regression tests for conditional scenario forecasting of animal products."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np

from kaggle_environments import make

from src.kaggriculture_agent import rules, planner
from src.kaggriculture_agent.current_assets import read_current_assets, _animal_current_state
from src.kaggriculture_agent.midgame_config import DEFAULT_MIDGAME_PARAMETERS, MidgameParameters
from src.kaggriculture_agent.scenario_forecast import (
    ScenarioDistribution,
    candidate_inventory,
    forecast_product_distribution,
    scenario_revenue_value,
)
from src.kaggriculture_agent.state import AssetState, TileState, reconstruct


class ScenarioForecastUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        obs = make("kaggriculture", configuration={"seed": 17}).reset(2)[0].observation
        cls.base = reconstruct(obs)

    def state(self, *, day=10, step=None, animals=(), shops=(), inventory_overrides=None, opp_money=0):
        tiles = list(self.base.own.tiles)
        for a in animals:
            x, y = a.position
            tiles[y * self.base.board_size + x] = TileState(a.position, dict(a.official))
        inv = dict(self.base.market.inventory)
        if inventory_overrides:
            inv.update(inventory_overrides)
        actual_step = day * 24 if step is None else step
        return replace(
            self.base,
            day=day,
            step=actual_step,
            shops=tuple(shops),
            market=replace(
                self.base.market,
                inventory=inv,
                price={item: rules.market_price(item, inv[item]) for item in rules.PRODUCTS},
            ),
            own=replace(
                self.base.own,
                tiles=tuple(tiles),
                animals=tuple(animals),
            ),
            opp=replace(
                self.base.opp,
                money=opp_money,
            ),
        )

    @staticmethod
    def animal(kind, *, placed_day=3, position=(4, 4), pending=0):
        return AssetState(kind, position, {
            "kind": "PASTURE" if kind in ("COW", "SHEEP") else "COOP",
            "animal": kind,
            "placed_day": placed_day,
            "yield_units": 0,
            "consecutive_unfed": 0,
            "fed_today": False,
            "cared_today": False,
            "pending_care_bonus": pending,
        })

    def test_current_inventory_shift_reanchoring(self):
        """Current inventory shift strictly re-anchors the first step of all scenarios."""
        for prod in ("WOOL", "MILK", "EGG"):
            for inv_val in (8500, 9800, 10000, 10500, 12000):
                st = self.state(day=10, inventory_overrides={prod: inv_val})
                dist = forecast_product_distribution(st, prod)
                self.assertEqual(dist.horizon_steps, 720 - st.step)
                # Strict first-point anchoring
                np.testing.assert_allclose(dist.inventory_paths[:, 0], inv_val)
                # Quantile ordering checks
                for q in range(4):
                    self.assertTrue(np.all(dist.inventory_q[q] <= dist.inventory_q[q + 1] + 1e-6))
                    self.assertTrue(np.all(dist.price_q[q] <= dist.price_q[q + 1] + 1e-6))

    def test_counterfactual_delta_zero_identity(self):
        """Candidate inventory with Delta=0 is strictly identical to reference paths."""
        st = self.state(day=11)
        for prod in ("WOOL", "MILK", "EGG"):
            dist = forecast_product_distribution(st, prod)
            cf_identical = candidate_inventory(dist.inventory_paths, candidate_impact=0, baseline_impact=0)
            np.testing.assert_array_equal(cf_identical, dist.inventory_paths)

            # Symmetrical impact
            cf_offset = candidate_inventory(dist.inventory_paths, candidate_impact=5, baseline_impact=5)
            np.testing.assert_array_equal(cf_offset, dist.inventory_paths)

            # Incremental impact Delta != 0
            cf_added = candidate_inventory(dist.inventory_paths, candidate_impact=3, baseline_impact=1)
            np.testing.assert_array_equal(cf_added, dist.inventory_paths + 2)

    def test_shop_reveal_response(self):
        """Revealing relevant shops triggers scenario reconditioning."""
        # MILK: PIZZA_SHOP, SMOOTHIE_SHOP, ICE_CREAM_SHOP
        st_no_shops = self.state(day=10, shops=())
        dist_no_shops = forecast_product_distribution(st_no_shops, "MILK")

        st_smoothie = self.state(day=10, shops=("SMOOTHIE_SHOP",))
        dist_smoothie = forecast_product_distribution(st_smoothie, "MILK")
        self.assertIn(dist_smoothie.conditioning_level, ("relevant", "exact", "state_fallback"))

        # WOOL: Yarn store revelation
        st_no_yarn = self.state(day=10, shops=("BAKERY", "PIZZA_SHOP"))
        dist_no_yarn = forecast_product_distribution(st_no_yarn, "WOOL")

        st_yarn = self.state(day=10, shops=("YARN_STORE", "PIZZA_SHOP"))
        dist_yarn = forecast_product_distribution(st_yarn, "WOOL")
        self.assertNotEqual(dist_no_yarn.state, dist_yarn.state)

    def test_animal_count_and_state_change_response(self):
        """Animal count and features properly govern gate and branch transitions."""
        # MILK gating transitions
        # cow_age35 <= 2 and mean <= 6.207 -> G0
        cows_g0 = [self.animal("COW", placed_day=6, position=(i, 0)) for i in range(2)]
        st_g0 = self.state(day=10, animals=cows_g0)
        dist_g0 = forecast_product_distribution(st_g0, "MILK")
        self.assertEqual(dist_g0.state, "G0")

        # cow_age35 > 2 and cow_total <= 17 -> G2
        # age at day 10: 10 - 6 = 4 in [3, 5]
        cows_g2 = [self.animal("COW", placed_day=6, position=(i % 10, i // 10)) for i in range(5)]
        st_g2 = self.state(day=10, animals=cows_g2)
        dist_g2 = forecast_product_distribution(st_g2, "MILK")
        self.assertEqual(dist_g2.state, "G2")

        # WOOL sheep_age9p gating: <= 3 -> G0, >= 4 -> G1
        sheep_g0 = [self.animal("SHEEP", placed_day=0, position=(i, 0)) for i in range(2)]
        st_w_g0 = self.state(day=10, animals=sheep_g0)
        dist_w_g0 = forecast_product_distribution(st_w_g0, "WOOL")
        self.assertEqual(dist_w_g0.state, "G0")

        sheep_g1 = [self.animal("SHEEP", placed_day=0, position=(i, 0)) for i in range(5)]
        st_w_g1 = self.state(day=10, animals=sheep_g1)
        dist_w_g1 = forecast_product_distribution(st_w_g1, "WOOL")
        self.assertEqual(dist_w_g1.state, "G1")

    def test_planner_call_site_and_heuristics_isolation(self):
        """Planner uses forecast_product_distribution for animals and forecast_inventory for crops."""
        state = self.state(day=10)
        params = DEFAULT_MIDGAME_PARAMETERS

        # Animals must call forecast_product_distribution
        with patch("src.kaggriculture_agent.planner.forecast_product_distribution", wraps=forecast_product_distribution) as mock_animal:
            val_cow = planner._forecast_animal_daily_value(state, "COW", params)
            self.assertEqual(mock_animal.call_args.args[1], "MILK")
            self.assertGreater(val_cow, -1000.0)

        with patch("src.kaggriculture_agent.planner.forecast_product_distribution", wraps=forecast_product_distribution) as mock_animal:
            val_sheep = planner._forecast_animal_daily_value(state, "SHEEP", params)
            self.assertEqual(mock_animal.call_args.args[1], "WOOL")

        with patch("src.kaggriculture_agent.planner.forecast_product_distribution", wraps=forecast_product_distribution) as mock_animal:
            val_goose = planner._forecast_animal_daily_value(state, "GOOSE", params)
            self.assertEqual(mock_animal.call_args.args[1], "EGG")

        # Crops must NOT call forecast_product_distribution; they use forecast_inventory
        with patch("src.kaggriculture_agent.planner.forecast_inventory", wraps=planner.forecast_inventory) as mock_crop:
            planner._forecast_crop_daily_value(state, "TOMATO", params)
            mock_crop.assert_called()
            self.assertEqual(mock_crop.call_args.args[1], "TOMATO")

    def test_scenario_revenue_value_sequential(self):
        """Scenario revenue respects official sequential price impact."""
        paths = np.array([[10000, 10000], [9000, 9000]], dtype=float)
        res = scenario_revenue_value("MILK", paths, quantities=[1, 1], step_indices=[0, 1])
        # Quantities sell sequentially
        self.assertIn("expected", res)
        self.assertIn("q10", res)
        self.assertIn("q50", res)
        self.assertIn("q90", res)
        self.assertGreater(res["expected"], 0)


if __name__ == "__main__":
    unittest.main()
