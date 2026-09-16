"""Unit, regression, and golden reference tests for conditional scenario forecasting of animal products."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np

from kaggle_environments import make

from src.kaggriculture_agent import rules, planner
from src.kaggriculture_agent.current_assets import (
    read_current_assets,
    _animal_current_state,
    existing_animal_future_production_days,
)
from src.kaggriculture_agent.market import sell_revenue
from src.kaggriculture_agent.midgame_config import DEFAULT_MIDGAME_PARAMETERS, MidgameParameters
from src.kaggriculture_agent.programme import AssetProgramme, Programme
from src.kaggriculture_agent.scenario_forecast import (
    AnimalMarketContext,
    ScenarioDistribution,
    _MilkEggForecaster,
    animal_batch_sale_events,
    animal_incremental_market_path,
    candidate_inventory,
    forecast_product_distribution,
    scenario_batch_sale_value,
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

    def test_exact_price_path_identity(self):
        """Test Point 4: price paths must strictly equal rules.market_price(product, round(inventory))."""
        for prod in ("WOOL", "MILK", "EGG"):
            st = self.state(day=11, inventory_overrides={prod: 9500})
            dist = forecast_product_distribution(st, prod)
            S, H = dist.inventory_paths.shape
            # Spot check sample of paths
            sample_s = min(S, 20)
            sample_h = min(H, 30)
            for s in range(sample_s):
                for h in range(sample_h):
                    inv_round = int(round(dist.inventory_paths[s, h]))
                    expected_p = rules.market_price(prod, inv_round)
                    self.assertEqual(dist.price_paths[s, h], expected_p)

    def test_counterfactual_planner_candidate_impact(self):
        """Test Point 5 & 10: planner candidate valuation reflects accepted animal impact."""
        st = self.state(day=11)
        params = DEFAULT_MIDGAME_PARAMETERS

        # Baseline: evaluate first cow with empty programme
        val_cow1 = planner._forecast_animal_daily_value(st, "COW", params, programme=None)

        # Now create programme with one already accepted cow
        rule = rules.ANIMALS["COW"]
        accepted_cow = AssetProgramme(
            asset_id="new:COW:0:0:265",
            asset_type="COW",
            tile=(0, 0),
            decision="NEW",
            existing=False,
        )
        prog_with_cow = Programme(
            formed_step=st.step,
            day=st.day,
            shops=st.shops,
            assets=(accepted_cow,),
        )

        # Evaluate second cow: its candidate scenarios see candidate 1's impact
        val_cow2 = planner._forecast_animal_daily_value(st, "COW", params, programme=prog_with_cow)

        # Additional supply depresses future price, so second cow value must be strictly lower
        self.assertLess(val_cow2, val_cow1)

    def test_shop_reveal_response(self):
        """Test Point 10: Shop reveal changes selected scenario set or count, with fallback when support is insufficient."""
        # Relevant shop selection at D12 anchor (step 311)
        st_no_shops = self.state(step=311, shops=())
        dist_no_shops = forecast_product_distribution(st_no_shops, "MILK")

        st_ice = self.state(step=311, shops=("ICE_CREAM_SHOP",))
        dist_ice = forecast_product_distribution(st_ice, "MILK")

        self.assertIn(dist_ice.conditioning_level, ("relevant", "exact"))
        # Relevant filter selects a specific subset of empirical scenarios
        self.assertLess(dist_ice.scenario_count, dist_no_shops.scenario_count)

        # Insufficient support fallback (e.g. at D10 where only 2 episodes have ICE_CREAM_SHOP as shop 0, threshold 10)
        st_fallback = self.state(day=10, shops=("ICE_CREAM_SHOP",))
        dist_fallback = forecast_product_distribution(st_fallback, "MILK")
        self.assertEqual(dist_fallback.conditioning_level, "state_fallback")
        self.assertTrue(dist_fallback.ood["reveal_ood"])

    def test_inventory_residual_rho_propagation(self):
        """Test Point 10: Inventory residual post-D12 propagates via rho, not constant shift."""
        st_a = self.state(day=13, step=335, inventory_overrides={"MILK": 260.0})
        st_b = self.state(day=13, step=335, inventory_overrides={"MILK": 280.0})

        # Persisted D12 anchor at step 311
        ctx = AnimalMarketContext()
        ctx.d12_anchors["MILK"] = (0, 250.0)

        dist_a = forecast_product_distribution(st_a, "MILK", context=ctx)
        dist_b = forecast_product_distribution(st_b, "MILK", context=ctx)

        # At h=0, strictly equal to current inventory
        self.assertEqual(dist_a.inventory_paths[0, 0], 260.0)
        self.assertEqual(dist_b.inventory_paths[0, 0], 280.0)

        diff = dist_b.inventory_paths - dist_a.inventory_paths
        # Difference at h=0 is 20.0
        np.testing.assert_allclose(diff[:, 0], 20.0)
        # Future difference is NOT constant 20 across all horizons; rho decays
        self.assertFalse(np.allclose(diff[:, 10], 20.0))
        self.assertFalse(np.allclose(diff[:, -1], 20.0))

    def test_sequential_revenue_hand_calculation(self):
        """Test Point 10: Hand-calculated case comparing against rules.market_price / sell_revenue."""
        candidate_inv = np.array([
            [100.0, 105.0],
            [200.0, 205.0],
        ])
        quantities = [2, 3]
        step_indices = [0, 1]

        # Scenario 0 manual calculation
        rev_s0_0 = sell_revenue("MILK", 2, 100)
        rev_s0_1 = sell_revenue("MILK", 3, 105)
        tot_s0 = rev_s0_0 + rev_s0_1

        # Scenario 1 manual calculation
        rev_s1_0 = sell_revenue("MILK", 2, 200)
        rev_s1_1 = sell_revenue("MILK", 3, 205)
        tot_s1 = rev_s1_0 + rev_s1_1

        expected_val = 0.5 * (tot_s0 + tot_s1)

        res = scenario_revenue_value("MILK", candidate_inv, quantities, step_indices)
        self.assertAlmostEqual(res["expected"], expected_val)

    def test_excluded_own_tiles_revalues_remaining_animals(self):
        """Test Point 8: When one animal exits, remaining animal valuation uses lowered supply."""
        params = DEFAULT_MIDGAME_PARAMETERS
        for kind, prod in (("COW", "MILK"), ("SHEEP", "WOOL"), ("GOOSE", "EGG")):
            held = 3 if kind == "SHEEP" else 0
            a1 = self.animal(kind, placed_day=3, position=(0, 0))
            a2 = self.animal(kind, placed_day=3, position=(1, 0))
            if held:
                a1.official["yield_units"] = held
                a2.official["yield_units"] = held
            st = self.state(day=12, animals=(a1, a2))

            # Both kept
            val_both = _animal_current_state(st, a2, prior=None, params=params, excluded_own_tiles=frozenset())

            # First animal exits
            val_excluded = _animal_current_state(
                st, a2, prior=None, params=params, excluded_own_tiles=frozenset([a1.position])
            )

            # When a1 exits, its future supply is removed, so market inventory drops and a2 daily value increases
            self.assertGreater(val_excluded.daily_value, val_both.daily_value)

    def test_late_attach_fallback(self):
        """Test Point 1: Agent attaching after D12 without anchor enters late_fallback."""
        st = self.state(day=14, step=336)
        dist = forecast_product_distribution(st, "MILK")
        self.assertEqual(dist.state, "late_fallback")
        self.assertEqual(dist.conditioning_level, "late_fallback")
        self.assertTrue(dist.ood.get("late_fallback", False))
        self.assertEqual(dist.confidence, 0.3)
        self.assertEqual(dist.inventory_paths[0, 0], float(st.market.inventory["MILK"]))

    def test_egg_insufficient_information_fallback(self):
        """Test Point 11: Missing opponent money triggers insufficient_info fallback, not branch 2."""
        st = self.state(day=13, step=311, shops=("BRUNCH_SPOT",), opp_money=None)
        # Ensure state.opp.money is None
        st = replace(st, opp=replace(st.opp, money=None))
        dist = forecast_product_distribution(st, "EGG")
        self.assertEqual(dist.state, "insufficient_info")
        self.assertTrue(dist.ood.get("insufficient_info", False))

    def test_reference_cache_single_computation(self):
        """Test Point 12: Reference distribution is cached and computed only once per (step, product)."""
        ctx = AnimalMarketContext()
        st = self.state(day=11)
        dist1 = forecast_product_distribution(st, "MILK", context=ctx)
        dist2 = forecast_product_distribution(st, "MILK", context=ctx)
        self.assertIs(dist1, dist2)

    def test_batch_schedule_new_cow_multi_batch(self):
        """12.1: New COW with 15 productions returns [(h1,6), (h2,6), (h3,3)], not 15 individual (h,1)."""
        st = self.state(day=0, step=0)
        # 15 productions for COW (first_yield=8, interval=2): end_day = 7 + 14 * 2 = 35
        events = animal_batch_sale_events(st, "COW", is_new=True, end_day=35)
        self.assertEqual(len(events), 3)
        self.assertEqual([q for _, q in events], [6, 6, 3])
        # Verify h1, h2, h3 are the steps when 6th, 12th, and 15th milk completed
        h1 = (7 + 5 * 2 + 1) * 24
        h2 = (7 + 11 * 2 + 1) * 24
        h3 = (7 + 14 * 2 + 1) * 24
        self.assertEqual([h for h, _ in events], [h1, h2, h3])

    def test_batch_schedule_existing_cow_held(self):
        """12.2: Existing COW with held=3 and 10 future productions yields [(h1,6), (h2,6), (h3,1)]."""
        st = self.state(day=0, step=0)
        # 10 productions: end_day = 7 + 9 * 2 = 25
        events = animal_batch_sale_events(
            st, "COW", raw={"placed_day": 0, "yield_units": 3}, end_day=25
        )
        self.assertEqual([q for _, q in events], [6, 6, 1])
        h1 = (7 + 2 * 2 + 1) * 24  # 3rd production completes 6th held
        h2 = (7 + 8 * 2 + 1) * 24  # 9th production completes 12th held
        h3 = (7 + 9 * 2 + 1) * 24  # 10th production completes terminal 1 held
        self.assertEqual([h for h, _ in events], [h1, h2, h3])

    def test_batch_schedule_goose(self):
        """12.3: GOOSE with 10 productions yields [(h1,4), (h2,4), (h3,2)]."""
        st = self.state(day=0, step=0)
        # GOOSE (first_yield=4, interval=1): 10 productions: end_day = 3 + 9 = 12
        events = animal_batch_sale_events(st, "GOOSE", is_new=True, end_day=12)
        self.assertEqual([q for _, q in events], [4, 4, 2])
        h1 = (3 + 3 + 1) * 24  # 4th egg
        h2 = (3 + 7 + 1) * 24  # 8th egg
        h3 = (3 + 9 + 1) * 24  # 10th egg
        self.assertEqual([h for h, _ in events], [h1, h2, h3])

    def test_production_does_not_affect_market_before_sale(self):
        """12.4: Production events do not alter market inventory before batch sale."""
        events = [(30, 3)]
        H = 100
        impact = animal_incremental_market_path(events, H)
        # For all h <= 30, impact must be 0
        self.assertTrue(np.all(impact[:31] == 0))
        # For all h > 30, impact must be 3
        self.assertTrue(np.all(impact[31:] == 3))

    def test_batch_sale_sequential_ordering(self):
        """12.5: Batch sale sequence: rev1 uses pre-sale1 inv; rev2 uses post-sale1 inv."""
        ref_paths = np.full((1, 100), 20.0)
        events = [(30, 6), (60, 4)]
        rev_dict = scenario_batch_sale_value("MILK", ref_paths, events)

        # Revenue 1 uses inventory 20.0
        exp_rev1 = sell_revenue("MILK", 6, 20)
        # Revenue 2 uses inventory 20 + 6 = 26 (containing batch 1 impact)
        exp_rev2 = sell_revenue("MILK", 4, 26)
        total_exp = exp_rev1 + exp_rev2

        self.assertAlmostEqual(rev_dict["expected"], float(total_exp))

    def test_exit_batch_subtraction(self):
        """12.6: Existing COW EXIT removes its batch schedule [(h1,6), (h2,3)] from reference scenario."""
        events = [(30, 6), (60, 3)]
        H = 100
        impact = animal_incremental_market_path(events, H)
        ref_paths = np.full((1, H), 30.0)
        active_inv = ref_paths - impact

        # At h <= 30: unchanged
        self.assertTrue(np.all(active_inv[0, :31] == 30.0))
        # At 30 < h <= 60: subtracted 6 (inv = 24.0)
        self.assertTrue(np.all(active_inv[0, 31:61] == 24.0))
        # At h > 60: subtracted 6 + 3 = 9 (inv = 21.0)
        self.assertTrue(np.all(active_inv[0, 61:] == 21.0))

    def test_shops_5_6_8_empirical_models_continue_working(self):
        """12.7: 5, 6, 8 shops allow empirical forecast for SHEEP, COW, and GOOSE."""
        from src.kaggriculture_agent.planner import _forecast_animal_daily_value
        from src.kaggriculture_agent.current_assets import _animal_current_state
        shops_5 = ("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP")
        shops_6 = shops_5 + ("PET_CAFE",)
        shops_8 = shops_6 + ("SMOOTHIE_SHOP", "FARMERS_MARKET")

        for sh in (shops_5, shops_6, shops_8):
            st = self.state(day=12, step=288, shops=sh)
            for kind in ("COW", "SHEEP", "GOOSE"):
                val = _forecast_animal_daily_value(st, kind, DEFAULT_MIDGAME_PARAMETERS)
                self.assertIsInstance(val, float)
                a = self.animal(kind, placed_day=3, position=(0, 0))
                cur = _animal_current_state(st, a, prior=None, params=DEFAULT_MIDGAME_PARAMETERS)
                self.assertIsInstance(cur.daily_value, float)

    def test_new_episode_context_reset(self):
        """12.8: Old D12 anchor and reference cache do not leak into new episode."""
        from src.kaggriculture_agent.operating import DailyPlanningSession
        session = DailyPlanningSession()
        # Episode 1 reaches D12 (step 311)
        st_ep1_d12 = self.state(day=12, step=311, shops=("BAKERY",), opp_money=5000)
        session.plan_for(st_ep1_d12)
        ctx1 = session._market_contexts[0]
        self.assertTrue(len(ctx1.d12_anchors) > 0 or len(ctx1.reference_cache) > 0)

        # Episode 2 starts at step 0 (state.step < last_step)
        st_ep2_start = self.state(day=0, step=0, shops=(), opp_money=100)
        session.plan_for(st_ep2_start)
        ctx2 = session._market_contexts[0]
        # Must be a fresh context with no old D12 anchors
        self.assertEqual(len(ctx2.d12_anchors), 0)


class GoldenReferenceTests(unittest.TestCase):
    """Test Point 3: Strict golden comparison against supplied reference runtimes."""

    def test_milk_golden_comparison_across_stages(self):
        repo_root = Path(__file__).resolve().parents[1]
        milk_dir = repo_root / "animal_calculation_value_model" / "milk"
        sys.path.insert(0, str(milk_dir))
        from milk_runtime_v3 import ConditionalMarketForecaster
        ref = ConditionalMarketForecaster(str(milk_dir), "milk")
        our = _MilkEggForecaster("MILK")

        # D9, D10, D11, D12 anchor, D12+1d, D12+3d, D12+5d
        steps = [240, 264, 288, 311, 335, 383, 431]
        for step in steps:
            s = {
                "step": step,
                "inventory": 250.0,
                "shops": ["ICE_CREAM_SHOP"],
                "cow_total": 5,
                "cow_age35": 2,
                "cow_age_mean": 5.0,
                "cow_bonus": 20.0,
            }
            if step >= 311:
                s["d12_branch"] = 0
                s["d12_inventory"] = 250.0

            r_ref = ref.forecast(s)
            r_our = our.forecast(s)

            self.assertEqual(r_ref["state"], r_our.state)
            self.assertEqual(r_ref["conditioning_level"], r_our.conditioning_level)
            self.assertEqual(r_ref["scenario_count"], r_our.scenario_count)
            np.testing.assert_allclose(r_ref["inventory_paths"], r_our.inventory_paths)
            np.testing.assert_allclose(r_ref["inventory_q"], r_our.inventory_q)
            self.assertEqual(r_ref["ood"]["reveal_ood"], r_our.ood["reveal_ood"])
            self.assertEqual(r_ref["ood"]["residual_ood"], r_our.ood["residual_ood"])

    def test_egg_golden_comparison_across_stages(self):
        repo_root = Path(__file__).resolve().parents[1]
        egg_dir = repo_root / "animal_calculation_value_model" / "egg"
        sys.path.insert(0, str(egg_dir))
        from egg_runtime_v3 import ConditionalMarketForecaster
        ref = ConditionalMarketForecaster(str(egg_dir), "egg")
        our = _MilkEggForecaster("EGG")

        # D9, D10, D11, D12 anchor, D12+1d, D12+3d, D12+5d
        steps = [240, 264, 288, 311, 335, 383, 431]
        for step in steps:
            s = {
                "step": step,
                "inventory": 300.0,
                "shops": ["BAKERY"],
                "goose_total": 5,
                "opponent_money": 8000,
            }
            if step >= 311:
                s["d12_branch"] = 0
                s["d12_inventory"] = 300.0

            r_ref = ref.forecast(s)
            r_our = our.forecast(s)

            self.assertEqual(r_ref["state"], r_our.state)
            self.assertEqual(r_ref["conditioning_level"], r_our.conditioning_level)
            self.assertEqual(r_ref["scenario_count"], r_our.scenario_count)
            np.testing.assert_allclose(r_ref["inventory_paths"], r_our.inventory_paths)
            np.testing.assert_allclose(r_ref["inventory_q"], r_our.inventory_q)
            self.assertEqual(r_ref["ood"]["reveal_ood"], r_our.ood["reveal_ood"])
            self.assertEqual(r_ref["ood"]["residual_ood"], r_our.ood["residual_ood"])


if __name__ == "__main__":
    unittest.main()
