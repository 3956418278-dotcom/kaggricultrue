"""Contract tests for the pure D5+ programme rules."""

import unittest

from src.kaggriculture_agent.programme import (
    BindingConstraint,
    PrebuildCaps,
    Programme,
    ThirdLandFacts,
    animal_addition_cap,
    block_addition_to_floor,
    carrot_floor,
    d10_animal_floor,
    exact_worker_requirement,
    fertilizer_should_sell,
    goose_bridge_target,
    operating_wheat_floor,
    optional_reduction_order,
    prebuild_count,
    programme_for,
    programme_pace,
    programme_priority,
    projected_next_day_cash,
    reserve_wheat,
    shop_counters,
    strawberry_floor,
    third_land_allowed,
    worker_limit,
)


class ProgrammeRulesTests(unittest.TestCase):
    def test_shop_counters_are_literal(self):
        counters = shop_counters((
            "PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP",
            "YARN_STORE", "BAKERY", "BRUNCH_SPOT",
            "PET_CAFE", "PET_CAFE", "FARMERS_MARKET",
        ))
        self.assertEqual(counters.milk, 3)
        self.assertEqual(counters.yarn, 1)
        self.assertEqual(counters.egg, 2)
        self.assertEqual(counters.strawberry, 4)
        self.assertEqual(counters.wheat, 5)
        self.assertEqual(counters.carrot_units, 5)

    def test_programme_selection_and_goose_bridge(self):
        empty = shop_counters(())
        bakery = shop_counters(("BAKERY",))
        brunches = shop_counters(("BRUNCH_SPOT", "BAKERY"))
        yarn = shop_counters(("YARN_STORE",))
        milk = shop_counters(("SMOOTHIE_SHOP",))
        self.assertEqual(programme_for(empty), Programme.GOOSE_BRIDGE_MIXED)
        self.assertEqual(programme_for(bakery), Programme.EGG_ST_GROWTH)
        self.assertEqual(programme_for(yarn), Programme.WOOL_GROWTH)
        self.assertEqual(programme_for(milk), Programme.DAIRY_GROWTH)
        self.assertEqual(goose_bridge_target(empty), 3)
        self.assertEqual(goose_bridge_target(bakery), 4)
        self.assertEqual(goose_bridge_target(brunches), 6)
        self.assertEqual(goose_bridge_target(milk), 0)

    def test_workforce_deadlines_and_animal_caps(self):
        self.assertEqual([worker_limit(day) for day in range(5, 12)],
                         [10, 10, 11, 11, 12, 12, 10])
        self.assertEqual(exact_worker_requirement(11), 10)
        self.assertIsNone(exact_worker_requirement(10))
        self.assertEqual([animal_addition_cap(day) for day in range(4, 11)],
                         [4, 3, 2, 2, 2, 3, 3])
        with self.assertRaises(ValueError):
            worker_limit(4)

    def test_d10_animal_floor_table(self):
        self.assertEqual(d10_animal_floor(shop_counters(("SMOOTHIE_SHOP",))),
                         (7, 0))
        self.assertEqual(d10_animal_floor(shop_counters(
            ("PIZZA_SHOP", "ICE_CREAM_SHOP"))), (10, 0))
        self.assertEqual(d10_animal_floor(shop_counters(
            ("PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP"))), (12, 0))
        self.assertEqual(d10_animal_floor(shop_counters(("YARN_STORE",))),
                         (0, 8))
        self.assertEqual(d10_animal_floor(shop_counters(
            ("YARN_STORE", "YARN_STORE"))), (0, 13))
        self.assertEqual(d10_animal_floor(shop_counters(
            ("SMOOTHIE_SHOP", "YARN_STORE"))), (6, 5))
        self.assertEqual(d10_animal_floor(shop_counters(
            ("PIZZA_SHOP", "YARN_STORE", "YARN_STORE"))), (6, 8))
        self.assertEqual(d10_animal_floor(shop_counters(
            ("PIZZA_SHOP", "ICE_CREAM_SHOP", "YARN_STORE"))), (9, 5))

    def test_crop_and_wheat_floors(self):
        self.assertEqual(strawberry_floor(
            6, shop_counters(("SMOOTHIE_SHOP",))), 16)
        self.assertEqual(strawberry_floor(
            6, shop_counters(("SMOOTHIE_SHOP", "BRUNCH_SPOT"))), 18)
        self.assertEqual(strawberry_floor(9, shop_counters(
            ("SMOOTHIE_SHOP", "BRUNCH_SPOT", "FARMERS_MARKET"))), 34)
        self.assertEqual(carrot_floor(("PET_CAFE",)), 14)
        self.assertEqual(carrot_floor(("PET_CAFE", "PET_CAFE")), 24)
        self.assertEqual(carrot_floor(("FARMERS_MARKET",)), 6)
        self.assertEqual(carrot_floor(("PET_CAFE", "FARMERS_MARKET")), 18)
        self.assertEqual(operating_wheat_floor(
            7, shop_counters(("ICE_CREAM_SHOP",))), 13)

    def test_typical_lines_and_floors_are_not_caps(self):
        ice = programme_pace(5, ("ICE_CREAM_SHOP",))
        self.assertEqual(ice.programme, Programme.DAIRY_GROWTH)
        self.assertEqual((ice.cow, ice.strawberry, ice.operating_wheat),
                         (6, 10, 12))
        self.assertEqual(ice.combined_cow_sheep, 6)
        yarn_two = programme_pace(7, ("YARN_STORE", "YARN_STORE"))
        self.assertEqual(yarn_two.sheep, 9)
        farmers = programme_pace(5, ("FARMERS_MARKET",))
        self.assertEqual((farmers.strawberry, farmers.carrot), (8, 6))

    def test_prebuild_cash_and_third_land_rules(self):
        self.assertEqual(prebuild_count(PrebuildCaps(5, 4, 3, 2)), 2)
        self.assertEqual(block_addition_to_floor(13, 20), 8)
        self.assertEqual(block_addition_to_floor(20, 20), 0)
        reservation = reserve_wheat((7, 7), "COW", 6, 2)
        self.assertEqual(
            (reservation.position, reservation.candidate_use,
             reservation.preferred_release_day),
            ((7, 7), "COW", 8),
        )
        with self.assertRaises(ValueError):
            reserve_wheat((7, 7), "COW", 6, 4)
        self.assertEqual(projected_next_day_cash(100, 500, 20, 80, 50, 10), 440)
        facts = ThirdLandFacts(
            game_day=7,
            two_land_space_insufficient=True,
            permanent_tiles_needed=8,
            cash_after_purchase=1000,
            day_worker_cost=100,
            next_day_feed_cost=200,
            planned_animal_cost=400,
            mandatory_seed_cost=300,
            has_same_day_layout=True,
        )
        self.assertTrue(third_land_allowed(facts))
        self.assertFalse(third_land_allowed(
            ThirdLandFacts(**{**facts.__dict__, "permanent_tiles_needed": 7})
        ))

    def test_priority_and_failure_vocabulary(self):
        self.assertEqual(programme_priority(("SMOOTHIE_SHOP",))[0], "COW")
        self.assertEqual(programme_priority(("YARN_STORE",))[0], "SHEEP")
        self.assertEqual(programme_priority(("BRUNCH_SPOT",))[:2],
                         ("STRAWBERRY", "GOOSE"))
        self.assertEqual(optional_reduction_order(), (
            "CARE", "MARGINAL_HARVEST", "RESERVATION_WHEAT",
            "LAST_MARGINAL_ASSET",
        ))
        self.assertNotIn("TARGET_REACHED", {item.value for item in BindingConstraint})
        self.assertTrue(fertilizer_should_sell("MELON", 1000, 100))
        self.assertFalse(fertilizer_should_sell("STRAWBERRY", 101, 100))


if __name__ == "__main__":
    unittest.main()
